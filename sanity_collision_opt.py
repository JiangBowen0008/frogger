#!/usr/bin/env python3
"""
Sanity check: starting from warmstart_single (good topology, bad collision),
optimize ONLY to fix collision using the box-grid SDF >= 0 formulation.

- All box-interior grid points: SDF >= 0 (no penetration)
- Fingertip + palm links: min(SDF) ≈ 0 (maintain contact)
- Other finger links: min(SDF) >= 0 (just don't penetrate)
- Regularize joints to stay near initial (preserve topology)

Run: conda run -n frogger python sanity_collision_opt.py
"""
import numpy as np
import trimesh
import torch
import torch.nn.functional as F
import pytorch_kinematics as pk
import os, sys, time
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as ScipyR

sys.path.insert(0, os.path.dirname(__file__))
from frogger.batched_pytorch_solver import BatchedSDF, _link_names

OBJ_MESH = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg/black_spray_bottle_single/object.obj"
HAND_DIR = os.path.join(os.path.dirname(__file__), "models/leap_rh")
URDF_PATH = os.path.join(HAND_DIR, "leap.urdf")
GRASPS = os.path.join(os.path.dirname(__file__), "output/grasps")

# Links where we want contact (min SDF ≈ 0)
CONTACT_LINKS = {
    "leap_rh_palm",
    "leap_rh_if_ds", "leap_rh_mf_ds", "leap_rh_rf_ds", "leap_rh_th_ds",
}

# Joint limits for LEAP
JOINT_LOWER = torch.tensor([
    -0.3142, -0.0471, -0.2793, -0.2967,
    -0.3142, -0.0471, -0.2793, -0.2967,
    -0.3142, -0.0471, -0.2793, -0.2967,
    0.2635, -0.1053, -0.1876, -0.1614,
], dtype=torch.float32)

JOINT_UPPER = torch.tensor([
    2.1817, 1.4961, 1.6929, 1.6144,
    2.1817, 1.4961, 1.6929, 1.6144,
    2.1817, 1.4961, 1.6929, 1.6144,
    1.3963, 1.1631, 1.6720, 1.6720,
], dtype=torch.float32)


def build_box_grids(spacing=0.003):
    """Build uniform grid inside URDF boxes for each link.
    Returns dict: link_name -> tensor [N, 4] homogeneous coords in link frame.
    """
    tree = ET.parse(URDF_PATH)
    grids = {}

    for le in tree.getroot().findall("link"):
        ln = le.get("name")
        boxes = le.findall("collision")
        if not boxes:
            continue

        all_pts = []
        for col in boxes:
            g = col.find("geometry")
            if g is None: continue
            b = g.find("box")
            if b is None: continue
            size = [float(x) for x in b.get("size").split()]
            o = col.find("origin")
            xyz = np.array([float(x) for x in o.get("xyz", "0 0 0").split()])
            rpy = np.array([float(x) for x in o.get("rpy", "0 0 0").split()])

            hx, hy, hz = [s/2 for s in size]
            gx = np.arange(-hx, hx + spacing/2, spacing)
            gy = np.arange(-hy, hy + spacing/2, spacing)
            gz = np.arange(-hz, hz + spacing/2, spacing)
            grid = np.stack(np.meshgrid(gx, gy, gz, indexing='ij'), axis=-1).reshape(-1, 3)

            if np.any(np.abs(rpy) > 1e-6):
                R = ScipyR.from_euler("xyz", rpy).as_matrix()
                grid = (R @ grid.T).T
            grid += xyz
            all_pts.append(grid)

        if all_pts:
            pts = np.vstack(all_pts).astype(np.float32)
            pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
            grids[ln] = pts_h

    return grids


def q_from_u(u, lower, upper):
    """Sigmoid parameterization: u ∈ R → q ∈ [lower, upper]."""
    return lower + (upper - lower) * torch.sigmoid(u)


def u_from_q(q, lower, upper):
    """Inverse sigmoid: q ∈ [lower, upper] → u ∈ R."""
    t = (q - lower) / (upper - lower)
    t = t.clamp(1e-6, 1 - 1e-6)
    return torch.log(t / (1 - t))


def evaluate(sdf, chain, box_grids, q, base_pos, base_rot, device):
    """Evaluate collision metric for a given configuration."""
    fk = chain.forward_kinematics(q.unsqueeze(0))

    T_base = torch.eye(4, device=device)
    T_base[:3, :3] = base_rot
    T_base[:3, 3] = base_pos
    bT = T_base.unsqueeze(0)

    results = {}
    for ln, pts_h_np in box_grids.items():
        if ln not in fk:
            continue
        pts_h = torch.tensor(pts_h_np, device=device)
        link_T = fk[ln].get_matrix().to(device)
        world_T = bT @ link_T
        pts_w = (world_T[0] @ pts_h.T).T[:, :3]
        sdfs = sdf.query(pts_w.unsqueeze(0)).squeeze(0)

        min_sdf = sdfs.min().item()
        n_inside = (sdfs < 0).sum().item()
        results[ln] = (min_sdf, n_inside, len(sdfs))

    return results


def print_status(results, label=""):
    """Print collision status grouped by finger."""
    groups = {
        "palm": ["leap_rh_palm"],
        "IF": [f"leap_rh_if_{s}" for s in ["bs","px","md","ds"]],
        "MF": [f"leap_rh_mf_{s}" for s in ["bs","px","md","ds"]],
        "RF": [f"leap_rh_rf_{s}" for s in ["bs","px","md","ds"]],
        "TH": [f"leap_rh_th_{s}" for s in ["mp","bs","px","ds"]],
    }
    print(f"\n  {label}")
    total_inside = 0
    total_pts = 0
    for gname, links in groups.items():
        g_min = float('inf')
        g_inside = 0
        g_total = 0
        details = []
        for ln in links:
            if ln not in results: continue
            min_sdf, n_in, n_tot = results[ln]
            g_min = min(g_min, min_sdf)
            g_inside += n_in
            g_total += n_tot
            if n_in > 0 or min_sdf < 0.003:
                short = ln.replace("leap_rh_", "")
                is_contact = ln in CONTACT_LINKS
                tag = "CONTACT" if is_contact else ""
                details.append(f"      {short:<10} min={min_sdf*1000:>7.1f}mm  in={n_in}/{n_tot} {tag}")
        total_inside += g_inside
        total_pts += g_total
        if g_total > 0:
            ok = "OK" if g_min >= -0.001 else "BAD"
            print(f"    {gname:<6} min={g_min*1000:>7.1f}mm  in={g_inside}/{g_total}  [{ok}]")
            for d in details:
                print(d)
    print(f"    TOTAL: {total_inside}/{total_pts} inside")


def main():
    device = "cuda"

    # Load object
    obj_mesh = trimesh.load(OBJ_MESH, force="mesh")
    bounds = obj_mesh.bounds
    offset = np.array([0.0, 0.0, -bounds[0, 2]])
    X_WO = np.eye(4); X_WO[:3, 3] = offset
    sdf = BatchedSDF(obj_mesh, X_WO, resolution=128, device=device)

    # FK chain (on GPU)
    with open(URDF_PATH) as f:
        chain = pk.build_chain_from_urdf(f.read()).to(device=device)

    # Box grids
    box_grids = build_box_grids(spacing=0.003)
    total = sum(len(v) for v in box_grids.values())
    print(f"Box grids: {len(box_grids)} links, {total} total points")

    # Load warmstart_best (collision-free, palm far — try to bring into contact)
    data = torch.load(os.path.join(GRASPS, "compare_warmstart_best.pt"),
                      weights_only=False, map_location="cpu")
    g = data[0]
    q_init = torch.tensor(g["q_joints"], dtype=torch.float32, device=device)
    base_pos = torch.tensor(g["base_pos"], dtype=torch.float32, device=device)
    base_rot = torch.tensor(g["base_rot"], dtype=torch.float32, device=device)

    lower = JOINT_LOWER.to(device)
    upper = JOINT_UPPER.to(device)

    # Evaluate BEFORE
    results_before = evaluate(sdf, chain, box_grids, q_init, base_pos, base_rot, device)
    print_status(results_before, "BEFORE optimization")

    # --- Optimization ---
    # Optimize joints + small base adjustment
    u = u_from_q(q_init, lower, upper).detach().requires_grad_(True)
    pos = base_pos.clone().detach().requires_grad_(True)
    # Keep rotation fixed to preserve approach direction
    rot = base_rot.clone().detach()

    # Pre-convert box grids to tensors
    grid_tensors = {}
    for ln, pts_np in box_grids.items():
        grid_tensors[ln] = torch.tensor(pts_np, device=device)

    opt = torch.optim.Adam([u, pos], lr=0.003)
    n_steps = 1500
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps, 1e-5)

    print(f"\n  Optimizing ({n_steps} steps)...")
    t0 = time.time()

    for step in range(n_steps):
        opt.zero_grad()

        q = q_from_u(u, lower, upper)
        fk = chain.forward_kinematics(q.unsqueeze(0))

        T_base = torch.eye(4, device=device)
        T_base[:3, :3] = rot
        T_base[:3, 3] = pos
        bT = T_base.unsqueeze(0)

        L_collision = torch.zeros(1, device=device)
        L_contact = torch.zeros(1, device=device)

        for ln, pts_h in grid_tensors.items():
            if ln not in fk:
                continue

            link_T = fk[ln].get_matrix().to(device)
            world_T = bT @ link_T
            pts_w = (world_T[0] @ pts_h.T).T[:, :3]
            link_sdfs = sdf.query(pts_w.unsqueeze(0)).squeeze(0)

            # Collision: penalize any SDF < 0
            violation = F.relu(-link_sdfs)
            L_collision = L_collision + violation.sum() + 10.0 * violation.max()

            # Contact: for contact links, drive min(SDF) toward 0
            if ln in CONTACT_LINKS:
                min_sdf = link_sdfs.min()
                L_contact = L_contact + min_sdf ** 2

        # Regularization: joints only, no position reg (let contact drive position)
        q_reg = (q - q_init) ** 2
        L_reg = 0.5 * q_reg.sum()

        total = 50.0 * L_collision + 10.0 * L_contact + L_reg
        total.backward()
        opt.step()
        scheduler.step()

        if step % 100 == 0 or step == n_steps - 1:
            with torch.no_grad():
                # Quick collision check
                q_eval = q_from_u(u, lower, upper)
                res = evaluate(sdf, chain, box_grids, q_eval, pos, rot, device)
                n_in = sum(r[1] for r in res.values())
                worst = min(r[0] for r in res.values())
            print(f"    step {step:3d}: L_col={L_collision.item():.4f} "
                  f"L_con={L_contact.item():.4f} L_reg={L_reg.item():.4f} "
                  f"inside={n_in} worst={worst*1000:.1f}mm")

    dt = time.time() - t0
    print(f"  Done in {dt:.1f}s")

    # Evaluate AFTER
    with torch.no_grad():
        q_final = q_from_u(u, lower, upper)
        results_after = evaluate(sdf, chain, box_grids, q_final, pos, rot, device)
    print_status(results_after, "AFTER optimization")

    # Print joint changes
    dq = (q_final - q_init).detach().cpu().numpy()
    dp = (pos - base_pos).detach().cpu().numpy()
    print(f"\n  Joint changes (deg): {np.rad2deg(dq).round(1)}")
    print(f"  Position change (mm): {dp * 1000}")
    print(f"  Position change norm: {np.linalg.norm(dp)*1000:.1f}mm")

    # Save result
    result = {
        "q_joints": q_final.detach().cpu().numpy(),
        "base_pos": pos.detach().cpu().numpy(),
        "base_rot": rot.detach().cpu().numpy(),
        "feasible": True,
    }
    out_path = os.path.join(GRASPS, "compare_collision_fixed.pt")
    torch.save([result], out_path)
    print(f"\n  Saved to {out_path}")
    print(f"  Run: conda run -n frogger python diagnose_grasp.py --grasp {out_path}")


if __name__ == "__main__":
    main()
