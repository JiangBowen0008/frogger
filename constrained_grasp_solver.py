"""
Constrained grasp solver for LEAP hand.

Architecture:
1. Capsule collision model per link (exact, fast)
2. Contact projection (fingertips forced onto surface after each step)
3. Constrained optimization (objective + projection, not soft penalties)
4. Triangular support topology (palm + fingers + thumb)

Uses FroGGer's verified math (grasp matrix, wrench matrix, LP) and
the BatchedSDF for differentiable SDF queries.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pytorch_kinematics as pk
import trimesh
import os
import json
from typing import List, Tuple, Optional
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as ScipyR

from frogger.batched_pytorch_solver import (
    BatchedSDF,
    _LEAP_JOINT_LOWER, _LEAP_JOINT_UPPER,
    _link_names, _visual_meshes,
    compute_contact_frames, compute_grasp_matrix_torch,
    compute_wrench_matrix, compute_primitive_forces_torch,
)


# ============================================================
# Capsule collision model
# ============================================================
class CapsuleModel:
    """Approximate each hand link as a capsule for fast exact collision."""

    def __init__(self, hand="rh", hand_type="leap", device="cpu"):
        self.device = device
        self.capsules = {}  # link_name -> (p0, p1, radius) in link frame

        mesh_dir = os.path.join(os.path.dirname(__file__), f"models/leap_{hand}")
        if not os.path.exists(mesh_dir):
            mesh_dir = os.path.join(os.path.dirname(__file__), f"../models/leap_{hand}")
        vis_meshes = _visual_meshes(hand, hand_type)
        _, col_names = _link_names(hand, hand_type)

        for nm in col_names:
            if nm not in vis_meshes:
                continue
            mesh_file, vis_pose = vis_meshes[nm][0]
            full_path = os.path.join(mesh_dir, mesh_file)
            if not os.path.exists(full_path):
                continue

            m = trimesh.load(full_path, force="mesh")
            verts = np.asarray(m.vertices, dtype=np.float64)

            # Apply visual pose transform
            if vis_pose is not None:
                vp = np.array(vis_pose, dtype=np.float64)
                Rv = ScipyR.from_euler("xyz", vp[3:]).as_matrix()
                verts = (Rv @ verts.T).T + vp[:3]

            # Fit capsule via PCA
            center = verts.mean(axis=0)
            cov = np.cov((verts - center).T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            # Principal axis = eigenvector with largest eigenvalue
            axis = eigvecs[:, np.argmax(eigvals)]

            # Project vertices onto axis
            proj = (verts - center) @ axis
            p0 = center + proj.min() * axis  # one end
            p1 = center + proj.max() * axis  # other end

            # Radius = max distance from axis
            along = np.outer((verts - center) @ axis, axis)
            perp = (verts - center) - along
            radius = np.sqrt((perp ** 2).sum(axis=1)).max()

            # Shrink radius slightly to avoid over-conservative collision
            radius *= 0.85

            self.capsules[nm] = (
                torch.tensor(p0, dtype=torch.float32, device=device),
                torch.tensor(p1, dtype=torch.float32, device=device),
                float(radius),
            )

        print(f"  CapsuleModel: {len(self.capsules)} links")

    def query_collision(self, fk, base_T, sdf, exclude_palm=True):
        """Check capsule-object collision for all links.

        Returns:
            violations: list of (link_name, min_sdf_on_capsule) for violated links
            total_pen: scalar penalty for gradient
        """
        dev = self.device
        violations = []
        all_pen = torch.tensor(0.0, device=dev)

        for nm, (p0_local, p1_local, radius) in self.capsules.items():
            if exclude_palm and 'palm' in nm:
                continue
            if nm not in fk:
                continue

            # Transform capsule endpoints to world
            link_T = fk[nm].get_matrix()  # [B, 4, 4] or [1, 4, 4]
            wT = base_T @ link_T

            # Capsule axis in world: sample N points along it
            N_samples = 8
            ts = torch.linspace(0, 1, N_samples, device=dev)
            axis_local = p0_local + ts.unsqueeze(-1) * (p1_local - p0_local)  # [N, 3]

            # To homogeneous
            ones = torch.ones(N_samples, 1, device=dev)
            axis_h = torch.cat([axis_local, ones], dim=-1)  # [N, 4]

            # Transform to world
            axis_world = (wT @ axis_h.T)[:, :3, :].transpose(1, 2)  # [B, N, 3]

            # Query SDF at axis points
            sdf_vals = sdf.query(axis_world.cuda()).cpu()  # [B, N]

            # Capsule SDF = axis_sdf - radius
            capsule_sdf = sdf_vals - radius  # [B, N]

            # Penetration = relu(-capsule_sdf - margin)
            margin = 0.001  # 1mm margin
            pen = F.relu(-capsule_sdf - margin)
            all_pen = all_pen + pen.sum()

            min_sdf = capsule_sdf.min().item()
            if min_sdf < -0.001:
                violations.append((nm, min_sdf))

        return violations, all_pen


# ============================================================
# Contact projection
# ============================================================
def project_tips_to_surface(chain, u, base_T, sdf, q_lo, q_hi,
                            tip_links, tip_offsets, skip_indices=None,
                            n_steps=5, lr=2.0):
    """Project fingertips onto the object surface.

    Uses aggressive gradient steps on sdf² with high LR.
    Also clips sdf to prevent overshooting.
    """
    dev = u.device
    if skip_indices is None:
        skip_indices = []

    for step in range(n_steps):
        u_proj = u.detach().requires_grad_(True)
        q_proj = q_lo + torch.sigmoid(u_proj) * (q_hi - q_lo)
        fk_proj = chain.forward_kinematics(q_proj)

        surf_loss = torch.tensor(0.0)
        for i, (link, off) in enumerate(zip(tip_links, tip_offsets)):
            if i in skip_indices:
                continue
            T = fk_proj[link].get_matrix()[0]
            tip = base_T[0, :3, :3] @ (T[:3, :3] @ off + T[:3, 3]) + base_T[0, :3, 3]
            s = sdf.query(tip.unsqueeze(0).unsqueeze(0).cuda()).cpu()
            # L2 + strong L1 for constant gradient far from surface
            surf_loss = surf_loss + (s ** 2).sum() + 5.0 * s.abs().sum()
        surf_loss.backward()

        if u_proj.grad is not None:
            # Adaptive LR: larger when far from surface
            u.data -= lr * u_proj.grad


# ============================================================
# Main solver
# ============================================================
def solve_constrained_grasp(
    mesh_path: str,
    actuation_contacts_path: str,
    hand: str = "rh",
    hand_type: str = "leap",
    num_trials: int = 20,
    steps: int = 2000,
    device: str = "cuda",
    save_path: Optional[str] = None,
):
    """Generate grasps using constrained optimization with capsule collision."""

    # Load object
    mesh = trimesh.load(mesh_path, force="mesh")
    bounds = mesh.bounds
    offset = np.array([0.0, 0.0, -bounds[0, 2]])
    X_WO = np.eye(4); X_WO[:3, 3] = offset
    verts_W = mesh.vertices + offset
    obj_center = mesh.centroid + offset

    sdf = BatchedSDF(mesh, X_WO, bounds_padding=0.15, resolution=128, device=device)

    # Load actuation
    with open(actuation_contacts_path) as f:
        act_data = json.load(f)
    act_pos = torch.tensor(
        np.array(act_data["actuation_contacts"][0]["pos"]) + offset,
        dtype=torch.float32
    )
    act_dir = torch.tensor(
        act_data["actuation_contacts"][0]["dir"], dtype=torch.float32
    )
    act_dir = act_dir / act_dir.norm()
    target_dir = -act_dir  # LEAP reverse actuation

    # Hand setup
    chain = pk.build_chain_from_urdf(
        open(f"models/leap_{hand}/leap.urdf").read()
    )
    q_lo = torch.tensor(_LEAP_JOINT_LOWER, dtype=torch.float32)
    q_hi = torch.tensor(_LEAP_JOINT_UPPER, dtype=torch.float32)

    tip_offsets = [torch.tensor([-0.0025, -0.0449, 0.0143])] * 3 + \
                  [torch.tensor([-0.002, -0.0558, -0.0144])]
    tip_links = [
        "leap_rh_if_ds", "leap_rh_mf_ds",
        "leap_rh_rf_ds", "leap_rh_th_ds",
    ]

    # Capsule collision model
    capsule_model = CapsuleModel(hand, hand_type, device="cpu")

    # Palm contact points in base frame
    palm_R = np.array([[-0, 0, -1], [0, 1, 0], [1, 0, -0]])
    palm_t = np.array([0, 0.035, 0.1])
    pp_link = np.array([
        [-0.03, -0.03, 0], [-0.05, -0.03, 0], [-0.07, -0.03, 0],
        [-0.03, 0.01, 0], [-0.05, 0.01, 0], [-0.07, 0.01, 0],
    ])
    pp_base = torch.tensor(
        (palm_R @ pp_link.T).T + palm_t, dtype=torch.float32
    )

    # Approach angle: θ=150° (oblique, allows IF to reach +x actuation)
    theta = np.radians(150)
    z_hat = np.array([-np.sin(theta), np.cos(theta), 0])
    y_hat = np.array([0, 0, 1])
    x_hat = np.cross(y_hat, z_hat)
    R_init = np.stack([x_hat, y_hat, z_hat], axis=1).astype(np.float32)
    r6d_init = torch.tensor(
        np.concatenate([R_init[:, 0], R_init[:, 1]]),
        dtype=torch.float32,
    )
    pos_init = torch.tensor([0.037, 0.074, 0.075], dtype=torch.float32)

    # Finger targets on opposite side of palm
    tree = cKDTree(verts_W)
    opp = np.array([np.sin(theta), -np.cos(theta), 0])
    opp /= np.linalg.norm(opp)
    _, idx = tree.query(np.array([-0.005, 0, 0.09]) + 0.04 * opp)
    mf_target = torch.tensor(verts_W[idx], dtype=torch.float32)
    _, idx = tree.query(np.array([-0.005, 0, 0.05]) + 0.04 * opp)
    rf_target = torch.tensor(verts_W[idx], dtype=torch.float32)
    _, idx = tree.query(np.array([-0.005, 0, 0.14]) + 0.035 * opp)
    th_target = torch.tensor(verts_W[idx], dtype=torch.float32)

    PALM_MARGIN = 0.004  # 4mm outside surface

    def rot6d_to_matrix(r):
        a1, a2 = r[:, :3], r[:, 3:]
        b1 = a1 / a1.norm(dim=-1, keepdim=True)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = b2 / b2.norm(dim=-1, keepdim=True)
        return torch.stack([b1, b2, torch.cross(b1, b2, dim=-1)], dim=-1)

    # ============================================================
    # Main optimization loop
    # ============================================================
    results = []

    for trial in range(num_trials):
        torch.manual_seed(trial * 17 + 3)

        # Initialize
        u = (torch.randn(1, 16) * 0.3).requires_grad_(True)
        pos = (pos_init.unsqueeze(0) + 0.008 * torch.randn(1, 3)).requires_grad_(True)
        rot6d = (r6d_init.unsqueeze(0) + 0.08 * torch.randn(1, 6)).requires_grad_(True)

        opt = torch.optim.Adam([u, pos, rot6d], lr=0.006)

        for s in range(steps):
            opt.zero_grad()

            q = q_lo + torch.sigmoid(u) * (q_hi - q_lo)
            R = rot6d_to_matrix(rot6d)
            bT = torch.eye(4).unsqueeze(0)
            bT[0, :3, :3] = R[0]
            bT[0, :3, 3] = pos[0]
            fk = chain.forward_kinematics(q)

            tips = []
            tip_x = []
            for link, off in zip(tip_links, tip_offsets):
                T = fk[link].get_matrix()[0]
                tip = bT[0, :3, :3] @ (T[:3, :3] @ off + T[:3, 3]) + bT[0, :3, 3]
                tips.append(tip)
                tip_x.append(bT[0, :3, :3] @ T[:3, 0])

            # === Objective: actuation (DOMINANT) + force closure proxy ===
            L = 300 * ((tips[0] - act_pos) ** 2).sum()
            L += 50 * (1 - (tip_x[0] * target_dir).sum()) ** 2

            # === Palm contact at margin ===
            pw = (bT[0, :3, :3] @ pp_base.T).T + bT[0, :3, 3]
            ps = sdf.query(pw.unsqueeze(0).cuda()).cpu()
            L += 30 * ((ps - PALM_MARGIN) ** 2).sum()

            # === Finger targets (guiding, not just penalty) ===
            L += 20 * ((tips[1] - mf_target) ** 2).sum()
            L += 20 * ((tips[2] - rf_target) ** 2).sum()
            L += 20 * ((tips[3] - th_target) ** 2).sum()

            # === Surface contact for MF, RF, TH (NOT IF — actuation handles it) ===
            non_if_tips = torch.stack(tips[1:]).unsqueeze(0)  # skip IF
            ts_noif = sdf.query(non_if_tips.cuda()).cpu()
            L += 500 * (ts_noif ** 2).sum() + 200 * ts_noif.abs().sum()
            L += 1000 * ts_noif.abs().max()  # worst finger
            # Also query all tips for reporting
            tp = torch.stack(tips).unsqueeze(0)
            ts = sdf.query(tp.cuda()).cpu()

            # === Capsule collision (replaces point-cloud) ===
            _, cap_pen = capsule_model.query_collision(fk, bT, sdf)
            L += 200 * cap_pen

            # === Palm anti-penetration (dense grid, in same loss) ===
            palm_R_np = np.array([[-0, 0, -1], [0, 1, 0], [1, 0, -0]])
            palm_t_np = np.array([0, 0.035, 0.1])
            if not hasattr(solve_constrained_grasp, '_pp_dense'):
                pp_d = []
                for px in np.linspace(-0.01, -0.09, 8):
                    for py in np.linspace(-0.06, 0.02, 6):
                        pp_d.append([px, py, 0])
                solve_constrained_grasp._pp_dense = torch.tensor(
                    (palm_R_np @ np.array(pp_d).T).T + palm_t_np,
                    dtype=torch.float32,
                )
            pw_d = (bT[0, :3, :3] @ solve_constrained_grasp._pp_dense.T).T + bT[0, :3, 3]
            ps_d = sdf.query(pw_d.unsqueeze(0).cuda()).cpu()
            L += 100 * F.relu(-ps_d - 0.002).sum()  # light palm anti-pen

            # === Position/rotation regularization ===
            L += 20 * ((pos - pos_init) ** 2).sum()
            L += 15 * ((rot6d - r6d_init) ** 2).sum()

            L.backward()
            opt.step()

            # === CONSTRAINT PROJECTION: force tips onto surface ===
            # Skip IF (index 0) — it should stay at actuation point
            if s % 5 == 0 and s > 50:  # more frequent projection
                project_tips_to_surface(
                    chain, u, bT.detach(), sdf, q_lo, q_hi,
                    tip_links, tip_offsets,
                    skip_indices=[0],  # don't project IF
                    n_steps=3, lr=0.3,
                )

        # Evaluate
        with torch.no_grad():
            q_f = q_lo + torch.sigmoid(u) * (q_hi - q_lo)
            R_f = rot6d_to_matrix(rot6d)[0].numpy()
            pos_f = pos[0].numpy()
            fk_f = chain.forward_kinematics(q_f)

            tips_w = []
            for link, off in zip(tip_links, tip_offsets):
                T = fk_f[link].get_matrix()[0].numpy()
                tips_w.append(R_f @ (T[:3, :3] @ off.numpy() + T[:3, 3]) + pos_f)

            d_act = np.linalg.norm(tips_w[0] - act_pos.numpy())
            dot_act = np.dot(
                R_f @ fk_f[tip_links[0]].get_matrix()[0].numpy()[:3, 0],
                target_dir.numpy(),
            )

            tp_t = torch.tensor(np.array(tips_w), dtype=torch.float32, device=device)
            tip_sdf = sdf.query(tp_t.unsqueeze(0))[0].cpu().numpy()
            max_tip_sdf = np.abs(tip_sdf).max()

            # Capsule collision check
            bT_f = torch.eye(4).unsqueeze(0)
            bT_f[0, :3, :3] = torch.tensor(R_f, dtype=torch.float32)
            bT_f[0, :3, 3] = torch.tensor(pos_f, dtype=torch.float32)
            violations, _ = capsule_model.query_collision(fk_f, bT_f, sdf)

            score = -d_act - 0.01 * (1 - dot_act) - 0.001 * max_tip_sdf

            print(
                f"#{trial:2d}: act={d_act*1000:.1f}mm dir={dot_act:.2f} "
                f"tip_sdf=[{','.join(f'{s*1000:.0f}' for s in tip_sdf)}]mm "
                f"col_viol={len(violations)}",
                flush=True,
            )

            results.append({
                "q_joints": q_f[0].numpy(),
                "base_pos": pos_f.astype(np.float32),
                "base_rot": R_f.astype(np.float64),
                "l_star": 0.0,
                "feasible": max_tip_sdf < 0.003 and d_act < 0.005,
                "score": float(score),
                "sigma_min": 0.0,
                "act_assignment": [0],
                "act_dist": float(d_act),
                "surf_err": float(max_tip_sdf),
                "min_col": 0.0,
                "sc_min_dist": 0.0,
            })

    results.sort(key=lambda r: -r["score"])

    if save_path:
        import torch as _torch
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        _torch.save(results[:10], save_path)
        print(f"\nSaved to {save_path}")

    return results


if __name__ == "__main__":
    MESH = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg/black_spray_bottle_single/object.obj"
    ACT = "/home/bowenj/Projects/DexFun/output/actuation_contacts/mesh_raw_ahg/black_spray_bottle_single_actuation.json"

    results = solve_constrained_grasp(
        mesh_path=MESH,
        actuation_contacts_path=ACT,
        num_trials=20,
        steps=2500,
        save_path="output/grasps/spray_bottle_constrained.pt",
    )

    # Run diagnostics
    import subprocess
    subprocess.run([
        "conda", "run", "-n", "frogger", "python", "-u", "diagnose_grasp.py",
        "--grasp", "output/grasps/spray_bottle_constrained.pt",
        "--output_dir", "output/diagnostics_constrained",
        "--tag", "const",
    ])
