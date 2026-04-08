"""Warm-start from actuation-only result + add surface constraints."""
import torch
import numpy as np
import pytorch_kinematics as pk
import trimesh
import sys
import json
import subprocess

sys.path.insert(0, ".")
from frogger.batched_pytorch_solver import (
    _LEAP_JOINT_LOWER, _LEAP_JOINT_UPPER, BatchedSDF,
)
from constrained_grasp_solver import project_tips_to_surface

np.set_printoptions(precision=3, suppress=True)

chain = pk.build_chain_from_urdf(open("models/leap_rh/leap.urdf").read())
q_lo = torch.tensor(_LEAP_JOINT_LOWER)
q_hi = torch.tensor(_LEAP_JOINT_UPPER)
mesh = trimesh.load(
    "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg/black_spray_bottle_single/object.obj",
    force="mesh",
)
bounds = mesh.bounds
offset = np.array([0.0, 0.0, -bounds[0, 2]])
X_WO = np.eye(4)
X_WO[:3, 3] = offset
sdf = BatchedSDF(mesh, X_WO, resolution=128, device="cuda")

act_pos = torch.tensor([0.039, 0.00033, 0.137])
act_dir = torch.tensor([-0.946, 0.265, -0.184])
act_dir = act_dir / act_dir.norm()
target_dir = -act_dir

tip_offsets = [torch.tensor([-0.0025, -0.0449, 0.0143])] * 3 + [
    torch.tensor([-0.002, -0.0558, -0.0144])
]
tip_links = [
    "leap_rh_if_ds", "leap_rh_mf_ds", "leap_rh_rf_ds", "leap_rh_th_ds",
]

# Load warm start
ws = torch.load("output/grasps/spray_bottle_good_warmstart.pt", weights_only=False)[0]
u_init = torch.log(
    (torch.tensor(ws["q_joints"]) - q_lo) / (q_hi - torch.tensor(ws["q_joints"]))
).clamp(-5, 5)
R_ws = ws["base_rot"]
pos_ws = ws["base_pos"]
r6d_ws = torch.tensor(
    np.concatenate([R_ws[:, 0], R_ws[:, 1]]), dtype=torch.float32
)


def rot6d_to_matrix(r):
    a1, a2 = r[:, :3], r[:, 3:]
    b1 = a1 / a1.norm(dim=-1, keepdim=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = b2 / b2.norm(dim=-1, keepdim=True)
    return torch.stack([b1, b2, torch.cross(b1, b2, dim=-1)], dim=-1)


results = []
for trial in range(10):
    torch.manual_seed(trial * 7 + 1)
    # Initialize IF at the specific curl that reaches actuation with correct dir
    # Found via exhaustive search: axl=0.2, mcp=0.67, pip=0.7, dip=0.7
    u_start = u_init.clone()
    # Convert percentage to u-space via inverse sigmoid
    if_pcts = torch.tensor([0.2, 0.667, 0.7, 0.7])
    u_start[:4] = torch.log(if_pcts / (1.0 - if_pcts))  # logit
    u = (u_start.unsqueeze(0) + 0.05 * torch.randn(1, 16)).requires_grad_(True)
    pos = (
        torch.tensor(pos_ws).unsqueeze(0) + 0.003 * torch.randn(1, 3)
    ).requires_grad_(True)
    rot6d = (r6d_ws.unsqueeze(0) + 0.03 * torch.randn(1, 6)).requires_grad_(True)

    # FREEZE base pose — only optimize joints
    opt = torch.optim.Adam([u], lr=0.005)

    for s in range(1500):
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
            tips.append(
                bT[0, :3, :3] @ (T[:3, :3] @ off + T[:3, 3]) + bT[0, :3, 3]
            )
            tip_x.append(bT[0, :3, :3] @ T[:3, 0])

        # ACTUATION (high weight)
        L = 200 * ((tips[0] - act_pos) ** 2).sum()
        L += 40 * (1 - (tip_x[0] * target_dir).sum()) ** 2

        # Surface for MF/RF/TH (strong)
        non_if = torch.stack(tips[1:]).unsqueeze(0)
        ts_noif = sdf.query(non_if.cuda()).cpu()
        L += 300 * (ts_noif ** 2).sum() + 100 * ts_noif.abs().sum()
        L += 500 * ts_noif.abs().max()

        # Palm contact at margin
        palm_R_np = np.array([[-0, 0, -1], [0, 1, 0], [1, 0, -0]])
        palm_t_np = np.array([0, 0.035, 0.1])
        pp_link = np.array([
            [-0.03, -0.03, 0], [-0.05, -0.03, 0], [-0.07, -0.03, 0],
            [-0.03, 0.01, 0], [-0.05, 0.01, 0], [-0.07, 0.01, 0],
        ])
        pp_base = torch.tensor(
            (palm_R_np @ pp_link.T).T + palm_t_np, dtype=torch.float32,
        )
        pw = (bT[0, :3, :3] @ pp_base.T).T + bT[0, :3, 3]
        ps = sdf.query(pw.unsqueeze(0).cuda()).cpu()
        L += 60 * ((ps - 0.008) ** 2).sum()  # palm at 8mm margin (prevents deep pen)

        # Palm anti-penetration
        pp_dense_link = []
        for px in np.linspace(-0.01, -0.09, 8):
            for py in np.linspace(-0.06, 0.02, 6):
                pp_dense_link.append([px, py, 0])
        pp_dense_base = torch.tensor(
            (palm_R_np @ np.array(pp_dense_link).T).T + palm_t_np,
            dtype=torch.float32,
        )
        pw_d = (bT[0, :3, :3] @ pp_dense_base.T).T + bT[0, :3, 3]
        ps_d = sdf.query(pw_d.unsqueeze(0).cuda()).cpu()
        L += 500 * torch.relu(-ps_d - 0.001).sum()  # VERY strong anti-pen

        # Tip penetration
        all_ts = sdf.query(torch.stack(tips).unsqueeze(0).cuda()).cpu()
        L += 50 * torch.relu(-all_ts - 0.001).sum()

        # Link body collision (capsule model)
        from constrained_grasp_solver import CapsuleModel
        if not hasattr(project_tips_to_surface, '_cap') or trial == 0 and s == 0:
            project_tips_to_surface._cap = CapsuleModel("rh", "leap", device="cpu")
        # Exclude only IF links (must reach actuation on surface)
        # Keep all other links including ds for collision — tips ARE projected
        # to surface but the link bodies behind them shouldn't penetrate
        _, cap_pen = project_tips_to_surface._cap.query_collision(
            fk, bT, sdf, exclude_links=["if_"]
        )
        L += 100 * cap_pen

        # Position/rotation reg (STRONG to keep palm on correct side)
        L += 80 * ((pos - torch.tensor(pos_ws)) ** 2).sum()
        L += 50 * ((rot6d - r6d_ws) ** 2).sum()

        L.backward()
        opt.step()

        # Project MF/RF/TH to surface
        if s % 5 == 0 and s > 50:
            project_tips_to_surface(
                chain, u, bT.detach(), sdf, q_lo, q_hi,
                tip_links, tip_offsets,
                skip_indices=[0], n_steps=3, lr=2.0,
            )

    with torch.no_grad():
        q_f = q_lo + torch.sigmoid(u) * (q_hi - q_lo)
        R_f = rot6d_to_matrix(rot6d)[0].numpy()
        pos_f = pos[0].numpy()
        fk_f = chain.forward_kinematics(q_f)
        tips_w = []
        for link, off in zip(tip_links, tip_offsets):
            T = fk_f[link].get_matrix()[0].numpy()
            tips_w.append(R_f @ (T[:3, :3] @ off.numpy() + T[:3, 3]) + pos_f)
        d = np.linalg.norm(tips_w[0] - act_pos.numpy())
        dot = np.dot(
            R_f @ fk_f[tip_links[0]].get_matrix()[0].numpy()[:3, 0],
            target_dir.numpy(),
        )
        tp_t = torch.tensor(np.array(tips_w), dtype=torch.float32, device="cuda")
        tip_sdf = sdf.query(tp_t.unsqueeze(0))[0].cpu().numpy()
        sdf_str = " ".join(f"{s*1000:.0f}" for s in tip_sdf)
        print(
            f"#{trial}: act={d*1000:.1f}mm dir={dot:.2f} "
            f"sdf=[{sdf_str}]mm",
            flush=True,
        )
        results.append({
            "q_joints": q_f[0].numpy(),
            "base_pos": pos_f.astype(np.float32),
            "base_rot": R_f.astype(np.float64),
            "l_star": 0.0, "feasible": True, "score": float(-d),
            "sigma_min": 0.0, "act_assignment": [0],
            "act_dist": float(d), "surf_err": 0.0,
            "min_col": 0.0, "sc_min_dist": 0.0,
        })

results.sort(key=lambda r: -r["score"])
torch.save(results[:10], "output/grasps/spray_bottle_warmstart.pt")
print("\nSaved.", flush=True)

subprocess.run([
    "conda", "run", "-n", "frogger", "python", "-u", "diagnose_grasp.py",
    "--grasp", "output/grasps/spray_bottle_warmstart.pt",
    "--output_dir", "output/diagnostics_warmstart", "--tag", "ws",
])
