#!/usr/bin/env python3
"""
Gradient-based IK solver for LEAP RH grasping a spray bottle.

Produces a single correct grasp with:
- Palm on -y side, inner surface contacting bottle body
- IF finger curling to reach actuation point [0.039, 0, 0.137]
- MF, RF curling around to +y side of bottle
- Thumb at bottle neck on +y side opposing palm
- No significant penetration

After solving, runs diagnose_grasp.py to verify.

Usage:
    conda run -n frogger python solve_spray_bottle_ik.py
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_kinematics as pk
import trimesh

sys.path.insert(0, os.path.dirname(__file__))
from frogger.batched_pytorch_solver import (
    BatchedSDF,
    _visual_meshes,
    _link_names,
    _LEAP_JOINT_LOWER,
    _LEAP_JOINT_UPPER,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MESH_PATH = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg/black_spray_bottle_single/object.obj"
ACTUATION_JSON = "/home/bowenj/Projects/DexFun/output/actuation_contacts/mesh_raw_ahg/black_spray_bottle_single_actuation.json"
URDF_PATH = os.path.join(os.path.dirname(__file__), "models/leap_rh/leap.urdf")
SAVE_PATH = os.path.join(os.path.dirname(__file__), "output/grasps/spray_bottle_handcrafted.pt")
DIAG_DIR = os.path.join(os.path.dirname(__file__), "output/diagnostics")

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Joint limits
Q_LO = torch.tensor(_LEAP_JOINT_LOWER, dtype=torch.float32, device=DEVICE)
Q_HI = torch.tensor(_LEAP_JOINT_UPPER, dtype=torch.float32, device=DEVICE)
Q_RANGE = Q_HI - Q_LO


def u2q(u):
    return Q_LO + torch.sigmoid(u) * Q_RANGE


def q2u(q):
    n = ((q - Q_LO) / Q_RANGE).clamp(0.01, 0.99)
    return torch.log(n / (1.0 - n))


def load_scene():
    """Load object mesh, compute SDF, set up FK chain."""
    # Object
    mesh = trimesh.load(MESH_PATH, force="mesh")
    bounds = mesh.bounds
    offset = np.array([0.0, 0.0, -bounds[0, 2]])
    X_WO = np.eye(4)
    X_WO[:3, 3] = offset
    obj_center = mesh.centroid + offset

    # Actuation target in world frame
    with open(ACTUATION_JSON) as f:
        act_data = json.load(f)
    act_contact = act_data["actuation_contacts"][0]
    act_pos = np.array(act_contact["pos"]) + offset
    act_dir = np.array(act_contact["dir"])
    act_dir = act_dir / np.linalg.norm(act_dir)

    print(f"Object bounds (world): z=[0, {bounds[1,2]-bounds[0,2]:.4f}]")
    print(f"Object center (world): {obj_center}")
    print(f"Actuation pos (world): {act_pos}")
    print(f"Actuation dir: {act_dir}")

    # SDF
    print("Building SDF...")
    sdf = BatchedSDF(mesh, X_WO, resolution=128, device=DEVICE)

    # FK chain
    with open(URDF_PATH) as f:
        chain = pk.build_chain_from_urdf(f.read()).to(device=DEVICE)

    return mesh, X_WO, offset, obj_center, act_pos, act_dir, sdf, chain


def compute_tip_positions_batch(chain, q, base_T, device):
    """Compute fingertip positions and pad x-axes for batch of configs.

    Returns:
        tip_pos: [B, 4, 3]
        tip_x: [B, 4, 3]
    """
    fk = chain.forward_kinematics(q)
    tip_names = ["leap_rh_if_ds", "leap_rh_mf_ds", "leap_rh_rf_ds", "leap_rh_th_ds"]
    f_off = torch.tensor([-0.0025, -0.0449, 0.0143], dtype=torch.float32, device=device)
    t_off = torch.tensor([-0.0020, -0.0558, -0.0144], dtype=torch.float32, device=device)
    offsets = [f_off, f_off, f_off, t_off]

    tips = []
    x_axes = []
    for name, off in zip(tip_names, offsets):
        wT = base_T @ fk[name].get_matrix()  # [B, 4, 4]
        oh = torch.cat([off, torch.ones(1, device=device)])
        pos = (wT @ oh.unsqueeze(-1)).squeeze(-1)[:, :3]
        tips.append(pos)
        x_axes.append(wT[:, :3, 0])
    return torch.stack(tips, dim=1), torch.stack(x_axes, dim=1), fk


def compute_palm_points_batch(chain, q, base_T, fk, device):
    """Compute palm inner surface sample points in world frame.

    Returns: [B, N, 3] palm surface points
    """
    # Grid of points on palm inner surface (base frame)
    # Palm inner surface faces +x, center ~[0.028, 0, 0.05], finger bases at z~0.093
    gy = torch.linspace(-0.025, 0.025, 5, device=device)
    gz = torch.linspace(0.03, 0.08, 5, device=device)
    yy, zz = torch.meshgrid(gy, gz, indexing='ij')
    # +x surface at about x=0.028
    pp = torch.stack([torch.full_like(yy, 0.028), yy, zz], dim=-1).reshape(-1, 3)
    palm_h = torch.cat([pp, torch.ones(pp.shape[0], 1, device=device)], dim=-1)

    # Points are in BASE frame, so apply base-to-world transform directly
    # (NOT through palm FK — that would double-transform)
    palm_world = (base_T @ palm_h.T)[:, :3, :].transpose(1, 2)
    return palm_world


_COL_CACHE = {}

def precompute_collision_points(device):
    """Load and cache collision points in link-local frames (call ONCE)."""
    if _COL_CACHE:
        return
    from scipy.spatial.transform import Rotation as ScipyR
    mesh_dir = os.path.join(os.path.dirname(__file__), "models/leap_rh")
    vis_meshes = _visual_meshes("rh", "leap")
    _, col_names = _link_names("rh", "leap")

    for nm in col_names:
        if nm not in vis_meshes:
            continue
        mesh_file, vis_pose = vis_meshes[nm][0]
        full_path = os.path.join(mesh_dir, mesh_file)
        if not os.path.exists(full_path):
            continue
        lm = trimesh.load(full_path, force="mesh")
        verts = np.asarray(lm.vertices, dtype=np.float64)
        if vis_pose is not None:
            vp = np.array(vis_pose, dtype=np.float64)
            Rv = ScipyR.from_euler("xyz", vp[3:]).as_matrix()
            verts = (Rv @ verts.T).T + vp[:3]
        n_pts = min(32, len(verts))
        idx = np.random.choice(len(verts), n_pts, replace=False)
        pts = verts[idx].astype(np.float32)
        pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
        _COL_CACHE[nm] = torch.tensor(pts_h, device=device)
    print(f"  Precomputed collision points for {len(_COL_CACHE)} links")


def compute_collision_points_batch(chain, q, base_T, fk, device):
    """Transform precomputed collision points to world frame (fast)."""
    if not _COL_CACHE:
        precompute_collision_points(device)
    all_pts = []
    for nm, pts_t in _COL_CACHE.items():
        if nm not in fk:
            continue
        # Skip palm — its contact with object is handled separately by L_palm
        if 'palm' in nm:
            continue
        wT = base_T @ fk[nm].get_matrix()
        wp = (wT @ pts_t.T)[:, :3, :].transpose(1, 2)
        all_pts.append(wp)
    if all_pts:
        return torch.cat(all_pts, dim=1)
    return None


def solve_ik(num_envs=512, steps=2000, lr=0.008):
    """Main IK solve loop."""
    mesh, X_WO, offset, obj_center, act_pos, act_dir, sdf, chain = load_scene()

    B = num_envs
    dev = DEVICE

    act_pos_t = torch.tensor(act_pos, dtype=torch.float32, device=dev)
    act_dir_t = torch.tensor(act_dir, dtype=torch.float32, device=dev)
    neg_act_dir_t = -act_dir_t
    obj_center_t = torch.tensor(obj_center, dtype=torch.float32, device=dev)

    # === Fixed base rotation ===
    # R_base columns: x=[0,1,0], y=[0,0,1], z=[1,0,0]
    # This puts palm inner surface (+x base) facing +y world
    R_base = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=np.float64)
    R_base_t = torch.tensor(R_base, dtype=torch.float32, device=dev)

    # 6D rotation (columns x, y of R_base)
    r6d_fixed = torch.cat([R_base_t[:, 0], R_base_t[:, 1]]).unsqueeze(0).expand(B, -1)

    # === Initial base position ===
    # Palm center in base frame: [0.028, 0, 0.05]
    # In world frame with R_base: R_base @ [0.028, 0, 0.05] = [0.05, 0.028, 0]
    # Palm should be on -y side of bottle at y ~ -0.035 (bottle radius ~3cm)
    # So: base_pos_y + 0.028 = -0.035 => base_pos_y ~ -0.063
    # Bottle center x ~ -0.005, palm x offset in world = R_base @ [0.028,0,0.05] -> x=0.05
    # base_pos_x + 0.05 = -0.005 => base_pos_x ~ -0.055
    # Bottle body z range [0, 0.182], palm center z offset in world = 0
    # We want palm center z ~ 0.11 (mid bottle), base_pos_z + 0 = 0.11
    # Actually: R_base @ palm_center = [0.05, 0.028, 0.0] -> so world_z contribution from palm is 0
    # The finger bases in base frame are at z~0.093, world frame: R_base@[0,0,0.093]=[0.093, 0, 0]
    # So base_pos_z should roughly align fingers with bottle: base_pos_z ~ 0.05 to 0.10

    # Let's compute more carefully:
    # world_point = R_base @ base_point + base_pos
    # R_base = [[0,0,1],[1,0,0],[0,1,0]]
    # So: world_x = base_z + pos_x
    #     world_y = base_x + pos_y
    #     world_z = base_y + pos_z

    # Palm inner center in base frame: [0.028, 0, 0.05]
    # -> world: [0.05 + pos_x, 0.028 + pos_y, 0 + pos_z]
    # Want palm at: world_y ~ -0.035 (on -y side, touching bottle at radius ~3cm)
    # => 0.028 + pos_y = -0.035 => pos_y = -0.063
    # Want palm_x ~ 0 (centered on bottle): 0.05 + pos_x = 0 => pos_x = -0.05
    # Want palm_z ~ 0.09 (middle of bottle body): pos_z = 0.09

    # Finger bases in base frame at z ~ 0.093
    # -> world: [0.093 + pos_x, 0 + pos_y, 0 + pos_z] = [0.043, -0.063, 0.09]
    # That's inside/near the bottle which is right

    # Actuation point: [0.039, 0, 0.137]
    # IF tip needs to reach there from the -y side, curling through +x
    # With IF at base z~0.093, world x ~ 0.043. Need to reach world x=0.039
    # The IF extends in +z base direction by ~7cm, world +x direction
    # But we need it to curl AROUND, not just extend straight

    # Base position: y is FROZEN at -0.063 so palm contacts bottle
    # (palm inner at base_x=0.028, world_y = 0.028 + (-0.063) = -0.035 = bottle surface)
    # Only x and z are optimizable
    init_pos = torch.tensor([-0.05, -0.063, 0.09], dtype=torch.float32, device=dev)
    pos_xz = (torch.tensor([[-0.05, 0.09]], device=dev).expand(B, -1)
              + 0.008 * torch.randn(B, 2, device=dev)).detach().requires_grad_(True)
    pos_y_fixed = -0.063  # frozen

    def get_base_pos():
        """Reconstruct [x, y_fixed, z] from optimizable xz."""
        y_col = torch.full((pos_xz.shape[0], 1), pos_y_fixed, device=dev)
        return torch.cat([pos_xz[:, :1], y_col, pos_xz[:, 1:]], dim=-1)

    base_pos_param = get_base_pos()  # initial value for reference

    # === Initial joint angles ===
    # For grasping, we want fingers moderately curled
    # Joint order per finger: axl, mcp, pip, dip
    # IF: needs to curl significantly to reach around to actuation on +x side
    # MF, RF: curl around to +y side
    # TH: reach to +y side at neck height
    q_init = torch.tensor([
        # IF: axl=0.5, mcp=0, pip=0.8, dip=0.8 - curl IF significantly
        0.5, 0.0, 0.8, 0.8,
        # MF: axl=0.5, mcp=0, pip=0.8, dip=0.8
        0.5, 0.0, 0.8, 0.8,
        # RF: axl=0.5, mcp=0, pip=0.8, dip=0.8
        0.5, 0.0, 0.8, 0.8,
        # TH: cmc=1.0, axl=1.0, mcp=0.3, ipl=0.0
        1.0, 1.0, 0.3, 0.0,
    ], dtype=torch.float32, device=dev)
    u_init = q2u(q_init)
    u_param = (u_init.unsqueeze(0).expand(B, -1) + 0.3 * torch.randn(B, 16, device=dev)).detach().requires_grad_(True)

    # Build base transform
    def make_base_T(pos, B):
        T = torch.eye(4, device=dev).unsqueeze(0).expand(B, -1, -1).clone()
        T[:, :3, :3] = R_base_t.unsqueeze(0).expand(B, -1, -1)
        T[:, :3, 3] = pos
        return T

    # =====================================================================
    # Phase 1: Get tips onto surface + IF to actuation
    # =====================================================================
    print("\n=== Phase 1: Surface contact + Actuation targeting ===")
    opt1 = torch.optim.Adam([u_param, pos_xz], lr=lr)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, steps // 2, lr * 0.1)

    for step in range(steps // 2):
        opt1.zero_grad()
        q = u2q(u_param)
        bT = make_base_T(get_base_pos(), B)
        tip_pos, tip_x, fk = compute_tip_positions_batch(chain, q, bT, dev)
        # tip_pos: [B, 4, 3] (IF, MF, RF, TH)

        # SDF at tips
        ts = sdf.query(tip_pos)  # [B, 4]

        # 1. Surface loss: all tips on surface
        ts_abs = ts.abs()
        L_surf = (ts ** 2).sum(-1) + 5.0 * ts_abs.sum(-1) + 20.0 * ts_abs.max(-1).values ** 2

        # 2. Actuation loss: IF tip at actuation point
        L_act = ((tip_pos[:, 0, :] - act_pos_t) ** 2).sum(-1)

        # 3. Actuation direction: IF pad aligned with -act_dir
        cos_align = (tip_x[:, 0, :] * neg_act_dir_t).sum(-1)
        L_dir = (1.0 - cos_align) ** 2

        # 4. Thumb target: should be on +y side at bottle neck height
        # Roughly at [0, 0.03, 0.13]
        th_target = torch.tensor([0.0, 0.03, 0.13], dtype=torch.float32, device=dev)
        L_th = ((tip_pos[:, 3, :] - th_target) ** 2).sum(-1)

        # 5. MF, RF targets: on +y side of bottle
        mf_target = torch.tensor([0.0, 0.03, 0.08], dtype=torch.float32, device=dev)
        rf_target = torch.tensor([0.0, 0.03, 0.05], dtype=torch.float32, device=dev)
        L_mf = ((tip_pos[:, 1, :] - mf_target) ** 2).sum(-1)
        L_rf = ((tip_pos[:, 2, :] - rf_target) ** 2).sum(-1)

        # 6. Palm proximity: palm points should be near surface
        palm_pts = compute_palm_points_batch(chain, q, bT, fk, dev)
        palm_sdf = sdf.query(palm_pts)
        L_palm = palm_sdf.abs().mean(-1) + 3.0 * F.relu(palm_sdf - 0.005).mean(-1)

        # 7. Collision avoidance: ALL link bodies must stay outside object
        col_pts = compute_collision_points_batch(chain, q, bT, fk, dev)
        L_col = torch.zeros(B, device=dev)
        if col_pts is not None:
            col_sdf = sdf.query(col_pts)
            L_col = F.relu(-col_sdf - 0.002).sum(-1)  # penalize penetration > 2mm

        # Tip penetration
        L_pen_tip = F.relu(-ts - 0.001).sum(-1)

        # 8. Position regularization: keep base position close to init (STRONG)
        L_pos_reg = ((get_base_pos() - init_pos) ** 2).sum(-1)

        # 9. Finger spread: MF, RF, TH should be on opposite side from palm
        L_spread = F.relu(-tip_pos[:, 1, 1]) + F.relu(-tip_pos[:, 2, 1]) + F.relu(-tip_pos[:, 3, 1])

        total = (500.0 * L_surf
                 + 300.0 * L_act
                 + 100.0 * L_dir
                 + 100.0 * L_th
                 + 80.0 * L_mf
                 + 80.0 * L_rf
                 + 500.0 * L_palm
                 + 500.0 * L_col           # STRONG collision avoidance
                 + 200.0 * L_pen_tip       # tip penetration
                 + 20.0 * L_pos_reg        # moderate position reg
                 + 50.0 * L_spread)

        total.mean().backward()
        opt1.step()
        sch1.step()

        if step % 100 == 0:
            with torch.no_grad():
                print(f"  Step {step:4d} | total={total.mean():.4f} "
                      f"surf={L_surf.mean():.3e} act={L_act.mean():.3e} "
                      f"palm={L_palm.mean():.3e} dir={L_dir.mean():.3e}")

    # =====================================================================
    # Phase 2: Refine - collision + force closure
    # =====================================================================
    print("\n=== Phase 2: Refinement ===")

    # Select top candidates from phase 1
    with torch.no_grad():
        q_p1 = u2q(u_param)
        bT_p1 = make_base_T(get_base_pos(), B)
        tp_p1, _, _ = compute_tip_positions_batch(chain, q_p1, bT_p1, dev)
        ts_p1 = sdf.query(tp_p1)

        # Score: surface error + actuation distance
        act_dist_p1 = ((tp_p1[:, 0, :] - act_pos_t) ** 2).sum(-1).sqrt()
        score_p1 = ts_p1.abs().sum(-1) + 5.0 * act_dist_p1

        K = min(64, B // 4)
        top_idx = score_p1.argsort()[:K]
        print(f"  Top-{K} mean surf: {ts_p1[top_idx].abs().mean():.4f}, "
              f"act_dist: {act_dist_p1[top_idx].mean():.4f}")

    # Expand top-K
    M = max(4, B // K)
    B2 = K * M
    with torch.no_grad():
        u2 = u_param[top_idx].repeat(M, 1) + 0.05 * torch.randn(K * M, 16, device=dev)
        p2_xz = pos_xz[top_idx].repeat(M, 1) + 0.002 * torch.randn(K * M, 2, device=dev)
    u_param = u2.detach().requires_grad_(True)
    pos_xz = p2_xz.detach().requires_grad_(True)

    opt2 = torch.optim.Adam([u_param, pos_xz], lr=lr * 0.5)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, steps // 2, lr * 0.05)

    best_score = torch.full((B2,), float("inf"), device=dev)
    best_u = u_param.clone().detach()
    best_p = get_base_pos().clone().detach()

    for step in range(steps // 2):
        opt2.zero_grad()
        q = u2q(u_param)
        bT = make_base_T(get_base_pos(), B2)
        tip_pos, tip_x, fk = compute_tip_positions_batch(chain, q, bT, dev)

        ts = sdf.query(tip_pos)
        ts_abs = ts.abs()

        # Surface
        L_surf = (ts ** 2).sum(-1) + 5.0 * ts_abs.sum(-1) + 20.0 * ts_abs.max(-1).values ** 2

        # Actuation
        L_act = ((tip_pos[:, 0, :] - act_pos_t) ** 2).sum(-1)
        act_dist = L_act.sqrt()
        L_act_smooth = torch.where(act_dist < 0.01, L_act, 0.01 * act_dist - 0.01**2 / 2)

        # Direction
        cos_align = (tip_x[:, 0, :] * neg_act_dir_t).sum(-1)
        L_dir = (1.0 - cos_align) ** 2

        # Palm
        palm_pts = compute_palm_points_batch(chain, q, bT, fk, dev)
        palm_sdf = sdf.query(palm_pts)
        L_palm = palm_sdf.abs().mean(-1) + 3.0 * F.relu(palm_sdf - 0.003).mean(-1)

        # Collision: ALL link bodies must stay outside
        col_pts = compute_collision_points_batch(chain, q, bT, fk, dev)
        L_col = torch.zeros(B2, device=dev)
        if col_pts is not None:
            col_sdf = sdf.query(col_pts)
            L_col = F.relu(-col_sdf - 0.002).sum(-1)

        # Penetration: penalize tips going inside
        L_pen_tip = F.relu(-ts - 0.001).sum(-1)

        # Finger targets (softer in phase 2)
        th_target = torch.tensor([0.0, 0.03, 0.13], dtype=torch.float32, device=dev)
        L_th = ((tip_pos[:, 3, :] - th_target) ** 2).sum(-1)

        # Opposing normals proxy
        eps_fd = 5e-4
        gx = (sdf.query(tip_pos + torch.tensor([eps_fd,0,0], device=dev)) -
              sdf.query(tip_pos - torch.tensor([eps_fd,0,0], device=dev))) / (2*eps_fd)
        gy = (sdf.query(tip_pos + torch.tensor([0,eps_fd,0], device=dev)) -
              sdf.query(tip_pos - torch.tensor([0,eps_fd,0], device=dev))) / (2*eps_fd)
        gz = (sdf.query(tip_pos + torch.tensor([0,0,eps_fd], device=dev)) -
              sdf.query(tip_pos - torch.tensor([0,0,eps_fd], device=dev))) / (2*eps_fd)
        sdf_grad = torch.stack([gx, gy, gz], dim=-1)
        tip_normals = -sdf_grad / sdf_grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Want opposing normals
        nc = 4
        dots = torch.bmm(tip_normals, tip_normals.transpose(1, 2))
        triu_mask = torch.triu(torch.ones(nc, nc, device=dev), diagonal=1).bool()
        pair_dots = dots[:, triu_mask]
        L_oppose = F.relu(pair_dots + 0.3).mean(-1)

        # Finger spread
        pw = torch.cdist(tip_pos, tip_pos)
        L_spread_dist = -pw[:, triu_mask].mean(-1)

        # Position reg
        L_pos_reg = ((get_base_pos() - init_pos) ** 2).sum(-1)

        t_frac = step / max(steps // 2 - 1, 1)
        w_act = 300.0
        w_surf = 500.0 + 500.0 * t_frac
        w_palm = 500.0
        w_oppose = 30.0 * min(1.0, t_frac / 0.3)

        total = (w_surf * L_surf
                 + w_act * L_act_smooth
                 + 100.0 * L_dir
                 + w_palm * L_palm
                 + 800.0 * L_col            # STRONG link collision
                 + 200.0 * L_pen_tip
                 + 50.0 * L_th
                 + w_oppose * L_oppose
                 + 3.0 * L_spread_dist
                 + 30.0 * L_pos_reg)

        total.mean().backward()
        opt2.step()
        sch2.step()

        # Track best
        with torch.no_grad():
            score = ts_abs.max(-1).values + 3.0 * act_dist + 0.5 * palm_sdf.abs().mean(-1)
            improved = score < best_score
            if improved.any():
                best_score[improved] = score[improved]
                best_u[improved] = u_param[improved]
                best_p[improved] = get_base_pos()[improved]

        if step % 100 == 0:
            with torch.no_grad():
                n_surf = (ts_abs.max(-1).values < 0.002).sum().item()
                n_act = (act_dist < 0.005).sum().item()
                print(f"  Step {step:4d} | total={total.mean():.4f} "
                      f"surf_ok={n_surf}/{B2} act_ok={n_act}/{B2} "
                      f"act={L_act_smooth.mean():.3e} palm={L_palm.mean():.3e}")

    # =====================================================================
    # Select best result
    # =====================================================================
    print("\n=== Selecting best result ===")
    with torch.no_grad():
        q_best = u2q(best_u)
        bT_best = make_base_T(best_p, B2)
        tp_best, tx_best, _ = compute_tip_positions_batch(chain, q_best, bT_best, dev)
        ts_best = sdf.query(tp_best)

        act_dist_best = torch.norm(tp_best[:, 0, :] - act_pos_t, dim=-1)
        dir_align_best = (tx_best[:, 0, :] * neg_act_dir_t).sum(-1)

        # Check palm
        fk_check = chain.forward_kinematics(q_best)
        # We'll check palm SDF for a subset
        palm_grid_y = torch.linspace(-0.025, 0.025, 4, device=dev)
        palm_grid_z = torch.linspace(0.03, 0.08, 4, device=dev)
        pyy, pzz = torch.meshgrid(palm_grid_y, palm_grid_z, indexing='ij')
        palm_pts_base = torch.stack([
            torch.full_like(pyy, 0.028), pyy, pzz
        ], dim=-1).reshape(-1, 3)
        palm_h = torch.cat([palm_pts_base, torch.ones(palm_pts_base.shape[0], 1, device=dev)], -1)

        wT_palm = bT_best @ fk_check["leap_rh_palm"].get_matrix()
        palm_world = (wT_palm @ palm_h.T)[:, :3, :].transpose(1, 2)
        palm_sdf_vals = sdf.query(palm_world)
        palm_near = (palm_sdf_vals.abs() < 0.005).float().mean(-1)

        # Feasibility score
        surf_err = ts_best.abs().max(-1).values
        feas_score = torch.where(
            (surf_err < 0.003) & (act_dist_best < 0.008) & (dir_align_best > 0.5),
            -act_dist_best - surf_err + 0.1 * palm_near,
            torch.tensor(-100.0, device=dev) - surf_err - 5.0 * act_dist_best
        )

        order = feas_score.argsort(descending=True)
        best_idx = order[0].item()

        print(f"Best idx: {best_idx}")
        print(f"  surf_err: {surf_err[best_idx]:.4f}")
        print(f"  act_dist: {act_dist_best[best_idx]:.4f}")
        print(f"  dir_align: {dir_align_best[best_idx]:.3f}")
        print(f"  palm_near: {palm_near[best_idx]:.3f}")
        print(f"  tip positions:")
        for i, name in enumerate(["IF", "MF", "RF", "TH"]):
            print(f"    {name}: {tp_best[best_idx, i].cpu().numpy()} sdf={ts_best[best_idx, i]:.4f}")

        # Convert best to result dict
        q_final = q_best[best_idx].cpu().numpy()
        base_pos_final = best_p[best_idx].cpu().numpy()
        base_rot_final = R_base

        result = {
            "q_joints": q_final,
            "base_pos": base_pos_final,
            "base_rot": base_rot_final,
            "l_star": 0.0,
            "feasible": True,
            "act_assignment": [0],
            "act_dist": float(act_dist_best[best_idx]),
            "surf_err": float(surf_err[best_idx]),
            "sigma_min": 0.0,
        }

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(SAVE_PATH)), exist_ok=True)
    torch.save([result], SAVE_PATH)
    print(f"\nSaved to {SAVE_PATH}")

    return result


def run_diagnostics():
    """Run the diagnostic tool on the saved grasp."""
    from diagnose_grasp import diagnose
    return diagnose(SAVE_PATH, DIAG_DIR, tag="_iter1")


def main():
    print("=" * 70)
    print("SPRAY BOTTLE IK SOLVER")
    print("=" * 70)

    # Iteration 1: solve and diagnose
    result = solve_ik(num_envs=512, steps=2000, lr=0.008)

    print("\n\n")
    all_pass, checks, pen_results = run_diagnostics()

    if not all_pass:
        print("\n\nSome checks failed. Analyzing failures...")
        failed = [(name, detail) for name, passed, detail in checks if not passed]
        for name, detail in failed:
            print(f"  FAILED: {name} - {detail}")

        # Iteration 2: retry with adjusted parameters
        print("\n\n=== Iteration 2: Adjusting parameters ===")
        result2 = solve_ik(num_envs=1024, steps=3000, lr=0.006)

        print("\n\n")
        all_pass2, checks2, pen_results2 = run_diagnostics()

        if not all_pass2:
            print("\n\nIteration 2 still has failures. Running iteration 3...")
            # Iteration 3: more aggressive
            print("\n\n=== Iteration 3: More aggressive optimization ===")
            result3 = solve_ik(num_envs=2048, steps=4000, lr=0.005)

            print("\n\n")
            all_pass3, checks3, pen_results3 = run_diagnostics()


if __name__ == "__main__":
    main()
