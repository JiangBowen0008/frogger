#!/usr/bin/env python3
"""Show init grasps in viser with object point, normal, and connecting line."""
import os, sys, numpy as np, torch, trimesh, viser, time
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(__file__))
from frogger.batched_pytorch_solver import _visual_meshes, BatchedSDF, BatchedGraspOptimizer

URDF = os.path.join(os.path.dirname(__file__), "models/leap_rh/leap.urdf")
MDIR = os.path.join(os.path.dirname(__file__), "models/leap_rh")
MESH_PATH = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg/hot_glue_gun/object.obj"
STAGES = {
    "1_init": "output/grasps/stage_after_init.pt",
    "2_support_ik": "output/grasps/stage_after_support_ik.pt",
    "3_optimized": "output/grasps/stage_after_optimization.pt",
}
GRASP_PATH = list(STAGES.values())[-1]  # default to latest

import subprocess, signal
# Kill any process on port 8090
try:
    out = subprocess.check_output(["lsof", "-ti", ":8090"]).decode().strip()
    for pid in out.split("\n"):
        if pid:
            os.kill(int(pid), signal.SIGKILL)
    import time as _t; _t.sleep(1)
except (subprocess.CalledProcessError, ProcessLookupError):
    pass

server = viser.ViserServer(host="0.0.0.0", port=8090)

# Actuation target (from annotation)
import json as _json
_act_path = "/home/bowenj/Projects/DexFun/assets/actuation_contacts/hot_glue_gun_actuation.json"
if os.path.exists(_act_path):
    with open(_act_path) as _f:
        _act_data = _json.load(_f)
    _obj_tmp = trimesh.load(MESH_PATH, force="mesh")
    _act_offset = np.array([0.0, 0.0, -_obj_tmp.bounds[0, 2]])
    ACT_POS = (np.array(_act_data["actuation_contacts"][0]["pos"]) + _act_offset).astype(np.float32)
    ACT_DIR = np.array(_act_data["actuation_contacts"][0]["dir"], dtype=np.float32)
else:
    ACT_POS = np.array([0, 0, 0.1], dtype=np.float32)
    ACT_DIR = np.array([0, 0, -1], dtype=np.float32)

# Load object
obj = trimesh.load(MESH_PATH, force="mesh")
offset = np.array([0.0, 0.0, -obj.bounds[0, 2]])
obj.apply_translation(offset)
server.scene.add_mesh_simple("/object",
    vertices=np.asarray(obj.vertices, dtype=np.float32),
    faces=np.asarray(obj.faces, dtype=np.int32),
    color=(180, 180, 180), opacity=0.6)

# Build SDF and optimizer for metrics computation
_obj_raw = trimesh.load(MESH_PATH, force="mesh")
_X_WO = np.eye(4)
_X_WO[:3, 3] = offset
sdf = BatchedSDF(_obj_raw, _X_WO, resolution=128, device="cuda")
opt = BatchedGraspOptimizer(sdf, num_envs=1, device="cuda", hand="rh",
                            hand_type="leap", palm_contact=True)
# Build GPU chain for metrics FK
_gpu_chain = opt.chain

# Finger name labels for display
FINGER_LABELS = ["IF", "MF", "RF", "TH"]

# FK chain + meshes
chain = pk.build_chain_from_urdf(open(URDF).read())
vis = _visual_meshes("rh", "leap")
link_cache = {}
for ln, ml in vis.items():
    for mi, (mf, vp) in enumerate(ml):
        p = os.path.join(MDIR, mf)
        if not os.path.exists(p): continue
        lm = trimesh.load(p, force="mesh")
        link_cache[(ln, mi)] = (np.asarray(lm.vertices, np.float32),
                                np.asarray(lm.faces, np.int32), vp)

# Load all stages
all_stages = {}
for sname, spath in STAGES.items():
    if os.path.exists(spath):
        all_stages[sname] = torch.load(spath, weights_only=False, map_location="cpu")
        print(f"  Loaded {sname}: {len(all_stages[sname])} grasps")

stage_names = list(all_stages.keys())
current_stage = [stage_names[-1]]  # start with latest
results = all_stages[current_stage[0]]
contact_base = np.array([0.023, 0, 0.048])

current = [0]

def show(idx, stage=None):
    if stage is None:
        stage = current_stage[0]
    # The latest stage (optimized) defines which grasps we show.
    # For earlier stages, match by env_idx.
    latest = all_stages[stage_names[-1]]
    if idx >= len(latest):
        idx = 0
    target_env = latest[idx].get("env_idx", idx)

    data = all_stages.get(stage, latest)
    # Find the grasp with matching env_idx
    g = None
    for entry in data:
        if entry.get("env_idx", -1) == target_env:
            g = entry
            break
    if g is None:
        g = data[min(idx, len(data)-1)]  # fallback
    R = g["base_rot"]; pos = g["base_pos"]
    q = torch.tensor(g["q_joints"], dtype=torch.float32).unsqueeze(0)
    fk = chain.forward_kinematics(q)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pos

    # Hand meshes
    for (ln, mi), (v, f, vp) in link_cache.items():
        if ln not in fk: continue
        wT = T @ fk[ln].get_matrix()[0].numpy()
        if vp is not None:
            vpa = np.array(vp)
            Rv = Rotation.from_euler("xyz", vpa[3:]).as_matrix()
            Tv = np.eye(4); Tv[:3, :3] = Rv; Tv[:3, 3] = vpa[:3]
            wT = wT @ Tv
        vw = (wT[:3, :3] @ v.T).T + wT[:3, 3]
        c = (50, 100, 255) if "palm" in ln else (255, 200, 100)
        server.scene.add_mesh_simple(f"/hand/{ln}_{mi}",
            vertices=vw.astype(np.float32), faces=f, color=c, opacity=0.85)

    # Actuation target (red sphere) + direction (red arrow tip)
    server.scene.add_icosphere("/markers/act_target", radius=0.008,
        color=(255, 0, 0), position=ACT_POS)
    act_dir_tip = ACT_POS + 0.03 * ACT_DIR
    server.scene.add_icosphere("/markers/act_dir", radius=0.004,
        color=(200, 0, 0), position=act_dir_tip)

    # Palm contact center
    palm_center = R @ contact_base + pos
    server.scene.add_icosphere("/markers/palm_center", radius=0.006,
        color=(255, 0, 255), position=palm_center.astype(np.float32))

    # Surface point
    surf = g.get("surf_pt")
    outward = g.get("outward_normal")
    if surf is not None:
        server.scene.add_icosphere("/markers/surf_pt", radius=0.006,
            color=(255, 255, 0), position=surf.astype(np.float32))

        # Object normal arrow (outward, 4cm)
        normal_tip = surf + 0.04 * outward
        server.scene.add_icosphere("/markers/normal_tip", radius=0.004,
            color=(255, 165, 0), position=normal_tip.astype(np.float32))

        # Connecting line: surface point → palm contact center
        # Use small spheres along the line
        n_pts = 20
        for li in range(n_pts):
            t = li / (n_pts - 1)
            pt = surf * (1 - t) + palm_center * t
            server.scene.add_icosphere(f"/markers/line_{li}", radius=0.002,
                color=(255, 0, 255), position=pt.astype(np.float32))

    # +X axis (palm normal, 4cm)
    x_tip = palm_center + 0.04 * R[:, 0]
    server.scene.add_icosphere("/markers/x_tip", radius=0.004,
        color=(255, 0, 0), position=x_tip.astype(np.float32))

    # ── Compute metrics ──────────────────────────────────────────────
    lines = [f"**Grasp {idx} (env {target_env}) | Stage: {stage}**\n"]

    # Run FK on GPU for metric queries
    _prev_grad = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    q_t = torch.tensor(g["q_joints"], dtype=torch.float32, device="cuda").unsqueeze(0)
    fk_gpu = _gpu_chain.forward_kinematics(q_t)
    bT_np = np.eye(4); bT_np[:3, :3] = R; bT_np[:3, 3] = pos
    bT_gpu = torch.tensor(bT_np, dtype=torch.float32, device="cuda").unsqueeze(0)

    # Get tip positions, collision points, SC points
    tip_pts, col_pts, tip_x_axes = opt._get_points(fk_gpu, bT_gpu)
    sc_pts = opt._get_sc_points(fk_gpu, bT_gpu)

    # --- 1. Actuation finger distance to target ---
    act_finger_idx = g.get("act_finger", g.get("act_assignment", [0])[0] if "act_assignment" in g else 0)
    if isinstance(act_finger_idx, list):
        act_finger_idx = act_finger_idx[0]
    act_tip_pos = tip_pts[0, act_finger_idx].cpu().numpy()
    act_dist = np.linalg.norm(act_tip_pos - ACT_POS) * 1000
    act_label = FINGER_LABELS[act_finger_idx] if act_finger_idx < 4 else f"F{act_finger_idx}"
    lines.append(f"**Actuation:** {act_label} dist={act_dist:.1f}mm")

    # --- 2. Actuation finger pad alignment with actuation direction ---
    act_pad_dir = tip_x_axes[0, act_finger_idx].cpu().numpy()
    # Pad push direction should align with -ACT_DIR (pushing in actuation direction)
    neg_act_dir = -ACT_DIR
    act_dot = np.dot(act_pad_dir, neg_act_dir)
    act_angle = np.degrees(np.arccos(np.clip(act_dot, -1, 1)))
    lines.append(f"  pad align: {act_angle:.1f}deg (dot={act_dot:.2f})")

    # --- 3. Per-support-finger tip SDF ---
    tip_sdf = sdf.query(tip_pts).cpu().numpy()[0]  # [nc]
    tip_sdf_line = "**Tip SDF (mm):**"
    for fi in range(4):
        label = FINGER_LABELS[fi]
        role = "[ACT]" if fi == act_finger_idx else "[SUP]"
        tip_sdf_line += f" {label}{role}={tip_sdf[fi]*1000:.1f}"
    lines.append(tip_sdf_line)

    # --- 4. Per-support-finger link collision count ---
    col_sdf = sdf.query(col_pts).cpu().numpy()[0]  # [N_col]
    lines.append("**Link collisions (pts inside):**")
    for nm, local_pts in opt._col_data:
        short = nm.replace("leap_rh_", "")
        # Transform this link's points to world frame
        if nm in fk_gpu:
            wT_link = bT_gpu @ fk_gpu[nm].get_matrix()
            wp = (wT_link @ local_pts.T)[:, :3, :].transpose(1, 2)
            link_sdf = sdf.query(wp).cpu().numpy()[0]
            n_inside = (link_sdf < 0).sum()
            n_total = len(link_sdf)
            worst = link_sdf.min() * 1000
            is_tip = "_ds" in nm
            role = ""
            for fi, tln in enumerate(opt.tip_link_names):
                if nm == tln:
                    role = "[ACT]" if fi == act_finger_idx else "[SUP]"
            if n_inside > 0:
                lines.append(f"  {short}{role}: {n_inside}/{n_total} inside, worst={worst:.1f}mm")

    # --- 5. Inter-finger minimum distances (from SC box points) ---
    if sc_pts is not None:
        lines.append("**Inter-finger min dist (mm):**")
        # Gather per-finger SC point indices
        finger_keys = ['if', 'mf', 'rf', 'th']
        finger_sc_pts = {}
        offset_sc = 0
        for nm, pts in opt._sc_data:
            n = pts.shape[0]
            for fk_name in finger_keys:
                if f'_{fk_name}_' in nm:
                    finger_sc_pts.setdefault(fk_name, []).append((offset_sc, offset_sc + n))
                    break
            if 'palm' in nm:
                finger_sc_pts.setdefault('palm', []).append((offset_sc, offset_sc + n))
            offset_sc += n

        def _get_finger_pts(key):
            ranges = finger_sc_pts.get(key, [])
            if not ranges:
                return None
            return torch.cat([sc_pts[0, s:e] for s, e in ranges], dim=0)

        all_keys = [k for k in ['if', 'mf', 'rf', 'th', 'palm'] if k in finger_sc_pts]
        dist_strs = []
        for i in range(len(all_keys)):
            for j in range(i + 1, len(all_keys)):
                p1 = _get_finger_pts(all_keys[i])
                p2 = _get_finger_pts(all_keys[j])
                if p1 is not None and p2 is not None:
                    dists = torch.cdist(p1.unsqueeze(0), p2.unsqueeze(0)).squeeze(0)
                    min_d = dists.min().item() * 1000
                    k1 = all_keys[i].upper()
                    k2 = all_keys[j].upper()
                    if min_d < 5.0:
                        dist_strs.append(f"  {k1}-{k2}: **{min_d:.1f}mm**")
                    else:
                        dist_strs.append(f"  {k1}-{k2}: {min_d:.1f}mm")
        for s in dist_strs:
            lines.append(s)

    # --- 6. sigma_min if available ---
    sigma_min = g.get("sigma_min", None)
    if sigma_min is not None:
        lines.append(f"**sigma_min:** {sigma_min:.4f}")
    l_star = g.get("l_star", None)
    if l_star is not None:
        lines.append(f"**l_star:** {l_star:.4f}")
    feasible = g.get("feasible", None)
    if feasible is not None:
        lines.append(f"**feasible:** {feasible}")

    # Palm-surface distance
    d = np.linalg.norm(palm_center - surf) if surf is not None else 0
    lines.append(f"palm-surf dist: {d*1000:.0f}mm")

    torch.set_grad_enabled(_prev_grad)
    info_md.content = "\n".join(lines)

with server.gui.add_folder("Grasp Browser"):
    info_md = server.gui.add_markdown("")
    stage_dd = server.gui.add_dropdown("Stage", options=stage_names, initial_value=stage_names[-1])
    gi_slider = server.gui.add_slider("Grasp", min=0, max=9, step=1, initial_value=0)
    btn_prev = server.gui.add_button("< Prev")
    btn_next = server.gui.add_button("Next >")

@stage_dd.on_update
def _(_):
    current_stage[0] = stage_dd.value
    show(int(gi_slider.value), stage_dd.value)

@gi_slider.on_update
def _(_):
    show(int(gi_slider.value), current_stage[0])

@btn_prev.on_click
def _(_):
    current[0] = (current[0] - 1) % len(all_stages[current_stage[0]])
    gi_slider.value = current[0]

@btn_next.on_click
def _(_):
    current[0] = (current[0] + 1) % len(all_stages[current_stage[0]])
    gi_slider.value = current[0]

show(0, stage_names[-1])
print("http://localhost:8090")
print("Magenta=palm contact center, Yellow=surface point, Orange=obj normal tip")
print("Magenta dots=connecting line, Red=+X axis")
try:
    while True: time.sleep(1)
except KeyboardInterrupt: pass
