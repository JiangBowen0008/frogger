#!/usr/bin/env python3
"""
Diagnose collision issues for grasps G0-G9 in stage_after_optimization.pt.

Prints a per-grasp table with:
  - Actuation finger: which finger, distance to target, per-link collision
  - Support fingers: per-finger tip SDF, per-link collision count
  - Inter-finger: all pairs min distance

Usage:
    conda run -n frogger python diagnose_init_grasps.py
"""

import os, sys, json, numpy as np, torch, trimesh
sys.path.insert(0, os.path.dirname(__file__))

from frogger.batched_pytorch_solver import BatchedSDF, BatchedGraspOptimizer

# ── Configuration ──────────────────────────────────────────────────────
MESH_PATH = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg/hot_glue_gun/object.obj"
GRASP_PATH = "output/grasps/stage_after_optimization.pt"
ACT_JSON = "/home/bowenj/Projects/DexFun/assets/actuation_contacts/hot_glue_gun_actuation.json"

FINGER_LABELS = ["IF", "MF", "RF", "TH"]
FINGER_KEYS = ["if", "mf", "rf", "th"]

# ── Load actuation target ─────────────────────────────────────────────
obj_tmp = trimesh.load(MESH_PATH, force="mesh")
act_offset = np.array([0.0, 0.0, -obj_tmp.bounds[0, 2]])

if os.path.exists(ACT_JSON):
    with open(ACT_JSON) as f:
        act_data = json.load(f)
    ACT_POS = (np.array(act_data["actuation_contacts"][0]["pos"]) + act_offset).astype(np.float32)
    ACT_DIR = np.array(act_data["actuation_contacts"][0]["dir"], dtype=np.float32)
else:
    ACT_POS = np.array([0, 0, 0.1], dtype=np.float32)
    ACT_DIR = np.array([0, 0, -1], dtype=np.float32)

print(f"Actuation target: {ACT_POS}, dir: {ACT_DIR}")

# ── Build SDF and optimizer ───────────────────────────────────────────
obj_raw = trimesh.load(MESH_PATH, force="mesh")
offset = np.array([0.0, 0.0, -obj_raw.bounds[0, 2]])
X_WO = np.eye(4)
X_WO[:3, 3] = offset

print("Building SDF...")
sdf = BatchedSDF(obj_raw, X_WO, resolution=128, device="cuda")

print("Building optimizer...")
opt = BatchedGraspOptimizer(sdf, num_envs=1, device="cuda", hand="rh",
                            hand_type="leap", palm_contact=True)
gpu_chain = opt.chain

# ── Load grasps ───────────────────────────────────────────────────────
grasps = torch.load(GRASP_PATH, weights_only=False, map_location="cpu")
n_grasps = min(10, len(grasps))
print(f"\nLoaded {len(grasps)} grasps from {GRASP_PATH}, diagnosing G0-G{n_grasps-1}")

# ── Build finger-to-SC-index mapping (once) ───────────────────────────
finger_sc_ranges = {}  # key -> list of (start, end) in sc_pts dimension
sc_offset = 0
for nm, pts in opt._sc_data:
    n = pts.shape[0]
    for fk_name in FINGER_KEYS:
        if f'_{fk_name}_' in nm:
            finger_sc_ranges.setdefault(fk_name, []).append((sc_offset, sc_offset + n))
            break
    if 'palm' in nm:
        finger_sc_ranges.setdefault('palm', []).append((sc_offset, sc_offset + n))
    sc_offset += n

# Build link-to-finger mapping for collision data
link_finger_map = {}  # link_name -> finger_key
for nm, _ in opt._col_data:
    for fk_name in FINGER_KEYS:
        if f'_{fk_name}_' in nm:
            link_finger_map[nm] = fk_name
            break
    if 'palm' in nm:
        link_finger_map[nm] = 'palm'


def get_sc_finger_pts(sc_pts_batch0, key):
    """Extract SC points for a given finger key from the concatenated sc_pts."""
    ranges = finger_sc_ranges.get(key, [])
    if not ranges:
        return None
    return torch.cat([sc_pts_batch0[s:e] for s, e in ranges], dim=0)


# ── Diagnose each grasp ──────────────────────────────────────────────
torch.set_grad_enabled(False)

print("\n" + "=" * 120)
print(f"{'G':>3} {'Feas':>5} {'sig_min':>8} {'ActF':>4} {'ActDist':>8} {'ActAng':>7} | "
      f"{'IF tip':>7} {'MF tip':>7} {'RF tip':>7} {'TH tip':>7} | "
      f"{'ColLinks':>30} | {'SC min (mm)':>12}")
print("-" * 120)

for gi in range(n_grasps):
    g = grasps[gi]
    R = g["base_rot"]
    pos = g["base_pos"]

    # FK on GPU
    q_t = torch.tensor(g["q_joints"], dtype=torch.float32, device="cuda").unsqueeze(0)
    fk_gpu = gpu_chain.forward_kinematics(q_t)
    bT_np = np.eye(4)
    bT_np[:3, :3] = R
    bT_np[:3, 3] = pos
    bT_gpu = torch.tensor(bT_np, dtype=torch.float32, device="cuda").unsqueeze(0)

    # Get points
    tip_pts, col_pts, tip_x_axes = opt._get_points(fk_gpu, bT_gpu)
    sc_pts = opt._get_sc_points(fk_gpu, bT_gpu)

    # Actuation finger
    act_fi = g.get("act_finger",
                    g.get("act_assignment", [0])[0] if "act_assignment" in g else 0)
    if isinstance(act_fi, list):
        act_fi = act_fi[0]
    act_tip = tip_pts[0, act_fi].cpu().numpy()
    act_dist = np.linalg.norm(act_tip - ACT_POS) * 1000

    # Pad alignment
    act_pad_dir = tip_x_axes[0, act_fi].cpu().numpy()
    neg_act_dir = -ACT_DIR
    act_dot = np.dot(act_pad_dir, neg_act_dir)
    act_angle = np.degrees(np.arccos(np.clip(act_dot, -1, 1)))

    # Tip SDF (all 4 fingers)
    tip_sdf = sdf.query(tip_pts[:, :4]).cpu().numpy()[0]  # just finger tips

    # Per-link collision
    col_links_inside = []
    for nm, local_pts in opt._col_data:
        if nm not in fk_gpu:
            continue
        wT_link = bT_gpu @ fk_gpu[nm].get_matrix()
        wp = (wT_link @ local_pts.T)[:, :3, :].transpose(1, 2)
        link_sdf_vals = sdf.query(wp).cpu().numpy()[0]
        n_inside = int((link_sdf_vals < 0).sum())
        n_total = len(link_sdf_vals)
        worst = link_sdf_vals.min() * 1000
        if n_inside > 0:
            short = nm.replace("leap_rh_", "")
            col_links_inside.append(f"{short}:{n_inside}/{n_total}({worst:.0f})")

    # Inter-finger min distances
    sc_min_str = ""
    if sc_pts is not None:
        all_sc_keys = [k for k in ['if', 'mf', 'rf', 'th', 'palm'] if k in finger_sc_ranges]
        worst_pair = ""
        worst_dist = 999.0
        for i in range(len(all_sc_keys)):
            for j in range(i + 1, len(all_sc_keys)):
                p1 = get_sc_finger_pts(sc_pts[0], all_sc_keys[i])
                p2 = get_sc_finger_pts(sc_pts[0], all_sc_keys[j])
                if p1 is not None and p2 is not None:
                    dists = torch.cdist(p1.unsqueeze(0), p2.unsqueeze(0)).squeeze(0)
                    min_d = dists.min().item() * 1000
                    if min_d < worst_dist:
                        worst_dist = min_d
                        worst_pair = f"{all_sc_keys[i].upper()}-{all_sc_keys[j].upper()}"
        sc_min_str = f"{worst_pair}:{worst_dist:.1f}"

    # Assemble summary line
    feas = g.get("feasible", "?")
    sigma = g.get("sigma_min", 0.0)
    act_label = FINGER_LABELS[act_fi] if act_fi < 4 else f"F{act_fi}"
    col_str = ", ".join(col_links_inside[:3]) if col_links_inside else "none"

    print(f"G{gi:>2} {str(feas):>5} {sigma:>8.4f} {act_label:>4} {act_dist:>7.1f}mm {act_angle:>6.1f}d | "
          f"{tip_sdf[0]*1000:>6.1f} {tip_sdf[1]*1000:>6.1f} {tip_sdf[2]*1000:>6.1f} {tip_sdf[3]*1000:>6.1f} | "
          f"{col_str:>30} | {sc_min_str:>12}")

# ── Detailed per-grasp breakdown ─────────────────────────────────────
print("\n\n" + "=" * 120)
print("DETAILED PER-GRASP BREAKDOWN")
print("=" * 120)

for gi in range(n_grasps):
    g = grasps[gi]
    R = g["base_rot"]
    pos = g["base_pos"]

    q_t = torch.tensor(g["q_joints"], dtype=torch.float32, device="cuda").unsqueeze(0)
    fk_gpu = gpu_chain.forward_kinematics(q_t)
    bT_np = np.eye(4)
    bT_np[:3, :3] = R
    bT_np[:3, 3] = pos
    bT_gpu = torch.tensor(bT_np, dtype=torch.float32, device="cuda").unsqueeze(0)

    tip_pts, col_pts, tip_x_axes = opt._get_points(fk_gpu, bT_gpu)
    sc_pts = opt._get_sc_points(fk_gpu, bT_gpu)

    act_fi = g.get("act_finger",
                    g.get("act_assignment", [0])[0] if "act_assignment" in g else 0)
    if isinstance(act_fi, list):
        act_fi = act_fi[0]

    feas = g.get("feasible", "?")
    sigma = g.get("sigma_min", 0.0)
    act_label = FINGER_LABELS[act_fi] if act_fi < 4 else f"F{act_fi}"
    act_tip = tip_pts[0, act_fi].cpu().numpy()
    act_dist = np.linalg.norm(act_tip - ACT_POS) * 1000

    print(f"\n--- G{gi} | feasible={feas} | sigma_min={sigma:.4f} | "
          f"act={act_label} dist={act_dist:.1f}mm ---")

    # Actuation finger details
    act_pad_dir = tip_x_axes[0, act_fi].cpu().numpy()
    neg_act_dir = -ACT_DIR
    act_dot = np.dot(act_pad_dir, neg_act_dir)
    act_angle = np.degrees(np.arccos(np.clip(act_dot, -1, 1)))
    print(f"  Actuation {act_label}: dist={act_dist:.1f}mm, pad_align={act_angle:.1f}deg (dot={act_dot:.2f})")

    # Per-link collision for actuation finger
    act_fk = FINGER_KEYS[act_fi] if act_fi < 4 else None
    print(f"  Actuation finger links:")
    for nm, local_pts in opt._col_data:
        if nm not in fk_gpu:
            continue
        fk_name = link_finger_map.get(nm)
        if fk_name != act_fk:
            continue
        wT_link = bT_gpu @ fk_gpu[nm].get_matrix()
        wp = (wT_link @ local_pts.T)[:, :3, :].transpose(1, 2)
        link_sdf_vals = sdf.query(wp).cpu().numpy()[0]
        n_inside = int((link_sdf_vals < 0).sum())
        n_total = len(link_sdf_vals)
        worst = link_sdf_vals.min() * 1000
        short = nm.replace("leap_rh_", "")
        flag = " ** COLLISION **" if n_inside > 0 else ""
        print(f"    {short}: {n_inside}/{n_total} inside, worst_sdf={worst:.1f}mm{flag}")

    # Support fingers
    tip_sdf = sdf.query(tip_pts[:, :4]).cpu().numpy()[0]
    print(f"  Support fingers (tip SDF):")
    for fi in range(4):
        if fi == act_fi:
            continue
        label = FINGER_LABELS[fi]
        print(f"    {label}: tip_sdf={tip_sdf[fi]*1000:.1f}mm")
        # Per-link collision for this support finger
        sup_fk = FINGER_KEYS[fi]
        for nm, local_pts in opt._col_data:
            if nm not in fk_gpu:
                continue
            fk_name = link_finger_map.get(nm)
            if fk_name != sup_fk:
                continue
            wT_link = bT_gpu @ fk_gpu[nm].get_matrix()
            wp = (wT_link @ local_pts.T)[:, :3, :].transpose(1, 2)
            link_sdf_vals = sdf.query(wp).cpu().numpy()[0]
            n_inside = int((link_sdf_vals < 0).sum())
            n_total = len(link_sdf_vals)
            worst = link_sdf_vals.min() * 1000
            short = nm.replace("leap_rh_", "")
            if n_inside > 0:
                print(f"      {short}: {n_inside}/{n_total} inside, worst={worst:.1f}mm")

    # Palm collision
    print(f"  Palm:")
    for nm, local_pts in opt._col_data:
        if 'palm' not in nm or nm not in fk_gpu:
            continue
        wT_link = bT_gpu @ fk_gpu[nm].get_matrix()
        wp = (wT_link @ local_pts.T)[:, :3, :].transpose(1, 2)
        link_sdf_vals = sdf.query(wp).cpu().numpy()[0]
        n_inside = int((link_sdf_vals < 0).sum())
        n_total = len(link_sdf_vals)
        worst = link_sdf_vals.min() * 1000
        short = nm.replace("leap_rh_", "")
        flag = " ** COLLISION **" if n_inside > 0 else ""
        print(f"    {short}: {n_inside}/{n_total} inside, worst_sdf={worst:.1f}mm{flag}")

    # Inter-finger distances (all pairs)
    if sc_pts is not None:
        print(f"  Inter-finger SC distances (mm):")
        all_sc_keys = [k for k in ['if', 'mf', 'rf', 'th', 'palm'] if k in finger_sc_ranges]
        for i in range(len(all_sc_keys)):
            for j in range(i + 1, len(all_sc_keys)):
                p1 = get_sc_finger_pts(sc_pts[0], all_sc_keys[i])
                p2 = get_sc_finger_pts(sc_pts[0], all_sc_keys[j])
                if p1 is not None and p2 is not None:
                    dists = torch.cdist(p1.unsqueeze(0), p2.unsqueeze(0)).squeeze(0)
                    min_d = dists.min().item() * 1000
                    k1 = all_sc_keys[i].upper()
                    k2 = all_sc_keys[j].upper()
                    flag = " ** CLOSE **" if min_d < 5.0 else ""
                    print(f"    {k1}-{k2}: {min_d:.1f}mm{flag}")

print("\n" + "=" * 120)
print("Done.")
