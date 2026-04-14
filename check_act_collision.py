"""Check actuation finger collision with object for saved grasp stages.

For each grasp, reconstruct FK from saved q_joints / base_pos / base_rot,
transform the actuation finger's collision box-grid points into world frame,
and query the object SDF to detect penetration.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frogger.batched_pytorch_solver import BatchedSDF, BatchedGraspOptimizer

# --- Setup SDF and optimizer (for collision geometry + FK chain) ---
mesh_path = '/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg/hot_glue_gun/object.obj'
mesh = trimesh.load(mesh_path, force='mesh')
offset = np.array([0.0, 0.0, -mesh.bounds[0, 2]])
X_WO = np.eye(4)
X_WO[:3, 3] = offset

print("Building SDF...")
sdf = BatchedSDF(mesh, X_WO, resolution=128, device='cuda')

print("Building optimizer (for FK chain + collision boxes)...")
opt = BatchedGraspOptimizer(sdf, num_envs=10, device='cuda', hand='rh',
                            hand_type='leap', palm_contact=True)

# Finger naming
finger_names = ['IF', 'MF', 'RF', 'TH']
finger_prefixes = ['if', 'mf', 'rf', 'th']
# Non-tip links per finger (bs/px/md for IF/MF/RF, mp/bs/px for TH)
finger_link_suffixes = {
    0: ['bs', 'px', 'md'],  # IF
    1: ['bs', 'px', 'md'],  # MF
    2: ['bs', 'px', 'md'],  # RF
    3: ['mp', 'bs', 'px'],  # TH
}

# Build a lookup from link name to collision box points (in link-local frame)
col_data_map = {nm: pts for nm, pts in opt._col_data}

# --- Check both stages ---
stages = ['stage_after_support_ik', 'stage_after_optimization']

for stage_name in stages:
    path = f'output/grasps/{stage_name}.pt'
    if not os.path.exists(path):
        print(f"\n!!! {path} not found, skipping")
        continue

    results = torch.load(path, weights_only=False, map_location='cpu')
    print(f'\n{"="*70}')
    print(f'  {stage_name}  ({len(results)} grasps)')
    print(f'{"="*70}')

    for gi, g in enumerate(results):
        act_fi = g['act_finger']  # 0=IF, 1=MF, 2=RF, 3=TH
        q_joints = torch.tensor(g['q_joints'], dtype=torch.float32, device='cuda').unsqueeze(0)
        base_pos = torch.tensor(g['base_pos'], dtype=torch.float32, device='cuda').unsqueeze(0)
        base_rot = torch.tensor(g['base_rot'], dtype=torch.float32, device='cuda').unsqueeze(0)  # [1,3,3]

        # Build base transform
        bT = torch.eye(4, device='cuda').unsqueeze(0)
        bT[0, :3, :3] = base_rot[0]
        bT[0, :3, 3] = base_pos[0]

        # Forward kinematics
        fk = opt.chain.forward_kinematics(q_joints)

        prefix = finger_prefixes[act_fi]
        suffixes = finger_link_suffixes[act_fi]

        print(f'\nG{gi+1} (idx {gi}): act_finger={act_fi} ({finger_names[act_fi]})')

        total_inside = 0
        worst_sdf_all = float('inf')

        for suf in suffixes:
            link_name = f"leap_rh_{prefix}_{suf}"

            if link_name not in col_data_map:
                print(f'  {link_name:30s}  -- no collision data')
                continue
            if link_name not in fk:
                print(f'  {link_name:30s}  -- not in FK chain')
                continue

            local_pts = col_data_map[link_name]  # [N, 4] homogeneous
            wT = bT @ fk[link_name].get_matrix()  # [1, 4, 4]
            world_pts = (wT @ local_pts.T)[:, :3, :].transpose(1, 2)  # [1, N, 3]

            sdf_vals = sdf.query(world_pts)  # [1, N]
            sdf_np = sdf_vals[0].cpu().numpy()

            n_inside = int((sdf_np < 0).sum())
            n_total = len(sdf_np)
            worst = float(sdf_np.min())
            mean_inside = float(sdf_np[sdf_np < 0].mean()) if n_inside > 0 else 0.0

            total_inside += n_inside
            worst_sdf_all = min(worst_sdf_all, worst)

            status = "COLLISION" if n_inside > 0 else "ok"
            print(f'  {link_name:30s}  {n_inside:3d}/{n_total:3d} inside  '
                  f'worst={worst*1000:+7.1f}mm  '
                  f'mean_inside={mean_inside*1000:+7.1f}mm  {status}')

        # Also check the ds (fingertip) — it should be touching but not deeply penetrating
        ds_suf = 'ds' if act_fi < 3 else 'ds'
        ds_link = f"leap_rh_{prefix}_{ds_suf}"
        if ds_link in col_data_map and ds_link in fk:
            local_pts = col_data_map[ds_link]
            wT = bT @ fk[ds_link].get_matrix()
            world_pts = (wT @ local_pts.T)[:, :3, :].transpose(1, 2)
            sdf_vals = sdf.query(world_pts)
            sdf_np = sdf_vals[0].cpu().numpy()
            n_inside = int((sdf_np < 0).sum())
            worst = float(sdf_np.min())
            print(f'  {ds_link:30s}  {n_inside:3d}/{len(sdf_np):3d} inside  '
                  f'worst={worst*1000:+7.1f}mm  (fingertip - expected near surface)')

        if total_inside > 0:
            print(f'  >>> TOTAL: {total_inside} non-tip points inside object, '
                  f'worst SDF = {worst_sdf_all*1000:+.1f}mm')
        else:
            print(f'  >>> CLEAN: no non-tip penetration')

# --- Summary comparison ---
print(f'\n{"="*70}')
print('  COMPARISON: did optimization make actuation collision worse?')
print(f'{"="*70}')

stage_data = {}
for stage_name in stages:
    path = f'output/grasps/{stage_name}.pt'
    if not os.path.exists(path):
        continue
    results = torch.load(path, weights_only=False, map_location='cpu')
    per_grasp = []
    for gi, g in enumerate(results):
        act_fi = g['act_finger']
        q_joints = torch.tensor(g['q_joints'], dtype=torch.float32, device='cuda').unsqueeze(0)
        base_pos = torch.tensor(g['base_pos'], dtype=torch.float32, device='cuda').unsqueeze(0)
        base_rot = torch.tensor(g['base_rot'], dtype=torch.float32, device='cuda').unsqueeze(0)
        bT = torch.eye(4, device='cuda').unsqueeze(0)
        bT[0, :3, :3] = base_rot[0]
        bT[0, :3, 3] = base_pos[0]
        fk = opt.chain.forward_kinematics(q_joints)
        prefix = finger_prefixes[act_fi]
        suffixes = finger_link_suffixes[act_fi]
        total_inside = 0
        worst_sdf = float('inf')
        for suf in suffixes:
            link_name = f"leap_rh_{prefix}_{suf}"
            if link_name not in col_data_map or link_name not in fk:
                continue
            local_pts = col_data_map[link_name]
            wT = bT @ fk[link_name].get_matrix()
            world_pts = (wT @ local_pts.T)[:, :3, :].transpose(1, 2)
            sdf_vals = sdf.query(world_pts)
            sdf_np = sdf_vals[0].cpu().numpy()
            total_inside += int((sdf_np < 0).sum())
            worst_sdf = min(worst_sdf, float(sdf_np.min()))
        per_grasp.append((total_inside, worst_sdf))
    stage_data[stage_name] = per_grasp

if len(stage_data) == 2:
    s1, s2 = stages
    print(f'\n{"Grasp":<8} {"Act":<4} '
          f'{"After IK: inside":<20} {"After IK: worst":<18} '
          f'{"After Opt: inside":<20} {"After Opt: worst":<18} '
          f'{"Delta inside":<14} {"Delta worst":<14}')
    print('-' * 130)
    r1 = torch.load(f'output/grasps/{s1}.pt', weights_only=False, map_location='cpu')
    r2 = torch.load(f'output/grasps/{s2}.pt', weights_only=False, map_location='cpu')
    for gi in range(min(len(stage_data[s1]), len(stage_data[s2]))):
        n1, w1 = stage_data[s1][gi]
        n2, w2 = stage_data[s2][gi]
        act_fi = r1[gi]['act_finger']
        delta_n = n2 - n1
        delta_w = (w2 - w1) * 1000
        flag = " <<< WORSE" if delta_n > 0 else (" <<< BETTER" if delta_n < 0 else "")
        print(f'G{gi+1:<7d} {finger_names[act_fi]:<4} '
              f'{n1:<20d} {w1*1000:+15.1f}mm  '
              f'{n2:<20d} {w2*1000:+15.1f}mm  '
              f'{delta_n:+13d} {delta_w:+13.1f}mm{flag}')
