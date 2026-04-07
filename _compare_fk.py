"""Compare pytorch_kinematics URDF FK vs render_grasp.py SDF FK for Allegro thumb."""
import numpy as np
import torch
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET
import re
import sys
sys.path.insert(0, '/home/bowenj/Projects/DexFun')
from dexfun.grasp_gen.render_grasp import GraspVisualizer
import os

LINKS = [
    'algr_rh_palm', 'algr_rh_th_mp', 'algr_rh_th_bs',
    'algr_rh_th_px', 'algr_rh_th_ds',
    'algr_rh_if_ds', 'algr_rh_mf_ds', 'algr_rh_rf_ds',
]

# Test with a non-zero thumb config
# Joint order: if_axl, if_mcp, if_pip, if_dip, mf_*, rf_*, th_cmc, th_axl, th_mcp, th_ipl
test_q = np.array([
    0.0, 0.5, 0.5, 0.5,   # if
    0.0, 0.5, 0.5, 0.5,   # mf
    0.0, 0.5, 0.5, 0.5,   # rf
    0.8, 0.5, 0.8, 0.5,   # th
], dtype=np.float32)

# 1. PK-based FK
chain = pk.build_chain_from_urdf(open('models/allegro/allegro_rh.urdf').read())
q = torch.tensor(test_q).unsqueeze(0)
fk = chain.forward_kinematics(q)
print('=== pytorch_kinematics FK (URDF) - non-zero config ===')
for name in LINKS:
    if name in fk:
        T = fk[name].get_matrix()[0].numpy()
        print(f'  {name}: pos={T[:3,3].round(6)}')

# 2. render_grasp.py SDF-based FK
sdf_path = '/home/bowenj/Projects/DexFun/models/allegro/allegro_rh.sdf'
vis = GraspVisualizer(sdf_path)
link_transforms = vis.compute_link_transforms({
    'algr_rh_if_axl': test_q[0], 'algr_rh_if_mcp': test_q[1],
    'algr_rh_if_pip': test_q[2], 'algr_rh_if_dip': test_q[3],
    'algr_rh_mf_axl': test_q[4], 'algr_rh_mf_mcp': test_q[5],
    'algr_rh_mf_pip': test_q[6], 'algr_rh_mf_dip': test_q[7],
    'algr_rh_rf_axl': test_q[8], 'algr_rh_rf_mcp': test_q[9],
    'algr_rh_rf_pip': test_q[10], 'algr_rh_rf_dip': test_q[11],
    'algr_rh_th_cmc': test_q[12], 'algr_rh_th_axl': test_q[13],
    'algr_rh_th_mcp': test_q[14], 'algr_rh_th_ipl': test_q[15],
})

print()
print('=== render_grasp.py SDF FK - non-zero config ===')
for name in LINKS:
    if name in link_transforms:
        T = link_transforms[name]
        print(f'  {name}: pos={T[:3,3].round(6)}')

# 3. Comparison
print()
print('=== Position difference (PK - SDF_FK) ===')
for name in LINKS:
    if name in fk and name in link_transforms:
        pk_pos = fk[name].get_matrix()[0].numpy()[:3, 3]
        sdf_pos = link_transforms[name][:3, 3]
        diff = pk_pos - sdf_pos
        err = np.linalg.norm(diff)
        flag = " <<<< MISMATCH" if err > 0.001 else ""
        print(f'  {name}: diff={diff.round(6)}, err={err:.6f}{flag}')

# 4. Orientation comparison
print()
print('=== Orientation difference ===')
for name in LINKS:
    if name in fk and name in link_transforms:
        pk_R = fk[name].get_matrix()[0].numpy()[:3, :3]
        sdf_R = link_transforms[name][:3, :3]
        R_diff = pk_R @ sdf_R.T
        angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
        flag = " <<<< MISMATCH" if angle > 0.01 else ""
        print(f'  {name}: angle_diff={np.degrees(angle):.4f} deg{flag}')
