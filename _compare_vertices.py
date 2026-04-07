"""Compare th_mp mesh vertex positions between PK/URDF and render_grasp.py/SDF approaches."""
import numpy as np
import sys
sys.path.insert(0, '/home/bowenj/Projects/DexFun')

import torch
import pytorch_kinematics as pk
import trimesh
from scipy.spatial.transform import Rotation as R
from dexfun.grasp_gen.render_grasp import GraspVisualizer

# Test config (non-zero thumb)
test_q = np.array([
    0.0, 0.5, 0.5, 0.5,
    0.0, 0.5, 0.5, 0.5,
    0.0, 0.5, 0.5, 0.5,
    0.8, 0.5, 0.8, 0.5,
], dtype=np.float32)

# Base transform (identity for simplicity)
T_base = np.eye(4)

# Load th_mp mesh
mesh = trimesh.load('models/allegro/meshes/link_12.0_right.obj', force='mesh')
verts = np.asarray(mesh.vertices, dtype=np.float64)

# ===== Method 1: PK URDF approach (our code) =====
chain = pk.build_chain_from_urdf(open('models/allegro/allegro_rh.urdf').read())
q = torch.tensor(test_q).unsqueeze(0)
fk = chain.forward_kinematics(q)
pk_T = fk['algr_rh_th_mp'].get_matrix()[0].numpy()
# For rh, th_mp has no visual pose, so world_T = T_base @ pk_T
world_T_pk = T_base @ pk_T
verts_pk = (world_T_pk[:3, :3] @ verts.T).T + world_T_pk[:3, 3]

# ===== Method 2: render_grasp.py SDF approach =====
sdf_path = '/home/bowenj/Projects/DexFun/models/allegro/allegro_rh.sdf'
vis = GraspVisualizer(sdf_path)
joint_configs = {
    'algr_rh_if_axl': test_q[0], 'algr_rh_if_mcp': test_q[1],
    'algr_rh_if_pip': test_q[2], 'algr_rh_if_dip': test_q[3],
    'algr_rh_mf_axl': test_q[4], 'algr_rh_mf_mcp': test_q[5],
    'algr_rh_mf_pip': test_q[6], 'algr_rh_mf_dip': test_q[7],
    'algr_rh_rf_axl': test_q[8], 'algr_rh_rf_mcp': test_q[9],
    'algr_rh_rf_pip': test_q[10], 'algr_rh_rf_dip': test_q[11],
    'algr_rh_th_cmc': test_q[12], 'algr_rh_th_axl': test_q[13],
    'algr_rh_th_mcp': test_q[14], 'algr_rh_th_ipl': test_q[15],
}
link_transforms = vis.compute_link_transforms(joint_configs)
sdf_T = link_transforms['algr_rh_th_mp']
# SDF visual for th_mp in rh has no visual pose offset
world_T_sdf = T_base @ sdf_T
verts_sdf = (world_T_sdf[:3, :3] @ verts.T).T + world_T_sdf[:3, 3]

# Compare
diff = verts_pk - verts_sdf
max_err = np.abs(diff).max()
mean_err = np.abs(diff).mean()
print(f"th_mp mesh vertex comparison (RH):")
print(f"  max error: {max_err:.10f}")
print(f"  mean error: {mean_err:.10f}")
print(f"  PK link T:\n{pk_T.round(6)}")
print(f"  SDF link T:\n{sdf_T.round(6)}")

# Now test th_ds (fingertip) - which has the tip offset
pk_T_ds = fk['algr_rh_th_ds'].get_matrix()[0].numpy()
sdf_T_ds = link_transforms['algr_rh_th_ds']

# Tip offset used in optimizer
r = 0.012
tip_offset = np.array([r * np.sin(np.pi/4), 0.0, 0.0423 + r * np.cos(np.pi/4)])
tip_pk = (T_base @ pk_T_ds)[:3, :3] @ tip_offset + (T_base @ pk_T_ds)[:3, 3]
tip_sdf = (T_base @ sdf_T_ds)[:3, :3] @ tip_offset + (T_base @ sdf_T_ds)[:3, 3]
print(f"\nth_ds tip position comparison:")
print(f"  PK: {tip_pk.round(6)}")
print(f"  SDF: {tip_sdf.round(6)}")
print(f"  diff: {np.linalg.norm(tip_pk - tip_sdf):.10f}")

# Also test LEFT hand
print("\n\n=== LEFT HAND ===")
chain_lh = pk.build_chain_from_urdf(open('models/allegro/allegro_lh.urdf').read())
q_lh = torch.tensor(test_q).unsqueeze(0)
fk_lh = chain_lh.forward_kinematics(q_lh)
pk_T_lh = fk_lh['algr_lh_th_mp'].get_matrix()[0].numpy()
print(f"  PK th_mp pos: {pk_T_lh[:3,3].round(6)}")
print(f"  PK th_mp R:\n{pk_T_lh[:3,:3].round(4)}")

# Check LH SDF
import xml.etree.ElementTree as ET, re
with open('models/allegro/allegro_lh.sdf') as f:
    xml_content = re.sub(r'drake:', '', f.read())
root = ET.fromstring(xml_content)
model = root.find('model')
for link in model.findall('link'):
    name = link.get('name')
    if name == 'algr_lh_th_mp':
        pose = [float(x) for x in link.find('pose').text.strip().split()]
        sdf_T_lh = np.eye(4)
        sdf_T_lh[:3, :3] = R.from_euler('xyz', pose[3:]).as_matrix()
        sdf_T_lh[:3, 3] = pose[:3]
        print(f"  SDF th_mp pos: {sdf_T_lh[:3,3].round(6)}")
        print(f"  SDF th_mp R:\n{sdf_T_lh[:3,:3].round(4)}")
        print(f"  pos diff: {np.linalg.norm(pk_T_lh[:3,3] - sdf_T_lh[:3,3]):.10f}")
        R_diff = pk_T_lh[:3,:3] @ sdf_T_lh[:3,:3].T
        angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
        print(f"  angle diff: {np.degrees(angle):.6f} deg")

# Check if render_grasp.py can handle LH
lh_sdf = 'models/allegro/allegro_lh.sdf'
vis_lh = GraspVisualizer(lh_sdf)
vis_lh.root_link = 'algr_lh_palm'
lh_joint_configs = {k.replace('_rh_', '_lh_'): v for k, v in joint_configs.items()}
lt = vis_lh.compute_link_transforms(lh_joint_configs)
if 'algr_lh_th_mp' in lt:
    sdf_fk_T = lt['algr_lh_th_mp']
    print(f"\n  SDF FK th_mp pos: {sdf_fk_T[:3,3].round(6)}")
    print(f"  PK FK th_mp pos:  {pk_T_lh[:3,3].round(6)}")
    diff_pos = np.linalg.norm(pk_T_lh[:3,3] - sdf_fk_T[:3,3])
    R_diff2 = pk_T_lh[:3,:3] @ sdf_fk_T[:3,:3].T
    angle2 = np.arccos(np.clip((np.trace(R_diff2) - 1) / 2, -1, 1))
    print(f"  pos diff: {diff_pos:.10f}")
    print(f"  angle diff: {np.degrees(angle2):.6f} deg")
    
    # Check all thumb links for LH
    for ln in ['algr_lh_th_mp', 'algr_lh_th_bs', 'algr_lh_th_px', 'algr_lh_th_ds']:
        if ln in lt and ln in fk_lh:
            pk_p = fk_lh[ln].get_matrix()[0].numpy()[:3,3]
            sdf_p = lt[ln][:3,3]
            print(f"  {ln}: pk={pk_p.round(6)}, sdf={sdf_p.round(6)}, diff={np.linalg.norm(pk_p-sdf_p):.10f}")
