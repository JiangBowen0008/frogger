import trimesh, numpy as np
from scipy.spatial.transform import Rotation as R
import torch, pytorch_kinematics as pk

# LEAP IF/MF/RF fingertip
ft = trimesh.load('models/leap_rh/meshes_obj/fingertip.obj', force='mesh')
vis_origin = np.array([0.013286, -0.006114, 0.0145])
vis_rpy = np.array([np.pi, 0, 0])
Rv = R.from_euler('xyz', vis_rpy).as_matrix()
verts_link = (Rv @ ft.vertices.T).T + vis_origin

y_min = verts_link[:, 1].min()
y_20 = y_min + 0.2 * (verts_link[:, 1].max() - y_min)
bottom_verts = verts_link[verts_link[:, 1] < y_20]
tip_center = bottom_verts.mean(axis=0)
print(f'Finger tip offset (link frame): {tip_center}')

# LEAP thumb tip
th = trimesh.load('models/leap_rh/meshes_obj/thumb_fingertip.obj', force='mesh')
th_origin = np.array([0.062560, 0.078460, 0.048993])
th_verts_link = th.vertices + th_origin

th_y_min = th_verts_link[:, 1].min()
th_y_20 = th_y_min + 0.2 * (th_verts_link[:, 1].max() - th_y_min)
th_bottom = th_verts_link[th_verts_link[:, 1] < th_y_20]
th_tip_center = th_bottom.mean(axis=0)
print(f'Thumb tip offset (link frame): {th_tip_center}')

# Verify at zero config
chain = pk.build_chain_from_urdf(open('models/leap_rh/leap.urdf').read())
q = torch.zeros(1, 16)
fk = chain.forward_kinematics(q)
for name in ['leap_rh_if_ds', 'leap_rh_mf_ds', 'leap_rh_rf_ds', 'leap_rh_th_ds']:
    T = fk[name].get_matrix()[0].numpy()
    if 'th' in name:
        tip_world = T[:3,:3] @ th_tip_center + T[:3,3]
    else:
        tip_world = T[:3,:3] @ tip_center + T[:3,3]
    print(f'{name}: link={T[:3,3].round(4)}, tip={tip_world.round(4)}')
