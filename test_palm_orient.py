#!/usr/bin/env python3
"""Test palm orientation: place hand with base +z toward a red sphere."""
import torch, numpy as np, trimesh, viser, time, sys, os
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation
sys.path.insert(0, os.path.dirname(__file__))
from frogger.batched_pytorch_solver import _visual_meshes

server = viser.ViserServer(host="0.0.0.0", port=8090)

base_pos = np.array([0.1, 0, 0.1])
target = np.array([0, 0, 0.1])
z_dir = (target - base_pos); z_dir /= np.linalg.norm(z_dir)
y_dir = np.array([0, 0, 1])
x_dir = np.cross(y_dir, z_dir); x_dir /= np.linalg.norm(x_dir)
y_dir = np.cross(z_dir, x_dir)
R = np.stack([x_dir, y_dir, z_dir], axis=-1)
T_base = np.eye(4); T_base[:3, :3] = R; T_base[:3, 3] = base_pos

chain = pk.build_chain_from_urdf(open("models/leap_rh/leap.urdf").read())
q = torch.zeros(1, 16)
fk = chain.forward_kinematics(q)
vis = _visual_meshes("rh", "leap")

server.scene.add_icosphere("/target", radius=0.02, color=(255, 0, 0),
                           position=np.array([0, 0, 0.1], dtype=np.float32))
server.scene.add_icosphere("/arrow", radius=0.005, color=(0, 255, 0),
                           position=(base_pos + 0.05 * z_dir).astype(np.float32))

for link_name, mesh_list in vis.items():
    if link_name not in fk: continue
    wT = T_base @ fk[link_name].get_matrix()[0].numpy()
    for mi, (mf, vp) in enumerate(mesh_list):
        path = os.path.join("models/leap_rh", mf)
        if not os.path.exists(path): continue
        lm = trimesh.load(path, force="mesh")
        v = np.asarray(lm.vertices, dtype=np.float32)
        f = np.asarray(lm.faces, dtype=np.int32)
        full_T = wT.copy()
        if vp is not None:
            vpa = np.array(vp)
            Rv = Rotation.from_euler("xyz", vpa[3:]).as_matrix()
            Tv = np.eye(4); Tv[:3, :3] = Rv; Tv[:3, 3] = vpa[:3]
            full_T = full_T @ Tv
        vw = (full_T[:3, :3] @ v.T).T + full_T[:3, 3]
        color = (50, 100, 255) if "palm" in link_name else (255, 200, 100)
        server.scene.add_mesh_simple(f"/hand/{link_name}_{mi}",
            vertices=vw.astype(np.float32), faces=f, color=color, opacity=0.85)

print("http://localhost:8090")
print("Red=target, Green=base+z direction. Blue=palm, Orange=fingers")
try:
    while True: time.sleep(1)
except KeyboardInterrupt: pass
