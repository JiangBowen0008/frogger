#!/usr/bin/env python3
"""Visualize LEAP hand collision boxes in viser (3D interactive)."""

import os, sys, numpy as np, torch, trimesh, time
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation as ScipyR
import xml.etree.ElementTree as ET
import viser

sys.path.insert(0, os.path.dirname(__file__))
from frogger.batched_pytorch_solver import _visual_meshes

URDF_PATH = os.path.join(os.path.dirname(__file__), "models/leap_rh/leap.urdf")
MESH_DIR = os.path.join(os.path.dirname(__file__), "models/leap_rh")

COLLISION_LINKS = [
    "leap_rh_palm",
    "leap_rh_if_bs", "leap_rh_if_px", "leap_rh_if_md", "leap_rh_if_ds",
    "leap_rh_mf_bs", "leap_rh_mf_px", "leap_rh_mf_md", "leap_rh_mf_ds",
    "leap_rh_rf_bs", "leap_rh_rf_px", "leap_rh_rf_md", "leap_rh_rf_ds",
    "leap_rh_th_mp", "leap_rh_th_bs", "leap_rh_th_px", "leap_rh_th_ds",
]

def make_box_mesh(half_extents):
    """Create a trimesh box."""
    return trimesh.creation.box(extents=np.array(half_extents) * 2)

server = viser.ViserServer(host="0.0.0.0", port=8090)
print("Hand anatomy viewer -> http://localhost:8090")

# FK at rest pose
with open(URDF_PATH) as f:
    chain = pk.build_chain_from_urdf(f.read())
q = torch.zeros(1, 16, dtype=torch.float32)
fk = chain.forward_kinematics(q)

tree = ET.parse(URDF_PATH)

# Visual meshes (translucent)
vis = _visual_meshes("rh", "leap")
for link_name in COLLISION_LINKS:
    if link_name not in vis or link_name not in fk:
        continue
    link_T = fk[link_name].get_matrix()[0].numpy()
    is_palm = "palm" in link_name
    for mi, (mf, vp) in enumerate(vis[link_name]):
        path = os.path.join(MESH_DIR, mf)
        if not os.path.exists(path): continue
        lm = trimesh.load(path, force="mesh")
        v = np.asarray(lm.vertices, dtype=np.float32)
        f = np.asarray(lm.faces, dtype=np.int32)
        wT = link_T.copy()
        if vp is not None:
            vpa = np.array(vp, dtype=np.float64)
            Rv = ScipyR.from_euler("xyz", vpa[3:]).as_matrix()
            Tv = np.eye(4); Tv[:3, :3] = Rv; Tv[:3, 3] = vpa[:3]
            wT = wT @ Tv
        vw = (wT[:3, :3] @ v.T).T + wT[:3, 3]
        color = (180, 180, 200) if is_palm else (200, 200, 220)
        server.scene.add_mesh_simple(
            f"/mesh/{link_name}_{mi}", vertices=vw.astype(np.float32), faces=f,
            color=color, opacity=0.25)

# Collision boxes
for link_name in COLLISION_LINKS:
    le = None
    for e in tree.getroot().findall("link"):
        if e.get("name") == link_name:
            le = e
            break
    if le is None or link_name not in fk:
        continue
    link_T = fk[link_name].get_matrix()[0].numpy()
    is_palm = "palm" in link_name

    for ci, col in enumerate(le.findall("collision")):
        g = col.find("geometry")
        if g is None: continue
        b = g.find("box")
        if b is None: continue
        sz = [float(x) for x in b.get("size").split()]
        o = col.find("origin")
        xyz = np.array([float(x) for x in o.get("xyz", "0 0 0").split()])
        rpy = np.array([float(x) for x in o.get("rpy", "0 0 0").split()])
        R_local = (ScipyR.from_euler("xyz", rpy).as_matrix()
                   if np.any(np.abs(rpy) > 1e-6) else np.eye(3))

        # Build local transform for this box
        box_T = np.eye(4)
        box_T[:3, :3] = R_local
        box_T[:3, 3] = xyz
        world_T = link_T @ box_T

        # Classify
        if is_palm and xyz[0] < -0.025:
            color = (255, 50, 50)   # red = back palm (skipped)
            cat = "back"
        elif is_palm:
            color = (50, 255, 50)   # green = front palm (checked)
            cat = "front"
        else:
            color = (50, 130, 255)  # blue = finger
            cat = "finger"

        # Create box mesh
        box = make_box_mesh(np.array(sz) / 2)
        v = np.asarray(box.vertices, dtype=np.float32)
        f = np.asarray(box.faces, dtype=np.int32)
        vw = (world_T[:3, :3] @ v.T).T + world_T[:3, 3]

        short = link_name.replace("leap_rh_", "")
        server.scene.add_mesh_simple(
            f"/boxes/{short}_{ci}_{cat}", vertices=vw.astype(np.float32), faces=f,
            color=color, opacity=0.4)

print("  Green = front palm (collision checked)")
print("  Red = back palm (skipped)")
print("  Blue = finger boxes")
print("  Translucent gray = visual mesh")
print("Press Ctrl+C to stop")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
