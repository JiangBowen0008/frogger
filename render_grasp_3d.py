#!/usr/bin/env python3
"""Render a grasp from 3 viewpoints for honest visual verification."""
import os, sys, numpy as np, torch, trimesh
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation as ScipyR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from frogger.batched_pytorch_solver import _visual_meshes

URDF_PATH = os.path.join(os.path.dirname(__file__), "models/leap_rh/leap.urdf")
MESH_DIR = os.path.join(os.path.dirname(__file__), "models/leap_rh")

def render(grasp_path, mesh_path, tag="", grasp_idx=0):
    out_dir = "output/diagnostics/render_3d"
    os.makedirs(out_dir, exist_ok=True)

    results = torch.load(grasp_path, weights_only=False, map_location="cpu")
    g = results[grasp_idx] if isinstance(results, list) else results

    obj_mesh = trimesh.load(mesh_path, force="mesh")
    offset = np.array([0.0, 0.0, -obj_mesh.bounds[0, 2]])
    X_WO = np.eye(4); X_WO[:3, 3] = offset

    # Create a scene
    scene = trimesh.Scene()

    # Add object
    obj_mesh.apply_transform(X_WO)
    obj_mesh.visual.face_colors = [180, 180, 180, 150]
    scene.add_geometry(obj_mesh, node_name="object")

    # Add hand meshes
    with open(URDF_PATH) as f:
        chain = pk.build_chain_from_urdf(f.read())
    q = torch.tensor(g["q_joints"], dtype=torch.float32).unsqueeze(0)
    fk = chain.forward_kinematics(q)
    T_base = np.eye(4)
    T_base[:3, :3] = g["base_rot"]
    T_base[:3, 3] = g["base_pos"]

    vis = _visual_meshes("rh", "leap")
    for link_name, mesh_list in vis.items():
        if link_name not in fk:
            continue
        link_T = fk[link_name].get_matrix()[0].numpy()
        wT = T_base @ link_T
        is_palm = "palm" in link_name
        for mi, (mf, vp) in enumerate(mesh_list):
            path = os.path.join(MESH_DIR, mf)
            if not os.path.exists(path): continue
            lm = trimesh.load(path, force="mesh")
            full_T = wT.copy()
            if vp is not None:
                vpa = np.array(vp, dtype=np.float64)
                Rv = ScipyR.from_euler("xyz", vpa[3:]).as_matrix()
                Tv = np.eye(4); Tv[:3, :3] = Rv; Tv[:3, 3] = vpa[:3]
                full_T = full_T @ Tv
            lm.apply_transform(full_T)
            if is_palm:
                lm.visual.face_colors = [50, 100, 255, 200]  # blue palm
            else:
                lm.visual.face_colors = [255, 200, 100, 200]  # orange fingers
            scene.add_geometry(lm, node_name=f"{link_name}_{mi}")

    # Render from 3 views
    views = {
        "front": {"angles": [np.pi/2, 0, 0], "center": [0, 0, 0.09]},
        "side": {"angles": [np.pi/2, 0, np.pi/2], "center": [0, 0, 0.09]},
        "top": {"angles": [0, 0, 0], "center": [0, 0, 0.09]},
    }

    # Use trimesh scene rendering
    for view_name, params in views.items():
        try:
            png = scene.save_image(resolution=[800, 600])
            path = os.path.join(out_dir, f"render_{tag}_{view_name}.png")
            with open(path, "wb") as f:
                f.write(png)
            print(f"Saved: {path}")
        except Exception as e:
            print(f"Render failed ({view_name}): {e}")

    # Also print palm orientation info
    palm_inward = -g["base_rot"][:, 2]
    obj_center = np.array([0, 0, offset[2] + (obj_mesh.bounds[1, 2] - obj_mesh.bounds[0, 2]) / 2])
    to_center = obj_center - g["base_pos"]
    to_center /= np.linalg.norm(to_center)
    dot = np.dot(palm_inward, to_center)
    print(f"\nGrasp {grasp_idx}: palm_inward·toward_center = {dot:.3f}")
    print(f"  {'PALM FACES OBJECT' if dot > 0.3 else 'PALM FACES AWAY' if dot < -0.3 else 'PALM SIDEWAYS'}")
    print(f"  feasible={g.get('feasible')}, σ_min={g.get('sigma_min', 0):.3f}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--grasp", required=True)
    p.add_argument("--mesh", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--idx", type=int, default=0)
    a = p.parse_args()
    render(a.grasp, a.mesh, a.tag, a.idx)
