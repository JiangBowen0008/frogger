#!/usr/bin/env python3
"""
Visualize a grasp with URDF collision boxes drawn as filled rectangles
overlaid on the object mesh. Shows exactly where collision occurs.

Run: conda run -n frogger python visualize_grasp_boxes.py --grasp output/grasps/compare_boxgrid.pt
"""
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import pytorch_kinematics as pk
import os, sys, argparse
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as ScipyR
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.dirname(__file__))
from frogger.batched_pytorch_solver import _visual_meshes

MESH_PATH = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg/black_spray_bottle_single/object.obj"
HAND_DIR = os.path.join(os.path.dirname(__file__), "models/leap_rh")
URDF_PATH = os.path.join(HAND_DIR, "leap.urdf")
OUT_DIR = os.path.join(os.path.dirname(__file__), "output/diagnostics/grasp_boxes")

FINGER_COLORS = {
    "palm": "#1f77b4",
    "if": "#d62728",
    "mf": "#2ca02c",
    "rf": "#ff7f0e",
    "th": "#9467bd",
}

def get_finger(link_name):
    for key in ["palm", "if", "mf", "rf", "th"]:
        if f"_{key}_" in link_name or link_name.endswith(f"_{key}") or f"_{key}" in link_name:
            if key == "if" and "mf" in link_name: continue
            if key == "rf" and "rf" not in link_name: continue
            return key
    if "palm" in link_name: return "palm"
    return "other"


def box_corners_world(size, origin_xyz, origin_rpy, world_T):
    """Get 8 box corners in world frame."""
    hx, hy, hz = [s/2 for s in size]
    corners = np.array([
        [-hx,-hy,-hz], [+hx,-hy,-hz], [+hx,+hy,-hz], [-hx,+hy,-hz],
        [-hx,-hy,+hz], [+hx,-hy,+hz], [+hx,+hy,+hz], [-hx,+hy,+hz],
    ])
    if np.any(np.abs(origin_rpy) > 1e-6):
        R = ScipyR.from_euler("xyz", origin_rpy).as_matrix()
        corners = (R @ corners.T).T
    corners += origin_xyz
    # To world
    corners_h = np.hstack([corners, np.ones((8,1))])
    corners_w = (world_T @ corners_h.T).T[:, :3]
    return corners_w


def project_hull(corners_3d, ax0, ax1):
    pts = corners_3d[:, [ax0, ax1]]
    try:
        hull = ConvexHull(pts)
        return pts[hull.vertices]
    except:
        return pts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grasp", required=True)
    parser.add_argument("--mesh", default=None)
    parser.add_argument("--idx", type=int, default=0, help="Grasp index in file")
    args = parser.parse_args()

    mesh_path = args.mesh or MESH_PATH
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load object
    obj_mesh = trimesh.load(mesh_path, force="mesh")
    bounds = obj_mesh.bounds
    offset = np.array([0.0, 0.0, -bounds[0, 2]])
    X_WO = np.eye(4); X_WO[:3, 3] = offset
    obj_verts = (X_WO[:3,:3] @ np.asarray(obj_mesh.vertices, dtype=np.float64).T).T + X_WO[:3, 3]

    # Load grasp
    data = torch.load(args.grasp, weights_only=False, map_location="cpu")
    g = data[args.idx] if isinstance(data, list) else data
    q = torch.tensor(g["q_joints"], dtype=torch.float32).unsqueeze(0)
    T_base = np.eye(4)
    T_base[:3, :3] = np.array(g["base_rot"])
    T_base[:3, 3] = np.array(g["base_pos"])

    # FK
    with open(URDF_PATH) as f:
        chain = pk.build_chain_from_urdf(f.read())
    fk = chain.forward_kinematics(q)

    # Parse URDF boxes
    tree = ET.parse(URDF_PATH)
    link_boxes = {}  # link_name -> list of (corners_world)
    for le in tree.getroot().findall("link"):
        ln = le.get("name")
        if ln not in fk: continue
        link_T = fk[ln].get_matrix()[0].numpy().astype(np.float64)
        world_T = T_base @ link_T

        for col in le.findall("collision"):
            geo = col.find("geometry")
            if geo is None: continue
            box = geo.find("box")
            if box is None: continue
            size = [float(x) for x in box.get("size").split()]
            o = col.find("origin")
            xyz = [float(x) for x in o.get("xyz", "0 0 0").split()]
            rpy = [float(x) for x in o.get("rpy", "0 0 0").split()]
            corners = box_corners_world(size, xyz, rpy, world_T)
            link_boxes.setdefault(ln, []).append(corners)

    # Also load visual mesh vertices for context
    vis = _visual_meshes("rh", "leap")
    vis_verts = {}
    for ln in vis:
        if ln not in fk: continue
        link_T = fk[ln].get_matrix()[0].numpy().astype(np.float64)
        world_T = T_base @ link_T
        for mesh_file, vis_pose in vis[ln]:
            path = os.path.join(HAND_DIR, mesh_file)
            if not os.path.exists(path): continue
            lm = trimesh.load(path, force="mesh")
            v = np.asarray(lm.vertices, dtype=np.float64)
            if vis_pose is not None:
                vp = np.array(vis_pose, dtype=np.float64)
                Rv = ScipyR.from_euler("xyz", vp[3:]).as_matrix()
                Tv = np.eye(4); Tv[:3,:3] = Rv; Tv[:3,3] = vp[:3]
                world_T_v = world_T @ Tv
            else:
                world_T_v = world_T
            vw = (world_T_v[:3,:3] @ v.T).T + world_T_v[:3,3]
            vis_verts.setdefault(ln, []).append(vw)
    for ln in vis_verts:
        vis_verts[ln] = np.vstack(vis_verts[ln])

    # Plot projections
    projections = [
        ("XY", 0, 1, "X (mm)", "Y (mm)"),
        ("YZ", 1, 2, "Y (mm)", "Z (mm)"),
        ("XZ", 0, 2, "X (mm)", "Z (mm)"),
    ]

    for proj_name, ax0, ax1, xlabel, ylabel in projections:
        fig, ax = plt.subplots(figsize=(14, 10))

        # Object (denser, more visible)
        ax.scatter(obj_verts[:, ax0]*1000, obj_verts[:, ax1]*1000,
                  c="lightgray", s=0.5, alpha=0.6, zorder=1, rasterized=True)

        # URDF boxes ONLY as filled polygons (no visual mesh)
        for ln, box_list in link_boxes.items():
            finger = get_finger(ln)
            color = FINGER_COLORS.get(finger, "gray")
            short = ln.replace("leap_rh_", "")
            for bi, corners in enumerate(box_list):
                polygon = project_hull(corners, ax0, ax1) * 1000
                poly = plt.Polygon(polygon, closed=True,
                                  facecolor=color, alpha=0.4,
                                  edgecolor=color, linewidth=2.0, zorder=4)
                ax.add_patch(poly)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Grasp with URDF boxes — {proj_name}")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Legend
        from matplotlib.patches import Patch
        legend = [Patch(facecolor="lightgray", label="object")]
        for name, color in FINGER_COLORS.items():
            legend.append(Patch(facecolor=color, alpha=0.5, edgecolor=color, label=name))
        ax.legend(handles=legend, loc="best", fontsize=9)

        path = os.path.join(OUT_DIR, f"grasp_boxes_{proj_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


if __name__ == "__main__":
    main()
