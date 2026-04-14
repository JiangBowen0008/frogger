#!/usr/bin/env python3
"""
Overlay URDF collision box primitives on top of visual meshes.
Draws actual box rectangles in each 2D projection.

Run: conda run -n frogger python compare_primitives_vs_visual.py
"""
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import os
import sys
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as ScipyR

sys.path.insert(0, os.path.dirname(__file__))
from frogger.batched_pytorch_solver import _visual_meshes

HAND_DIR = os.path.join(os.path.dirname(__file__), "models/leap_rh")
URDF_PATH = os.path.join(HAND_DIR, "leap.urdf")
OUT_DIR = os.path.join(os.path.dirname(__file__), "output/diagnostics/primitives_vs_visual")


def get_box_corners(size, origin_xyz, origin_rpy):
    """Get 8 corners of a box in link-local frame."""
    hx, hy, hz = [s/2 for s in size]
    corners = np.array([
        [-hx, -hy, -hz], [+hx, -hy, -hz], [+hx, +hy, -hz], [-hx, +hy, -hz],
        [-hx, -hy, +hz], [+hx, -hy, +hz], [+hx, +hy, +hz], [-hx, +hy, +hz],
    ])
    if np.any(np.abs(origin_rpy) > 1e-6):
        R = ScipyR.from_euler("xyz", origin_rpy).as_matrix()
        corners = (R @ corners.T).T
    corners += origin_xyz
    return corners


def project_box_to_2d(corners, ax0, ax1):
    """Project 8 box corners to 2D and return the convex hull polygon."""
    from scipy.spatial import ConvexHull
    pts_2d = corners[:, [ax0, ax1]]
    try:
        hull = ConvexHull(pts_2d)
        polygon = pts_2d[hull.vertices]
        return polygon
    except:
        return pts_2d


def load_visual_mesh_in_link_frame(link_name):
    vis = _visual_meshes("rh", "leap")
    if link_name not in vis:
        return None
    all_verts = []
    for mesh_file, vis_pose in vis[link_name]:
        path = os.path.join(HAND_DIR, mesh_file)
        if not os.path.exists(path):
            continue
        m = trimesh.load(path, force="mesh")
        verts = np.asarray(m.vertices, dtype=np.float64)
        if vis_pose is not None:
            vp = np.array(vis_pose, dtype=np.float64)
            R = ScipyR.from_euler("xyz", vp[3:]).as_matrix()
            verts = (R @ verts.T).T + vp[:3]
        all_verts.append(verts)
    return np.vstack(all_verts) if all_verts else None


def load_collision_boxes(link_name):
    tree = ET.parse(URDF_PATH)
    boxes = []
    for le in tree.getroot().findall("link"):
        if le.get("name") != link_name:
            continue
        for col in le.findall("collision"):
            g = col.find("geometry")
            if g is None: continue
            b = g.find("box")
            if b is None: continue
            size = [float(x) for x in b.get("size").split()]
            o = col.find("origin")
            xyz = [float(x) for x in o.get("xyz", "0 0 0").split()] if o is not None else [0,0,0]
            rpy = [float(x) for x in o.get("rpy", "0 0 0").split()] if o is not None else [0,0,0]
            name = col.get("name", "")
            corners = get_box_corners(size, xyz, rpy)
            boxes.append({"size": size, "xyz": xyz, "rpy": rpy, "name": name, "corners": corners})
    return boxes


def plot_link(link_name, vis_verts, boxes, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    short = link_name.replace("leap_rh_", "")

    projections = [
        ("XY", 0, 1, "X (m)", "Y (m)"),
        ("YZ", 1, 2, "Y (m)", "Z (m)"),
        ("XZ", 0, 2, "X (m)", "Z (m)"),
    ]

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
              '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62']

    for proj_name, ax0, ax1, xlabel, ylabel in projections:
        fig, ax = plt.subplots(figsize=(12, 9))

        # Visual mesh as scatter
        if vis_verts is not None:
            ax.scatter(vis_verts[:, ax0] * 1000, vis_verts[:, ax1] * 1000,
                      c="lightblue", s=0.3, alpha=0.4, zorder=1, rasterized=True)

        # Draw boxes as filled rectangles with edges
        for bi, box in enumerate(boxes):
            polygon = project_box_to_2d(box["corners"], ax0, ax1) * 1000  # to mm
            color = colors[bi % len(colors)]
            poly_patch = plt.Polygon(polygon, closed=True,
                                     facecolor=color, alpha=0.25,
                                     edgecolor=color, linewidth=2,
                                     zorder=3)
            ax.add_patch(poly_patch)
            # Label at center
            cx, cy = polygon.mean(axis=0)
            ax.text(cx, cy, str(bi), fontsize=8, ha='center', va='center',
                   fontweight='bold', color=color, zorder=4)

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor='lightblue', markersize=8,
                                  label='visual mesh')]
        for bi, box in enumerate(boxes):
            color = colors[bi % len(colors)]
            legend_elements.append(
                patches.Patch(facecolor=color, alpha=0.4, edgecolor=color,
                             label=f'box {bi}'))
        ax.legend(handles=legend_elements, loc="best", fontsize=8)

        ax.set_xlabel(f"{xlabel} (mm)")
        ax.set_ylabel(f"{ylabel} (mm)")
        ax.set_title(f"{short} — {proj_name}: visual mesh vs URDF collision boxes")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        path = os.path.join(out_dir, f"{short}_{proj_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  {short}: saved")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    tree = ET.parse(URDF_PATH)
    link_names = [le.get("name") for le in tree.getroot().findall("link")
                  if le.findall("collision")]

    print(f"Links with collision geometry: {len(link_names)}")

    for link_name in link_names:
        vis_verts = load_visual_mesh_in_link_frame(link_name)
        boxes = load_collision_boxes(link_name)
        if vis_verts is None and not boxes:
            continue
        plot_link(link_name, vis_verts, boxes, OUT_DIR)

    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
