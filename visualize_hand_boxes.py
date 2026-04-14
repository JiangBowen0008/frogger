#!/usr/bin/env python3
"""
Visualize the LEAP hand at rest pose with collision boxes color-coded:
- GREEN: front palm boxes (used for collision)
- RED: back palm boxes (skipped — structural/motor housing)
- BLUE: finger boxes

Shows the hand mesh (translucent) with box outlines overlaid,
so you can see what the collision primitives actually cover.
"""

import os, sys, numpy as np, torch, trimesh
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation as ScipyR
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.dirname(__file__))
from frogger.batched_pytorch_solver import _visual_meshes

URDF_PATH = os.path.join(os.path.dirname(__file__), "models/leap_rh/leap.urdf")
MESH_DIR = os.path.join(os.path.dirname(__file__), "models/leap_rh")
OUTPUT_DIR = "output/diagnostics/hand_anatomy"

COLLISION_LINKS = [
    "leap_rh_palm",
    "leap_rh_if_bs", "leap_rh_if_px", "leap_rh_if_md", "leap_rh_if_ds",
    "leap_rh_mf_bs", "leap_rh_mf_px", "leap_rh_mf_md", "leap_rh_mf_ds",
    "leap_rh_rf_bs", "leap_rh_rf_px", "leap_rh_rf_md", "leap_rh_rf_ds",
    "leap_rh_th_mp", "leap_rh_th_bs", "leap_rh_th_px", "leap_rh_th_ds",
]


def box_edges(center, half_extents, R):
    """Return 12 line segments (pairs of points) for a 3D box."""
    c = np.array(center)
    h = np.array(half_extents)
    # 8 corners in local frame
    signs = np.array([[s0, s1, s2] for s0 in [-1, 1] for s1 in [-1, 1] for s2 in [-1, 1]])
    corners_local = signs * h
    corners_world = (R @ corners_local.T).T + c
    # 12 edges: connect corners that differ in exactly 1 coordinate
    edges = []
    for i in range(8):
        for j in range(i+1, 8):
            diff = np.sum(signs[i] != signs[j])
            if diff == 1:
                edges.append((corners_world[i], corners_world[j]))
    return edges


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # FK at rest pose (all joints = 0)
    with open(URDF_PATH) as f:
        chain = pk.build_chain_from_urdf(f.read())
    q = torch.zeros(1, 16, dtype=torch.float32)
    fk = chain.forward_kinematics(q)

    # Parse URDF boxes
    tree = ET.parse(URDF_PATH)

    # Collect boxes with classification
    boxes = []  # (link_name, center_world, half_extents, R_world, category)
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

        for col in le.findall("collision"):
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

            # Transform to world frame
            center_local = xyz
            center_world = link_T[:3, :3] @ center_local + link_T[:3, 3]
            R_world = link_T[:3, :3] @ R_local
            half_ext = np.array(sz) / 2

            if is_palm and xyz[0] < -0.025:
                cat = "back_palm"
            elif is_palm:
                cat = "front_palm"
            else:
                cat = "finger"

            boxes.append((link_name, center_world, half_ext, R_world, cat))

    # Load visual meshes for hand outline
    vis = _visual_meshes("rh", "leap")
    mesh_pts = {"palm": [], "finger": []}
    for link_name in COLLISION_LINKS:
        if link_name not in vis or link_name not in fk:
            continue
        link_T = fk[link_name].get_matrix()[0].numpy()
        is_palm = "palm" in link_name
        for mf, vp in vis[link_name]:
            path = os.path.join(MESH_DIR, mf)
            if not os.path.exists(path): continue
            lm = trimesh.load(path, force="mesh")
            v = np.asarray(lm.vertices, dtype=np.float64)
            wT = link_T.copy()
            if vp is not None:
                vpa = np.array(vp, dtype=np.float64)
                Rv = ScipyR.from_euler("xyz", vpa[3:]).as_matrix()
                Tv = np.eye(4); Tv[:3, :3] = Rv; Tv[:3, 3] = vpa[:3]
                wT = wT @ Tv
            vw = (wT[:3, :3] @ v.T).T + wT[:3, 3]
            key = "palm" if is_palm else "finger"
            mesh_pts[key].append(vw)

    palm_mesh = np.vstack(mesh_pts["palm"]) if mesh_pts["palm"] else np.zeros((0, 3))
    finger_mesh = np.vstack(mesh_pts["finger"]) if mesh_pts["finger"] else np.zeros((0, 3))

    # Color map
    colors = {"front_palm": "green", "back_palm": "red", "finger": "dodgerblue"}
    labels = {"front_palm": "Front palm (collision ON)",
              "back_palm": "Back palm (skipped)",
              "finger": "Finger boxes"}

    # Draw projections
    projections = [("XY", 0, 1), ("YZ", 1, 2), ("XZ", 0, 2)]
    ax_labels = {0: "X (palm inner→)", 1: "Y (width)", 2: "Z (height)"}

    for pname, a0, a1 in projections:
        fig, ax = plt.subplots(figsize=(14, 10))

        # Hand mesh outline (very light)
        if len(palm_mesh) > 0:
            ax.scatter(palm_mesh[:, a0], palm_mesh[:, a1],
                       c="lightyellow", s=0.3, alpha=0.15, zorder=1, edgecolors="none")
        if len(finger_mesh) > 0:
            ax.scatter(finger_mesh[:, a0], finger_mesh[:, a1],
                       c="lightcyan", s=0.3, alpha=0.15, zorder=1, edgecolors="none")

        # Draw box edges
        drawn_labels = set()
        for link_name, center, half_ext, R, cat in boxes:
            edges = box_edges(center, half_ext, R)
            c = colors[cat]
            lbl = labels[cat] if cat not in drawn_labels else None
            drawn_labels.add(cat)
            for p0, p1 in edges:
                ax.plot([p0[a0], p1[a0]], [p0[a1], p1[a1]],
                        c=c, linewidth=1.5, alpha=0.7, label=lbl, zorder=3)
                lbl = None  # only label once

        # Draw box centers
        for link_name, center, half_ext, R, cat in boxes:
            ax.plot(center[a0], center[a1], 'o', c=colors[cat],
                    markersize=4, zorder=4)

        # Annotate palm +x direction (inner surface)
        palm_link_T = fk["leap_rh_palm"].get_matrix()[0].numpy()
        palm_origin = palm_link_T[:3, 3]
        palm_x_dir = palm_link_T[:3, 0]  # +x axis = inner surface direction
        arrow_len = 0.03
        ax.annotate("", xy=(palm_origin[a0] + palm_x_dir[a0]*arrow_len,
                            palm_origin[a1] + palm_x_dir[a1]*arrow_len),
                     xytext=(palm_origin[a0], palm_origin[a1]),
                     arrowprops=dict(arrowstyle="->", color="black", lw=2),
                     zorder=5)
        ax.text(palm_origin[a0] + palm_x_dir[a0]*arrow_len*1.2,
                palm_origin[a1] + palm_x_dir[a1]*arrow_len*1.2,
                "+x (inner\nsurface)", fontsize=8, ha="center", zorder=5)

        ax.set_xlabel(ax_labels[a0], fontsize=11)
        ax.set_ylabel(ax_labels[a1], fontsize=11)
        ax.set_title(f"LEAP RH — {pname} (rest pose, q=0)\n"
                     f"Green=front palm (checked), Red=back palm (skipped), Blue=fingers",
                     fontsize=12)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")

        path = os.path.join(OUTPUT_DIR, f"hand_anatomy_{pname}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    print(f"\nBox counts: front_palm={sum(1 for b in boxes if b[4]=='front_palm')}, "
          f"back_palm={sum(1 for b in boxes if b[4]=='back_palm')}, "
          f"finger={sum(1 for b in boxes if b[4]=='finger')}")
    print("Done.")


if __name__ == "__main__":
    main()
