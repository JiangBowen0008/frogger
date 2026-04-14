#!/usr/bin/env python3
"""
Exp 4: Visualize what the optimizer's collision check actually sees.

Overlays the box-grid collision points (as used by the solver) on the grasp
projections, colored by SDF sign. Shows which points are inside (red) vs
outside (blue), and which are the face-only palm points vs full-volume finger points.

This answers: "Does the optimizer know the palm is going through the object?"
"""

import os, sys, numpy as np, torch, trimesh
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation as ScipyR
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from frogger.batched_pytorch_solver import BatchedSDF, _visual_meshes

MESH_PATH = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg/black_spray_bottle_single/object.obj"
URDF_PATH = os.path.join(os.path.dirname(__file__), "models/leap_rh/leap.urdf")
MESH_DIR = os.path.join(os.path.dirname(__file__), "models/leap_rh")
GRASP_PATH = "output/grasps/compare_collision_fixed.pt"
OUTPUT_DIR = "output/diagnostics/collision_overlay"

COLLISION_LINKS = [
    "leap_rh_palm",
    "leap_rh_if_bs", "leap_rh_if_px", "leap_rh_if_md", "leap_rh_if_ds",
    "leap_rh_mf_bs", "leap_rh_mf_px", "leap_rh_mf_md", "leap_rh_mf_ds",
    "leap_rh_rf_bs", "leap_rh_rf_px", "leap_rh_rf_md", "leap_rh_rf_ds",
    "leap_rh_th_mp", "leap_rh_th_bs", "leap_rh_th_px", "leap_rh_th_ds",
]


def build_collision_points(pitch=0.008, palm_face_only=True):
    """Build collision points exactly as the solver does."""
    tree = ET.parse(URDF_PATH)
    col_data = {}  # link_name -> (pts_local, is_face_only)

    for link_name in COLLISION_LINKS:
        le = None
        for e in tree.getroot().findall("link"):
            if e.get("name") == link_name:
                le = e
                break
        if le is None:
            continue

        is_palm = "palm" in link_name
        link_pts = []
        for col in le.findall("collision"):
            g = col.find("geometry")
            if g is None: continue
            b = g.find("box")
            if b is None: continue
            sz = [float(x) for x in b.get("size").split()]
            o = col.find("origin")
            xyz = np.array([float(x) for x in o.get("xyz", "0 0 0").split()])
            rpy = np.array([float(x) for x in o.get("rpy", "0 0 0").split()])
            R = (ScipyR.from_euler("xyz", rpy).as_matrix()
                 if np.any(np.abs(rpy) > 1e-6) else np.eye(3))
            if is_palm and xyz[0] < -0.025:
                continue
            hx, hy, hz = sz[0]/2, sz[1]/2, sz[2]/2
            if is_palm and palm_face_only:
                gx = np.array([hx])  # face only
            else:
                gx = np.arange(-hx, hx + pitch/2, pitch)
            gy = np.arange(-hy, hy + pitch/2, pitch)
            gz = np.arange(-hz, hz + pitch/2, pitch)
            grid = np.stack(np.meshgrid(gx, gy, gz, indexing='ij'),
                            axis=-1).reshape(-1, 3)
            grid = ((R @ grid.T).T + xyz).astype(np.float32)
            link_pts.append(grid)

        if link_pts:
            col_data[link_name] = np.vstack(link_pts)

    return col_data


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load grasp
    g = torch.load(GRASP_PATH, weights_only=False, map_location="cpu")
    if isinstance(g, list):
        g = g[0]
    q_joints = np.array(g["q_joints"], dtype=np.float64)
    base_pos = np.array(g["base_pos"], dtype=np.float64)
    base_rot = np.array(g["base_rot"], dtype=np.float64)

    # Load mesh + SDF
    mesh = trimesh.load(MESH_PATH, force="mesh")
    bounds = mesh.bounds
    offset = np.array([0.0, 0.0, -bounds[0, 2]])
    X_WO = np.eye(4); X_WO[:3, 3] = offset
    verts_W = (X_WO[:3, :3] @ np.asarray(mesh.vertices, dtype=np.float64).T).T + X_WO[:3, 3]
    sdf = BatchedSDF(mesh, X_WO, resolution=128, device="cuda")

    # FK
    with open(URDF_PATH) as f:
        chain = pk.build_chain_from_urdf(f.read())
    q = torch.tensor(q_joints, dtype=torch.float32).unsqueeze(0)
    fk = chain.forward_kinematics(q)
    T_base = np.eye(4); T_base[:3, :3] = base_rot; T_base[:3, 3] = base_pos
    bT = torch.tensor(T_base, dtype=torch.float32, device="cuda").unsqueeze(0)

    # Also get visual mesh vertices for hand outline (visualization only)
    vis = _visual_meshes("rh", "leap")
    hand_verts_all = []
    for chain_name, link_list in [
        ("palm", ["leap_rh_palm"]),
        ("fingers", ["leap_rh_if_bs","leap_rh_if_px","leap_rh_if_md","leap_rh_if_ds",
                     "leap_rh_mf_bs","leap_rh_mf_px","leap_rh_mf_md","leap_rh_mf_ds",
                     "leap_rh_rf_bs","leap_rh_rf_px","leap_rh_rf_md","leap_rh_rf_ds",
                     "leap_rh_th_mp","leap_rh_th_bs","leap_rh_th_px","leap_rh_th_ds"]),
    ]:
        for ln in link_list:
            if ln not in vis or ln not in fk:
                continue
            for mf, vp in vis[ln]:
                path = os.path.join(MESH_DIR, mf)
                if not os.path.exists(path): continue
                lm = trimesh.load(path, force="mesh")
                v = np.asarray(lm.vertices, dtype=np.float64)
                link_T = fk[ln].get_matrix()[0].numpy().astype(np.float64)
                wT = T_base @ link_T
                if vp is not None:
                    vpa = np.array(vp, dtype=np.float64)
                    Rv = ScipyR.from_euler("xyz", vpa[3:]).as_matrix()
                    Tv = np.eye(4); Tv[:3, :3] = Rv; Tv[:3, 3] = vpa[:3]
                    wT = wT @ Tv
                vw = (wT[:3, :3] @ v.T).T + wT[:3, 3]
                hand_verts_all.append(vw)
    hand_verts = np.vstack(hand_verts_all)

    # Build collision points — CURRENT solver (face-only palm)
    col_face = build_collision_points(pitch=0.008, palm_face_only=True)
    # Build collision points — FULL VOLUME palm for comparison
    col_full = build_collision_points(pitch=0.005, palm_face_only=False)

    # Transform and query SDF for both
    for label, col_data, pitch_label in [
        ("solver_sees", col_face, "8mm, face-only palm"),
        ("full_volume", col_full, "5mm, full volume"),
    ]:
        all_pts = []
        all_sdf = []
        all_is_palm = []
        for ln, pts_local in col_data.items():
            if ln not in fk: continue
            pts_h = torch.tensor(
                np.hstack([pts_local, np.ones((len(pts_local), 1), dtype=np.float32)]),
                device="cuda")
            link_T = fk[ln].get_matrix().to("cuda")
            world_T = bT @ link_T
            pts_w = (world_T[0] @ pts_h.T).T[:, :3]
            sdf_vals = sdf.query(pts_w.unsqueeze(0)).squeeze(0).cpu().numpy()
            all_pts.append(pts_w.cpu().numpy())
            all_sdf.append(sdf_vals)
            all_is_palm.extend(["palm" in ln] * len(sdf_vals))

        pts = np.vstack(all_pts)
        sdf_v = np.concatenate(all_sdf)
        is_palm = np.array(all_is_palm)

        n_inside = (sdf_v < 0).sum()
        n_palm_inside = ((sdf_v < 0) & is_palm).sum()
        print(f"\n  {label} ({pitch_label}): {len(pts)} pts, "
              f"{n_inside} inside ({n_inside/len(pts)*100:.1f}%), "
              f"palm inside: {n_palm_inside}")

        # Draw projections
        projections = [("XY", 0, 1), ("YZ", 1, 2), ("XZ", 0, 2)]
        ax_labels = {0: "X", 1: "Y", 2: "Z"}
        for pname, a0, a1 in projections:
            fig, ax = plt.subplots(figsize=(12, 10))

            # Object outline (gray)
            ax.scatter(verts_W[:, a0], verts_W[:, a1], c="lightgray", s=0.3, alpha=0.3, zorder=1)

            # Hand outline (very light, for context)
            ax.scatter(hand_verts[:, a0], hand_verts[:, a1], c="lightyellow", s=0.1, alpha=0.15, zorder=2,
                       edgecolors="none")

            # Collision points — outside (blue)
            outside = sdf_v >= 0
            if outside.sum() > 0:
                ax.scatter(pts[outside, a0], pts[outside, a1],
                           c="blue", s=6, alpha=0.5, zorder=3, label=f"outside ({outside.sum()})")
            # Collision points — inside (red)
            inside = sdf_v < 0
            if inside.sum() > 0:
                ax.scatter(pts[inside, a0], pts[inside, a1],
                           c="red", s=8, alpha=0.7, zorder=4, label=f"INSIDE ({inside.sum()})")

            # Palm points highlighted with special marker
            palm_pts = pts[is_palm]
            palm_sdf = sdf_v[is_palm]
            if len(palm_pts) > 0:
                palm_in = palm_sdf < 0
                ax.scatter(palm_pts[~palm_in, a0], palm_pts[~palm_in, a1],
                           c="cyan", s=20, marker="s", alpha=0.8, zorder=5,
                           label=f"palm outside ({(~palm_in).sum()})")
                if palm_in.sum() > 0:
                    ax.scatter(palm_pts[palm_in, a0], palm_pts[palm_in, a1],
                               c="magenta", s=20, marker="s", alpha=0.8, zorder=6,
                               label=f"PALM INSIDE ({palm_in.sum()})")

            ax.set_xlabel(ax_labels[a0])
            ax.set_ylabel(ax_labels[a1])
            ax.set_title(f"{pname} — {label} ({pitch_label})\n"
                         f"{n_inside}/{len(pts)} inside, palm_inside={n_palm_inside}")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            path = os.path.join(OUTPUT_DIR, f"{label}_{pname}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"    Saved: {path}")


if __name__ == "__main__":
    main()
