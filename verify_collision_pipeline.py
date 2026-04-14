#!/usr/bin/env python3
"""
Verify the collision pipeline is correct end-to-end.

For a given grasp, independently computes:
1. Box-grid collision points in link-local frame (from URDF)
2. FK + base transform → world frame
3. SDF query at world-frame points
4. Visualizes points colored by SDF sign on 2D projections

If the pipeline is correct:
- Points visually INSIDE the object mesh should have SDF < 0 (red)
- Points visually OUTSIDE should have SDF > 0 (blue)
- Any mismatch = transform or SDF bug

Usage:
    conda run -n frogger python verify_collision_pipeline.py
"""

import os
import sys
import numpy as np
import torch
import trimesh
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation as ScipyR
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from frogger.batched_pytorch_solver import BatchedSDF, _visual_meshes

# Config
MESH_PATH = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg/black_spray_bottle_single/object.obj"
URDF_PATH = os.path.join(os.path.dirname(__file__), "models/leap_rh/leap.urdf")
MESH_DIR = os.path.join(os.path.dirname(__file__), "models/leap_rh")
GRASP_PATH = "output/grasps/compare_warmstart_single.pt"
OUTPUT_DIR = "output/diagnostics/pipeline_verify"

COLLISION_LINKS = [
    "leap_rh_palm",
    "leap_rh_if_bs", "leap_rh_if_px", "leap_rh_if_md", "leap_rh_if_ds",
    "leap_rh_mf_bs", "leap_rh_mf_px", "leap_rh_mf_md", "leap_rh_mf_ds",
    "leap_rh_rf_bs", "leap_rh_rf_px", "leap_rh_rf_md", "leap_rh_rf_ds",
    "leap_rh_th_mp", "leap_rh_th_bs", "leap_rh_th_px", "leap_rh_th_ds",
]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load grasp
    g = torch.load(GRASP_PATH, weights_only=False, map_location="cpu")
    if isinstance(g, list):
        g = g[0]
    q_joints = np.array(g["q_joints"], dtype=np.float64)
    base_pos = np.array(g["base_pos"], dtype=np.float64)
    base_rot = np.array(g["base_rot"], dtype=np.float64)
    print(f"Grasp: {GRASP_PATH}")
    print(f"  base_pos = {base_pos}")

    # Load mesh + SDF
    mesh = trimesh.load(MESH_PATH, force="mesh")
    bounds = mesh.bounds
    offset = np.array([0.0, 0.0, -bounds[0, 2]])
    X_WO = np.eye(4)
    X_WO[:3, 3] = offset
    verts_O = np.asarray(mesh.vertices, dtype=np.float64)
    verts_W = (X_WO[:3, :3] @ verts_O.T).T + X_WO[:3, 3]

    sdf = BatchedSDF(mesh, X_WO, resolution=128, device="cuda")
    print(f"  Object bounds: z=[{verts_W[:,2].min():.4f}, {verts_W[:,2].max():.4f}]")

    # Verify SDF at known points
    obj_center = verts_W.mean(axis=0)
    center_sdf = sdf.query(torch.tensor(obj_center, dtype=torch.float32, device="cuda").reshape(1,1,3))
    far_pt = obj_center + np.array([0.5, 0, 0])
    far_sdf = sdf.query(torch.tensor(far_pt, dtype=torch.float32, device="cuda").reshape(1,1,3))
    print(f"\n  SDF sanity check:")
    print(f"    Object center: SDF = {center_sdf.item()*1000:.1f}mm (should be negative)")
    print(f"    Far point (+0.5m): SDF = {far_sdf.item()*1000:.1f}mm (should be positive)")

    # FK
    with open(URDF_PATH) as f:
        chain = pk.build_chain_from_urdf(f.read())
    q = torch.tensor(q_joints, dtype=torch.float32).unsqueeze(0)
    fk = chain.forward_kinematics(q)

    T_base = np.eye(4)
    T_base[:3, :3] = base_rot
    T_base[:3, 3] = base_pos
    bT = torch.tensor(T_base, dtype=torch.float32, device="cuda").unsqueeze(0)

    # Parse URDF boxes and build grid per link (same as solver)
    tree = ET.parse(URDF_PATH)
    pitch = 0.005  # 5mm for detailed verification

    all_world_pts = []
    all_sdf_vals = []
    all_link_names = []
    all_margins = []

    print(f"\n  Per-link collision check (5mm grid):")
    print(f"  {'Link':<25} {'Pts':>5} {'Inside%':>8} {'MinSDF':>9} {'Margin':>7}")
    print(f"  {'-'*25} {'-'*5} {'-'*8} {'-'*9} {'-'*7}")

    for link_name in COLLISION_LINKS:
        le = None
        for e in tree.getroot().findall("link"):
            if e.get("name") == link_name:
                le = e
                break
        if le is None:
            continue
        if link_name not in fk:
            continue

        is_palm = "palm" in link_name
        link_pts = []
        for col in le.findall("collision"):
            g_el = col.find("geometry")
            if g_el is None: continue
            b = g_el.find("box")
            if b is None: continue
            sz = [float(x) for x in b.get("size").split()]
            o = col.find("origin")
            xyz = np.array([float(x) for x in o.get("xyz", "0 0 0").split()])
            rpy = np.array([float(x) for x in o.get("rpy", "0 0 0").split()])
            R = (ScipyR.from_euler("xyz", rpy).as_matrix()
                 if np.any(np.abs(rpy) > 1e-6) else np.eye(3))

            # Same skip as solver
            if is_palm and xyz[0] < -0.025:
                print(f"  [SKIP] {link_name} box at x={xyz[0]:.3f} (back palm)")
                continue

            hx, hy, hz = sz[0]/2, sz[1]/2, sz[2]/2
            gx = np.arange(-hx, hx + pitch/2, pitch)
            gy = np.arange(-hy, hy + pitch/2, pitch)
            gz = np.arange(-hz, hz + pitch/2, pitch)
            grid = np.stack(np.meshgrid(gx, gy, gz, indexing='ij'), axis=-1).reshape(-1, 3)
            grid = ((R @ grid.T).T + xyz).astype(np.float32)
            link_pts.append(grid)

        if not link_pts:
            continue
        pts = np.vstack(link_pts)
        pts_h = torch.tensor(
            np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)]),
            device="cuda")

        # Transform to world frame (same as solver _get_points)
        link_T = fk[link_name].get_matrix().to("cuda")
        world_T = bT @ link_T
        pts_w = (world_T[0] @ pts_h.T).T[:, :3]

        # Query SDF
        sdf_vals = sdf.query(pts_w.unsqueeze(0)).squeeze(0)

        # Margin (same as solver)
        if "palm" in link_name:
            margin = -0.005
        elif "_ds" in link_name:
            margin = -0.001
        else:
            margin = 0.0

        sdf_np = sdf_vals.cpu().numpy()
        n_inside = (sdf_np < 0).sum()
        pct = n_inside / len(sdf_np) * 100
        min_sdf = sdf_np.min()
        violation = (margin - sdf_np)
        n_violated = (violation > 0).sum()
        short = link_name.replace("leap_rh_", "")
        print(f"  {short:<25} {len(sdf_np):>5} {pct:>7.1f}% {min_sdf*1000:>8.1f}mm {margin*1000:>6.1f}mm")

        pts_w_np = pts_w.cpu().numpy()
        all_world_pts.append(pts_w_np)
        all_sdf_vals.append(sdf_np)
        all_link_names.extend([link_name] * len(sdf_np))
        all_margins.extend([margin] * len(sdf_np))

    # Combine all points
    all_pts = np.vstack(all_world_pts)
    all_sdf = np.concatenate(all_sdf_vals)
    all_margins = np.array(all_margins)

    n_total = len(all_sdf)
    n_inside = (all_sdf < 0).sum()
    n_violated = ((all_margins - all_sdf) > 0).sum()
    print(f"\n  TOTAL: {n_total} points, {n_inside} inside ({n_inside/n_total*100:.1f}%), "
          f"{n_violated} margin-violated ({n_violated/n_total*100:.1f}%)")

    # Visualization: plot collision points colored by SDF sign
    # Red = inside object (SDF < 0), Blue = outside (SDF > 0)
    projections = [("XY", 0, 1), ("YZ", 1, 2), ("XZ", 0, 2)]
    labels = {0: "X", 1: "Y", 2: "Z"}

    for name, ax0, ax1 in projections:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        for idx, (ax, title_suffix) in enumerate(zip(axes, ["SDF sign", "SDF value"])):
            # Object outline
            ax.scatter(verts_W[:, ax0], verts_W[:, ax1],
                       c="lightgray", s=0.5, alpha=0.3, zorder=1)

            if idx == 0:
                # Binary: inside (red) vs outside (blue)
                inside = all_sdf < 0
                outside = ~inside
                if outside.sum() > 0:
                    ax.scatter(all_pts[outside, ax0], all_pts[outside, ax1],
                               c="blue", s=3, alpha=0.6, label=f"outside ({outside.sum()})", zorder=2)
                if inside.sum() > 0:
                    ax.scatter(all_pts[inside, ax0], all_pts[inside, ax1],
                               c="red", s=3, alpha=0.6, label=f"inside ({inside.sum()})", zorder=3)
            else:
                # Continuous SDF value
                vmin, vmax = -0.03, 0.03
                sc = ax.scatter(all_pts[:, ax0], all_pts[:, ax1],
                                c=all_sdf, cmap="RdBu", vmin=vmin, vmax=vmax,
                                s=3, alpha=0.6, zorder=2)
                plt.colorbar(sc, ax=ax, label="SDF (m)")

            ax.set_xlabel(labels[ax0])
            ax.set_ylabel(labels[ax1])
            ax.set_title(f"{name} — {title_suffix}")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=8)

        fig.suptitle(f"Collision Pipeline Verification — {GRASP_PATH}", fontsize=11)
        path = os.path.join(OUTPUT_DIR, f"verify_{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # Also make per-link detail plots for palm (the main concern)
    palm_mask = np.array(["palm" in n for n in all_link_names])
    if palm_mask.sum() > 0:
        palm_pts = all_pts[palm_mask]
        palm_sdf = all_sdf[palm_mask]

        for name, ax0, ax1 in projections:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(verts_W[:, ax0], verts_W[:, ax1],
                       c="lightgray", s=0.5, alpha=0.3, zorder=1)
            sc = ax.scatter(palm_pts[:, ax0], palm_pts[:, ax1],
                            c=palm_sdf, cmap="RdBu", vmin=-0.03, vmax=0.03,
                            s=8, alpha=0.8, zorder=2)
            plt.colorbar(sc, ax=ax, label="SDF (m)")
            n_in = (palm_sdf < 0).sum()
            ax.set_title(f"PALM {name} — {n_in}/{len(palm_sdf)} inside, "
                         f"min={palm_sdf.min()*1000:.1f}mm")
            ax.set_xlabel(labels[ax0])
            ax.set_ylabel(labels[ax1])
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            path = os.path.join(OUTPUT_DIR, f"verify_palm_{name}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {path}")

    print("\nDone. Check images in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
