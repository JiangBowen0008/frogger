#!/usr/bin/env python3
"""
Diagnostic tool for verifying dexterous hand grasp poses.

Generates 2D projection plots (XY, YZ, XZ) for each hand part + object,
runs numerical checks, and reports penetration analysis.

Usage:
    conda run -n frogger python diagnose_grasp.py --grasp output/grasps/spray_bottle_handcrafted.pt
"""

import argparse
import os
import sys
import numpy as np
import trimesh
import torch
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation as ScipyR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add frogger to path
sys.path.insert(0, os.path.dirname(__file__))
from frogger.batched_pytorch_solver import (
    BatchedSDF,
    _visual_meshes,
    _link_names,
    _LEAP_JOINT_LOWER,
    _LEAP_JOINT_UPPER,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MESH_PATH = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg/black_spray_bottle_single/object.obj"
ACTUATION_POS = np.array([0.039, 0, 0.137])
ACTUATION_DIR = np.array([-0.946, 0.265, -0.184])
ACTUATION_DIR = ACTUATION_DIR / np.linalg.norm(ACTUATION_DIR)
URDF_PATH = os.path.join(os.path.dirname(__file__), "models/leap_rh/leap.urdf")
MESH_DIR = os.path.join(os.path.dirname(__file__), "models/leap_rh")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output/diagnostics")

# Finger chain grouping
FINGER_CHAINS = {
    "palm": ["leap_rh_palm"],
    "IF": ["leap_rh_if_bs", "leap_rh_if_px", "leap_rh_if_md", "leap_rh_if_ds"],
    "MF": ["leap_rh_mf_bs", "leap_rh_mf_px", "leap_rh_mf_md", "leap_rh_mf_ds"],
    "RF": ["leap_rh_rf_bs", "leap_rh_rf_px", "leap_rh_rf_md", "leap_rh_rf_ds"],
    "TH": ["leap_rh_th_mp", "leap_rh_th_bs", "leap_rh_th_px", "leap_rh_th_ds"],
}

CHAIN_COLORS = {
    "palm": "blue",
    "IF": "red",
    "MF": "green",
    "RF": "orange",
    "TH": "purple",
}


def load_object_mesh():
    """Load object mesh and compute offset to sit on z=0."""
    mesh = trimesh.load(MESH_PATH, force="mesh")
    bounds = mesh.bounds
    offset = np.array([0.0, 0.0, -bounds[0, 2]])
    X_WO = np.eye(4)
    X_WO[:3, 3] = offset
    verts_O = np.asarray(mesh.vertices, dtype=np.float64)
    verts_W = (X_WO[:3, :3] @ verts_O.T).T + X_WO[:3, 3]
    return mesh, X_WO, verts_W, offset


def load_grasp(path):
    """Load a grasp result from .pt file."""
    results = torch.load(path, weights_only=False, map_location="cpu")
    if isinstance(results, list):
        g = results[0]
    else:
        g = results
    return g


def compute_hand_world_vertices(q_joints, base_pos, base_rot):
    """Compute world-frame vertices for all hand links.

    Returns dict: chain_name -> np.array (N, 3) of world vertices.
    """
    # Build FK chain
    with open(URDF_PATH) as f:
        chain = pk.build_chain_from_urdf(f.read())

    q = torch.tensor(q_joints, dtype=torch.float32).unsqueeze(0)
    fk = chain.forward_kinematics(q)

    T_base = np.eye(4)
    T_base[:3, :3] = base_rot
    T_base[:3, 3] = base_pos

    vis_meshes = _visual_meshes("rh", "leap")

    chain_verts = {}
    link_verts = {}  # per-link for penetration analysis

    for chain_name, link_list in FINGER_CHAINS.items():
        all_verts = []
        for link_name in link_list:
            if link_name not in vis_meshes or link_name not in fk:
                continue
            for mesh_file, vis_pose in vis_meshes[link_name]:
                full_path = os.path.join(MESH_DIR, mesh_file)
                if not os.path.exists(full_path):
                    continue
                lm = trimesh.load(full_path, force="mesh")
                verts = np.asarray(lm.vertices, dtype=np.float64)

                # Apply visual origin transform
                link_T = fk[link_name].get_matrix()[0].numpy().astype(np.float64)
                world_T = T_base @ link_T
                if vis_pose is not None:
                    vp = np.array(vis_pose, dtype=np.float64)
                    Rv = ScipyR.from_euler("xyz", vp[3:]).as_matrix()
                    Tv = np.eye(4)
                    Tv[:3, :3] = Rv
                    Tv[:3, 3] = vp[:3]
                    world_T = world_T @ Tv

                verts_w = (world_T[:3, :3] @ verts.T).T + world_T[:3, 3]
                all_verts.append(verts_w)

                # Store per-link for penetration
                key = link_name
                if key not in link_verts:
                    link_verts[key] = []
                link_verts[key].append(verts_w)

        if all_verts:
            chain_verts[chain_name] = np.vstack(all_verts)

    # Merge per-link verts
    for k in link_verts:
        link_verts[k] = np.vstack(link_verts[k])

    return chain_verts, link_verts


def compute_tip_positions(q_joints, base_pos, base_rot):
    """Compute fingertip positions and pad directions in world frame."""
    with open(URDF_PATH) as f:
        chain = pk.build_chain_from_urdf(f.read())

    q = torch.tensor(q_joints, dtype=torch.float32).unsqueeze(0)
    fk = chain.forward_kinematics(q)

    T_base = np.eye(4)
    T_base[:3, :3] = base_rot
    T_base[:3, 3] = base_pos

    tip_names = ["leap_rh_if_ds", "leap_rh_mf_ds", "leap_rh_rf_ds", "leap_rh_th_ds"]
    f_off = np.array([-0.0025, -0.0449, 0.0143])
    t_off = np.array([-0.0020, -0.0558, -0.0144])
    offsets = [f_off, f_off, f_off, t_off]

    tips = {}
    tip_x_axes = {}
    for name, off in zip(tip_names, offsets):
        if name not in fk:
            continue
        link_T = fk[name].get_matrix()[0].numpy().astype(np.float64)
        wT = T_base @ link_T
        pos = wT[:3, :3] @ off + wT[:3, 3]
        x_axis = wT[:3, 0]  # pad push direction
        tips[name] = pos
        tip_x_axes[name] = x_axis

    return tips, tip_x_axes


# ---------------------------------------------------------------------------
# 2D Projection plots
# ---------------------------------------------------------------------------
def plot_projections(chain_verts, obj_verts_W, output_dir, tag=""):
    """Generate 2D projection plots for each hand part + object."""
    os.makedirs(output_dir, exist_ok=True)

    projections = {
        "XY": (0, 1, "X", "Y"),
        "YZ": (1, 2, "Y", "Z"),
        "XZ": (0, 2, "X", "Z"),
    }

    # Overall figure with all parts
    for proj_name, (ax0, ax1, xlabel, ylabel) in projections.items():
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Object outline
        ax.scatter(obj_verts_W[:, ax0], obj_verts_W[:, ax1],
                   c="lightgray", s=0.5, alpha=0.5, label="object", zorder=1)

        # Hand parts
        for chain_name, verts in chain_verts.items():
            color = CHAIN_COLORS.get(chain_name, "black")
            ax.scatter(verts[:, ax0], verts[:, ax1],
                       c=color, s=2, alpha=0.7, label=chain_name, zorder=2)

        # Actuation point
        ax.scatter([ACTUATION_POS[ax0]], [ACTUATION_POS[ax1]],
                   c="red", s=100, marker="*", zorder=5, label="actuation")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{proj_name} Projection (all parts) {tag}")
        ax.legend(loc="best", fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        path = os.path.join(output_dir, f"projection_all_{proj_name}{tag}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # Individual per-part figures
    for chain_name, verts in chain_verts.items():
        for proj_name, (ax0, ax1, xlabel, ylabel) in projections.items():
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.scatter(obj_verts_W[:, ax0], obj_verts_W[:, ax1],
                       c="lightgray", s=0.5, alpha=0.5, zorder=1)
            color = CHAIN_COLORS.get(chain_name, "black")
            ax.scatter(verts[:, ax0], verts[:, ax1],
                       c=color, s=3, alpha=0.8, zorder=2)
            ax.scatter([ACTUATION_POS[ax0]], [ACTUATION_POS[ax1]],
                       c="red", s=100, marker="*", zorder=5)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{proj_name}: {chain_name} {tag}")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            path = os.path.join(output_dir, f"projection_{chain_name}_{proj_name}{tag}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)


# ---------------------------------------------------------------------------
# Numerical checks
# ---------------------------------------------------------------------------
def run_numerical_checks(q_joints, base_pos, base_rot, link_verts, obj_mesh, X_WO):
    """Run all 7 numerical checks. Returns list of (name, passed, detail)."""
    tips, tip_x_axes = compute_tip_positions(q_joints, base_pos, base_rot)
    results = []

    # Check 1: IF tip within 5mm of actuation point
    if_tip = tips.get("leap_rh_if_ds")
    if if_tip is not None:
        dist = np.linalg.norm(if_tip - ACTUATION_POS)
        passed = dist < 0.005
        results.append(("IF tip near actuation (<5mm)", passed, f"dist={dist*1000:.1f}mm"))
    else:
        results.append(("IF tip near actuation (<5mm)", False, "IF tip not found"))

    # Check 2: IF pad direction aligned with -actuation_dir (dot > 0.7)
    if_x = tip_x_axes.get("leap_rh_if_ds")
    if if_x is not None:
        neg_act_dir = -ACTUATION_DIR
        dot = np.dot(if_x, neg_act_dir)
        passed = dot > 0.7
        results.append(("IF pad aligned with -act_dir (dot>0.7)", passed, f"dot={dot:.3f}"))
    else:
        results.append(("IF pad aligned with -act_dir (dot>0.7)", False, "IF x-axis not found"))

    # Check 3: Thumb on opposite side of object from palm
    palm_center_y = None
    if "leap_rh_palm" in link_verts:
        palm_center_y = link_verts["leap_rh_palm"][:, 1].mean()
    th_tip = tips.get("leap_rh_th_ds")
    if palm_center_y is not None and th_tip is not None:
        # Palm at -y means thumb should be at +y (or at least positive side)
        palm_side = "negative-y" if palm_center_y < 0 else "positive-y"
        thumb_y = th_tip[1]
        if palm_center_y < 0:
            passed = thumb_y > 0
        else:
            passed = thumb_y < 0
        results.append(("Thumb opposite palm", passed,
                        f"palm_y={palm_center_y:.4f} ({palm_side}), thumb_y={thumb_y:.4f}"))
    else:
        results.append(("Thumb opposite palm", False, "missing data"))

    # Check 4: Palm inner surface points have |SDF| < 5mm
    if "leap_rh_palm" in link_verts:
        palm_verts = link_verts["leap_rh_palm"]
        # Build SDF for distance queries
        sdf = BatchedSDF(obj_mesh, X_WO, resolution=128, device="cuda")
        palm_pts = torch.tensor(palm_verts, dtype=torch.float32, device="cuda").unsqueeze(0)
        palm_sdf = sdf.query(palm_pts).cpu().numpy()[0]
        near_surface = np.abs(palm_sdf) < 0.005
        frac_near = near_surface.sum() / len(palm_sdf)
        min_sdf = np.abs(palm_sdf).min()
        passed = frac_near > 0.1  # at least 10% near surface
        results.append(("Palm contact (>10% within 5mm)", passed,
                        f"{frac_near*100:.1f}% within 5mm, min|SDF|={min_sdf*1000:.1f}mm"))
    else:
        results.append(("Palm contact (>10% within 5mm)", False, "palm not found"))
        sdf = BatchedSDF(obj_mesh, X_WO, resolution=128, device="cuda")

    # Check 5: No significant penetration (< 10% vertices with SDF < -3mm)
    all_ok = True
    pen_details = []
    for link_name, verts in link_verts.items():
        pts = torch.tensor(verts, dtype=torch.float32, device="cuda").unsqueeze(0)
        sdf_vals = sdf.query(pts).cpu().numpy()[0]
        deep_pen = (sdf_vals < -0.003).sum() / len(sdf_vals)
        if deep_pen > 0.10:
            all_ok = False
            pen_details.append(f"{link_name}: {deep_pen*100:.1f}% deep")
    detail = "OK" if all_ok else "; ".join(pen_details)
    results.append(("No significant penetration (<10% at -3mm)", all_ok, detail))

    # Check 6: At least one pair of opposing contact normals (dot < -0.3)
    tip_positions = []
    for name in ["leap_rh_if_ds", "leap_rh_mf_ds", "leap_rh_rf_ds", "leap_rh_th_ds"]:
        if name in tips:
            tip_positions.append(tips[name])
    if len(tip_positions) >= 2:
        tp_tensor = torch.tensor(np.array(tip_positions), dtype=torch.float32, device="cuda")
        tp_tensor = tp_tensor.unsqueeze(0)  # [1, nc, 3]
        _, normals = sdf.query_with_normals(tp_tensor)
        normals = normals[0].cpu().numpy()  # [nc, 3]
        min_dot = 1.0
        has_opposing = False
        for i in range(len(normals)):
            for j in range(i+1, len(normals)):
                d = np.dot(normals[i], normals[j])
                min_dot = min(min_dot, d)
                if d < -0.3:
                    has_opposing = True
        results.append(("Opposing contact normals (dot < -0.3)", has_opposing,
                        f"min_dot={min_dot:.3f}"))
    else:
        results.append(("Opposing contact normals (dot < -0.3)", False, "not enough tips"))

    # Check 7: Palm aligned with bottle axis
    if "leap_rh_palm" in link_verts:
        palm_verts = link_verts["leap_rh_palm"]
        palm_z_extent = palm_verts[:, 2].max() - palm_verts[:, 2].min()
        palm_y_extent = palm_verts[:, 1].max() - palm_verts[:, 1].min()
        # Palm long axis should be vertical (z), so z_extent > y_extent
        # Actually the palm y-extent in base frame maps to world z, so check
        # that the palm's world z-extent is the largest
        palm_x_extent = palm_verts[:, 0].max() - palm_verts[:, 0].min()
        max_extent = max(palm_x_extent, palm_y_extent, palm_z_extent)
        aligned = palm_z_extent == max_extent or (palm_z_extent > 0.5 * max_extent)
        results.append(("Palm aligned with bottle axis", aligned,
                        f"extents: x={palm_x_extent*1000:.1f}mm, "
                        f"y={palm_y_extent*1000:.1f}mm, z={palm_z_extent*1000:.1f}mm"))
    else:
        results.append(("Palm aligned with bottle axis", False, "palm not found"))

    return results


# ---------------------------------------------------------------------------
# Penetration analysis
# ---------------------------------------------------------------------------
def penetration_analysis(link_verts, obj_mesh, X_WO):
    """For each link, compute % of vertices inside object (SDF < 0)."""
    sdf = BatchedSDF(obj_mesh, X_WO, resolution=128, device="cuda")
    results = []
    for link_name, verts in link_verts.items():
        pts = torch.tensor(verts, dtype=torch.float32, device="cuda").unsqueeze(0)
        sdf_vals = sdf.query(pts).cpu().numpy()[0]
        inside = (sdf_vals < 0).sum()
        pct = inside / len(sdf_vals) * 100
        flag = "FLAG" if pct > 20 else "OK"
        min_sdf = sdf_vals.min()
        results.append((link_name, pct, min_sdf, flag))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def diagnose(grasp_path, output_dir=None, tag=""):
    """Run full diagnostics on a grasp."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("GRASP DIAGNOSTICS")
    print("=" * 70)

    # Load data
    print("\n[1] Loading grasp and mesh...")
    g = load_grasp(grasp_path)
    q_joints = np.array(g["q_joints"], dtype=np.float64)
    base_pos = np.array(g["base_pos"], dtype=np.float64)
    base_rot = np.array(g["base_rot"], dtype=np.float64)

    print(f"  q_joints = {q_joints}")
    print(f"  base_pos = {base_pos}")
    print(f"  base_rot =\n{base_rot}")

    obj_mesh, X_WO, obj_verts_W, offset = load_object_mesh()
    print(f"  Object bounds: {obj_mesh.bounds}")
    print(f"  Offset: {offset}")
    print(f"  Object verts in world: z=[{obj_verts_W[:,2].min():.4f}, {obj_verts_W[:,2].max():.4f}]")

    # Compute hand vertices
    print("\n[2] Computing hand FK...")
    chain_verts, link_verts = compute_hand_world_vertices(q_joints, base_pos, base_rot)
    for name, verts in chain_verts.items():
        print(f"  {name}: {len(verts)} vertices, "
              f"center={verts.mean(axis=0)}, "
              f"z=[{verts[:,2].min():.4f}, {verts[:,2].max():.4f}]")

    # Tip positions
    tips, tip_x_axes = compute_tip_positions(q_joints, base_pos, base_rot)
    print("\n  Tip positions:")
    for name, pos in tips.items():
        short = name.split("_")[-2]  # e.g., "if", "mf", etc.
        x_ax = tip_x_axes[name]
        print(f"    {short} tip: {pos}, pad_dir={x_ax}")

    # 2D projections
    print(f"\n[3] Generating 2D projections...")
    plot_projections(chain_verts, obj_verts_W, output_dir, tag=tag)

    # Numerical checks
    print(f"\n[4] Numerical checks:")
    checks = run_numerical_checks(q_joints, base_pos, base_rot, link_verts, obj_mesh, X_WO)
    all_pass = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}: {detail}")

    # Penetration analysis
    print(f"\n[5] Penetration analysis:")
    pen_results = penetration_analysis(link_verts, obj_mesh, X_WO)
    for link_name, pct, min_sdf, flag in pen_results:
        print(f"  [{flag}] {link_name}: {pct:.1f}% inside, min_SDF={min_sdf*1000:.1f}mm")

    print(f"\n{'=' * 70}")
    if all_pass:
        print("ALL CHECKS PASSED")
    else:
        n_pass = sum(1 for _, p, _ in checks if p)
        print(f"{n_pass}/{len(checks)} CHECKS PASSED")
    print("=" * 70)

    return all_pass, checks, pen_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose a grasp pose")
    parser.add_argument("--grasp", required=True, help="Path to .pt grasp file")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--tag", default="", help="Tag suffix for filenames")
    args = parser.parse_args()
    diagnose(args.grasp, args.output_dir, args.tag)
