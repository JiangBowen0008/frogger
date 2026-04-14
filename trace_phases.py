#!/usr/bin/env python3
"""
Trace optimisation phases — run solver with snapshot logging to understand
where collision develops.

Runs with 500 envs (fast) and saves phase snapshots for analysis.
"""

import os
import sys
import numpy as np
import trimesh
import torch

sys.path.insert(0, os.path.dirname(__file__))
from frogger.batched_pytorch_solver import BatchedSDF, BatchedGraspOptimizer

MESH_PATH = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg/black_spray_bottle_single/object.obj"
OUTPUT_DIR = "output/diagnostics/phase_trace"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading mesh...")
    mesh = trimesh.load(MESH_PATH, force="mesh")
    bounds = mesh.bounds
    offset = np.array([0.0, 0.0, -bounds[0, 2]])
    X_WO = np.eye(4)
    X_WO[:3, 3] = offset
    obj_center = mesh.centroid + offset

    # Default actuation target (80% height)
    verts_W = (X_WO[:3, :3] @ np.asarray(mesh.vertices).T).T + X_WO[:3, 3]
    mesh_W = trimesh.Trimesh(vertices=verts_W, faces=mesh.faces)
    candidate = np.array([[0.0, 0.0, offset[2] + (bounds[1, 2] - bounds[0, 2]) * 0.8]])
    closest_pts, _, _ = trimesh.proximity.closest_point(mesh_W, candidate)
    act_pos = closest_pts[0]
    actuation_targets = [(act_pos, None)]
    print(f"  Actuation: {act_pos}")

    print("\nBuilding SDF...")
    sdf = BatchedSDF(mesh, X_WO, bounds_padding=0.15, resolution=128, device="cuda")

    print("\nRunning optimizer (500 envs, 600 steps)...")
    opt = BatchedGraspOptimizer(
        sdf, num_envs=500, device="cuda",
        hand="rh", hand_type="leap",
        palm_contact=True,
    )
    results = opt.optimize(
        actuation_targets=actuation_targets,
        object_center=obj_center,
        steps=600,
        lr=0.005,
        save_path=os.path.join(OUTPUT_DIR, "trace_result.pt"),
    )

    # Print snapshot summary
    print("\n" + "=" * 70)
    print("PHASE SNAPSHOT SUMMARY")
    print("=" * 70)
    print(f"{'Phase':<15} {'Tip SDF':>10} {'Col Inside%':>12} {'Col Min':>10} {'Margin Viol%':>13}")
    print("-" * 60)
    for snap in opt._phase_snapshots:
        print(f"{snap['tag']:<15} {snap['tip_sdf_abs_mean']*1000:>9.1f}mm "
              f"{snap['col_inside_pct']:>11.1f}% "
              f"{snap['col_min_sdf']*1000:>9.1f}mm "
              f"{snap['col_margin_violated']:>12.1f}%")

    # Save snapshots for further analysis
    torch.save(opt._phase_snapshots, os.path.join(OUTPUT_DIR, "phase_snapshots.pt"))

    # Also save each snapshot as a grasp file for the verification script
    for snap in opt._phase_snapshots:
        grasp = {
            "q_joints": snap["q_joints"],
            "base_pos": snap["base_pos"],
            "base_rot": snap["base_rot"],
        }
        tag = snap["tag"]
        path = os.path.join(OUTPUT_DIR, f"snap_{tag}.pt")
        torch.save([grasp], path)
        print(f"  Saved: {path}")

    print("\nDone.")

if __name__ == "__main__":
    main()
