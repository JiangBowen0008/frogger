"""Test projected gradient descent solver on 3 test objects.

Usage:
    conda run -n frogger python test_pgd_solver.py
"""

import numpy as np
import trimesh
import torch
import json
import os
import time

from frogger.batched_pytorch_solver import (
    BatchedSDF,
    BatchedGraspOptimizer,
)

MESH_DIR = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg"
ACT_DIR = "/home/bowenj/Projects/DexFun/output/actuation_contacts/mesh_raw_ahg"
SAVE_DIR = "output/grasps/pgd_test"

OBJECTS = [
    "black_spray_bottle_single",
    "hot_glue_gun",
    "syrup_pourer_single",
]


def run_object(obj_name, num_envs=4000, steps=1200):
    mesh_path = os.path.join(MESH_DIR, obj_name, "object.obj")
    act_path = os.path.join(ACT_DIR, f"{obj_name}_actuation.json")

    print(f"\n{'='*70}")
    print(f"  OBJECT: {obj_name}")
    print(f"{'='*70}")

    mesh = trimesh.load(mesh_path, force="mesh")
    bounds = mesh.bounds
    offset = np.array([0.0, 0.0, -bounds[0, 2]])
    X_WO = np.eye(4)
    X_WO[:3, 3] = offset
    obj_center = mesh.centroid + offset

    with open(act_path) as f:
        act_data = json.load(f)
    actuation_targets = [
        (np.array(c["pos"]) + offset, np.array(c["dir"]))
        for c in act_data["actuation_contacts"]
    ]
    for i, (pos, d) in enumerate(actuation_targets):
        print(f"  Actuation[{i}]: pos={pos}, dir={d}")

    print("  Building SDF ...")
    sdf = BatchedSDF(mesh, X_WO, bounds_padding=0.15, resolution=128, device="cuda")

    opt = BatchedGraspOptimizer(
        sdf, num_envs=num_envs, device="cuda",
        hand="rh", hand_type="leap",
    )

    save_path = os.path.join(SAVE_DIR, f"{obj_name}_leap_rh_pgd.pt")
    t0 = time.time()
    results = opt.optimize(
        actuation_targets=actuation_targets,
        object_center=obj_center,
        steps=steps,
        lr=0.005,
        save_path=save_path,
    )
    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Results saved to {save_path}")
    return results


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    for obj in OBJECTS:
        try:
            run_object(obj, num_envs=4000, steps=1200)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
    print("\n\nDone.")
