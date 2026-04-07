"""Test the batched solver on multiple real objects."""
import numpy as np
import trimesh
import json
import os
import sys
from frogger.batched_pytorch_solver import BatchedSDF, BatchedGraspOptimizer

MESH_DIR = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg"
ACT_DIR = "/home/bowenj/Projects/DexFun/output/actuation_contacts/mesh_raw_ahg"

OBJECTS = [
    "black_spray_bottle_single",
    "scissors_single",
    "hot_glue_gun",
    "knife_single",
    "marker_single",
    "pliers_single",
]

results = []
for obj_name in OBJECTS:
    mesh_path = os.path.join(MESH_DIR, obj_name, "object.obj")
    act_path = os.path.join(ACT_DIR, f"{obj_name}_actuation.json")
    if not os.path.exists(mesh_path) or not os.path.exists(act_path):
        print(f"SKIP {obj_name}: missing mesh or actuation")
        continue

    mesh = trimesh.load(mesh_path, force='mesh')
    bounds = mesh.bounds
    offset = np.array([0., 0., -bounds[0, 2]])
    X_WO = np.eye(4); X_WO[:3, 3] = offset
    obj_center = mesh.centroid + offset

    with open(act_path) as f:
        data = json.load(f)
    actuation_targets = [
        (np.array(c['pos']) + offset, np.array(c['dir']))
        for c in data['actuation_contacts']
    ]
    n_act = len(actuation_targets)
    print(f"\n{'='*60}")
    print(f"Object: {obj_name} ({n_act} actuation contacts)")
    print(f"{'='*60}")

    sdf = BatchedSDF(mesh, X_WO, bounds_padding=0.15, resolution=128, device='cuda')
    opt = BatchedGraspOptimizer(sdf, num_envs=4000, device='cuda', hand='rh')
    res = opt.optimize(
        actuation_targets=actuation_targets,
        object_center=obj_center,
        steps=1200,
        lr=0.005,
    )
    best = res[0]
    results.append((obj_name, n_act, best['loss']))

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for name, n_act, loss in results:
    print(f"  {name:35s}  act={n_act}  loss={loss:.4f}")
