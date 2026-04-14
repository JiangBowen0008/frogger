#!/usr/bin/env python3
"""Exp 2: Run with palm_contact=False to test H2."""
import os, sys, numpy as np, trimesh, torch
sys.path.insert(0, os.path.dirname(__file__))
from frogger.batched_pytorch_solver import BatchedSDF, BatchedGraspOptimizer

MESH_PATH = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg/black_spray_bottle_single/object.obj"

mesh = trimesh.load(MESH_PATH, force="mesh")
bounds = mesh.bounds
offset = np.array([0.0, 0.0, -bounds[0, 2]])
X_WO = np.eye(4); X_WO[:3, 3] = offset
obj_center = mesh.centroid + offset
verts_W = (X_WO[:3, :3] @ np.asarray(mesh.vertices).T).T + X_WO[:3, 3]
mesh_W = trimesh.Trimesh(vertices=verts_W, faces=mesh.faces)
candidate = np.array([[0.0, 0.0, offset[2] + (bounds[1, 2] - bounds[0, 2]) * 0.8]])
closest_pts, _, _ = trimesh.proximity.closest_point(mesh_W, candidate)
act_pos = closest_pts[0]

sdf = BatchedSDF(mesh, X_WO, bounds_padding=0.15, resolution=128, device="cuda")
opt = BatchedGraspOptimizer(sdf, num_envs=500, device="cuda", hand="rh", hand_type="leap",
                            palm_contact=False)  # KEY: no palm contact
results = opt.optimize(actuation_targets=[(act_pos, None)], object_center=obj_center,
                       steps=600, lr=0.005,
                       save_path="output/grasps/compare_exp2_no_palm.pt")
print("\nExp 2 done.")
