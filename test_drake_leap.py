"""Test FroGGer's original Drake solver with LEAP hand + box collision primitives.

Uses the full FroGGer pipeline: hybrid sampler + actuation contacts + FunctionalFrogger.
Based on examples/test_leap.py but with LeapModelConfig and simplified mesh.
"""
import numpy as np
import trimesh
import json
import torch
import os
import sys

from pydrake.math import RigidTransform, RotationMatrix
from frogger.objects import MeshObjectConfig
from frogger.robots.custom_robots import LeapModelConfig
from frogger.solvers import FroggerConfig
from frogger.sampling import HeuristicAlgrICSampler
from frogger.custom_robot_model import FunctionalRobotModel
from frogger.custom_solver import FunctionalFrogger
from frogger.utils import timeout
from frogger.custom_sampling import create_hybrid_sampler
from frogger.learning_based_heuristics import ContactMixedHeuristic
from scipy.spatial.transform import Rotation as ScipyR

# Load simplified mesh (5K faces for Drake speed)
mesh_path = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg/black_spray_bottle_single/object_simple.obj"
mesh = trimesh.load(mesh_path, force="mesh")
bounds = mesh.bounds
offset = np.array([0.0, 0.0, -bounds[0, 2]])
X_WO = RigidTransform(RotationMatrix(), offset)

obj = MeshObjectConfig(X_WO=X_WO, mesh=mesh, name="black_spray_bottle_single", clean=False).create()

# Load actuation contacts
act_path = "/home/bowenj/Projects/DexFun/output/actuation_contacts/mesh_raw_ahg/black_spray_bottle_single_actuation.json"
with open(act_path) as f:
    act_info = json.load(f)
actuation_contacts = [
    (np.array(c["pos"]) + offset, np.array(c["dir"]))
    for c in act_info["actuation_contacts"]
]

print(f"Object: spray_bottle ({len(mesh.vertices)} verts)")
print(f"Actuation: pos={actuation_contacts[0][0].round(4)}, dir={actuation_contacts[0][1].round(4)}")

# Create LEAP model
model_cfg = LeapModelConfig(
    obj=obj,
    ns=4,
    mu=0.9,
    d_min=0.001,
    d_pen=0.005,
    l_bar_cutoff=0.3,
    hand="rh",
)

reverse_actuation = True  # LEAP uses reversed actuation

print(f"Creating FunctionalRobotModel (LEAP RH)...")
model = FunctionalRobotModel(model_cfg)

# Hybrid sampler with contact predictor + actuation
contact_predictor = ContactMixedHeuristic(heatmap_dir="output/mixed_heatmap")
contact_predictor.load_object(mesh_name="black_spray_bottle_single")
sampler = create_hybrid_sampler(
    model, contact_predictor, actuation_contacts,
    reverse_actuation=reverse_actuation,
    palm_offset=(0.005, 0.02),
)

# Solver with relaxed tolerances (same as test_leap.py)
frogger = FunctionalFrogger(
    cfg=FroggerConfig(
        model=model,
        sampler=sampler,
        tol_surf=5e-3,
        tol_joint=1e-2,
        tol_col=7e-3,
        tol_fclosure=2e-1,
        xtol_rel=1e-5,
        xtol_abs=1e-5,
        maxeval=500,
    ),
    actuation_contacts=actuation_contacts,
)

print("Generating grasps...")
results = []
for i in range(5):
    print(f"  Grasp {i}...", end=" ", flush=True)
    try:
        q_star, q0 = timeout(300.0)(frogger.generate_grasp)(
            optimize=True, check_constraints=False, tol_pos=0.05, tol_ang=0.2,
        )
        f_val = model.compute_f(q_star)
        print(f"f={f_val:.4f}")
        results.append(q_star)
    except Exception as e:
        print(f"Error: {e}")

print(f"\nGenerated {len(results)} grasps")

# Save in visualization-compatible format
# Drake floating base: [qw, qx, qy, qz, x, y, z, joint1, ..., joint16]
# Joint order: Drake URDF order (mcp, axl, ...) -> pk order (axl, mcp, ...)
drake_to_pk = [1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15]

saved = []
for q_star in results:
    quat_wxyz = q_star[0:4]
    base_pos = q_star[4:7]
    q_drake = q_star[7:]
    base_rot = ScipyR.from_quat(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    ).as_matrix()
    q_joints = q_drake[drake_to_pk]

    saved.append({
        "q_joints": q_joints.astype(np.float32),
        "base_pos": base_pos.astype(np.float32),
        "base_rot": base_rot.astype(np.float64),
        "l_star": 0.0, "feasible": True, "score": 0.0,
        "sigma_min": 0.0, "act_assignment": [0],
        "act_dist": 0.0, "surf_err": 0.0,
        "min_col": 0.0, "sc_min_dist": 0.0,
    })

torch.save(saved, "output/grasps/spray_bottle_drake_leap.pt")
print(f"Saved {len(saved)} grasps to output/grasps/spray_bottle_drake_leap.pt")
