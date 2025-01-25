import numpy as np
import trimesh
from pydrake.math import RigidTransform, RotationMatrix

from frogger import ROOT
from frogger.objects import MeshObject, MeshObjectConfig
from frogger.robots.robots import AlgrModelConfig, BH280ModelConfig, FR3AlgrModelConfig
from frogger.sampling import (
    HeuristicAlgrICSampler,
    HeuristicBH280ICSampler,
    HeuristicFR3AlgrICSampler,
)
from frogger.custom_sampling import FnHeuristicAlgrICSampler
from frogger.solvers import Frogger, FroggerConfig
from frogger.utils import timeout

# loading object
obj_name = "hot_glue_gun"
mesh = trimesh.load(
    f"/home/bowenj/Projects/DexFun/reconstruction/mesh_raw/{obj_name}.stl",
    file_type="stl")
mesh.apply_scale(0.2)
bounds = mesh.bounds
lb_O = bounds[0, :]
ub_O = bounds[1, :]
X_WO = RigidTransform(
    RotationMatrix(),
    np.array([0.0, 0.0, -lb_O[-1]]),
)
obj = MeshObjectConfig(X_WO=X_WO, mesh=mesh, name=obj_name, clean=False).create()
processed_mesh = obj.mesh.as_open3d

# loading model
model = AlgrModelConfig(
    obj=obj,
    ns=4,
    mu=0.7,
    d_min=0.001,
    d_pen=0.005,
    l_bar_cutoff=0.3,
    hand="rh",
).create()
frogger = FroggerConfig(
    model=model,
    sampler=HeuristicAlgrICSampler(model),
    tol_surf=1e-3,
    tol_joint=1e-2,
    tol_col=1e-3,
    tol_fclosure=1e-5,
    xtol_rel=1e-6,
    xtol_abs=1e-6,
    maxeval=1000,
).create()
print("Model compiled! Generating grasp...")
q_star = timeout(1000.0)(frogger.generate_grasp)(tol_pos=0.02, tol_ang=0.25)
print("Grasp generated!")
model.viz_config(q_star)
