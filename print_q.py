from frogger.objects import MeshObject, MeshObjectConfig
from frogger.robots.robots import AlgrModelConfig
from pydrake.math import RigidTransform, RotationMatrix
import numpy as np
import trimesh

obj_name = "004_sugar_box"
mesh = trimesh.load(f"data/{obj_name}/{obj_name}.obj")
X_WO = RigidTransform(RotationMatrix(), np.array([0, 0, 0]))
obj = MeshObjectConfig(X_WO=X_WO, mesh=mesh, name=obj_name, clean=False).create()

model = AlgrModelConfig(
    obj=obj,
    ns=4,
    hand="lh",
    viz=False,
).create()

print("n:", model.n)
q = model.plant.GetPositions(model.plant.GetMyContextFromRoot(model.context), model.robot_instance)
print("q length:", len(q))
print("q format:", q)
