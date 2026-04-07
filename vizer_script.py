from pydrake.math import RigidTransform, RotationMatrix
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import time

from frogger.objects import MeshObject, MeshObjectConfig
from frogger.robots.robots import AlgrModelConfig

# Latest output from the batched PyTorch solver (two-phase, 6D rotation)
# base_rot is a 3x3 rotation matrix; convert to quaternion for Drake
q_joints = np.array([-0.02933099865913391, 0.6130782961845398, 1.2290832996368408, 0.7652813196182251, -0.12292981147766113, 0.7613885998725891, 0.9029645919799805, 1.116683006286621, -0.3228648602962494, 1.3570634126663208, 0.9923062324523926, 0.847716212272644, 1.3491367101669312, 0.6701356768608093, 1.3108885288238525, 1.3183786869049072], dtype=np.float32)
base_pos = np.array([-0.03885858505964279, -0.07152803987264633, 0.1030796691775322], dtype=np.float32)
base_rot = np.array([[-0.19557598233222961, -0.05597853660583496, 0.9790895581245422], [0.9799742102622986, -0.04925362765789032, 0.19293665885925293], [0.03742334991693497, 0.9972163438796997, 0.06449034810066223]], dtype=np.float64)

obj_name = "004_sugar_box"
mesh = trimesh.load(f"data/{obj_name}/{obj_name}.obj")
bounds = mesh.bounds
offset = np.array([0, 0, -bounds[0, 2]])
X_WO = RigidTransform(RotationMatrix(), offset)
obj = MeshObjectConfig(X_WO=X_WO, mesh=mesh, name=obj_name, clean=False).create()

model = AlgrModelConfig(
    obj=obj,
    ns=4,
    hand="lh",
    viz=True,
).create()

# Convert rotation matrix to Drake quaternion [w, x, y, z]
r = Rotation.from_matrix(base_rot)
quat = r.as_quat()  # scipy: [x, y, z, w]
drake_quat = np.array([quat[3], quat[0], quat[1], quat[2]])

q = np.concatenate([drake_quat, base_pos, q_joints])

model.viz_config(q)
print("Meshcat opened! Check the terminal for URL (usually http://localhost:7000)")
print("Press Ctrl+C to close")
while True:
    time.sleep(1)
