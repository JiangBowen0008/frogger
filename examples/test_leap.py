import numpy as np
import trimesh
import open3d as o3d
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.geometry import Sphere, Rgba

from frogger.objects import MeshObjectConfig
from frogger.robots.robots import AlgrModelConfig
from frogger.robots.custom_robots import LeapModelConfig
from frogger.sampling import HeuristicAlgrICSampler
from frogger.custom_sampling import FnHeuristicAlgrICSampler
from frogger.solvers import FroggerConfig, Frogger
# Import our custom classes
from frogger.custom_robot_model import FunctionalRobotModel  # Updated import
from frogger.custom_solver import FunctionalFrogger         # Updated import
from frogger.utils import timeout

from frogger.custom_sampling import ContactHeuristicAlgrICSampler, PalmHeuristicAlgrICSampler
from frogger.learning_based_heuristics import ContactDBHeuristic, ContactGenHeuristic
from frogger.utils import add_marker

def select_functional_points(o3d_mesh: o3d.geometry.TriangleMesh, offset=None):
    """
    Converts the mesh to a point cloud by sampling points uniformly, estimates normals,
    and then allows the user to select functional contact points by clicking on the point cloud.

    SHIFT + Left-click to pick points.
    Press 'Q' or close the window to confirm selection.

    Parameters
    ----------
    o3d_mesh : o3d.geometry.TriangleMesh
        The already processed (scaled, transformed) mesh.
    num_points : int, optional
        Number of points to sample from the mesh for the point cloud.

    Returns
    -------
    functional_contacts : list of (pos, dir)
        Each element is a tuple (position, direction) where:
        - position is the 3D coordinates of the picked point (np.array of shape (3,))
        - direction is the estimated normal at that point (np.array of shape (3,))
    """

    # Sample a point cloud from the mesh
    pcd = o3d_mesh.sample_points_uniformly(number_of_points=10000)

    # Estimate normals for the sampled point cloud
    pcd.estimate_normals()
    # Optionally, orient the normals consistently
    pcd.orient_normals_consistent_tangent_plane(30)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window("Select Functional Points (PCD)", width=1024, height=768)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    picked_ids = vis.get_picked_points()
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    functional_contacts = []
    for pid in picked_ids:
        pos = points[pid]
        if offset is not None:
            pos += offset
        direction = -normals[pid]
        # direction = None
        functional_contacts.append((pos, direction))

    print(f"{len(functional_contacts)} points selected.")
    return functional_contacts


# -------------------- Main Code --------------------

obj_name = "hot_glue_gun"

# Load and process mesh - this part remains the same
mesh = trimesh.load(
    f"/home/bowenj/Projects/DexFun/reconstruction/mesh_raw/{obj_name}.stl",
    file_type="stl"
)
mesh.apply_scale(0.2)
bounds = mesh.bounds
lb_O = bounds[0, :]
ub_O = bounds[1, :]
X_WO = RigidTransform(
    RotationMatrix(),
    np.array([0.0, 0.0, -lb_O[-1]]),
)
obj = MeshObjectConfig(X_WO=X_WO, mesh=mesh, name=obj_name, clean=False).create()

# Convert mesh and get functional contacts - this remains the same
processed_mesh = obj.mesh.as_open3d
# functional_contacts = select_functional_points(
#     processed_mesh, 
#     offset=np.array([0.0, 0.0, -lb_O[-1]])
# )
# print(functional_contacts)
support_contacts = [
    (np.array([-0.01386208,  0.04284091,  0.07107334]), np.array([ 0.98006238,  0.05186685, -0.19180085])),
    (np.array([0.02397209, 0.03807753, 0.06341889]), np.array([-0.97386469,  0.22416271, -0.03658755])),
    ]
# support_contacts = [
#     (np.array([-0.01386208,  0.04284091,  0.07107334]), None),
#     (np.array([0.02397209, 0.03807753, 0.06341889]), None)
#     ]
functional_contacts = [
    (np.array([ 0.00308108, -0.02312746,  0.07774291]), np.array([ 0.10211614,  0.75565293, -0.64696287])),
    # *support_contacts,
    ]

# Create the configuration
model_cfg = AlgrModelConfig(
# model_cfg = LeapModelConfig(
    obj=obj,
    ns=4,
    mu=0.7,
    d_min=0.001,
    d_pen=0.005,
    l_bar_cutoff=0.3,
    hand="lh",
)


# Compute the unconstrained pose
# model = model_cfg.create()
model = FunctionalRobotModel(model_cfg)  # # Create our functional robot model instead of regular model

# Sampler selection remains the same
if len(functional_contacts) > 0:
    # sampler = FnHeuristicAlgrICSampler(model, functional_contacts)
    contact_predictor = ContactDBHeuristic()
    sampler = PalmHeuristicAlgrICSampler(model, functional_contacts, contact_predictor, palm_offset=0.05)
    # sampler = ContactHeuristicAlgrICSampler(model=model, functional_contacts=functional_contacts, contact_predictor=contact_predictor)
else:
    sampler = HeuristicAlgrICSampler(model)

# Create our functional solver
frogger = FunctionalFrogger(  # Changed from FnFrogger
    cfg=FroggerConfig(
        model=model,
        sampler=sampler,
        tol_surf=1e-2,      # 1e-3
        tol_joint=1e-2,     # 1e-2
        tol_col=1e-2,       # 1e-3
        tol_fclosure=2e-1,  # relaxed
        xtol_rel=1e-6,      # relaxed 1e-4
        xtol_abs=1e-6,      # relaxed 1e-4
        maxeval=1000,       # 1000
    ),
    functional_contacts=functional_contacts,
)

# Generate grasp
print("Model compiled! Generating grasp...")
# q_star, q0 = timeout(1000.0)(frogger.generate_grasp)(optimize=False, check_constraints=False, tol_pos=0.15, tol_ang=0.5)
q_star, q0 = timeout(1000.0)(frogger.generate_grasp)(optimize=False, check_constraints=False, tol_pos=0.01, tol_ang=0.1)
print("Grasp generated!")

# fingertip_poses = model.compute_fingertip_poses()
# fingertip_poses = model.compute_contact_poses(q=q0)

# # Visualize the fingertips
# for i, pose in enumerate(fingertip_poses):
#     pos, _ = pose
#     # sphere = Sphere(0.0025)
#     model.meshcat.SetObject(
#         path=f"fingertip_{i}",
#         shape=Sphere(0.01),
#         rgba=Rgba(1.0, 0.0, 0.0, 1.0),
#     )
#     model.meshcat.SetTransform(
#         path=f"fingertip_{i}",
#         X_ParentPath=RigidTransform(pos)
#     )
add_marker(model, sampler.X_WPalm_des.translation(), color=[0,1,0,1], radius=0.01, name="palm")
if hasattr(frogger.sampler, "add_visualization"):
    frogger.sampler.add_visualization(model)

# Visualize configurations
model.viz_config(q0)
if q_star is not None:
    model.viz_config(q_star)

# Generate grasps
# from copy import deepcopy
# results = []
# for i in range(10):
#     print("Model compiled! Generating grasp...")
#     q_star, q0 = timeout(1000.0)(frogger.generate_grasp)(optimize=False, tol_pos=0.12, tol_ang=0.5)
#     print("Grasp generated!")

#     # model_copy = deepcopy(model)
#     if hasattr(frogger.sampler, "add_visualization"):
#         frogger.sampler.add_visualization(model)

#     # results.append((model, q0))
#     model.viz_config(q0)

    


