import numpy as np
import trimesh
import open3d as o3d
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.geometry import Sphere, Rgba

from frogger.objects import MeshObjectConfig
from frogger.robots.robots import AlgrModelConfig
from frogger.sampling import HeuristicAlgrICSampler
from frogger.custom_sampling import FnHeuristicAlgrICSampler
from frogger.solvers import FroggerConfig, Frogger
# Import our custom classes
from frogger.custom_robot_model import FunctionalRobotModel  # Updated import
from frogger.custom_solver_parallel import FunctionalFrogger         # Updated import
from frogger.utils import timeout

from frogger.custom_sampling import ContactHeuristicAlgrICSampler
from frogger.learning_based_heuristics import ContactDBHeuristic, ContactGenHeuristic

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

# Load and process mesh
mesh = trimesh.load(
    f"/home/bowenj/Projects/DexFun/reconstruction/mesh_raw/{obj_name}.stl",
    file_type="stl"
)
rotation = trimesh.transformations.rotation_matrix(
    angle=np.pi,  # 180 degrees in radians
    direction=[0, 0, 1]  # z-axis
)
# mesh.apply_transform(rotation)
mesh.apply_scale(0.2)
bounds = mesh.bounds
lb_O = bounds[0, :]
ub_O = bounds[1, :]
X_WO = RigidTransform(
    RotationMatrix(),
    np.array([0.0, 0.0, -lb_O[-1]]),
)
obj = MeshObjectConfig(X_WO=X_WO, mesh=mesh, name=obj_name, clean=False).create()

# Convert mesh and get functional contacts
processed_mesh = obj.mesh.as_open3d
# functional_contacts = select_functional_points(
#     processed_mesh, 
#     offset=np.array([0.0, 0.0, -lb_O[-1]])
# )
# print(functional_contacts)
functional_contacts = [
    (np.array([ 0.00308108, -0.02312746,  0.07774291]), np.array([ 0.10211614,  0.75565293, -0.64696287])),
    ]
# for i in range(len(functional_contacts)):
#     loc, dir = functional_contacts[i]
#     loc[:2] *= -1
#     dir[:2] *= -1
#     functional_contacts[i] = (loc, dir)

success = False
while not success:
    # Create the configuration
    model_cfg = AlgrModelConfig(
        obj=obj,
        ns=4,
        mu=0.7,
        d_min=0.001,
        d_pen=0.005,
        l_bar_cutoff=0.3,
        hand="rh",
    )

    # Compute the unconstrained pose
    # model = model_cfg.create()
    model = FunctionalRobotModel(model_cfg)  # # Create our functional robot model instead of regular model
    sampler = HeuristicAlgrICSampler(model)
    frogger = Frogger(  # Changed from FnFrogger
        cfg=FroggerConfig(
            model=model,
            sampler=sampler,
            tol_surf=1e-3,      # 1e-3
            tol_joint=1e-2,     # 1e-2
            tol_col=1e-3,       # 1e-3
            tol_fclosure=1e-5,  # relaxed
            xtol_rel=1e-5,      # relaxed
            xtol_abs=1e-5,      # relaxed
            maxeval=1000,
        )
    )
    print("Model compiled! Generating grasp...")
    q_star, q0 = timeout(1000.0)(frogger.generate_grasp)(optimize=True, tol_pos=0.15, tol_ang=0.5)
    print("Initial grasp generated!")

    # Convert the poses into pos + normal
    fingertip_poses = model.compute_fingertip_poses()
    fingertip_pos = [pose[0] for pose in fingertip_poses]
    # fingertip dir is the normal (along x axis)
    fingertip_dir = [pose[1] for pose in fingertip_poses]
    initial_poses = list(zip(fingertip_pos, fingertip_dir))
    # initial_poses = list(zip(fingertip_pos, [None for _ in fingertip_poses]))


    # Now create the functional solver
    # Try replacing each finger with the functional contact
    q_finals = []
    for i  in range(4):
        contacts = initial_poses.copy()
        contacts[i] = functional_contacts[0]

        # Create the configuration
        sampler = FnHeuristicAlgrICSampler(model, contacts)
        # contact_predictor = ContactDBHeuristic()
        # sampler = ContactHeuristicAlgrICSampler(model=model, functional_contacts=functional_contacts, contact_predictor=contact_predictor)

        # Create our functional solver
        frogger = FunctionalFrogger(  # Changed from FnFrogger
            cfg=FroggerConfig(
                model=model,
                sampler=sampler,
                tol_surf=1e-2,      # 1e-3
                tol_joint=1e-1,     # 1e-2
                tol_col=1e-2,       # 1e-3
                tol_fclosure=5e-3,  # relaxed
                xtol_rel=1e-4,      # relaxed
                xtol_abs=1e-4,      # relaxed
                maxeval=1000,
            ),
            functional_contacts=contacts,
        )

        # Generate grasp
        print(f"Trying reset finger {i}! Generating grasp...")
        try:
            _, q0 = timeout(30.0)(frogger.generate_grasp)(optimize=False, tol_pos=0.15, tol_ang=0.5, set_palm=False)
        except Exception as e:
            print(f"Failed to generate grasp for finger {i}! Error: {e}")
            continue

        print("Final grasp generated!")
        if hasattr(frogger.sampler, "add_visualization"):
            frogger.sampler.add_visualization(model)

        # Visualize configurations
        q_finals.append(q0)
        # model.viz_config(q0)
        # if q_star is not None:
        #     model.viz_config(q_star)

        if len(q_finals) == 0:
            print("No valid grasps found!")
        else:
            success = True

print(f"Found {len(q_finals)} valid grasps!")
model.viz_config(q_star)
for i, q in enumerate(q_finals):
    model.viz_config(q)

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

    


