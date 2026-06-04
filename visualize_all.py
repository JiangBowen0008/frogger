"""Multi-object grasp visualizer with object and grasp dropdowns."""
import argparse
import numpy as np
import trimesh
import torch
import pytorch_kinematics as pk
import viser
import os
import time
from scipy.spatial.transform import Rotation

from frogger.batched_pytorch_solver import _visual_meshes, _link_names

ASSETS = "/media/bowenj/DATA/projects/DexFun/assets"
# Default: latest 3-batch run with all wins (palm-d 0-5cm, IK 250, surf_pt 70/30)
GRASP_DIR = "output/v34_split_70_30_3batch"

OBJECTS = {
    "grinder": {
        "mesh": f"{ASSETS}/mesh_obj/grinder/object.obj",
        "grasp": f"{GRASP_DIR}/grinder/grasps_pooled.pt",
    },
    "spray_bottle": {
        "mesh": f"{ASSETS}/mesh_obj/funky_clear_spray_bottle/object.obj",
        "grasp": f"{GRASP_DIR}/funky_clear_spray_bottle/grasps_pooled.pt",
    },
    "flashlight": {
        "mesh": f"{ASSETS}/mesh_obj/flashlight/object.obj",
        "grasp": f"{GRASP_DIR}/flashlight/grasps_pooled.pt",
    },
    "air_blower": {
        "mesh": f"{ASSETS}/mesh_obj/air_blower/object.obj",
        "grasp": f"{GRASP_DIR}/air_blower/grasps_pooled.pt",
    },
    "hot_glue_gun": {
        "mesh": f"{ASSETS}/mesh_obj/hot_glue_gun/object.obj",
        "grasp": f"{GRASP_DIR}/hot_glue_gun/grasps_pooled.pt",
    },
}


def is_real_feasible(g):
    """REAL feasibility: passes all gates including FC LP (l_star > 0)."""
    return g.get("feasible", False) and g.get("l_star", -1) > 0


def load_object(mesh_path):
    mesh = trimesh.load(mesh_path, force="mesh")
    bounds = mesh.bounds
    offset = np.array([0.0, 0.0, -bounds[0, 2]])
    X_WO = np.eye(4)
    X_WO[:3, 3] = offset
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    R_WO = X_WO[:3, :3]
    t_WO = X_WO[:3, 3]
    verts_W = (R_WO @ verts.astype(np.float64).T).T + t_WO
    return verts_W.astype(np.float32), faces, X_WO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"Viser running -> http://localhost:{args.port}")

    # FK chain
    hand, hand_type = "rh", "leap"
    urdf_path = os.path.join(os.path.dirname(__file__), f"models/leap_{hand}/leap.urdf")
    mesh_dir = os.path.join(os.path.dirname(__file__), f"models/leap_{hand}")
    chain = pk.build_chain_from_urdf(open(urdf_path).read())
    vis_meshes = _visual_meshes(hand, hand_type)
    tip_names, _ = _link_names(hand, hand_type)

    f_off = np.array([-0.0025, -0.0449, 0.0143])
    t_off = np.array([-0.0020, -0.0558, -0.0144])
    tip_offsets = [f_off, f_off, f_off, t_off]
    tip_colors = [(0, 180, 255), (0, 255, 0), (255, 165, 0), (255, 255, 0)]

    # Pre-load link meshes
    link_mesh_cache = {}
    for link_name, mesh_list in vis_meshes.items():
        for mi, (mesh_file, vis_pose) in enumerate(mesh_list):
            full_path = os.path.join(mesh_dir, mesh_file)
            if not os.path.exists(full_path):
                continue
            lm = trimesh.load(full_path, force="mesh")
            link_mesh_cache[(link_name, mi)] = (
                np.asarray(lm.vertices, dtype=np.float32),
                np.asarray(lm.faces, dtype=np.int32),
                vis_pose,
            )

    # Pre-load all objects and grasps
    obj_data = {}
    for name, info in OBJECTS.items():
        if not os.path.exists(info["mesh"]) or not os.path.exists(info["grasp"]):
            print(f"  Skipping {name}: missing files")
            continue
        verts_W, faces, X_WO = load_object(info["mesh"])
        all_results = torch.load(info["grasp"], weights_only=False)
        # Filter to REAL feasibles only — passes all gates including FC LP
        results = [g for g in all_results if is_real_feasible(g)]
        if not results:
            print(f"  Skipping {name}: 0 REAL feasibles (of {len(all_results)} total)")
            continue
        obj_data[name] = {
            "verts": verts_W, "faces": faces, "results": results,
        }
        print(f"  Loaded {name}: {len(results)} REAL feasible grasps (of {len(all_results)} total)")

    obj_names = list(obj_data.keys())
    if not obj_names:
        print("No objects found!")
        return

    # State
    current_obj = [obj_names[0]]
    current_grasp = [0]

    def clear_scene():
        """Remove all scene elements by adding empty/invisible replacements."""
        # Viser doesn't have scene.remove — overwrite with tiny invisible meshes
        dummy_v = np.zeros((3, 3), dtype=np.float32)
        dummy_f = np.array([[0, 1, 2]], dtype=np.int32)
        server.scene.add_mesh_simple("/object", vertices=dummy_v, faces=dummy_f,
                                     color=(0, 0, 0), opacity=0.0)
        for (link_name, mi) in link_mesh_cache:
            server.scene.add_mesh_simple(f"/hand/{link_name}_{mi}",
                                         vertices=dummy_v, faces=dummy_f,
                                         color=(0, 0, 0), opacity=0.0)
        for link in tip_names:
            server.scene.add_icosphere(f"/tips/{link}", radius=0.0001,
                                       color=(0, 0, 0), position=(0, 0, 0))

    def show_object(name):
        data = obj_data[name]
        server.scene.add_mesh_simple(
            "/object", vertices=data["verts"], faces=data["faces"],
            color=(180, 180, 180), opacity=0.7,
        )

    def show_grasp(obj_name, grasp_idx):
        data = obj_data[obj_name]
        results = data["results"]
        if grasp_idx >= len(results):
            return
        result = results[grasp_idx]

        q = torch.tensor(result["q_joints"], dtype=torch.float32).unsqueeze(0)
        fk = chain.forward_kinematics(q)

        T_base = np.eye(4)
        T_base[:3, :3] = result["base_rot"]
        T_base[:3, 3] = result["base_pos"]

        # Link meshes
        for (link_name, mi), (lv, lf, vis_pose) in link_mesh_cache.items():
            if link_name not in fk:
                continue
            link_T = fk[link_name].get_matrix()[0].numpy()
            world_T = T_base @ link_T
            if vis_pose is not None:
                vp = np.array(vis_pose)
                Rv = Rotation.from_euler("xyz", vp[3:]).as_matrix()
                Tv = np.eye(4)
                Tv[:3, :3] = Rv
                Tv[:3, 3] = vp[:3]
                world_T = world_T @ Tv
            lv_w = (world_T[:3, :3] @ lv.T).T + world_T[:3, 3]
            server.scene.add_mesh_simple(
                f"/hand/{link_name}_{mi}",
                vertices=lv_w.astype(np.float32), faces=lf,
                color=(200, 200, 220), opacity=0.85,
            )

        # Fingertip spheres
        for i, (link, off) in enumerate(zip(tip_names, tip_offsets)):
            if link not in fk:
                continue
            link_T = fk[link].get_matrix()[0].numpy()
            w_T = T_base @ link_T
            p = w_T[:3, :3] @ off + w_T[:3, 3]
            server.scene.add_icosphere(
                f"/tips/{link}", radius=0.006, color=tip_colors[i],
                position=p.astype(np.float32),
            )

        # Info
        feas = "FEAS" if result.get("feasible", False) else "----"
        sigma = result.get("sigma_min", result.get("l_star", 0))
        info_md.content = (
            f"**{obj_name}** | Grasp {grasp_idx+1}/{len(results)} | "
            f"[{feas}] σ_min={sigma:.4f}"
        )

    def refresh():
        clear_scene()
        show_object(current_obj[0])
        show_grasp(current_obj[0], current_grasp[0])

    # GUI
    with server.gui.add_folder("Browse"):
        info_md = server.gui.add_markdown("")

        obj_dropdown = server.gui.add_dropdown(
            "Object", options=obj_names, initial_value=obj_names[0],
        )

        grasp_options = [f"#{i+1}" for i in range(len(obj_data[obj_names[0]]["results"]))]
        grasp_dropdown = server.gui.add_dropdown(
            "Grasp", options=grasp_options, initial_value=grasp_options[0],
        )

        btn_prev = server.gui.add_button("< Prev Grasp")
        btn_next = server.gui.add_button("Next Grasp >")

    @obj_dropdown.on_update
    def _on_obj(_):
        current_obj[0] = obj_dropdown.value
        current_grasp[0] = 0
        n = len(obj_data[current_obj[0]]["results"])
        new_options = [f"#{i+1}" for i in range(n)]
        grasp_dropdown.options = new_options
        grasp_dropdown.value = new_options[0]

    @grasp_dropdown.on_update
    def _on_grasp(_):
        idx = int(grasp_dropdown.value.replace("#", "")) - 1
        current_grasp[0] = idx
        refresh()

    @btn_prev.on_click
    def _on_prev(_):
        n = len(obj_data[current_obj[0]]["results"])
        current_grasp[0] = (current_grasp[0] - 1) % n
        grasp_dropdown.value = f"#{current_grasp[0]+1}"

    @btn_next.on_click
    def _on_next(_):
        n = len(obj_data[current_obj[0]]["results"])
        current_grasp[0] = (current_grasp[0] + 1) % n
        grasp_dropdown.value = f"#{current_grasp[0]+1}"

    # Show first
    refresh()

    print("Press Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
