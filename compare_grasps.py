"""Compare grasps from different commits/configs side by side."""
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

ASSETS = "/home/bowenj/Projects/DexFun/assets"
MESH_PATH = f"{ASSETS}/mesh_obj/black_spray_bottle_single/object.obj"

VERSIONS = {
    "01_warmstart_good": "output/grasps/compare_warmstart_good.pt",
    "02_warmstart_best": "output/grasps/compare_warmstart_best.pt",
    "03_warmstart_single": "output/grasps/compare_warmstart_single.pt",
    "04_contact_zones": "output/grasps/compare_contact_zones.pt",
    "05_multipoint_palm": "output/grasps/compare_multipoint_palm.pt",
    "06_iterative_batch": "output/grasps/compare_iterative_batch.pt",
    "07_3obj_first": "output/grasps/compare_3obj_first.pt",
    "08_box_sc_routing": "output/grasps/compare_box_sc_routing.pt",
    "09_thumb_5mm": "output/grasps/compare_thumb_5mm.pt",
    "10_3obj_tested": "output/grasps/compare_3obj_tested.pt",
    "11_batched_curated": "output/grasps/compare_batched_curated.pt",
    "12_augmented_lagrangian": "output/grasps/compare_augmented_lagrangian.pt",
    "13_volumetric_box": "output/grasps/compare_volumetric_box.pt",
    "14_visual_mesh": "output/grasps/compare_visual_mesh.pt",
    "15_urdf_box": "output/grasps/compare_urdf_box.pt",
    "16_current_spheres": "output/grasps/compare_current.pt",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8092)
    args = parser.parse_args()

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"Comparison visualizer -> http://localhost:{args.port}")

    # Load object
    mesh = trimesh.load(MESH_PATH, force="mesh")
    bounds = mesh.bounds
    offset = np.array([0.0, 0.0, -bounds[0, 2]])
    X_WO = np.eye(4); X_WO[:3, 3] = offset
    obj_verts = (X_WO[:3, :3] @ np.asarray(mesh.vertices, dtype=np.float64).T).T + X_WO[:3, 3]
    obj_faces = np.asarray(mesh.faces, dtype=np.int32)

    # FK chain
    hand, hand_type = "rh", "leap"
    urdf_path = os.path.join(os.path.dirname(__file__), f"models/leap_{hand}/leap.urdf")
    mesh_dir = os.path.join(os.path.dirname(__file__), f"models/leap_{hand}")
    chain = pk.build_chain_from_urdf(open(urdf_path).read())
    vis_meshes = _visual_meshes(hand, hand_type)

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

    # Load all versions
    versions = {}
    for name, path in VERSIONS.items():
        if os.path.exists(path):
            results = torch.load(path, weights_only=False)
            versions[name] = results
            print(f"  {name}: {len(results)} grasps")

    version_names = list(versions.keys())
    current_version = [version_names[0]]
    current_grasp = [0]

    def show(ver_name, grasp_idx):
        data = versions[ver_name]
        if grasp_idx >= len(data):
            grasp_idx = 0
        g = data[grasp_idx]

        # Object
        server.scene.add_mesh_simple(
            "/object", vertices=obj_verts.astype(np.float32), faces=obj_faces,
            color=(180, 180, 180), opacity=0.7,
        )

        q = torch.tensor(g["q_joints"], dtype=torch.float32).unsqueeze(0)
        fk = chain.forward_kinematics(q)
        T_base = np.eye(4)
        T_base[:3, :3] = g["base_rot"]
        T_base[:3, 3] = g["base_pos"]

        for (link_name, mi), (lv, lf, vis_pose) in link_mesh_cache.items():
            if link_name not in fk:
                continue
            link_T = fk[link_name].get_matrix()[0].numpy()
            world_T = T_base @ link_T
            if vis_pose is not None:
                vp = np.array(vis_pose)
                Rv = Rotation.from_euler("xyz", vp[3:]).as_matrix()
                Tv = np.eye(4); Tv[:3, :3] = Rv; Tv[:3, 3] = vp[:3]
                world_T = world_T @ Tv
            lv_w = (world_T[:3, :3] @ lv.T).T + world_T[:3, 3]
            server.scene.add_mesh_simple(
                f"/hand/{link_name}_{mi}",
                vertices=lv_w.astype(np.float32), faces=lf,
                color=(200, 200, 220), opacity=0.85,
            )

        feas = g.get("feasible", "?")
        sigma = g.get("sigma_min", g.get("l_star", 0))
        info_md.content = (
            f"**{ver_name}** | Grasp {grasp_idx+1}/{len(data)} | "
            f"feas={feas} σ={sigma:.3f}"
        )

    # GUI
    with server.gui.add_folder("Compare"):
        info_md = server.gui.add_markdown("")
        ver_dropdown = server.gui.add_dropdown(
            "Version", options=version_names, initial_value=version_names[0],
        )
        grasp_dd = server.gui.add_dropdown(
            "Grasp", options=[f"#{i+1}" for i in range(10)], initial_value="#1",
        )
        btn_prev = server.gui.add_button("< Prev")
        btn_next = server.gui.add_button("Next >")

    @ver_dropdown.on_update
    def _on_ver(_):
        current_version[0] = ver_dropdown.value
        current_grasp[0] = 0
        n = len(versions[current_version[0]])
        grasp_dd.options = [f"#{i+1}" for i in range(n)]
        grasp_dd.value = "#1"

    @grasp_dd.on_update
    def _on_grasp(_):
        idx = int(grasp_dd.value.replace("#", "")) - 1
        current_grasp[0] = idx
        show(current_version[0], idx)

    @btn_prev.on_click
    def _(_ev):
        n = len(versions[current_version[0]])
        current_grasp[0] = (current_grasp[0] - 1) % n
        grasp_dd.value = f"#{current_grasp[0]+1}"

    @btn_next.on_click
    def _(_ev):
        n = len(versions[current_version[0]])
        current_grasp[0] = (current_grasp[0] + 1) % n
        grasp_dd.value = f"#{current_grasp[0]+1}"

    show(version_names[0], 0)
    print("Press Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
