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

MESHES_DIR = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg"
ASSETS = "/home/bowenj/Projects/DexFun/assets"

# Default mesh for old versions
DEFAULT_MESH = f"{ASSETS}/mesh_obj/black_spray_bottle_single/object.obj"

# (grasp_path, mesh_path_or_None_for_default)
VERSIONS = {
    "glue_00_init": ("output/grasps/stage_after_init.pt", f"{MESHES_DIR}/hot_glue_gun/object.obj"),
    "glue_01_after_P0": ("output/grasps/stage_after_P0.pt", f"{MESHES_DIR}/hot_glue_gun/object.obj"),
    "glue_02_after_P1": ("output/grasps/stage_after_P1.pt", f"{MESHES_DIR}/hot_glue_gun/object.obj"),
    "glue_03_final": ("output/grasps/stage_final.pt", f"{MESHES_DIR}/hot_glue_gun/object.obj"),
    "spray_warmstart": ("output/grasps/compare_warmstart_single.pt", None),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8092)
    args = parser.parse_args()

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"Comparison visualizer -> http://localhost:{args.port}")

    # Pre-load all unique meshes
    mesh_cache = {}  # mesh_path -> (obj_verts, obj_faces)
    for name, (grasp_path, mesh_path) in VERSIONS.items():
        mp = mesh_path or DEFAULT_MESH
        if mp not in mesh_cache and os.path.exists(mp):
            m = trimesh.load(mp, force="mesh")
            off = np.array([0.0, 0.0, -m.bounds[0, 2]])
            xwo = np.eye(4); xwo[:3, 3] = off
            verts = (xwo[:3, :3] @ np.asarray(m.vertices, dtype=np.float64).T).T + xwo[:3, 3]
            mesh_cache[mp] = (verts.astype(np.float32), np.asarray(m.faces, dtype=np.int32))

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
    version_meshes = {}  # name -> mesh_path
    for name, (grasp_path, mesh_path) in VERSIONS.items():
        mp = mesh_path or DEFAULT_MESH
        if os.path.exists(grasp_path) and mp in mesh_cache:
            results = torch.load(grasp_path, weights_only=False)
            versions[name] = results
            version_meshes[name] = mp
            print(f"  {name}: {len(results)} grasps ({os.path.basename(os.path.dirname(mp))})")

    version_names = list(versions.keys())
    current_version = [version_names[0]]
    current_grasp = [0]

    def show(ver_name, grasp_idx):
        data = versions[ver_name]
        if grasp_idx >= len(data):
            grasp_idx = 0
        g = data[grasp_idx]

        # Object (per-version mesh)
        mp = version_meshes[ver_name]
        ov, of = mesh_cache[mp]
        server.scene.add_mesh_simple(
            "/object", vertices=ov, faces=of,
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
