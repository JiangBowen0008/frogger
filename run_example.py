"""End-to-end test of the batched PyTorch grasp optimiser + viser visualisation.

Supports both Allegro and LEAP hands.
Usage:
    python run_example.py                    # Allegro LH, sugar box
    python run_example.py --hand_type leap   # LEAP RH, sugar box
"""

import argparse
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import time
import os

from frogger.batched_pytorch_solver import (
    BatchedSDF,
    BatchedGraspOptimizer,
    _visual_meshes,
    _link_names,
    _JOINT_LOWER,
    _JOINT_UPPER,
    _LEAP_JOINT_LOWER,
    _LEAP_JOINT_UPPER,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand_type", default="allegro", choices=["allegro", "leap"])
    parser.add_argument("--hand", default=None, help="lh or rh (default: lh for allegro, rh for leap)")
    parser.add_argument("--obj", default="004_sugar_box")
    parser.add_argument("--mesh", default=None, help="Path to custom mesh .obj file")
    parser.add_argument("--actuation", default=None, help="Path to actuation contacts JSON file")
    parser.add_argument("--num_envs", type=int, default=4000)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--resolution", type=int, default=128, help="SDF grid resolution (128 or 256)")
    parser.add_argument("--load", default=None, help="Load cached results from .pt file instead of optimising")
    parser.add_argument("--save_dir", default="output/grasps", help="Directory to save results")
    parser.add_argument("--port", type=int, default=8090, help="Viser visualization port")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization (batch mode)")
    args = parser.parse_args()

    hand = args.hand
    if hand is None:
        hand = "rh" if args.hand_type == "leap" else "lh"

    import json as _json

    if args.mesh:
        # Custom mesh + actuation contacts
        mesh_path = args.mesh
        obj_name = os.path.splitext(os.path.basename(os.path.dirname(mesh_path)))[0] or \
                   os.path.splitext(os.path.basename(mesh_path))[0]
    else:
        obj_name = args.obj
        mesh_path = f"data/{obj_name}/{obj_name}.obj"

    print(f"Hand: {args.hand_type} {hand}")
    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path, force="mesh")

    bounds = mesh.bounds
    offset = np.array([0.0, 0.0, -bounds[0, 2]])
    X_WO = np.eye(4)
    X_WO[:3, 3] = offset
    obj_center = mesh.centroid + offset

    if args.actuation:
        # Load actuation contacts from JSON
        with open(args.actuation) as f:
            act_data = _json.load(f)
        actuation_targets = [
            (np.array(c["pos"]) + offset, np.array(c["dir"]))
            for c in act_data["actuation_contacts"]
        ]
        for i, (pos, d) in enumerate(actuation_targets):
            print(f"  Actuation[{i}]: pos={pos}, dir={d}")
    else:
        # Default: surface-point actuation target
        verts_W = (X_WO[:3, :3] @ np.asarray(mesh.vertices).T).T + X_WO[:3, 3]
        mesh_W = trimesh.Trimesh(vertices=verts_W, faces=mesh.faces)
        candidate = np.array([[0.0, 0.0, offset[2] + (bounds[1, 2] - bounds[0, 2]) * 0.8]])
        closest_pts, _, _ = trimesh.proximity.closest_point(mesh_W, candidate)
        act_pos = closest_pts[0]
        actuation_targets = [(act_pos, None)]
        print(f"Actuation target: {act_pos}")

    import torch
    save_file = os.path.join(args.save_dir, f"{obj_name}_{args.hand_type}_{hand}.pt")
    if args.load:
        save_file = args.load

    if args.load and os.path.exists(args.load):
        print(f"Loading cached results from {args.load}")
        results = torch.load(args.load, weights_only=False)
        print(f"  Loaded {len(results)} grasps")
    else:
        print("Building SDF ...")
        sdf = BatchedSDF(mesh, X_WO, bounds_padding=0.15, resolution=args.resolution, device="cuda")

        opt = BatchedGraspOptimizer(
            sdf, num_envs=args.num_envs, device="cuda",
            hand=hand, hand_type=args.hand_type,
        )
        results = opt.optimize(
            actuation_targets=actuation_targets,
            object_center=obj_center,
            steps=args.steps,
            lr=0.005,
            save_path=save_file,
        )

    best = results[0]
    jl = _LEAP_JOINT_LOWER if args.hand_type == "leap" else _JOINT_LOWER
    jh = _LEAP_JOINT_UPPER if args.hand_type == "leap" else _JOINT_UPPER
    violations = [
        (i, v, lo, hi) for i, (lo, hi, v)
        in enumerate(zip(jl, jh, best["q_joints"]))
        if v < lo - 1e-6 or v > hi + 1e-6
    ]
    if violations:
        print("\n  Joint limit violations:")
        for i, v, lo, hi in violations:
            print(f"    joint {i}: {v:.4f} not in [{lo:.4f}, {hi:.4f}]")
    else:
        print("\n  All joints within limits")

    if not args.no_viz:
        visualize(mesh, X_WO, results, actuation_targets, hand, args.hand_type, port=args.port)


def visualize(mesh, X_WO, results, actuation_targets, hand, hand_type="allegro", port=8090):
    """Interactive 3-D visualisation using viser with multi-grasp browsing."""
    import viser
    import pytorch_kinematics as pk
    import torch

    server = viser.ViserServer(host="0.0.0.0", port=port)
    print(f"\n  Viser running -> http://localhost:{port}  ({len(results)} grasps)")

    # Object mesh (static)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    R_WO = X_WO[:3, :3]
    t_WO = X_WO[:3, 3]
    verts_W = (R_WO @ verts.astype(np.float64).T).T + t_WO
    server.scene.add_mesh_simple(
        "/object", vertices=verts_W.astype(np.float32), faces=faces,
        color=(180, 180, 180), opacity=0.7,
    )

    # Actuation targets (static)
    for i, (pos, _) in enumerate(actuation_targets):
        server.scene.add_icosphere(
            f"/act/{i}", radius=0.008, color=(255, 0, 0),
            position=pos.astype(np.float32),
        )

    # FK chain (shared)
    if hand_type == "leap":
        urdf_path = os.path.join(os.path.dirname(__file__), f"models/leap_{hand}/leap.urdf")
        mesh_dir = os.path.join(os.path.dirname(__file__), f"models/leap_{hand}")
    else:
        urdf_path = os.path.join(os.path.dirname(__file__), f"models/allegro/allegro_{hand}.urdf")
        mesh_dir = os.path.join(os.path.dirname(__file__), "models/allegro")
    chain = pk.build_chain_from_urdf(open(urdf_path).read())
    vis_meshes = _visual_meshes(hand, hand_type)
    tip_names, _ = _link_names(hand, hand_type)

    if hand_type == "leap":
        f_off = np.array([-0.0025, -0.0449, 0.0143])
        t_off = np.array([-0.0020, -0.0558, -0.0144])
    else:
        th_a = np.pi / 4.0
        r = 0.012
        f_off = np.array([r * np.sin(th_a), 0.0, 0.0267 + r * np.cos(th_a)])
        t_off = np.array([r * np.sin(th_a), 0.0, 0.0423 + r * np.cos(th_a)])
    tip_offsets = [f_off, f_off, f_off, t_off]
    tip_colors = [(0, 180, 255), (0, 255, 0), (255, 165, 0), (255, 255, 0)]

    # Pre-load link meshes (vertices/faces) so we don't reload from disk each time
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

    # -- Render a single grasp -------------------------------------------
    def show_grasp(idx):
        result = results[idx]
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

        # Update info text
        info = (f"Grasp {idx + 1}/{len(results)}  |  "
                f"l*={result['l_star']:.4f}  |  "
                f"feasible={result['feasible']}")
        info_md.content = info

    # -- GUI controls ----------------------------------------------------
    n = len(results)
    dropdown_options = [
        f"#{i+1}  l*={r['l_star']:.4f}  {'OK' if r['feasible'] else '--'}"
        for i, r in enumerate(results)
    ]

    with server.gui.add_folder("Grasp Browser"):
        info_md = server.gui.add_markdown("")
        dropdown = server.gui.add_dropdown(
            "Select grasp", options=dropdown_options, initial_value=dropdown_options[0],
        )
        btn_prev = server.gui.add_button("< Prev")
        btn_next = server.gui.add_button("Next >")

    current_idx = [0]  # mutable container for closure

    @dropdown.on_update
    def _on_dropdown(_):
        idx = dropdown_options.index(dropdown.value)
        current_idx[0] = idx
        show_grasp(idx)

    @btn_prev.on_click
    def _on_prev(_):
        current_idx[0] = (current_idx[0] - 1) % n
        dropdown.value = dropdown_options[current_idx[0]]

    @btn_next.on_click
    def _on_next(_):
        current_idx[0] = (current_idx[0] + 1) % n
        dropdown.value = dropdown_options[current_idx[0]]

    # Show first grasp
    show_grasp(0)

    print("  Press Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
