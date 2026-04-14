#!/usr/bin/env python3
"""Render grasp using Open3D offscreen renderer."""
import os, sys, numpy as np, torch, trimesh
import open3d as o3d
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation as ScipyR

sys.path.insert(0, os.path.dirname(__file__))
from frogger.batched_pytorch_solver import _visual_meshes, BatchedSDF

URDF = os.path.join(os.path.dirname(__file__), "models/leap_rh/leap.urdf")
MDIR = os.path.join(os.path.dirname(__file__), "models/leap_rh")
OUT = "output/diagnostics/renders"

def render_grasp(grasp_path, mesh_path, tag="", idx=0):
    os.makedirs(OUT, exist_ok=True)
    results = torch.load(grasp_path, weights_only=False, map_location="cpu")
    g = results[idx] if isinstance(results, list) else results

    # Object mesh + SDF
    obj = trimesh.load(mesh_path, force="mesh")
    offset = np.array([0.0, 0.0, -obj.bounds[0, 2]])
    X_WO = np.eye(4); X_WO[:3, 3] = offset
    sdf = BatchedSDF(obj, X_WO, resolution=128, device="cuda")
    obj.apply_translation(offset)

    # FK
    chain = pk.build_chain_from_urdf(open(URDF).read())
    q = torch.tensor(g["q_joints"], dtype=torch.float32).unsqueeze(0)
    fk = chain.forward_kinematics(q)
    T_base = np.eye(4); T_base[:3, :3] = g["base_rot"]; T_base[:3, 3] = g["base_pos"]

    vis = _visual_meshes("rh", "leap")

    # Build O3D meshes
    # Object
    o3d_obj = o3d.geometry.TriangleMesh()
    o3d_obj.vertices = o3d.utility.Vector3dVector(np.asarray(obj.vertices))
    o3d_obj.triangles = o3d.utility.Vector3iVector(np.asarray(obj.faces))
    o3d_obj.compute_vertex_normals()
    o3d_obj.paint_uniform_color([0.7, 0.7, 0.7])

    hand_meshes = []
    for link_name, mesh_list in vis.items():
        if link_name not in fk: continue
        wT = T_base @ fk[link_name].get_matrix()[0].numpy()
        is_palm = "palm" in link_name
        for mi, (mf, vp) in enumerate(mesh_list):
            path = os.path.join(MDIR, mf)
            if not os.path.exists(path): continue
            lm = trimesh.load(path, force="mesh")
            full_T = wT.copy()
            if vp is not None:
                vpa = np.array(vp, dtype=np.float64)
                Rv = ScipyR.from_euler("xyz", vpa[3:]).as_matrix()
                Tv = np.eye(4); Tv[:3, :3] = Rv; Tv[:3, 3] = vpa[:3]
                full_T = full_T @ Tv
            v = np.asarray(lm.vertices, dtype=np.float64)
            vw = (full_T[:3, :3] @ v.T).T + full_T[:3, 3]
            m = o3d.geometry.TriangleMesh()
            m.vertices = o3d.utility.Vector3dVector(vw)
            m.triangles = o3d.utility.Vector3iVector(np.asarray(lm.faces))
            m.compute_vertex_normals()
            if is_palm:
                m.paint_uniform_color([0.2, 0.4, 1.0])
            else:
                m.paint_uniform_color([1.0, 0.8, 0.3])
            hand_meshes.append(m)

    # Add coordinate frame axes at the base — as cylinders for visibility
    base_origin = g["base_pos"]
    R = g["base_rot"]
    axis_len = 0.06
    axis_colors = [(1, 0, 0), (0, 0.8, 0), (0, 0, 1)]  # R=x, G=y, B=z

    frame_cyls = []
    for ai in range(3):
        tip = base_origin + axis_len * R[:, ai]
        # Create cylinder from base_origin to tip
        direction = tip - base_origin
        length = np.linalg.norm(direction)
        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=0.003, height=length)
        # Align cylinder along direction
        mid = (base_origin + tip) / 2
        z_axis = direction / length
        # Find rotation from [0,0,1] to z_axis
        v = np.cross([0, 0, 1], z_axis)
        s = np.linalg.norm(v)
        c = np.dot([0, 0, 1], z_axis)
        if s < 1e-8:
            rot = np.eye(3) if c > 0 else np.diag([1, -1, -1])
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            rot = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)
        cyl.rotate(rot, center=[0, 0, 0])
        cyl.translate(mid)
        cyl.paint_uniform_color(axis_colors[ai])
        frame_cyls.append(cyl)
        # Add sphere at tip
        tip_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        tip_sphere.translate(tip)
        tip_sphere.paint_uniform_color(axis_colors[ai])
        frame_cyls.append(tip_sphere)

    # Palm contact center in world
    palm_contact_local = np.array([0.010, -0.007, 0.051])  # in BASE frame (not link frame)
    palm_center_world = R @ palm_contact_local + base_origin

    # Find the object surface point: project palm center along +x until SDF=0
    # The surface point = palm_center + t * x_hat where SDF(point) ≈ 0
    x_dir = R[:, 0]  # +x = toward object
    import torch as _torch
    _sdf_dev = "cuda" if _torch.cuda.is_available() else "cpu"
    # March along +x from palm center to find surface
    best_pt = palm_center_world.copy()
    for t_step in np.arange(0, 0.10, 0.001):
        test_pt = palm_center_world + t_step * x_dir
        test_t = _torch.tensor(test_pt, dtype=_torch.float32, device=_sdf_dev).reshape(1, 1, 3)
        sdf_val = sdf.query(test_t).item()
        if sdf_val <= 0:
            best_pt = test_pt
            break

    # Object surface point sphere (yellow)
    surf_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
    surf_sphere.translate(best_pt)
    surf_sphere.paint_uniform_color([1, 1, 0])

    # Line from palm center to surface point (should be collinear with +x and surface normal)
    def make_cylinder_between(p0, p1, radius=0.002, color=[1, 0, 1]):
        d = p1 - p0
        length = np.linalg.norm(d)
        if length < 1e-6:
            return o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
        mid = (p0 + p1) / 2
        z_ax = d / length
        v = np.cross([0, 0, 1], z_ax)
        s = np.linalg.norm(v)
        c_val = np.dot([0, 0, 1], z_ax)
        if s < 1e-8:
            rot = np.eye(3) if c_val > 0 else np.diag([1, -1, -1])
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            rot = np.eye(3) + vx + vx @ vx * (1 - c_val) / (s * s)
        cyl.rotate(rot, center=[0, 0, 0])
        cyl.translate(mid)
        cyl.paint_uniform_color(color)
        return cyl

    # Magenta line: palm center → surface point
    approach_cyl = make_cylinder_between(palm_center_world, best_pt, 0.002, [1, 0, 1])

    # Palm center sphere (magenta)
    palm_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
    palm_sphere.translate(palm_center_world)
    palm_sphere.paint_uniform_color([1, 0, 1])

    # Base origin sphere (white)
    base_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
    base_sphere.translate(base_origin)
    base_sphere.paint_uniform_color([1, 1, 1])

    # Render from multiple views
    renderer = o3d.visualization.rendering.OffscreenRenderer(800, 600)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat_line = o3d.visualization.rendering.MaterialRecord()
    mat_line.shader = "unlitLine"
    mat_line.line_width = 3.0

    renderer.scene.add_geometry("object", o3d_obj, mat)
    for i, hm in enumerate(hand_meshes):
        renderer.scene.add_geometry(f"hand_{i}", hm, mat)
    for ci, fc in enumerate(frame_cyls):
        renderer.scene.add_geometry(f"frame_{ci}", fc, mat)
    renderer.scene.add_geometry("approach", approach_cyl, mat)
    renderer.scene.add_geometry("palm_sphere", palm_sphere, mat)
    renderer.scene.add_geometry("base_sphere", base_sphere, mat)

    # Compute scene center
    all_verts = np.asarray(obj.vertices)
    center = all_verts.mean(axis=0)

    views = {
        "front": [0, 0, 0.4],
        "side": [0.4, 0, 0.15],
        "top": [0, -0.01, 0.5],
    }

    for vname, eye_offset in views.items():
        eye = center + np.array(eye_offset)
        up = np.array([0, 0, 1]) if vname != "top" else np.array([0, -1, 0])
        renderer.setup_camera(60.0, center, eye, up)
        renderer.scene.set_background([1, 1, 1, 1])
        img = renderer.render_to_image()
        path = os.path.join(OUT, f"{tag}_g{idx}_{vname}.png")
        o3d.io.write_image(path, img)
        print(f"Saved: {path}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--grasp", required=True)
    p.add_argument("--mesh", required=True)
    p.add_argument("--tag", default="render")
    p.add_argument("--idx", type=int, default=0)
    a = p.parse_args()
    render_grasp(a.grasp, a.mesh, a.tag, a.idx)
