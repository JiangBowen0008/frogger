#!/usr/bin/env python3
"""
Fundamental tests for SDF computation and collision detection.

Tests:
1. Object mesh watertightness and SDF sign convention
2. Open3D SDF sanity at known points
3. Known geometry (sphere) to verify Open3D conventions
4. Hand link mesh watertightness
5. Palm mesh analysis
6. Warmstart vs batched: per-link vertex penetration
7. BatchedSDF grid vs O3D direct query accuracy

Run: conda run -n frogger python test_sdf_fundamentals.py
"""
import numpy as np
import trimesh
import open3d as o3d
import torch
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from frogger.batched_pytorch_solver import BatchedSDF, _visual_meshes

ASSETS = "/home/bowenj/Projects/DexFun"
OBJ_MESH = f"{ASSETS}/output/meshes/mesh_raw_ahg/black_spray_bottle_single/object.obj"
HAND_DIR = os.path.join(os.path.dirname(__file__), "models/leap_rh")
GRASPS = os.path.join(os.path.dirname(__file__), "output/grasps")

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

# ============================================================================
# TEST 1: Object mesh properties
# ============================================================================
def test_object_mesh():
    section("TEST 1: Object Mesh Properties")
    mesh = trimesh.load(OBJ_MESH, force="mesh")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")
    print(f"  Bounds: {mesh.bounds}")
    print(f"  Is watertight: {mesh.is_watertight}")
    print(f"  Is volume: {mesh.is_volume}")
    print(f"  Euler number: {mesh.euler_number}")
    if mesh.is_watertight:
        print(f"  Volume: {mesh.volume:.6f} m³ ({mesh.volume*1e9:.1f} mm³)")
    return mesh

# ============================================================================
# TEST 2: Open3D SDF sanity on the object
# ============================================================================
def test_object_sdf_sanity(mesh):
    section("TEST 2: Object SDF Sanity")

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    centroid = verts.mean(axis=0)
    bbox_min, bbox_max = verts.min(axis=0), verts.max(axis=0)

    mesh_o3d = o3d.t.geometry.TriangleMesh()
    mesh_o3d.vertex.positions = o3d.core.Tensor(verts)
    mesh_o3d.triangle.indices = o3d.core.Tensor(faces)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_o3d)

    print(f"  Centroid: {centroid}")
    print(f"  Bbox: [{bbox_min}] to [{bbox_max}]")
    print(f"  Size: {(bbox_max-bbox_min)*1000} mm")

    # Key test points
    test_points = [
        ("centroid", centroid, "INSIDE"),
        ("far +x (1m)", centroid + [1, 0, 0], "OUT"),
        ("far +y", centroid + [0, 1, 0], "OUT"),
        ("far -z", centroid + [0, 0, -1], "OUT"),
        ("bbox_min - 50mm", bbox_min - 0.05, "OUT"),
        ("bbox_max + 50mm", bbox_max + 0.05, "OUT"),
        ("surface vert 0", verts[0], "≈0"),
        ("surface vert mid", verts[len(verts)//2], "≈0"),
    ]
    # Ray along +x from centroid
    for d_mm in [5, 10, 20, 30, 50, 80, 100]:
        d = d_mm / 1000.0
        test_points.append((f"c+{d_mm}mm_x", centroid + [d, 0, 0], "?"))
        test_points.append((f"c-{d_mm}mm_x", centroid - [d, 0, 0], "?"))

    pts = np.array([p[1] for p in test_points], dtype=np.float32)
    sdfs = scene.compute_signed_distance(o3d.core.Tensor(pts, dtype=o3d.core.float32)).numpy()

    print(f"\n  {'Name':<25} {'SDF(mm)':>8} {'Sign':>6} {'Expected':>8}")
    print(f"  {'-'*55}")
    for (name, _, exp), sdf_val in zip(test_points, sdfs):
        sign = "IN" if sdf_val < 0 else "OUT"
        flag = ""
        if exp == "INSIDE" and sdf_val >= 0: flag = " ***WRONG***"
        elif exp == "OUT" and sdf_val < 0: flag = " ***WRONG***"
        elif exp == "≈0" and abs(sdf_val) > 0.002: flag = f" (off by {abs(sdf_val)*1000:.1f}mm)"
        print(f"  {name:<25} {sdf_val*1000:>8.2f} {sign:>6} {exp:>8}{flag}")

    # Monotonicity check: ray from centroid along +x
    print(f"\n  Monotonicity (centroid → +x, every 2mm):")
    ray_dists = np.arange(0, 0.15, 0.002)
    ray_pts = np.array([centroid + [d, 0, 0] for d in ray_dists], dtype=np.float32)
    ray_sdfs = scene.compute_signed_distance(o3d.core.Tensor(ray_pts, dtype=o3d.core.float32)).numpy()

    crossed_surface = False
    for i, (d, s) in enumerate(zip(ray_dists, ray_sdfs)):
        if not crossed_surface and s > 0:
            crossed_surface = True
            print(f"    ** Surface crossing at d≈{d*1000:.1f}mm (SDF went positive) **")
        if i % 5 == 0 or (i > 0 and np.sign(s) != np.sign(ray_sdfs[i-1])):
            print(f"    d={d*1000:6.1f}mm: SDF={s*1000:>8.2f}mm {'IN' if s<0 else 'OUT'}")

    return scene

# ============================================================================
# TEST 3: Known geometry (sphere r=50mm)
# ============================================================================
def test_known_geometry():
    section("TEST 3: Known Geometry (sphere r=50mm)")
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.05)
    print(f"  watertight={sphere.is_watertight}")

    verts = np.asarray(sphere.vertices, dtype=np.float32)
    faces = np.asarray(sphere.faces, dtype=np.int32)
    mesh_o3d = o3d.t.geometry.TriangleMesh()
    mesh_o3d.vertex.positions = o3d.core.Tensor(verts)
    mesh_o3d.triangle.indices = o3d.core.Tensor(faces)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_o3d)

    test_r = [0, 0.010, 0.025, 0.040, 0.049, 0.050, 0.051, 0.060, 0.075, 0.100]
    pts = np.array([[r, 0, 0] for r in test_r], dtype=np.float32)
    sdfs = scene.compute_signed_distance(o3d.core.Tensor(pts, dtype=o3d.core.float32)).numpy()

    print(f"  {'r(mm)':<10} {'SDF(mm)':>8} {'Expected':>8} {'Error(mm)':>10}")
    print(f"  {'-'*40}")
    for r, sdf_val in zip(test_r, sdfs):
        expected = r - 0.05
        err = sdf_val - expected
        print(f"  {r*1000:<10.0f} {sdf_val*1000:>8.2f} {expected*1000:>8.2f} {err*1000:>10.3f}")

    print(f"\n  Confirmed: Open3D convention is SDF < 0 = INSIDE")

# ============================================================================
# TEST 4: Hand link mesh watertightness
# ============================================================================
def test_hand_meshes():
    section("TEST 4: Hand Link Mesh Properties")
    mesh_dir = os.path.join(HAND_DIR, "meshes_obj")
    for f in sorted(os.listdir(mesh_dir)):
        if f.endswith(".obj"):
            m = trimesh.load(os.path.join(mesh_dir, f), force="mesh")
            print(f"  {f:<25} V={len(m.vertices):>5} F={len(m.faces):>5} "
                  f"wt={str(m.is_watertight):<5} euler={m.euler_number}")

# ============================================================================
# TEST 5: Palm mesh analysis
# ============================================================================
def test_palm_mesh():
    section("TEST 5: Palm Mesh Analysis")
    palm = trimesh.load(os.path.join(HAND_DIR, "meshes_obj", "palm_lower.obj"), force="mesh")
    print(f"  V={len(palm.vertices)} F={len(palm.faces)} wt={palm.is_watertight}")
    print(f"  Bounds: {palm.bounds}")
    size = palm.bounds[1] - palm.bounds[0]
    print(f"  Size: {size*1000} mm")

    # Count boundary edges
    from collections import Counter
    ec = Counter()
    for face in palm.faces:
        for i in range(3):
            ec[tuple(sorted([face[i], face[(i+1)%3]]))] += 1
    boundary = sum(1 for _,c in ec.items() if c == 1)
    print(f"  Boundary edges: {boundary} (0 = watertight)")

    if not palm.is_watertight:
        print(f"  *** NOT WATERTIGHT — Open3D signed distance unreliable ***")

    # Test Open3D SDF on palm
    verts = np.asarray(palm.vertices, dtype=np.float32)
    faces_np = np.asarray(palm.faces, dtype=np.int32)
    center = verts.mean(axis=0)

    mesh_o3d = o3d.t.geometry.TriangleMesh()
    mesh_o3d.vertex.positions = o3d.core.Tensor(verts)
    mesh_o3d.triangle.indices = o3d.core.Tensor(faces_np)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_o3d)

    pts_list = [("center", center),
                ("far+x", center + [0.1, 0, 0]),
                ("far-x", center - [0.1, 0, 0])]
    for d in [-20, -10, -5, 5, 10, 20]:
        for ax, nm in enumerate(["x","y","z"]):
            off = np.zeros(3, dtype=np.float32)
            off[ax] = d / 1000.0
            pts_list.append((f"c{d:+d}mm_{nm}", center + off))

    pts = np.array([p[1] for p in pts_list], dtype=np.float32)
    sdfs = scene.compute_signed_distance(o3d.core.Tensor(pts, dtype=o3d.core.float32)).numpy()

    print(f"\n  Palm SDF at test points:")
    for (name, _), s in zip(pts_list, sdfs):
        print(f"    {name:<18} SDF={s*1000:>8.2f}mm {'IN' if s<0 else 'OUT'}")

    return palm

# ============================================================================
# TEST 6: Grasp comparison — per-link vertex penetration
# ============================================================================
def test_grasp_comparison(obj_mesh):
    section("TEST 6: Warmstart vs Batched — Per-Link Vertex Penetration")

    import pytorch_kinematics as pk
    from scipy.spatial.transform import Rotation as ScipyR

    bounds = obj_mesh.bounds
    offset = np.array([0.0, 0.0, -bounds[0, 2]])
    X_WO = np.eye(4); X_WO[:3, 3] = offset

    verts_O = np.asarray(obj_mesh.vertices, dtype=np.float32)
    faces = np.asarray(obj_mesh.faces, dtype=np.int32)
    mesh_o3d = o3d.t.geometry.TriangleMesh()
    mesh_o3d.vertex.positions = o3d.core.Tensor(verts_O)
    mesh_o3d.triangle.indices = o3d.core.Tensor(faces)
    obj_scene = o3d.t.geometry.RaycastingScene()
    scene_id = obj_scene.add_triangles(mesh_o3d)

    urdf_path = os.path.join(HAND_DIR, "leap.urdf")
    with open(urdf_path) as f:
        chain = pk.build_chain_from_urdf(f.read())
    vis_meshes = _visual_meshes("rh", "leap")

    link_mesh_files = {}
    for ln, ml in vis_meshes.items():
        for mf, vp in ml:
            fp = os.path.join(HAND_DIR, mf)
            if os.path.exists(fp):
                link_mesh_files[ln] = (fp, vp)
                break

    grasp_files = [
        ("warmstart_single", "compare_warmstart_single.pt"),
        ("warmstart_best", "compare_warmstart_best.pt"),
        ("batched_curated", "compare_batched_curated.pt"),
    ]

    for grasp_name, filename in grasp_files:
        path = os.path.join(GRASPS, filename)
        if not os.path.exists(path):
            continue
        data = torch.load(path, weights_only=False, map_location="cpu")
        g = data[0]

        print(f"\n  --- {grasp_name} ---")

        q = torch.tensor(g["q_joints"], dtype=torch.float32).unsqueeze(0)
        fk = chain.forward_kinematics(q)
        T_base = np.eye(4)
        T_base[:3, :3] = np.array(g["base_rot"])
        T_base[:3, 3] = np.array(g["base_pos"])

        R_OW = X_WO[:3, :3].T
        t_OW = -R_OW @ X_WO[:3, 3]

        palm_inside_total = 0
        palm_verts_total = 0
        finger_inside_total = 0
        finger_verts_total = 0

        for link_name in vis_meshes:
            if link_name not in fk or link_name not in link_mesh_files:
                continue
            mesh_path, vis_pose = link_mesh_files[link_name]
            lm = trimesh.load(mesh_path, force="mesh")
            verts = np.asarray(lm.vertices, dtype=np.float64)

            link_T = fk[link_name].get_matrix()[0].numpy().astype(np.float64)
            world_T = T_base @ link_T
            if vis_pose is not None:
                vp_arr = np.array(vis_pose, dtype=np.float64)
                Rv = ScipyR.from_euler("xyz", vp_arr[3:]).as_matrix()
                Tv = np.eye(4); Tv[:3,:3] = Rv; Tv[:3,3] = vp_arr[:3]
                world_T = world_T @ Tv

            verts_w = (world_T[:3,:3] @ verts.T).T + world_T[:3,3]
            verts_obj = (R_OW.astype(np.float64) @ verts_w.T).T + t_OW

            sdfs = obj_scene.compute_signed_distance(
                o3d.core.Tensor(verts_obj.astype(np.float32), dtype=o3d.core.float32)
            ).numpy()

            n_in = (sdfs < 0).sum()
            n_deep = (sdfs < -0.001).sum()
            min_sdf = sdfs.min()

            is_palm = "palm" in link_name
            if is_palm:
                palm_inside_total += n_in
                palm_verts_total += len(sdfs)
            else:
                finger_inside_total += n_in
                finger_verts_total += len(sdfs)

            short = link_name.replace("leap_rh_", "")
            if n_in > 0 or n_deep > 0:
                print(f"    {short:<12} {n_in:>4}/{len(sdfs):>4} in ({n_in/len(sdfs)*100:5.1f}%) "
                      f"deep(>1mm)={n_deep:>3} worst={min_sdf*1000:.1f}mm")

        print(f"\n  Summary:")
        if palm_verts_total > 0:
            print(f"    Palm: {palm_inside_total}/{palm_verts_total} ({palm_inside_total/palm_verts_total*100:.1f}%)")
        if finger_verts_total > 0:
            print(f"    Fingers: {finger_inside_total}/{finger_verts_total} ({finger_inside_total/finger_verts_total*100:.1f}%)")

# ============================================================================
# TEST 7: BatchedSDF grid accuracy
# ============================================================================
def test_batchedsdf_accuracy(obj_mesh):
    section("TEST 7: BatchedSDF Grid Accuracy")

    bounds = obj_mesh.bounds
    offset = np.array([0.0, 0.0, -bounds[0, 2]])
    X_WO = np.eye(4); X_WO[:3, 3] = offset

    sdf = BatchedSDF(obj_mesh, X_WO, resolution=128, device="cuda")

    # Direct O3D
    verts_O = np.asarray(obj_mesh.vertices, dtype=np.float32)
    faces = np.asarray(obj_mesh.faces, dtype=np.int32)
    mesh_o3d = o3d.t.geometry.TriangleMesh()
    mesh_o3d.vertex.positions = o3d.core.Tensor(verts_O)
    mesh_o3d.triangle.indices = o3d.core.Tensor(faces)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_o3d)

    verts_W = (X_WO[:3,:3] @ verts_O.astype(np.float64).T).T + X_WO[:3,3]
    center_w = verts_W.mean(axis=0)

    np.random.seed(42)
    test_pts = (center_w + np.random.randn(200, 3) * 0.05).astype(np.float64)

    # Grid query
    pts_torch = torch.tensor(test_pts, dtype=torch.float32, device="cuda").unsqueeze(0)
    sdf_grid = sdf.query(pts_torch).cpu().numpy()[0]

    # O3D query in object frame
    R_OW = X_WO[:3,:3].T
    t_OW = -R_OW @ X_WO[:3,3]
    pts_obj = (R_OW @ test_pts.T).T + t_OW
    sdf_o3d = scene.compute_signed_distance(
        o3d.core.Tensor(pts_obj.astype(np.float32), dtype=o3d.core.float32)
    ).numpy()

    errors = np.abs(sdf_grid - sdf_o3d)
    sign_match = (np.sign(sdf_grid) == np.sign(sdf_o3d)).sum()
    print(f"  200 random points near object:")
    print(f"    Max error:  {errors.max()*1000:.3f}mm")
    print(f"    Mean error: {errors.mean()*1000:.3f}mm")
    print(f"    Sign match: {sign_match}/200")

    # Focus on points near surface (where sign errors matter most)
    near_surf = np.abs(sdf_o3d) < 0.005  # within 5mm of surface
    if near_surf.sum() > 0:
        near_errors = errors[near_surf]
        near_sign = (np.sign(sdf_grid[near_surf]) == np.sign(sdf_o3d[near_surf])).sum()
        print(f"\n    Near-surface ({near_surf.sum()} pts within 5mm):")
        print(f"      Max error:  {near_errors.max()*1000:.3f}mm")
        print(f"      Mean error: {near_errors.mean()*1000:.3f}mm")
        print(f"      Sign match: {near_sign}/{near_surf.sum()}")

# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("SDF Fundamentals Test Suite")
    print("="*70)

    obj_mesh = test_object_mesh()
    test_object_sdf_sanity(obj_mesh)
    test_known_geometry()
    test_hand_meshes()
    test_palm_mesh()
    test_grasp_comparison(obj_mesh)
    test_batchedsdf_accuracy(obj_mesh)

    print(f"\n{'='*70}")
    print("ALL TESTS COMPLETE")
    print("="*70)
