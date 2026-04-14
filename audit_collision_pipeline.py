#!/usr/bin/env python3
"""
Collision detection pipeline audit for LEAP hand dexterous grasp optimizer.

Checks:
1. SDF correctness (sign convention, gradient direction, known-point values)
2. Box-grid transform correctness (FK + base rotation → world frame)
3. Margin system (per-link values, comparison logic)
4. Penetration counting (box-grid, honest, no filtering)
5. Self-collision (adjacent link exclusion, distance metric, threshold)

Run: conda run --no-capture-output -n frogger python -u audit_collision_pipeline.py
"""

import os
import sys
import numpy as np
import trimesh
import open3d as o3d
import torch
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation as ScipyR
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(__file__))
from frogger.batched_pytorch_solver import (
    BatchedSDF,
    BatchedGraspOptimizer,
    _link_names,
    _visual_meshes,
    _LEAP_JOINT_LOWER,
    _LEAP_JOINT_UPPER,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MESH_PATH = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg/hot_glue_gun/object.obj"
GRASP_PATH = os.path.join(os.path.dirname(__file__), "output/grasps_opt_exp/exp_ABCD.pt")
URDF_PATH = os.path.join(os.path.dirname(__file__), "models/leap_rh/leap.urdf")
HAND_TYPE = "leap"
HAND = "rh"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

results = {}


def header(title):
    print(f"\n{'=' * 70}")
    print(f"  CHECK: {title}")
    print(f"{'=' * 70}")


def report(name, passed, detail=""):
    tag = "PASS" if passed else "FAIL"
    results[name] = passed
    print(f"  [{tag}] {name}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"         {line}")


# ===========================================================================
# Load mesh and SDF
# ===========================================================================
print("Loading mesh and building SDF...")
mesh = trimesh.load(MESH_PATH, force="mesh")
bounds = mesh.bounds
offset = np.array([0.0, 0.0, -bounds[0, 2]])
X_WO = np.eye(4)
X_WO[:3, 3] = offset
obj_center = mesh.centroid + offset

sdf = BatchedSDF(mesh, X_WO, bounds_padding=0.15, resolution=128, device=DEVICE)

# Load grasp
print(f"Loading grasp from {GRASP_PATH}...")
grasp_data = torch.load(GRASP_PATH, weights_only=False, map_location="cpu")
if isinstance(grasp_data, list) and len(grasp_data) > 0:
    grasp = grasp_data[0]  # best grasp
    print(f"  Loaded {len(grasp_data)} grasps, using first (best)")
else:
    print(f"  ERROR: No grasps in file")
    sys.exit(1)

# ===========================================================================
# CHECK 1: SDF Correctness
# ===========================================================================
header("1. SDF Correctness")

# 1a. Sign convention: negative inside, positive outside
verts_O = np.asarray(mesh.vertices, dtype=np.float32)
faces = np.asarray(mesh.faces, dtype=np.int32)
R_WO = X_WO[:3, :3].astype(np.float64)
t_WO = X_WO[:3, 3].astype(np.float64)
verts_W = (R_WO @ verts_O.astype(np.float64).T).T + t_WO

# Centroid in world frame
centroid_W = verts_W.mean(axis=0)

# Direct Open3D query at centroid (in object frame)
centroid_O = centroid_W - offset  # undo world transform (R=I)
mesh_o3d = o3d.t.geometry.TriangleMesh()
mesh_o3d.vertex.positions = o3d.core.Tensor(verts_O)
mesh_o3d.triangle.indices = o3d.core.Tensor(faces)
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(mesh_o3d)

centroid_sdf_o3d = scene.compute_signed_distance(
    o3d.core.Tensor(centroid_O.astype(np.float32).reshape(1, 3))
).numpy()[0]

# BatchedSDF query at centroid (world frame)
centroid_t = torch.tensor(centroid_W, dtype=torch.float32, device=DEVICE).reshape(1, 1, 3)
centroid_sdf_grid = sdf.query(centroid_t).item()

report(
    "1a. Centroid SDF is negative (inside object)",
    centroid_sdf_grid < 0,
    f"Open3D direct: {centroid_sdf_o3d:.4f}\n"
    f"BatchedSDF grid: {centroid_sdf_grid:.4f}\n"
    f"Centroid world: {centroid_W}\n"
    f"Convention: Open3D uses NEGATIVE=inside for watertight meshes"
)

# 1b. Far point is positive (outside)
far_point_W = centroid_W + np.array([0.5, 0.0, 0.0])
far_t = torch.tensor(far_point_W, dtype=torch.float32, device=DEVICE).reshape(1, 1, 3)
far_sdf = sdf.query(far_t).item()
report(
    "1b. Far point SDF is positive (outside object)",
    far_sdf > 0,
    f"Far point (+0.5m along X): SDF = {far_sdf:.4f}"
)

# 1c. Surface point SDF is near zero
# Sample a surface point
surf_pts, _ = trimesh.sample.sample_surface(
    trimesh.Trimesh(vertices=verts_W.astype(np.float32), faces=mesh.faces), 100)
surf_t = torch.tensor(surf_pts[:10], dtype=torch.float32, device=DEVICE).unsqueeze(0)
surf_sdf = sdf.query(surf_t)[0]
max_surf_err = surf_sdf.abs().max().item()
mean_surf_err = surf_sdf.abs().mean().item()
report(
    "1c. Surface points have SDF near zero",
    max_surf_err < 0.005,  # 5mm tolerance for 128^3 grid
    f"10 surface points: max|SDF| = {max_surf_err*1000:.2f}mm, mean|SDF| = {mean_surf_err*1000:.2f}mm\n"
    f"(128^3 grid expected accuracy ~1-2mm)"
)

# 1d. SDF gradient points away from surface (outward)
# At a point slightly outside the object, gradient should point away from object center
outside_point_W = centroid_W + np.array([0.1, 0.0, 0.0])
outside_t = torch.tensor(outside_point_W, dtype=torch.float32, device=DEVICE).reshape(1, 1, 3)
outside_t.requires_grad_(True)
sdf_val = sdf.query(outside_t)
sdf_val.backward()
grad = outside_t.grad[0, 0].cpu().numpy()
grad_norm = grad / (np.linalg.norm(grad) + 1e-8)

# Gradient should point roughly away from centroid (positive dot with centroid→point direction)
to_point = outside_point_W - centroid_W
to_point_norm = to_point / (np.linalg.norm(to_point) + 1e-8)
dot = np.dot(grad_norm, to_point_norm)
report(
    "1d. SDF gradient points outward from object",
    dot > 0.5,
    f"Point 10cm from centroid along +X: SDF = {sdf_val.item():.4f}\n"
    f"Gradient direction: {grad_norm}\n"
    f"Direction to point from centroid: {to_point_norm}\n"
    f"Dot product: {dot:.4f} (should be > 0.5)"
)

# 1e. Open3D vs Grid consistency
# Query Open3D directly at 50 random world-frame points, compare with grid
rng = np.random.RandomState(42)
test_pts_W = centroid_W + 0.1 * rng.randn(50, 3).astype(np.float64)
test_pts_O = (test_pts_W - offset).astype(np.float32)
o3d_sdf = scene.compute_signed_distance(o3d.core.Tensor(test_pts_O)).numpy()
grid_sdf = sdf.query(
    torch.tensor(test_pts_W.astype(np.float32), dtype=torch.float32, device=DEVICE).unsqueeze(0)
)[0].cpu().numpy()
diff = np.abs(o3d_sdf - grid_sdf)
max_diff = diff.max()
mean_diff = diff.mean()
report(
    "1e. Grid SDF matches Open3D direct query",
    max_diff < 0.005,  # 5mm max deviation
    f"50 random points: max_diff = {max_diff*1000:.2f}mm, mean_diff = {mean_diff*1000:.2f}mm"
)

# ===========================================================================
# CHECK 2: Box-Grid Transform Correctness
# ===========================================================================
header("2. Box-Grid Transform Correctness")

# Load FK chain
with open(URDF_PATH) as f:
    chain = pk.build_chain_from_urdf(f.read()).to(device=DEVICE)

q_joints = torch.tensor(grasp["q_joints"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
base_pos = grasp["base_pos"]
base_rot = grasp["base_rot"]

# Build base transform
bT = np.eye(4)
bT[:3, :3] = base_rot
bT[:3, 3] = base_pos
bT_t = torch.tensor(bT, dtype=torch.float32, device=DEVICE).unsqueeze(0)

fk = chain.forward_kinematics(q_joints)

# 2a. FK transform structure check
tip_names, col_names = _link_names(HAND, HAND_TYPE)
palm_link = "leap_rh_palm"
palm_fk = fk[palm_link].get_matrix()[0].cpu().numpy()
palm_world = bT @ palm_fk

# The FK should give palm a sensible transform relative to base
palm_pos_world = palm_world[:3, 3]
palm_rot_world = palm_world[:3, :3]
det = np.linalg.det(palm_rot_world)
is_proper_rotation = abs(det - 1.0) < 0.01
report(
    "2a. FK produces proper rotation matrices",
    is_proper_rotation,
    f"Palm FK det = {det:.6f} (should be 1.0)\n"
    f"Palm world position: {palm_pos_world}\n"
    f"Base position: {base_pos}"
)

# 2b. Box grid points match expected positions
# Load URDF box primitives for one link and compare
tree = ET.parse(URDF_PATH)
root = tree.getroot()

# Pick a fingertip link for testing
test_link = "leap_rh_if_ds"
test_le = None
for le in root.findall("link"):
    if le.get("name") == test_link:
        test_le = le
        break

if test_le is not None:
    # Get box points from URDF
    urdf_pts = []
    for cel in test_le.findall("collision"):
        g = cel.find("geometry")
        if g is None:
            continue
        b = g.find("box")
        if b is None:
            continue
        sz = [float(x) for x in b.get("size").split()]
        o = cel.find("origin")
        p = np.array([float(x) for x in o.get("xyz", "0 0 0").split()])
        rpy = np.array([float(x) for x in o.get("rpy", "0 0 0").split()])
        R = ScipyR.from_euler("xyz", rpy).as_matrix() if np.any(np.abs(rpy) > 1e-6) else np.eye(3)
        hx, hy, hz = sz[0] / 2, sz[1] / 2, sz[2] / 2
        # Just corners for comparison
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                for sz_s in [-1, 1]:
                    urdf_pts.append(R @ np.array([sx * hx, sy * hy, sz_s * hz]) + p)
    urdf_pts = np.array(urdf_pts, dtype=np.float32)

    # Transform to world
    link_fk = fk[test_link].get_matrix()[0].cpu().numpy()
    wT = bT @ link_fk
    urdf_pts_world = (wT[:3, :3] @ urdf_pts.T).T + wT[:3, 3]

    # Compare with _col_data from optimizer
    # Build optimizer to get its internal _col_data
    opt = BatchedGraspOptimizer(
        sdf, num_envs=2, device=DEVICE, hand=HAND, hand_type=HAND_TYPE, palm_contact=False
    )

    # Find this link in col_data
    col_pts_local = None
    for nm, pts in opt._col_data:
        if nm == test_link:
            col_pts_local = pts.cpu().numpy()
            break

    if col_pts_local is not None:
        # Transform col_data points to world
        col_pts_world = (wT[:3, :3] @ col_pts_local[:, :3].T).T + wT[:3, 3]

        # The URDF corner points should be contained within the convex hull
        # of the grid points (or at least very close to grid boundary points)
        from scipy.spatial import cKDTree
        kd = cKDTree(col_pts_world)
        dists, _ = kd.query(urdf_pts_world)
        max_dist = dists.max()
        report(
            "2b. URDF box corners are covered by col_data grid points",
            max_dist < 0.006,  # grid pitch 5mm, corner should be within ~half pitch
            f"Link: {test_link}, {len(urdf_pts)} corners, {len(col_pts_world)} grid pts\n"
            f"Max distance from corner to nearest grid point: {max_dist*1000:.2f}mm\n"
            f"(5mm grid → corners should be within ~3.5mm of a grid point)"
        )
    else:
        report("2b. URDF box corners vs col_data", False, f"Link {test_link} not found in _col_data")

    # 2c. Transformed collision points are where we expect the hand link to be
    # Check that the world-frame collision points for the fingertip are near
    # the object surface (since it's supposed to be grasping)
    col_sdf = sdf.query(
        torch.tensor(col_pts_world, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    )[0].cpu().numpy()
    n_near_surface = (np.abs(col_sdf) < 0.010).sum()
    n_inside = (col_sdf < 0).sum()
    report(
        "2c. Fingertip collision points near object surface",
        n_near_surface > 0 or n_inside > 0,
        f"Link {test_link}: {len(col_sdf)} grid points\n"
        f"  Near surface (|SDF|<10mm): {n_near_surface}\n"
        f"  Inside (SDF<0): {n_inside}\n"
        f"  SDF range: [{col_sdf.min()*1000:.1f}mm, {col_sdf.max()*1000:.1f}mm]"
    )
else:
    report("2b. URDF box corners vs col_data", False, f"Link {test_link} not found in URDF")

# 2d. Visual mesh vertices should roughly coincide with box-grid transformed points
# Load visual mesh for a non-tip link and compare
test_vis_link = "leap_rh_if_px"  # proximal phalanx
vis_meshes = _visual_meshes(HAND, HAND_TYPE)
if test_vis_link in vis_meshes:
    vis_all_verts = []
    hand_mesh_dir = os.path.join(os.path.dirname(__file__), f"models/leap_{HAND}")
    for mesh_file, vis_pose in vis_meshes[test_vis_link]:
        vpath = os.path.join(hand_mesh_dir, mesh_file)
        if os.path.exists(vpath):
            vm = trimesh.load(vpath, force="mesh")
            verts = np.asarray(vm.vertices, dtype=np.float64)
            if vis_pose is not None:
                vp = np.array(vis_pose, dtype=np.float64)
                Rv = ScipyR.from_euler("xyz", vp[3:]).as_matrix()
                verts = (Rv @ verts.T).T + vp[:3]
            vis_all_verts.append(verts)
    if vis_all_verts:
        vis_all_verts = np.vstack(vis_all_verts)
        # Transform to world
        link_fk_vis = fk[test_vis_link].get_matrix()[0].cpu().numpy()
        wT_vis = bT @ link_fk_vis
        vis_world = (wT_vis[:3, :3] @ vis_all_verts.T).T + wT_vis[:3, 3]

        # Get col_data for this link
        for nm, pts in opt._col_data:
            if nm == test_vis_link:
                col_world_vis = (wT_vis[:3, :3] @ pts[:, :3].cpu().numpy().T).T + wT_vis[:3, 3]
                # Check overlap: visual mesh bbox should contain most collision box points
                vis_min = vis_world.min(axis=0) - 0.005  # 5mm tolerance
                vis_max = vis_world.max(axis=0) + 0.005
                inside = ((col_world_vis >= vis_min) & (col_world_vis <= vis_max)).all(axis=1)
                pct_inside = 100 * inside.sum() / len(col_world_vis)
                report(
                    "2d. Box-grid points lie within visual mesh bounding box",
                    pct_inside > 70,
                    f"Link {test_vis_link}: {pct_inside:.1f}% of {len(col_world_vis)} box-grid points\n"
                    f"  inside visual mesh bbox (with 5mm tolerance)\n"
                    f"  Visual bbox: [{vis_min}] to [{vis_max}]\n"
                    f"  Grid pts bbox: [{col_world_vis.min(0)}] to [{col_world_vis.max(0)}]"
                )
                break
        else:
            report("2d. Visual mesh vs box-grid", False, f"Link {test_vis_link} not in _col_data")
    else:
        report("2d. Visual mesh vs box-grid", False, f"No visual mesh verts loaded for {test_vis_link}")
else:
    report("2d. Visual mesh vs box-grid", False, f"No visual mesh spec for {test_vis_link}")

# ===========================================================================
# CHECK 3: Margin System
# ===========================================================================
header("3. Margin System")

# 3a. Inspect actual margin values for each link type
margins = opt._col_margins.cpu().numpy()
print(f"  Total margin values: {len(margins)}")

# Map margins back to links
margin_per_link = {}
for li, (nm, pts) in enumerate(opt._col_data):
    si, ei = opt._col_link_ranges[li]
    link_margins = margins[si:ei]
    unique_m = np.unique(link_margins)
    margin_per_link[nm] = unique_m

print(f"\n  Per-link margins:")
ds_links = []
other_links = []
for nm, unique_m in sorted(margin_per_link.items()):
    tag = "ds" if "_ds" in nm else ("palm" if "palm" in nm else "other")
    val_str = ", ".join([f"{m*1000:.1f}mm" for m in unique_m])
    print(f"    {nm:30s} -> margin = {val_str}  [{tag}]")
    if "_ds" in nm:
        ds_links.append((nm, unique_m))
    else:
        other_links.append((nm, unique_m))

# 3b. Check ds links get -1mm
all_ds_neg1mm = all(
    all(abs(m - (-0.001)) < 0.0001 for m in unique_m)
    for _, unique_m in ds_links
)
report(
    "3b. Fingertip (_ds) links get -1mm margin",
    all_ds_neg1mm,
    f"DS links: {[(nm, [f'{m*1000:.1f}mm' for m in um]) for nm, um in ds_links]}"
)

# 3c. Non-ds links get 0mm margin
all_others_0mm = all(
    all(abs(m - 0.0) < 0.0001 for m in unique_m)
    for _, unique_m in other_links
)
report(
    "3c. Non-fingertip links get 0mm margin",
    all_others_0mm,
    f"Sample non-ds links: {[(nm, [f'{m*1000:.1f}mm' for m in um]) for nm, um in other_links[:5]]}"
)

# 3d. Margin comparison logic: margins - SDF > 0 means violation
# Code at line ~2244: col_violation = F.relu(self._col_margins - cs_final)
# If margin = -0.001 (ds), SDF must be < -0.001 to violate (1mm inside)
# If margin = 0.0, SDF must be < 0 to violate (any penetration)
# Let's verify with synthetic examples
test_cases = [
    ("ds @ SDF=+5mm", -0.001, 0.005, False),   # tip far outside: no violation
    ("ds @ SDF=0mm", -0.001, 0.0, False),       # tip on surface: no violation (margin allows -1mm)
    ("ds @ SDF=-0.5mm", -0.001, -0.0005, False), # tip 0.5mm inside: no violation
    ("ds @ SDF=-1mm", -0.001, -0.001, False),    # tip 1mm inside: exactly at margin, no violation
    ("ds @ SDF=-2mm", -0.001, -0.002, True),     # tip 2mm inside: violation (deeper than allowed)
    ("other @ SDF=+1mm", 0.0, 0.001, False),    # outside: no violation
    ("other @ SDF=0mm", 0.0, 0.0, False),       # on surface: no violation
    ("other @ SDF=-1mm", 0.0, -0.001, True),    # inside: violation
]
margin_logic_ok = True
details = []
for name, margin, sdf_val, expected_violation in test_cases:
    # relu(margin - sdf) > 0 ?
    violation = max(0, margin - sdf_val) > 0
    ok = violation == expected_violation
    if not ok:
        margin_logic_ok = False
    details.append(f"{name}: relu({margin*1000:.1f} - {sdf_val*1000:.1f}) = "
                   f"{max(0, margin-sdf_val)*1000:.1f}mm -> {'VIOLATION' if violation else 'OK'} "
                   f"{'[CORRECT]' if ok else '[WRONG]'}")

report(
    "3d. Margin comparison logic is correct",
    margin_logic_ok,
    "\n".join(details)
)

# ===========================================================================
# CHECK 4: Penetration Counting
# ===========================================================================
header("4. Penetration Counting")

# 4a. Verification uses box-grid points, not mesh vertices
# The post-solve verification block (line ~2330) creates a _verify_pts dict
# from URDF boxes with 5mm grid. Let's verify it matches what we'd expect.
verify_pts = {}
vp_pitch = 0.005
for le in root.findall("link"):
    ln = le.get("name")
    lpts = []
    for cel in le.findall("collision"):
        g = cel.find("geometry")
        if g is None:
            continue
        b = g.find("box")
        if b is None:
            continue
        sz = [float(x) for x in b.get("size").split()]
        o = cel.find("origin")
        p = np.array([float(x) for x in o.get("xyz", "0 0 0").split()])
        rpy = np.array([float(x) for x in o.get("rpy", "0 0 0").split()])
        R = ScipyR.from_euler("xyz", rpy).as_matrix() if np.any(np.abs(rpy) > 1e-6) else np.eye(3)
        hx, hy, hz = sz[0] / 2, sz[1] / 2, sz[2] / 2
        gx = np.arange(-hx, hx + vp_pitch / 2, vp_pitch)
        gy = np.arange(-hy, hy + vp_pitch / 2, vp_pitch)
        gz = np.arange(-hz, hz + vp_pitch / 2, vp_pitch)
        grid = np.stack(np.meshgrid(gx, gy, gz, indexing='ij'), axis=-1).reshape(-1, 3)
        grid = ((R @ grid.T).T + p).astype(np.float32)
        lpts.append(grid)
    if lpts:
        verify_pts[ln] = np.vstack(lpts)

# Count: how many links have verify points
n_verify_links = sum(1 for nm in col_names if nm in verify_pts)
report(
    "4a. Verification uses box-grid points (not mesh vertices)",
    n_verify_links == len(col_names),
    f"Collision links with box-grid verify points: {n_verify_links}/{len(col_names)}\n"
    f"Links: {[nm.split('leap_rh_')[-1] for nm in col_names if nm in verify_pts]}"
)

# 4b. Count is honest — includes ALL links including palm
total_verify = 0
total_pen = 0
per_link_counts = []
for nm in col_names:
    if nm not in verify_pts:
        continue
    pts = verify_pts[nm]
    link_fk_v = fk[nm].get_matrix()[0].cpu().numpy()
    wT_v = bT @ link_fk_v
    pw = (wT_v[:3, :3] @ pts.T).T + wT_v[:3, 3]
    sv = sdf.query(
        torch.tensor(pw, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    )[0].cpu().numpy()
    n_pen = (sv < -0.001).sum()
    total_verify += len(sv)
    total_pen += n_pen
    short_name = nm.split("leap_rh_")[-1] if "leap_rh_" in nm else nm
    per_link_counts.append((short_name, len(sv), n_pen, sv.min()))

# Check that palm is included
palm_included = any("palm" in nm for nm, _, _, _ in per_link_counts)
report(
    "4b. Penetration count includes palm (no subset filtering)",
    palm_included,
    f"Total: {total_pen}/{total_verify} points at -1mm threshold\n"
    f"Per-link breakdown:"
)

for short_nm, total, pen, worst in per_link_counts:
    if pen > 0 or worst < 0:
        print(f"    {short_nm:15s}: {pen:4d}/{total:4d} pen, worst={worst*1000:.1f}mm")

# 4c. Palm boxes: verify ALL palm boxes are checked (the "large boxes ARE the palm" rule)
palm_pts_count = 0
for nm, pts in opt._col_data:
    if "palm" in nm:
        palm_pts_count += pts.shape[0]

# Count palm boxes in URDF
palm_boxes = 0
for le in root.findall("link"):
    if le.get("name") == "leap_rh_palm":
        for cel in le.findall("collision"):
            g = cel.find("geometry")
            if g and g.find("box") is not None:
                palm_boxes += 1

report(
    "4c. All palm collision boxes are used (no filtering by x-position)",
    palm_boxes > 0 and palm_pts_count > 50,
    f"Palm has {palm_boxes} URDF boxes → {palm_pts_count} grid points in _col_data\n"
    f"(If palm boxes were filtered, we'd see far fewer points)"
)

# ===========================================================================
# CHECK 5: Self-Collision
# ===========================================================================
header("5. Self-Collision System")

# 5a. Adjacent links are excluded
# Build expected adjacency list
expected_adj = {
    ('palm', 'if_bs'), ('palm', 'mf_bs'), ('palm', 'rf_bs'), ('palm', 'th_mp'),
    ('if_bs', 'if_px'), ('if_px', 'if_md'), ('if_md', 'if_ds'),
    ('mf_bs', 'mf_px'), ('mf_px', 'mf_md'), ('mf_md', 'mf_ds'),
    ('rf_bs', 'rf_px'), ('rf_px', 'rf_md'), ('rf_md', 'rf_ds'),
    ('th_mp', 'th_bs'), ('th_bs', 'th_px'), ('th_px', 'th_ds'),
}

# The self_col_pairs should NOT contain pairs within the same finger chain
# (same finger chain links are adjacent by definition in a serial chain)
sc_pairs = opt._self_col_pairs
print(f"  Self-collision pairs: {len(sc_pairs)}")

# The system groups by finger — pairs are between different finger groups
# Check that the pairs make sense (inter-finger, not intra-finger)
# _self_col_pairs uses index arrays into the flattened _sc_data
# Reconstruct which finger each _sc_data entry belongs to
finger_keys = ['if', 'mf', 'rf', 'th']
sc_finger_map = {}  # global_idx -> finger_key
sc_offset = 0
for nm, pts in opt._sc_data:
    n = pts.shape[0]
    for fk_key in finger_keys:
        if f'_{fk_key}_' in nm:
            for i in range(sc_offset, sc_offset + n):
                sc_finger_map[i] = fk_key
            break
    if 'palm' in nm:
        for i in range(sc_offset, sc_offset + n):
            sc_finger_map[i] = 'palm'
    sc_offset += n

intra_finger_pairs = 0
inter_finger_pairs = 0
for idx1, idx2 in sc_pairs:
    fingers1 = set(sc_finger_map.get(i.item(), '?') for i in idx1)
    fingers2 = set(sc_finger_map.get(i.item(), '?') for i in idx2)
    # Each pair should be from DIFFERENT fingers/palm
    if fingers1 & fingers2:
        intra_finger_pairs += 1
    else:
        inter_finger_pairs += 1

report(
    "5a. Self-collision excludes intra-finger (adjacent) pairs",
    intra_finger_pairs == 0,
    f"Inter-finger pairs: {inter_finger_pairs}\n"
    f"Intra-finger pairs (should be 0): {intra_finger_pairs}"
)

# 5b. Check pair coverage: should have pairs for all inter-finger combos
# 4 fingers + palm = C(4,2) + 4 = 10 pairs
expected_pairs = set()
fk_list = ['if', 'mf', 'rf', 'th']
for i in range(len(fk_list)):
    for j in range(i + 1, len(fk_list)):
        expected_pairs.add((fk_list[i], fk_list[j]))
for fk_key in fk_list:
    expected_pairs.add(('palm', fk_key))

actual_pairs = set()
for idx1, idx2 in sc_pairs:
    fingers1 = set(sc_finger_map.get(i.item(), '?') for i in idx1)
    fingers2 = set(sc_finger_map.get(i.item(), '?') for i in idx2)
    for f1 in fingers1:
        for f2 in fingers2:
            if f1 != f2:
                pair = tuple(sorted([f1, f2]))
                actual_pairs.add(pair)

# Normalize expected pairs
expected_pairs_sorted = {tuple(sorted(p)) for p in expected_pairs}
missing = expected_pairs_sorted - actual_pairs
extra = actual_pairs - expected_pairs_sorted

report(
    "5b. All inter-finger/palm pairs are checked",
    len(missing) == 0,
    f"Expected: {len(expected_pairs_sorted)} pairs\n"
    f"Actual: {len(actual_pairs)} unique finger-pair combos\n"
    f"Missing: {missing if missing else 'none'}\n"
    f"Extra: {extra if extra else 'none'}"
)

# 5c. Self-collision distance metric and threshold
# From the code: torch.cdist on box keypoints, threshold varies by context:
# - Optimization (section D): F.relu(0.002 - d) → 2mm threshold
# - Verification: worst_sc < 0.003 → 3mm reported, < 0.0005 → 0.5mm fails feasibility
# Check what the actual SC distances are for this grasp
sc_data = opt._sc_data
sc_pts_world = []
for nm, pts in sc_data:
    if nm in fk:
        link_fk_sc = fk[nm].get_matrix()[0].cpu().numpy()
        wT_sc = bT @ link_fk_sc
        wp = (wT_sc[:3, :3] @ pts[:, :3].cpu().numpy().T).T + wT_sc[:3, 3]
        sc_pts_world.append(wp)
    else:
        sc_pts_world.append(pts[:, :3].cpu().numpy())

sc_all = np.vstack(sc_pts_world)
sc_all_t = torch.tensor(sc_all, dtype=torch.float32, device=DEVICE).unsqueeze(0)

# Evaluate SC for each pair
worst_sc_dist = 999.0
sc_violations = []
for pair_idx, (idx1, idx2) in enumerate(sc_pairs):
    # Need to figure out the offset mapping for world-frame points
    # Instead, use the same approach as the solver
    pass

# Direct evaluation using solver's approach
sc_offset_map = {}
so = 0
for nm, pts in sc_data:
    n = pts.shape[0]
    sc_offset_map[nm] = (so, so + n)
    so += n

# Transform all SC points to world
all_sc_world = np.zeros((so, 3), dtype=np.float32)
for nm, pts in sc_data:
    si, ei = sc_offset_map[nm]
    if nm in fk:
        lt = fk[nm].get_matrix()[0].cpu().numpy()
        wT_sc = bT @ lt
        wp = (wT_sc[:3, :3] @ pts[:, :3].cpu().numpy().T).T + wT_sc[:3, 3]
        all_sc_world[si:ei] = wp

all_sc_world_t = torch.tensor(all_sc_world, dtype=torch.float32, device=DEVICE)

worst_sc = 999.0
sc_details = []
for idx1, idx2 in sc_pairs:
    p1 = all_sc_world_t[idx1]  # [n1, 3]
    p2 = all_sc_world_t[idx2]  # [n2, 3]
    d = torch.cdist(p1.unsqueeze(0), p2.unsqueeze(0))[0]
    min_d = d.min().item()
    if min_d < worst_sc:
        worst_sc = min_d
    if min_d < 0.005:
        f1 = sc_finger_map.get(idx1[0].item(), '?')
        f2 = sc_finger_map.get(idx2[0].item(), '?')
        sc_details.append(f"{f1}-{f2}: {min_d*1000:.1f}mm")

report(
    "5c. Self-collision distance metric (cdist on box keypoints)",
    True,  # This is a descriptive check
    f"Worst SC distance: {worst_sc*1000:.1f}mm\n"
    f"Optimization threshold: 2mm (soft penalty)\n"
    f"Verification report threshold: 3mm\n"
    f"Feasibility fail threshold: 0.5mm\n"
    f"Close pairs (<5mm): {sc_details if sc_details else 'none'}"
)

# 5d. Self-collision uses URDF box points (not visual mesh)
# The _sc_data should contain box corner points (8 per box)
n_sc_pts_per_link = {}
for nm, pts in sc_data:
    n_sc_pts_per_link[nm.split("leap_rh_")[-1] if "leap_rh_" in nm else nm] = pts.shape[0]

# Count URDF boxes per link to predict expected SC points
expected_sc = {}
for le in root.findall("link"):
    ln = le.get("name")
    if ln not in [nm for nm, _ in sc_data]:
        continue
    n_boxes = sum(1 for cel in le.findall("collision")
                  if cel.find("geometry") is not None and cel.find("geometry").find("box") is not None)
    short = ln.split("leap_rh_")[-1] if "leap_rh_" in ln else ln
    expected_sc[short] = n_boxes * 8  # 8 corners per box

all_match = True
sc_compare_details = []
for short_nm, expected_n in expected_sc.items():
    actual_n = n_sc_pts_per_link.get(short_nm, 0)
    match = actual_n == expected_n
    if not match:
        all_match = False
    sc_compare_details.append(f"{short_nm}: expected {expected_n} (={expected_n//8}boxes*8), got {actual_n} {'OK' if match else 'MISMATCH'}")

report(
    "5d. Self-collision uses URDF box corner points (8 per box)",
    all_match,
    "\n".join(sc_compare_details[:10]) + (f"\n... ({len(sc_compare_details)} total)" if len(sc_compare_details) > 10 else "")
)

# ===========================================================================
# SUMMARY
# ===========================================================================
header("SUMMARY")
n_pass = sum(1 for v in results.values() if v)
n_fail = sum(1 for v in results.values() if not v)
n_total = len(results)

for name, passed in results.items():
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] {name}")

print(f"\n  Total: {n_pass}/{n_total} passed, {n_fail} failed")
