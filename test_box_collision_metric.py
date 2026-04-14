#!/usr/bin/env python3
"""
Test the correct collision metric: uniform grid inside URDF boxes,
min(object_SDF) per link.

Expected results:
  warmstart_single (GOOD):  palm min(SDF) ≈ 0  (touching, minimal penetration)
  warmstart_best (OK, far): palm min(SDF) > 0   (not touching)
  batched_curated (BAD):    palm min(SDF) << 0   (through the middle)

Run: conda run -n frogger python test_box_collision_metric.py
"""
import numpy as np
import trimesh
import open3d as o3d
import torch
import pytorch_kinematics as pk
import os, sys
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as ScipyR

sys.path.insert(0, os.path.dirname(__file__))
from frogger.batched_pytorch_solver import BatchedSDF, _visual_meshes

OBJ_MESH = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg/black_spray_bottle_single/object.obj"
HAND_DIR = os.path.join(os.path.dirname(__file__), "models/leap_rh")
URDF_PATH = os.path.join(HAND_DIR, "leap.urdf")
GRASPS = os.path.join(os.path.dirname(__file__), "output/grasps")

FINGER_GROUPS = {
    "palm": ["leap_rh_palm"],
    "IF": ["leap_rh_if_bs", "leap_rh_if_px", "leap_rh_if_md", "leap_rh_if_ds"],
    "MF": ["leap_rh_mf_bs", "leap_rh_mf_px", "leap_rh_mf_md", "leap_rh_mf_ds"],
    "RF": ["leap_rh_rf_bs", "leap_rh_rf_px", "leap_rh_rf_md", "leap_rh_rf_ds"],
    "TH": ["leap_rh_th_mp", "leap_rh_th_bs", "leap_rh_th_px", "leap_rh_th_ds"],
}


def build_box_grid_points(link_name, grid_spacing=0.003):
    """Build uniform grid points inside all URDF collision boxes for a link.

    Returns points in link-local frame as [N, 4] homogeneous coordinates.
    """
    tree = ET.parse(URDF_PATH)
    all_pts = []

    for le in tree.getroot().findall("link"):
        if le.get("name") != link_name:
            continue
        for col in le.findall("collision"):
            g = col.find("geometry")
            if g is None: continue
            b = g.find("box")
            if b is None: continue
            size = [float(x) for x in b.get("size").split()]
            o = col.find("origin")
            xyz = np.array([float(x) for x in o.get("xyz", "0 0 0").split()])
            rpy = np.array([float(x) for x in o.get("rpy", "0 0 0").split()])

            hx, hy, hz = [s/2 for s in size]
            # Uniform grid inside the box
            gx = np.arange(-hx, hx + grid_spacing/2, grid_spacing)
            gy = np.arange(-hy, hy + grid_spacing/2, grid_spacing)
            gz = np.arange(-hz, hz + grid_spacing/2, grid_spacing)
            grid = np.stack(np.meshgrid(gx, gy, gz, indexing='ij'), axis=-1).reshape(-1, 3)

            # Apply box rotation and translation
            if np.any(np.abs(rpy) > 1e-6):
                R = ScipyR.from_euler("xyz", rpy).as_matrix()
                grid = (R @ grid.T).T
            grid += xyz
            all_pts.append(grid)

    if not all_pts:
        return np.zeros((0, 4), dtype=np.float32)

    pts = np.vstack(all_pts).astype(np.float32)
    # Convert to homogeneous
    pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
    return pts_h


def main():
    # Load object
    obj_mesh = trimesh.load(OBJ_MESH, force="mesh")
    bounds = obj_mesh.bounds
    offset = np.array([0.0, 0.0, -bounds[0, 2]])
    X_WO = np.eye(4); X_WO[:3, 3] = offset

    sdf = BatchedSDF(obj_mesh, X_WO, resolution=128, device="cuda")

    # FK setup
    with open(URDF_PATH) as f:
        chain = pk.build_chain_from_urdf(f.read())

    # Pre-build box grid for each link
    all_links = set()
    for group_links in FINGER_GROUPS.values():
        all_links.update(group_links)

    box_grids = {}
    total_pts = 0
    for ln in sorted(all_links):
        pts = build_box_grid_points(ln, grid_spacing=0.003)
        if len(pts) > 0:
            box_grids[ln] = torch.tensor(pts, device="cuda")
            total_pts += len(pts)
            short = ln.replace("leap_rh_", "")
            print(f"  {short:<12} {len(pts):>5} grid points")
    print(f"  Total: {total_pts} points across {len(box_grids)} links\n")

    # Test grasps
    grasp_files = [
        ("warmstart_single (GOOD)", "compare_warmstart_single.pt"),
        ("warmstart_best (OK-far)", "compare_warmstart_best.pt"),
        ("batched_curated (BAD)", "compare_batched_curated.pt"),
    ]

    for grasp_name, filename in grasp_files:
        path = os.path.join(GRASPS, filename)
        if not os.path.exists(path):
            continue
        data = torch.load(path, weights_only=False, map_location="cpu")
        g = data[0]

        print(f"{'='*70}")
        print(f"  {grasp_name}")
        print(f"{'='*70}")

        q = torch.tensor(g["q_joints"], dtype=torch.float32).unsqueeze(0)
        fk = chain.forward_kinematics(q)
        T_base = np.eye(4)
        T_base[:3, :3] = np.array(g["base_rot"])
        T_base[:3, 3] = np.array(g["base_pos"])
        bT = torch.tensor(T_base, dtype=torch.float32, device="cuda").unsqueeze(0)

        # Per-group metrics
        for group_name, group_links in FINGER_GROUPS.items():
            group_min_sdf = float('inf')
            group_n_inside = 0
            group_n_total = 0

            link_details = []
            for ln in group_links:
                if ln not in box_grids or ln not in fk:
                    continue
                pts_h = box_grids[ln]  # [N, 4]

                # Transform to world frame
                link_T = fk[ln].get_matrix().to("cuda")  # [1, 4, 4]
                world_T = bT @ link_T  # [1, 4, 4]
                pts_world = (world_T[0] @ pts_h.T).T[:, :3]  # [N, 3]

                # Query object SDF
                sdfs = sdf.query(pts_world.unsqueeze(0)).squeeze(0)  # [N]

                min_sdf = sdfs.min().item()
                n_inside = (sdfs < 0).sum().item()
                n_total = len(sdfs)
                mean_sdf = sdfs.mean().item()

                group_min_sdf = min(group_min_sdf, min_sdf)
                group_n_inside += n_inside
                group_n_total += n_total

                short = ln.replace("leap_rh_", "")
                if n_inside > 0 or min_sdf < 0.005:
                    link_details.append(
                        f"    {short:<12} min={min_sdf*1000:>7.1f}mm  "
                        f"inside={n_inside:>4}/{n_total}  mean={mean_sdf*1000:.1f}mm")

            # Summary per group
            if group_n_total > 0:
                status = "CONTACT" if abs(group_min_sdf) < 0.003 else \
                         "FAR" if group_min_sdf > 0.003 else "PENETRATING"
                flag = "OK" if group_min_sdf >= -0.001 else "BAD"
                print(f"  {group_name:<6} min(SDF)={group_min_sdf*1000:>7.1f}mm  "
                      f"inside={group_n_inside}/{group_n_total}  [{status}] [{flag}]")
                for d in link_details:
                    print(d)

        print()


if __name__ == "__main__":
    main()
