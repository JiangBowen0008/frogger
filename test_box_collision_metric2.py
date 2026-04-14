#!/usr/bin/env python3
"""
Deeper analysis of box collision metric — distribution of SDF values,
not just min/count. Compare where the penetration occurs.
"""
import numpy as np
import trimesh
import torch
import pytorch_kinematics as pk
import os, sys
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as ScipyR

sys.path.insert(0, os.path.dirname(__file__))
from frogger.batched_pytorch_solver import BatchedSDF

OBJ_MESH = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg/black_spray_bottle_single/object.obj"
HAND_DIR = os.path.join(os.path.dirname(__file__), "models/leap_rh")
URDF_PATH = os.path.join(HAND_DIR, "leap.urdf")
GRASPS = os.path.join(os.path.dirname(__file__), "output/grasps")


def build_box_grid(link_name, spacing=0.003):
    tree = ET.parse(URDF_PATH)
    all_pts = []
    for le in tree.getroot().findall("link"):
        if le.get("name") != link_name: continue
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
            gx = np.arange(-hx, hx + spacing/2, spacing)
            gy = np.arange(-hy, hy + spacing/2, spacing)
            gz = np.arange(-hz, hz + spacing/2, spacing)
            grid = np.stack(np.meshgrid(gx, gy, gz, indexing='ij'), axis=-1).reshape(-1, 3)
            if np.any(np.abs(rpy) > 1e-6):
                R = ScipyR.from_euler("xyz", rpy).as_matrix()
                grid = (R @ grid.T).T
            grid += xyz
            all_pts.append(grid)
    if not all_pts: return None
    pts = np.vstack(all_pts).astype(np.float32)
    return np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])


def main():
    obj_mesh = trimesh.load(OBJ_MESH, force="mesh")
    bounds = obj_mesh.bounds
    offset = np.array([0.0, 0.0, -bounds[0, 2]])
    X_WO = np.eye(4); X_WO[:3, 3] = offset
    sdf = BatchedSDF(obj_mesh, X_WO, resolution=128, device="cuda")

    with open(URDF_PATH) as f:
        chain = pk.build_chain_from_urdf(f.read())

    palm_grid = build_box_grid("leap_rh_palm", spacing=0.003)
    palm_grid_t = torch.tensor(palm_grid, device="cuda")
    print(f"Palm grid: {len(palm_grid)} points\n")

    grasp_files = [
        ("warmstart_single", "compare_warmstart_single.pt"),
        ("warmstart_best", "compare_warmstart_best.pt"),
        ("batched_curated", "compare_batched_curated.pt"),
    ]

    for name, filename in grasp_files:
        path = os.path.join(GRASPS, filename)
        if not os.path.exists(path): continue
        data = torch.load(path, weights_only=False, map_location="cpu")
        g = data[0]

        q = torch.tensor(g["q_joints"], dtype=torch.float32).unsqueeze(0)
        fk = chain.forward_kinematics(q)
        T_base = np.eye(4)
        T_base[:3, :3] = np.array(g["base_rot"])
        T_base[:3, 3] = np.array(g["base_pos"])
        bT = torch.tensor(T_base, dtype=torch.float32, device="cuda").unsqueeze(0)

        link_T = fk["leap_rh_palm"].get_matrix().to("cuda")
        world_T = bT @ link_T
        pts_w = (world_T[0] @ palm_grid_t.T).T[:, :3]
        sdfs = sdf.query(pts_w.unsqueeze(0)).squeeze(0).cpu().numpy()

        print(f"  {name}:")
        print(f"    min(SDF) = {sdfs.min()*1000:.1f}mm")
        print(f"    mean(SDF) = {sdfs.mean()*1000:.1f}mm")
        print(f"    median(SDF) = {np.median(sdfs)*1000:.1f}mm")

        # Distribution at various thresholds
        thresholds = [0, -1, -3, -5, -10, -15, -20]
        print(f"    SDF distribution (of {len(sdfs)} total):")
        for t in thresholds:
            n = (sdfs < t/1000.0).sum()
            pct = n / len(sdfs) * 100
            print(f"      SDF < {t:>3d}mm: {n:>5} ({pct:>5.1f}%)")

        # Volume of intersection (approximate)
        vol_inside = (sdfs < 0).sum() * (0.003 ** 3) * 1e9  # mm³
        vol_deep = (sdfs < -0.005).sum() * (0.003 ** 3) * 1e9
        print(f"    Intersection volume: {vol_inside:.0f} mm³")
        print(f"    Deep (>5mm) volume:  {vol_deep:.0f} mm³")
        print()


if __name__ == "__main__":
    main()
