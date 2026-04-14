#!/usr/bin/env python3
"""
Run collision experiments with different grid spacing and loss weights.
Patches the optimizer at runtime without modifying the source.

Usage:
  conda run -n frogger python run_collision_experiment.py --pitch 0.005 --col_weight 10 --pen_scale 0.1 --tag expA
"""
import argparse
import numpy as np
import trimesh
import torch
import os, sys

sys.path.insert(0, os.path.dirname(__file__))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pitch", type=float, default=0.005)
    parser.add_argument("--col_weight", type=float, default=10.0)
    parser.add_argument("--pen_scale", type=float, default=0.1, help="Scale for pen.sum (replaces pen.mean*3)")
    parser.add_argument("--tag", default="exp")
    parser.add_argument("--num_envs", type=int, default=4000)
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()

    # Monkey-patch the grid pitch before importing the optimizer
    import frogger.batched_pytorch_solver as bps

    # Store original method
    _orig_precompute = bps.BatchedGraspOptimizer._precompute_collision_points

    def _patched_precompute(self):
        """Patched to use custom grid pitch."""
        # Temporarily replace the pitch constant
        # We'll call the original but it reads _pitch from the method body
        # Instead, just re-implement the LEAP branch with our pitch
        from scipy.spatial.transform import Rotation as ScipyR
        import xml.etree.ElementTree as ET

        if self.hand_type != "leap":
            _orig_precompute(self)
            return

        _pitch = args.pitch
        _urdf = os.path.join(os.path.dirname(bps.__file__),
                             f"../models/leap_{self.hand}/leap.urdf")
        _tree = ET.parse(_urdf)
        _col_link_names = list(self.collision_link_names)
        col_data = []
        col_link_ranges = []
        _offset = 0

        for nm in _col_link_names:
            link_pts = []
            _le = None
            for _e in _tree.getroot().findall("link"):
                if _e.get("name") == nm:
                    _le = _e
                    break
            if _le is not None:
                for _cel in _le.findall("collision"):
                    _g = _cel.find("geometry")
                    if _g is None: continue
                    _b = _g.find("box")
                    if _b is None: continue
                    _sz = [float(x) for x in _b.get("size").split()]
                    _o = _cel.find("origin")
                    _p = np.array([float(x) for x in _o.get("xyz", "0 0 0").split()])
                    _rpy = np.array([float(x) for x in _o.get("rpy", "0 0 0").split()])
                    _R = (ScipyR.from_euler("xyz", _rpy).as_matrix()
                          if np.any(np.abs(_rpy) > 1e-6) else np.eye(3))
                    hx, hy, hz = _sz[0]/2, _sz[1]/2, _sz[2]/2
                    gx = np.arange(-hx, hx + _pitch/2, _pitch)
                    gy = np.arange(-hy, hy + _pitch/2, _pitch)
                    gz = np.arange(-hz, hz + _pitch/2, _pitch)
                    grid = np.stack(np.meshgrid(gx, gy, gz, indexing='ij'),
                                    axis=-1).reshape(-1, 3)
                    grid = ((_R @ grid.T).T + _p).astype(np.float32)
                    link_pts.append(grid)

            if link_pts:
                pts = np.vstack(link_pts)
            else:
                pts = np.array([[0, 0, 0]], dtype=np.float32)
            pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
            col_data.append((nm, torch.tensor(pts_h, device=self.device)))
            col_link_ranges.append((_offset, _offset + len(pts)))
            _offset += len(pts)

        self._col_data = col_data
        self._col_link_ranges = col_link_ranges
        self._n_col_links = len(col_link_ranges)
        n_total = sum(p.shape[0] for _, p in col_data)
        print(f"  Box-grid collision: {len(col_data)} links, "
              f"{n_total} points ({_pitch*1000:.0f}mm grid)")

        # Margins
        margins = []
        for li, (nm, pts) in enumerate(col_data):
            si, ei = col_link_ranges[li]
            m = -0.001 if "_ds" in nm else 0.0
            margins.extend([m] * (ei - si))
        self._col_margins = torch.tensor(margins, dtype=torch.float32,
                                         device=self.device)

        # Self-collision (call original SC setup by running the rest)
        # We need to replicate the SC setup from the original code
        from scipy.spatial.transform import Rotation as _ScipyR
        sc_data = []
        if os.path.exists(_urdf):
            for _le in _tree.getroot().findall("link"):
                _ln = _le.get("name")
                if _ln not in self.collision_link_names:
                    continue
                _bpts = []
                for _col_elem in _le.findall("collision"):
                    _g = _col_elem.find("geometry")
                    if _g is None: continue
                    _b = _g.find("box")
                    if _b is None: continue
                    _sx, _sy, _sz = [float(x) for x in _b.get("size").split()]
                    _h = np.array([_sx/2, _sy/2, _sz/2])
                    _o = _col_elem.find("origin")
                    _p = np.array([float(x) for x in _o.get("xyz", "0 0 0").split()])
                    _rpy = np.array([float(x) for x in _o.get("rpy", "0 0 0").split()])
                    _R = _ScipyR.from_euler("xyz", _rpy).as_matrix() if np.any(np.abs(_rpy) > 1e-6) else np.eye(3)
                    for sx in [-1, 1]:
                        for sy in [-1, 1]:
                            for sz in [-1, 1]:
                                _bpts.append(_R @ np.array([sx*_h[0], sy*_h[1], sz*_h[2]]) + _p)
                if _bpts:
                    _pts = np.array(_bpts, dtype=np.float32)
                else:
                    _pts = np.array([[0, 0, 0]], dtype=np.float32)
                _pts_h = np.hstack([_pts, np.ones((len(_pts), 1), dtype=np.float32)])
                sc_data.append((_ln, torch.tensor(_pts_h, device=self.device)))
        self._sc_data = sc_data

        _SC_MAX = 60
        finger_keys = ['if', 'mf', 'rf', 'th']
        _fcol = {}
        offset = 0
        palm_idx = []
        for nm, pts in sc_data:
            n = pts.shape[0]
            if 'palm' in nm:
                palm_idx.extend(range(offset, offset + n))
            for fk in finger_keys:
                if f'_{fk}_' in nm:
                    _fcol.setdefault(fk, []).extend(range(offset, offset + n))
                    break
            offset += n

        def _subsample(idx_list):
            if len(idx_list) > _SC_MAX:
                step = len(idx_list) / _SC_MAX
                return [idx_list[int(k * step)] for k in range(_SC_MAX)]
            return idx_list

        self._self_col_pairs = []
        fk_list = [k for k in finger_keys if k in _fcol]
        for i in range(len(fk_list)):
            for j in range(i + 1, len(fk_list)):
                self._self_col_pairs.append((
                    torch.tensor(_subsample(_fcol[fk_list[i]]), dtype=torch.long, device=self.device),
                    torch.tensor(_subsample(_fcol[fk_list[j]]), dtype=torch.long, device=self.device),
                ))
        if palm_idx:
            for fk in fk_list:
                self._self_col_pairs.append((
                    torch.tensor(_subsample(palm_idx), dtype=torch.long, device=self.device),
                    torch.tensor(_subsample(_fcol[fk]), dtype=torch.long, device=self.device),
                ))
        n_sc_pts = sum(p.shape[0] for _, p in sc_data)
        print(f"  Self-collision: {len(self._self_col_pairs)} pairs, {n_sc_pts} box pts")

    # Apply monkey-patch
    bps.BatchedGraspOptimizer._precompute_collision_points = _patched_precompute

    # Now monkey-patch the optimize method to use custom weights
    _orig_optimize = bps.BatchedGraspOptimizer.optimize

    # We can't easily patch loss weights inside optimize() without rewriting it.
    # Instead, store the params as instance attributes and read them in a patched version.
    # For simplicity, let's just store the experiment params globally and use them.
    bps._EXP_COL_WEIGHT = args.col_weight
    bps._EXP_PEN_SCALE = args.pen_scale

    # Load mesh and run
    mesh_path = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg/black_spray_bottle_single/object.obj"
    mesh = trimesh.load(mesh_path, force="mesh")
    bounds = mesh.bounds
    offset = np.array([0.0, 0.0, -bounds[0, 2]])
    X_WO = np.eye(4); X_WO[:3, 3] = offset

    from frogger.batched_pytorch_solver import BatchedSDF, BatchedGraspOptimizer
    sdf = BatchedSDF(mesh, X_WO, resolution=128, device="cuda")
    opt = BatchedGraspOptimizer(sdf, num_envs=args.num_envs, device="cuda",
                                 hand="rh", hand_type="leap", palm_contact=True)

    # Compute actuation target
    verts_W = (X_WO[:3,:3] @ np.asarray(mesh.vertices, dtype=np.float64).T).T + X_WO[:3,3]
    act_height = offset[2] + (bounds[1,2] - bounds[0,2]) * 0.8
    act_candidate = np.array([[0.0, 0.0, act_height]])
    mesh_W = trimesh.Trimesh(vertices=verts_W, faces=mesh.faces)
    closest, _, _ = trimesh.proximity.closest_point(mesh_W, act_candidate)
    act_pos = closest[0]
    obj_center = verts_W.mean(axis=0)

    actuation_targets = [(act_pos, np.array([0.0, 0.0, -1.0]))]

    print(f"\n=== Experiment {args.tag}: pitch={args.pitch*1000:.0f}mm, "
          f"col_weight={args.col_weight}, pen_scale={args.pen_scale} ===\n")

    results = opt.optimize(actuation_targets=actuation_targets,
                           object_center=obj_center, steps=args.steps)
    n_feas = sum(1 for r in results if r.get("feasible", False))
    print(f"\n=== Results: {n_feas}/{len(results)} feasible ===")

    save_dir = f"output/grasps_{args.tag}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "result.pt")
    torch.save(results, save_path)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
