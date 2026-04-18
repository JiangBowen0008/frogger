"""Compare baseline-IK (150 steps pos+dir+col) vs two-phase IK (100 pos + 50 full)
on per-stage metrics across 3 objects. Measures act_dist, dir_align, palm_worst,
filter pass counts, and opt candidate counts.
"""
import os, sys, json, numpy as np, trimesh, torch

sys.path.insert(0, "/home/bowenj/Projects/DexFun/third_parties/frogger")
from frogger.batched_pytorch_solver import BatchedSDF, BatchedGraspOptimizer

MESH_BASE = "/home/bowenj/Projects/DexFun/assets/mesh_obj"
ACT_BASE = "/home/bowenj/Projects/DexFun/assets/actuation_contacts"
NUM_ENVS = 4000
OBJECTS = ["hot_glue_gun", "grinder", "air_blower"]


def run(obj_name, two_phase, out_dir):
    mesh_path = os.path.join(MESH_BASE, obj_name, "object.obj")
    if not os.path.exists(mesh_path): return None
    mesh = trimesh.load(mesh_path, force="mesh")
    bounds = mesh.bounds
    offset = np.array([0.0, 0.0, -bounds[0, 2]])
    X_WO = np.eye(4); X_WO[:3, 3] = offset
    obj_center = mesh.centroid + offset
    with open(f"{ACT_BASE}/{obj_name}_actuation.json") as f:
        c = json.load(f)["actuation_contacts"][0]
    actuation_targets = [(np.array(c["pos"]) + offset, np.array(c["dir"]))]

    sdf = BatchedSDF(mesh, X_WO, bounds_padding=0.15, resolution=128, device="cuda")
    sdf.add_clearance_volume(actuation_targets[0][0], actuation_targets[0][1],
                             radius=0.020, height=0.03)
    sdf.add_floor(0.0)
    opt = BatchedGraspOptimizer(sdf, num_envs=NUM_ENVS, device="cuda",
                                hand="rh", hand_type="leap", palm_contact=True)

    save_path = os.path.join(out_dir, f"{obj_name}_grasps.pt")
    os.makedirs(out_dir, exist_ok=True)
    results = opt.optimize(
        actuation_targets=actuation_targets, object_center=obj_center,
        steps=300, lr=0.005, save_path=save_path,
        opt_sections="ABCD", opt_variant="P",
    )
    n_feas = sum(1 for r in results if r.get("feasible", False))
    return {"n_grasps": len(results), "n_feasible": n_feas,
            "n_lstar_pos": sum(1 for r in results if r.get("l_star", -1) > 0)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "twophase"], required=True)
    args = ap.parse_args()
    out = f"output/ik_compare_{args.mode}"
    print(f"\n{'='*60}\n  IK {args.mode}\n{'='*60}")
    results = {}
    for obj in OBJECTS:
        r = run(obj, args.mode == "twophase", out)
        if r:
            results[obj] = r
            print(f"  {obj:<20}: {r['n_feasible']} feasible, {r['n_lstar_pos']} l*>0")
    torch.save(results, os.path.join(out, "summary.pt"))


if __name__ == "__main__":
    main()
