#!/usr/bin/env python3
"""Multi-object generalization test: run ABCD pipeline on diverse objects.

Tests whether parameters tuned on hot_glue_gun work on other objects.
Uses assets/mesh_obj for meshes (has all objects).
"""
import os, sys, json, numpy as np, trimesh, torch, time

sys.path.insert(0, os.path.dirname(__file__))
from frogger.batched_pytorch_solver import BatchedSDF, BatchedGraspOptimizer

MESH_BASE = "/home/bowenj/Projects/DexFun/assets/mesh_obj"
MESH_ALT = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg"
ACT_BASE = "/home/bowenj/Projects/DexFun/assets/actuation_contacts"
OUT_DIR = "output/grasps_multi_object"
NUM_ENVS = 8000

# Diverse test set: different shapes, sizes, actuation types
# Only use first actuation contact (single-trigger objects)
TEST_OBJECTS = [
    "hot_glue_gun",               # trigger (current dev object)
    "black_spray_bottle_single",  # spray trigger
    "syrup_pourer_single",        # lever/button
    "pen_single",                 # small cylindrical, click button
    "pump_spray_single",          # pump top
]


def find_mesh(name):
    """Find mesh file, preferring mesh_raw_ahg, falling back to assets/mesh_obj."""
    p1 = os.path.join(MESH_ALT, name, "object.obj")
    p2 = os.path.join(MESH_BASE, name, "object.obj")
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    return None


def run_object(name):
    """Run ABCD optimization on one object. Returns metrics dict."""
    mesh_path = find_mesh(name)
    act_path = os.path.join(ACT_BASE, f"{name}_actuation.json")

    if mesh_path is None:
        print(f"  SKIP {name}: no mesh found")
        return None
    if not os.path.exists(act_path):
        print(f"  SKIP {name}: no actuation JSON")
        return None

    print(f"\n{'='*60}")
    print(f"  OBJECT: {name}")
    print(f"  Mesh: {mesh_path}")
    print(f"  Actuation: {act_path}")
    print(f"{'='*60}")

    mesh = trimesh.load(mesh_path, force="mesh")
    bounds = mesh.bounds
    offset = np.array([0.0, 0.0, -bounds[0, 2]])
    X_WO = np.eye(4)
    X_WO[:3, 3] = offset
    obj_center = mesh.centroid + offset

    with open(act_path) as f:
        act_data = json.load(f)
    # Use only first actuation contact
    c = act_data["actuation_contacts"][0]
    actuation_targets = [(np.array(c["pos"]) + offset, np.array(c["dir"]))]
    print(f"  Actuation pos: {actuation_targets[0][0]}")
    print(f"  Actuation dir: {actuation_targets[0][1]}")
    print(f"  Object size: {(bounds[1] - bounds[0]) * 1000} mm")

    save_file = os.path.join(OUT_DIR, f"{name}_ABCD.pt")

    try:
        sdf = BatchedSDF(mesh, X_WO, bounds_padding=0.15, resolution=128, device="cuda")
        opt = BatchedGraspOptimizer(
            sdf, num_envs=NUM_ENVS, device="cuda",
            hand="rh", hand_type="leap", palm_contact=True,
        )
        t0 = time.time()
        results = opt.optimize(
            actuation_targets=actuation_targets,
            object_center=obj_center,
            steps=300,
            lr=0.005,
            save_path=save_file,
            opt_sections="ABCD",
        )
        elapsed = time.time() - t0
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        return {"object": name, "error": str(e)}

    metrics = {
        "object": name,
        "n_grasps": len(results),
        "n_feasible": sum(1 for r in results if r.get("feasible", False)),
        "time_s": elapsed,
    }
    if results:
        metrics["sigma_min_best"] = max(r["sigma_min"] for r in results)
        metrics["sigma_min_mean"] = np.mean([r["sigma_min"] for r in results])
        metrics["l_star_best"] = max(r["l_star"] for r in results)
        metrics["surf_err_mean_mm"] = np.mean([r["surf_err"] for r in results]) * 1000
        metrics["surf_err_best_mm"] = min(r["surf_err"] for r in results) * 1000
        metrics["pen_pct_mean"] = np.mean([r.get("mesh_pen_pct", 0) for r in results])
        metrics["pen_pct_best"] = min(r.get("mesh_pen_pct", 0) for r in results)
    else:
        metrics["sigma_min_best"] = 0
        metrics["sigma_min_mean"] = 0
        metrics["l_star_best"] = -1
        metrics["surf_err_mean_mm"] = 999
        metrics["surf_err_best_mm"] = 999
        metrics["pen_pct_mean"] = 100
        metrics["pen_pct_best"] = 100

    return metrics


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_metrics = []
    for name in TEST_OBJECTS:
        m = run_object(name)
        if m is not None:
            all_metrics.append(m)

    # Print comparison table
    print(f"\n{'='*90}")
    print(f"  MULTI-OBJECT COMPARISON (ABCD pipeline, {NUM_ENVS} envs)")
    print(f"{'='*90}")
    header = f"{'Object':<30} {'Feas':<6} {'σ_best':<8} {'σ_mean':<8} {'l*_best':<8} {'surf_mm':<8} {'pen_%':<8} {'time':<6}"
    print(header)
    print("-" * len(header))
    for m in all_metrics:
        if "error" in m:
            print(f"{m['object']:<30} ERROR: {m['error'][:40]}")
            continue
        print(f"{m['object']:<30} "
              f"{m['n_feasible']:<6} "
              f"{m.get('sigma_min_best', 0):<8.4f} "
              f"{m.get('sigma_min_mean', 0):<8.4f} "
              f"{m.get('l_star_best', -1):<8.4f} "
              f"{m.get('surf_err_best_mm', 999):<8.1f} "
              f"{m.get('pen_pct_best', 100):<8.1f} "
              f"{m.get('time_s', 0):<6.0f}s")

    # Save metrics
    torch.save(all_metrics, os.path.join(OUT_DIR, "comparison.pt"))
    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
