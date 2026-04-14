#!/usr/bin/env python3
"""Compare three optimization variants on spray_bottle.

Alternative A (DA): Direct-q parameterization (no sigmoid gradient compression)
Alternative B (SA): Side-aware ds collision (pad-hemisphere only)
Alternative C (CA): Contact-aware IK initialization (straight fingers, anti-penetration)
Baseline (A): Original variant A

Each variant runs 4000 envs on funky_clear_spray_bottle.
Reports: support IK yield, optimization trajectory, ds_worst, feasibility, CMC separation.
"""
import os, sys, json, numpy as np, trimesh, torch, time

sys.path.insert(0, os.path.dirname(__file__))
from frogger.batched_pytorch_solver import BatchedSDF, BatchedGraspOptimizer

MESH_BASE = "/home/bowenj/Projects/DexFun/assets/mesh_obj"
MESH_ALT = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg"
ACT_BASE = "/home/bowenj/Projects/DexFun/assets/actuation_contacts"
OUT_DIR = "output/grasps_variants"
NUM_ENVS = 4000
OBJECT_NAME = "funky_clear_spray_bottle"

# Variants to compare
VARIANTS = ["A", "DA", "SA", "CA"]


def find_mesh(name):
    p1 = os.path.join(MESH_ALT, name, "object.obj")
    p2 = os.path.join(MESH_BASE, name, "object.obj")
    return p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else None)


def run_variant(variant_name, mesh_path, act_path, offset, obj_center, actuation_targets):
    print(f"\n{'='*70}")
    print(f"  VARIANT: {variant_name} ({NUM_ENVS} envs)")
    print(f"{'='*70}")

    mesh = trimesh.load(mesh_path, force="mesh")
    bounds = mesh.bounds
    X_WO = np.eye(4); X_WO[:3, 3] = offset

    var_dir = os.path.join(OUT_DIR, f"variant_{variant_name}")
    os.makedirs(var_dir, exist_ok=True)
    save_file = os.path.join(var_dir, "grasps.pt")

    trajectory = []
    try:
        sdf = BatchedSDF(mesh, X_WO, bounds_padding=0.15, resolution=128, device="cuda")
        sdf.add_clearance_volume(actuation_targets[0][0], actuation_targets[0][1],
                                 radius=0.015, height=0.05)
        sdf.add_floor(0.0)
        opt = BatchedGraspOptimizer(
            sdf, num_envs=NUM_ENVS, device="cuda",
            hand="rh", hand_type="leap", palm_contact=True,
        )
        t0 = time.time()
        results = opt.optimize(
            actuation_targets=actuation_targets,
            object_center=obj_center,
            steps=300, lr=0.005,
            save_path=save_file,
            opt_sections="ABCD",
            opt_variant=variant_name,
            trajectory_log=trajectory,
        )
        elapsed = time.time() - t0

        # Copy stage snapshots
        stage_dir = os.path.join(os.path.dirname(__file__), "output/grasps")
        for stage_name in ["stage_after_init.pt", "stage_after_support_ik.pt", "stage_after_optimization.pt"]:
            src = os.path.join(stage_dir, stage_name)
            if os.path.exists(src):
                import shutil
                dst = os.path.join(var_dir, stage_name)
                shutil.copy2(src, dst)

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        return {"variant": variant_name, "error": str(e)}

    metrics = {
        "variant": variant_name,
        "n_grasps": len(results),
        "n_feasible": sum(1 for r in results if r.get("feasible", False)),
        "time_s": elapsed,
        "trajectory": trajectory,
    }
    if results:
        metrics["sigma_best"] = max(r["sigma_min"] for r in results)
        metrics["lstar_best"] = max(r["l_star"] for r in results)
        metrics["surf_best_mm"] = min(r["surf_err"] for r in results) * 1000
        metrics["min_col_best"] = min(r["min_col"] for r in results)
        n_lstar = sum(1 for r in results if r["l_star"] > 0)
        metrics["n_lstar_pos"] = n_lstar

        # Detailed per-result metrics
        for i, r in enumerate(results[:5]):
            prefix = f"top{i+1}"
            metrics[f"{prefix}_sigma"] = r["sigma_min"]
            metrics[f"{prefix}_lstar"] = r["l_star"]
            metrics[f"{prefix}_surf_mm"] = r["surf_err"] * 1000
            metrics[f"{prefix}_min_col"] = r["min_col"]
            metrics[f"{prefix}_feasible"] = r.get("feasible", False)
            metrics[f"{prefix}_sc_min"] = r.get("sc_min_dist", 999)

    return metrics


def main():
    mesh_path = find_mesh(OBJECT_NAME)
    act_path = os.path.join(ACT_BASE, f"{OBJECT_NAME}_actuation.json")
    if mesh_path is None or not os.path.exists(act_path):
        print(f"Missing mesh or actuation for {OBJECT_NAME}")
        return

    mesh = trimesh.load(mesh_path, force="mesh")
    bounds = mesh.bounds
    offset = np.array([0.0, 0.0, -bounds[0, 2]])
    obj_center = mesh.centroid + offset

    with open(act_path) as f:
        act_data = json.load(f)
    c = act_data["actuation_contacts"][0]
    actuation_targets = [(np.array(c["pos"]) + offset, np.array(c["dir"]))]

    os.makedirs(OUT_DIR, exist_ok=True)
    all_metrics = []

    for variant in VARIANTS:
        m = run_variant(variant, mesh_path, act_path, offset, obj_center, actuation_targets)
        all_metrics.append(m)
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*90}")
    print(f"  VARIANT COMPARISON ({OBJECT_NAME}, {NUM_ENVS} envs)")
    print(f"{'='*90}")
    header = (f"{'Variant':<10} {'Feas':<6} {'l*>0':<6} {'sigma':<8} {'l*':<8} "
              f"{'surf_mm':<8} {'min_col':<10} {'time':<6}")
    print(header)
    print("-" * len(header))
    for m in all_metrics:
        if "error" in m:
            print(f"{m['variant']:<10} ERROR: {m['error'][:50]}")
            continue
        print(f"{m['variant']:<10} "
              f"{m['n_feasible']:<6} "
              f"{m.get('n_lstar_pos', 0):<6} "
              f"{m.get('sigma_best', 0):<8.4f} "
              f"{m.get('lstar_best', -1):<8.4f} "
              f"{m.get('surf_best_mm', 999):<8.1f} "
              f"{m.get('min_col_best', -1):<10.4f} "
              f"{m.get('time_s', 0):<6.0f}s")

    # Detailed top-5 per variant
    print(f"\n  --- Top 5 per variant ---")
    for m in all_metrics:
        if "error" in m:
            continue
        print(f"\n  {m['variant']}:")
        for i in range(5):
            prefix = f"top{i+1}"
            if f"{prefix}_sigma" not in m:
                break
            feas = "FEAS" if m.get(f"{prefix}_feasible", False) else "----"
            print(f"    #{i+1}: {feas} sigma={m[f'{prefix}_sigma']:.4f} "
                  f"l*={m[f'{prefix}_lstar']:.4f} "
                  f"surf={m[f'{prefix}_surf_mm']:.1f}mm "
                  f"min_col={m[f'{prefix}_min_col']:.4f} "
                  f"sc={m.get(f'{prefix}_sc_min', 999):.4f}")

    # Trajectory comparison
    print(f"\n  --- Optimization Trajectories ---")
    for m in all_metrics:
        if "error" in m or not m.get("trajectory"):
            continue
        traj = m["trajectory"]
        print(f"\n  {m['variant']}:")
        for t in traj:
            print(f"    step {t['step']:3d}: surf={t['surface_mm']:.1f}mm sigma={t['sigma']:.4f}")

    torch.save(all_metrics, os.path.join(OUT_DIR, "comparison.pt"))
    print(f"\nAll results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
