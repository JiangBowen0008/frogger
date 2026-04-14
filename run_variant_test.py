"""Compare 3 optimization variants on hot_glue_gun.

Variant A: Baseline soft penalty (Adam, sections A+B+C+D)
Variant B: Min-k unified surface/collision
Variant C: Min-k with adaptive FC contacts

Usage:
    PYTHONUNBUFFERED=1 conda run --no-capture-output -n frogger python -u run_variant_test.py
"""

import numpy as np
import trimesh
import torch
import json
import time
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from frogger.batched_pytorch_solver import (
    BatchedSDF,
    BatchedGraspOptimizer,
)


def run_variant(variant_name, mesh, X_WO, obj_center, actuation_targets, num_envs=4000, steps=300):
    """Run a single optimization variant and collect metrics."""
    print(f"\n{'='*70}")
    print(f"  VARIANT {variant_name}")
    print(f"{'='*70}")

    t0 = time.time()

    sdf = BatchedSDF(mesh, X_WO, bounds_padding=0.15, resolution=128, device="cuda")

    opt = BatchedGraspOptimizer(
        sdf, num_envs=num_envs, device="cuda",
        hand="rh", hand_type="leap",
        palm_contact=True,
    )

    trajectory = []
    save_path = f"output/grasps/variant_{variant_name}.pt"

    results = opt.optimize(
        actuation_targets=actuation_targets,
        object_center=obj_center,
        steps=steps,
        lr=0.005,
        save_path=save_path,
        opt_variant=variant_name,
        trajectory_log=trajectory,
    )

    elapsed = time.time() - t0

    # Collect summary metrics from results
    metrics = {
        "variant": variant_name,
        "elapsed_s": elapsed,
        "n_results": len(results),
        "trajectory": trajectory,
    }

    if results:
        surf_errs = [r["surf_err"] for r in results]
        sigma_mins = [r["sigma_min"] for r in results]
        l_stars = [r["l_star"] for r in results]
        feasible_count = sum(1 for r in results if r.get("feasible", False))
        pen_pcts = [r.get("mesh_pen_pct", 0) for r in results]
        sc_worsts = [r.get("sc_worst", 999) for r in results]

        metrics["surface_mm_mean"] = np.mean(surf_errs) * 1000
        metrics["surface_mm_best"] = np.min(surf_errs) * 1000
        metrics["sigma_min_mean"] = np.mean(sigma_mins)
        metrics["sigma_min_best"] = np.max(sigma_mins)
        metrics["l_star_mean"] = np.mean(l_stars)
        metrics["l_star_best"] = np.max(l_stars)
        metrics["feasible_count"] = feasible_count
        metrics["pen_pct_mean"] = np.mean(pen_pcts)
        metrics["sc_worst_mean"] = np.mean(sc_worsts) * 1000
    else:
        metrics["surface_mm_mean"] = float("nan")
        metrics["surface_mm_best"] = float("nan")
        metrics["sigma_min_mean"] = float("nan")
        metrics["sigma_min_best"] = float("nan")
        metrics["l_star_mean"] = float("nan")
        metrics["l_star_best"] = float("nan")
        metrics["feasible_count"] = 0
        metrics["pen_pct_mean"] = float("nan")
        metrics["sc_worst_mean"] = float("nan")

    return metrics, results


def print_trajectory(metrics_list):
    """Print optimization trajectory comparison."""
    print(f"\n{'='*70}")
    print("  OPTIMIZATION TRAJECTORY")
    print(f"{'='*70}")

    # Collect all unique steps across variants
    all_steps = set()
    for m in metrics_list:
        for t in m["trajectory"]:
            all_steps.add(t["step"])
    all_steps = sorted(all_steps)

    # Header
    header = f"{'Step':>6}"
    for m in metrics_list:
        v = m["variant"]
        header += f" | {v} surf(mm)"
        header += f" | {v} sigma"
    print(header)
    print("-" * len(header))

    # Build lookup per variant
    for step in all_steps:
        row = f"{step:>6}"
        for m in metrics_list:
            traj = {t["step"]: t for t in m["trajectory"]}
            if step in traj:
                row += f" | {traj[step]['surface_mm']:>10.1f}"
                row += f" | {traj[step]['sigma']:>8.4f}"
            else:
                row += f" | {'---':>10}"
                row += f" | {'---':>8}"
        print(row)


def print_comparison(metrics_list):
    """Print final comparison table."""
    print(f"\n{'='*70}")
    print("  FINAL COMPARISON TABLE")
    print(f"{'='*70}")

    header = (f"{'Variant':>8} | {'Surf(mm)':>9} | {'Surf best':>9} | {'sigma':>8} | "
              f"{'sigma best':>10} | {'l*':>8} | {'l* best':>8} | "
              f"{'Feasible':>8} | {'Pen%':>6} | {'SC(mm)':>7} | {'Time(s)':>7}")
    print(header)
    print("-" * len(header))

    for m in metrics_list:
        row = (f"{m['variant']:>8} | "
               f"{m['surface_mm_mean']:>9.1f} | "
               f"{m['surface_mm_best']:>9.1f} | "
               f"{m['sigma_min_mean']:>8.4f} | "
               f"{m['sigma_min_best']:>10.4f} | "
               f"{m['l_star_mean']:>8.4f} | "
               f"{m['l_star_best']:>8.4f} | "
               f"{m['feasible_count']:>8d} | "
               f"{m['pen_pct_mean']:>6.1f} | "
               f"{m['sc_worst_mean']:>7.1f} | "
               f"{m['elapsed_s']:>7.0f}")
        print(row)

    # Key question: does surface stay low while sigma increases?
    print(f"\n  Key question: surface < 5mm AND sigma_min > 0?")
    for m in metrics_list:
        surf_ok = m["surface_mm_mean"] < 5.0
        sig_ok = m["sigma_min_mean"] > 0.0
        verdict = "YES" if (surf_ok and sig_ok) else "NO"
        print(f"    Variant {m['variant']}: surf={m['surface_mm_mean']:.1f}mm "
              f"sigma={m['sigma_min_mean']:.4f} -> {verdict}")


def main():
    mesh_path = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg/hot_glue_gun/object.obj"
    act_path = "/home/bowenj/Projects/DexFun/assets/actuation_contacts/hot_glue_gun_actuation.json"

    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path, force="mesh")

    bounds = mesh.bounds
    offset = np.array([0.0, 0.0, -bounds[0, 2]])
    X_WO = np.eye(4)
    X_WO[:3, 3] = offset
    obj_center = mesh.centroid + offset

    with open(act_path) as f:
        act_data = json.load(f)
    actuation_targets = [
        (np.array(c["pos"]) + offset, np.array(c["dir"]))
        for c in act_data["actuation_contacts"]
    ]
    for i, (pos, d) in enumerate(actuation_targets):
        print(f"  Actuation[{i}]: pos={pos}, dir={d}")

    num_envs = 4000
    steps = 300

    all_metrics = []

    for variant in ["A", "B", "C"]:
        # Clear GPU cache between variants
        torch.cuda.empty_cache()

        metrics, results = run_variant(
            variant, mesh, X_WO, obj_center, actuation_targets,
            num_envs=num_envs, steps=steps,
        )
        all_metrics.append(metrics)

        # Print per-grasp details for this variant
        print(f"\n  Variant {variant} top grasps:")
        for i, r in enumerate(results[:5]):
            f_tag = "FEAS" if r.get("feasible") else "FAIL"
            print(f"    G{i} [{f_tag}] surf={r['surf_err']*1000:.1f}mm "
                  f"sigma={r['sigma_min']:.4f} l*={r['l_star']:.4f} "
                  f"pen={r.get('mesh_pen_pct', 0):.1f}% "
                  f"sc={r.get('sc_worst', 999)*1000:.1f}mm")

    print_trajectory(all_metrics)
    print_comparison(all_metrics)


if __name__ == "__main__":
    main()
