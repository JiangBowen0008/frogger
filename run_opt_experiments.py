#!/usr/bin/env python3
"""Run targeted optimization experiments A→AB→ABC→ABCD.

Each experiment isolates one additional loss component to verify
it doesn't break previous components.

Exp A:    Surface only (tips converge to SDF=0)
Exp AB:   + Link-object collision (tips stay on surface, links stay out)
Exp ABC:  + Force closure (σ_min improves, surface/collision don't degrade)
Exp ABCD: + Inter-finger collision (full pipeline)
"""
import os, sys, json, numpy as np, trimesh, torch, time

sys.path.insert(0, os.path.dirname(__file__))
from frogger.batched_pytorch_solver import BatchedSDF, BatchedGraspOptimizer

# ── Configuration ──
MESH_PATH = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg/hot_glue_gun/object.obj"
ACT_PATH = "/home/bowenj/Projects/DexFun/assets/actuation_contacts/hot_glue_gun_actuation.json"
NUM_ENVS = 8000
OUT_DIR = "output/grasps_opt_exp"

EXPERIMENTS = [
    ("A",    "Surface only"),
    ("AB",   "Surface + collision"),
    ("ABC",  "Surface + collision + FC"),
    ("ABCD", "Surface + collision + FC + inter-finger"),
]


def run_experiment(exp_name, sections, sdf, mesh, X_WO, obj_center, actuation_targets):
    """Run one experiment and return results + metrics."""
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT {exp_name}: sections={sections}")
    print(f"{'='*60}")

    save_file = os.path.join(OUT_DIR, f"exp_{exp_name}.pt")
    opt = BatchedGraspOptimizer(
        sdf, num_envs=NUM_ENVS, device="cuda",
        hand="rh", hand_type="leap", palm_contact=True,
    )
    t0 = time.time()
    results = opt.optimize(
        actuation_targets=actuation_targets,
        object_center=obj_center,
        steps=1200,
        lr=0.005,
        save_path=save_file,
        opt_sections=sections,
    )
    elapsed = time.time() - t0

    # Summarize
    metrics = {
        "experiment": exp_name,
        "sections": sections,
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
        metrics["sc_worst_mean_mm"] = np.mean([r.get("sc_worst", 999) for r in results]) * 1000
        metrics["sc_worst_best_mm"] = min(r.get("sc_worst", 999) for r in results) * 1000

    return results, metrics


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load mesh
    mesh = trimesh.load(MESH_PATH, force="mesh")
    bounds = mesh.bounds
    offset = np.array([0.0, 0.0, -bounds[0, 2]])
    X_WO = np.eye(4)
    X_WO[:3, 3] = offset
    obj_center = mesh.centroid + offset

    # Load actuation
    with open(ACT_PATH) as f:
        act_data = json.load(f)
    actuation_targets = [
        (np.array(c["pos"]) + offset, np.array(c["dir"]))
        for c in act_data["actuation_contacts"]
    ]

    # Build SDF once (expensive)
    print("Building SDF ...")
    sdf = BatchedSDF(mesh, X_WO, bounds_padding=0.15, resolution=128, device="cuda")

    # Run experiments
    all_metrics = []
    for exp_name, desc in EXPERIMENTS:
        results, metrics = run_experiment(
            exp_name, exp_name, sdf, mesh, X_WO, obj_center, actuation_targets)
        all_metrics.append(metrics)

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"  EXPERIMENT COMPARISON")
    print(f"{'='*80}")
    header = f"{'Exp':<8} {'Sections':<8} {'Feas':<6} {'σ_best':<8} {'σ_mean':<8} {'l*_best':<8} {'surf_mm':<8} {'pen_%':<8} {'sc_mm':<8} {'time':<6}"
    print(header)
    print("-" * len(header))
    for m in all_metrics:
        print(f"{m['experiment']:<8} {m['sections']:<8} "
              f"{m['n_feasible']:<6} "
              f"{m.get('sigma_min_best', 0):<8.4f} "
              f"{m.get('sigma_min_mean', 0):<8.4f} "
              f"{m.get('l_star_best', 0):<8.4f} "
              f"{m.get('surf_err_best_mm', 0):<8.1f} "
              f"{m.get('pen_pct_best', 0):<8.1f} "
              f"{m.get('sc_worst_best_mm', 0):<8.1f} "
              f"{m['time_s']:<6.0f}s")

    # Save metrics
    torch.save(all_metrics, os.path.join(OUT_DIR, "comparison.pt"))
    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
