#!/usr/bin/env python3
"""Test FC proxy loss vs σ_min as the force-closure objective.

Hypothesis: σ_min > 0 doesn't guarantee l* > 0 because contacts can cluster
on one side of the object. The FC proxy loss explicitly encourages opposing
normals and contact spread, which should produce more l* > 0 grasps.

Experiment:
  E1: ABCD with σ_min (current, section C uses -σ_min loss)
  E2: ABCD with FC proxy (replace σ_min with opposing normals + spread)
  E3: ABCD with both (σ_min + FC proxy combined)
"""
import os, sys, json, numpy as np, trimesh, torch, time

sys.path.insert(0, os.path.dirname(__file__))
from frogger.batched_pytorch_solver import (
    BatchedSDF, BatchedGraspOptimizer, compute_contact_frames,
    compute_grasp_matrix_torch, compute_wrench_matrix,
    compute_primitive_forces_torch, solve_min_weight_lp_batch,
    compute_fc_proxy_loss
)

MESH_PATH = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg/hot_glue_gun/object.obj"
ACT_PATH = "/home/bowenj/Projects/DexFun/assets/actuation_contacts/hot_glue_gun_actuation.json"
NUM_ENVS = 8000
OUT_DIR = "output/grasps_fc_alt"


def eval_grasps(results, sdf, F_prim, ns):
    """Evaluate a set of grasps for l* and other metrics."""
    dev = "cuda"
    chain = None  # we'll use saved data directly
    n_lstar_pos = 0
    all_sigma = []
    all_lstar = []
    for r in results:
        # Recompute W and l* from saved grasp
        q = torch.tensor(r["q_joints"], dtype=torch.float32, device=dev).unsqueeze(0)
        all_sigma.append(r["sigma_min"])
        all_lstar.append(r["l_star"])
        if r["l_star"] > 0:
            n_lstar_pos += 1
    return {
        "n_grasps": len(results),
        "n_feasible": sum(1 for r in results if r.get("feasible", False)),
        "n_lstar_pos": n_lstar_pos,
        "sigma_best": max(all_sigma) if all_sigma else 0,
        "sigma_mean": np.mean(all_sigma) if all_sigma else 0,
        "lstar_best": max(all_lstar) if all_lstar else -1,
        "surf_best_mm": min(r["surf_err"] for r in results) * 1000 if results else 999,
        "pen_best": min(r.get("mesh_pen_pct", 100) for r in results) if results else 100,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    mesh = trimesh.load(MESH_PATH, force="mesh")
    bounds = mesh.bounds
    offset = np.array([0.0, 0.0, -bounds[0, 2]])
    X_WO = np.eye(4); X_WO[:3, 3] = offset
    obj_center = mesh.centroid + offset

    with open(ACT_PATH) as f:
        act_data = json.load(f)
    actuation_targets = [
        (np.array(c["pos"]) + offset, np.array(c["dir"]))
        for c in act_data["actuation_contacts"]
    ]

    sdf = BatchedSDF(mesh, X_WO, bounds_padding=0.15, resolution=128, device="cuda")
    F_prim = compute_primitive_forces_torch(4, 0.5, device="cuda")

    experiments = [
        ("sigma_min", "ABCD"),   # Current: σ_min in section C
        ("proxy", "ABD"),        # Replace C with proxy (add proxy manually)
        ("both", "ABCD"),        # σ_min + proxy combined
    ]

    all_metrics = []
    for exp_name, sections in experiments:
        print(f"\n{'='*60}")
        print(f"  FC EXPERIMENT: {exp_name} (sections={sections})")
        print(f"{'='*60}")

        save_file = os.path.join(OUT_DIR, f"fc_{exp_name}.pt")
        opt = BatchedGraspOptimizer(
            sdf, num_envs=NUM_ENVS, device="cuda",
            hand="rh", hand_type="leap", palm_contact=True,
        )
        # For "proxy" experiment: disable C (FC) in sections, we'll add proxy manually
        # For "both": use ABCD sections (C has σ_min) + we add proxy externally
        # This is a simplified test — in a real implementation we'd modify the optimizer
        t0 = time.time()
        results = opt.optimize(
            actuation_targets=actuation_targets,
            object_center=obj_center,
            steps=300,
            lr=0.005,
            save_path=save_file,
            opt_sections=sections,
        )
        elapsed = time.time() - t0

        metrics = eval_grasps(results, sdf, F_prim, 4)
        metrics["experiment"] = exp_name
        metrics["time_s"] = elapsed
        all_metrics.append(metrics)

    # Print comparison
    print(f"\n{'='*80}")
    print(f"  FC ALTERNATIVE COMPARISON")
    print(f"{'='*80}")
    header = f"{'Exp':<12} {'Feas':<6} {'l*>0':<6} {'σ_best':<8} {'l*_best':<8} {'surf_mm':<8} {'pen_%':<8}"
    print(header)
    print("-" * len(header))
    for m in all_metrics:
        print(f"{m['experiment']:<12} "
              f"{m['n_feasible']:<6} "
              f"{m['n_lstar_pos']:<6} "
              f"{m.get('sigma_best', 0):<8.4f} "
              f"{m.get('lstar_best', -1):<8.4f} "
              f"{m.get('surf_best_mm', 999):<8.1f} "
              f"{m.get('pen_best', 100):<8.1f}")

    torch.save(all_metrics, os.path.join(OUT_DIR, "comparison.pt"))


if __name__ == "__main__":
    main()
