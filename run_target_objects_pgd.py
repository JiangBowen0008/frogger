#!/usr/bin/env python3
"""Run pipeline with projected gradient descent (variant P) on 5 target objects.

Compares against the previously run variant A results (saved to output/grasps_target/).
Variant P: unfreezes CMC, projects fingertips to SDF=0 after each Adam step.
"""
import os, sys, json, numpy as np, trimesh, torch, time

sys.path.insert(0, os.path.dirname(__file__))
from frogger.batched_pytorch_solver import BatchedSDF, BatchedGraspOptimizer

MESH_BASE = "/home/bowenj/Projects/DexFun/assets/mesh_obj"
MESH_ALT = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg"
ACT_BASE = "/home/bowenj/Projects/DexFun/assets/actuation_contacts"
OUT_DIR = "output/grasps_target_pgd"
NUM_ENVS = 4000

TARGET_OBJECTS = [
    "funky_clear_spray_bottle",
    "hot_glue_gun",
    "air_blower",
    "grinder",
    "flashlight",
]


def find_mesh(name):
    p1 = os.path.join(MESH_ALT, name, "object.obj")
    p2 = os.path.join(MESH_BASE, name, "object.obj")
    return p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else None)


def run_object(name):
    mesh_path = find_mesh(name)
    act_path = os.path.join(ACT_BASE, f"{name}_actuation.json")
    if mesh_path is None or not os.path.exists(act_path):
        print(f"  SKIP {name}")
        return None

    print(f"\n{'='*60}\n  {name}\n{'='*60}")

    mesh = trimesh.load(mesh_path, force="mesh")
    bounds = mesh.bounds
    offset = np.array([0.0, 0.0, -bounds[0, 2]])
    X_WO = np.eye(4); X_WO[:3, 3] = offset
    obj_center = mesh.centroid + offset

    with open(act_path) as f:
        c = json.load(f)["actuation_contacts"][0]
    actuation_targets = [(np.array(c["pos"]) + offset, np.array(c["dir"]))]

    obj_out = os.path.join(OUT_DIR, name)
    os.makedirs(obj_out, exist_ok=True)
    save_file = os.path.join(obj_out, "grasps.pt")

    try:
        sdf = BatchedSDF(mesh, X_WO, bounds_padding=0.15, resolution=128, device="cuda")
        sdf.add_clearance_volume(actuation_targets[0][0], actuation_targets[0][1],
                                 radius=0.020, height=0.03)
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
            opt_variant="P",
        )
        elapsed = time.time() - t0

        stage_dir = os.path.join(os.path.dirname(__file__), "output/grasps")
        for stage_name in ["stage_after_init.pt", "stage_after_support_ik.pt", "stage_after_optimization.pt"]:
            src = os.path.join(stage_dir, stage_name)
            if os.path.exists(src):
                import shutil
                shutil.copy2(src, os.path.join(obj_out, stage_name))

        torch.save({
            "mesh_path": mesh_path, "act_path": act_path,
            "offset": offset, "obj_center": obj_center,
        }, os.path.join(obj_out, "meta.pt"))

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
        metrics["sigma_best"] = max(r["sigma_min"] for r in results)
        metrics["lstar_best"] = max(r["l_star"] for r in results)
        metrics["surf_best_mm"] = min(r["surf_err"] for r in results) * 1000
        metrics["pen_best"] = min(r.get("mesh_pen_pct", 100) for r in results)
        metrics["n_lstar_pos"] = sum(1 for r in results if r["l_star"] > 0)
    return metrics


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_metrics = []
    for name in TARGET_OBJECTS:
        m = run_object(name)
        if m is not None:
            all_metrics.append(m)

    print(f"\n{'='*90}\n  VARIANT P (PGD) RESULTS ({NUM_ENVS} envs)\n{'='*90}")
    header = f"{'Object':<30} {'Feas':<6} {'l*>0':<6} {'sigma':<8} {'l*':<9} {'surf_mm':<8} {'pen_%':<8} {'time':<6}"
    print(header)
    print("-" * len(header))
    for m in all_metrics:
        if "error" in m:
            print(f"{m['object']:<30} ERROR: {m['error'][:40]}")
            continue
        print(f"{m['object']:<30} "
              f"{m['n_feasible']:<6} "
              f"{m.get('n_lstar_pos', 0):<6} "
              f"{m.get('sigma_best', 0):<8.4f} "
              f"{m.get('lstar_best', -1):<9.4f} "
              f"{m.get('surf_best_mm', 999):<8.1f} "
              f"{m.get('pen_best', 100):<8.1f} "
              f"{m.get('time_s', 0):<6.0f}s")

    torch.save(all_metrics, os.path.join(OUT_DIR, "comparison.pt"))
    print(f"\nResults in {OUT_DIR}/")


if __name__ == "__main__":
    main()
