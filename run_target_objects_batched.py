#!/usr/bin/env python3
"""Multi-batch sampling variant: run pipeline N times per object, pool results.

Instead of one 20k-env batch (memory heavy), run N batches of 4000 envs with
independent random seeds and pool the feasible grasps. Each batch re-samples
the base pose init distribution so we get better coverage.
"""
import os, sys, json, numpy as np, trimesh, torch, time, argparse, contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from frogger.batched_pytorch_solver import BatchedSDF, BatchedGraspOptimizer

# Asset paths resolved relative to the repo location. The third_parties/frogger
# folder lives at <DEXFUN>/third_parties/frogger, so assets/ is two levels up.
_DEXFUN_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
MESH_BASE = os.path.join(_DEXFUN_ROOT, "assets", "mesh_obj")
MESH_ALT = os.path.join(_DEXFUN_ROOT, "output", "meshes", "mesh_raw_ahg")
ACT_BASE = os.path.join(_DEXFUN_ROOT, "assets", "actuation_contacts")
TARGET_OBJECTS = [
    "funky_clear_spray_bottle", "hot_glue_gun", "air_blower",
    "grinder", "flashlight",
]


def find_mesh(name):
    p1 = os.path.join(MESH_ALT, name, "object.obj")
    p2 = os.path.join(MESH_BASE, name, "object.obj")
    return p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else None)


def run_batch(name, mesh_path, actuation_targets, X_WO, obj_center, offset,
              batch_i, num_envs, out_dir, seed_offset=0):
    mesh = trimesh.load(mesh_path, force="mesh")
    sdf = BatchedSDF(mesh, X_WO, bounds_padding=0.15, resolution=128, device="cuda")
    sdf.add_clearance_volume(actuation_targets[0][0], actuation_targets[0][1],
                             radius=0.020, height=0.03)
    sdf.add_floor(0.0)

    # Reseed so each batch samples independently. seed_offset enables cross-seed
    # ablation reruns ("Consistent" evidence per CLAUDE.md requires independent
    # draws, not just deterministic reproduction).
    seed = 1000 * batch_i + 7 + seed_offset
    torch.manual_seed(seed)
    np.random.seed(seed)

    opt = BatchedGraspOptimizer(
        sdf, num_envs=num_envs, device="cuda",
        hand="rh", hand_type="leap", palm_contact=True,
    )
    batch_out = os.path.join(out_dir, name, f"batch_{batch_i}")
    os.makedirs(batch_out, exist_ok=True)
    results = opt.optimize(
        actuation_targets=actuation_targets,
        object_center=obj_center,
        steps=300, lr=0.005,
        save_path=os.path.join(batch_out, "grasps.pt"),
        opt_sections="ABCD",
        opt_variant="P",
    )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=4000)
    ap.add_argument("--n-batches", type=int, default=5)
    ap.add_argument("--out-dir", default="output/batched_sampling")
    ap.add_argument("--objects", nargs="*", default=TARGET_OBJECTS)
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="Add this offset to the per-batch seed. Use a "
                         "different value to get cross-seed independent runs "
                         "for 'Consistent' evidence per CLAUDE.md.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {}
    for name in args.objects:
        mesh_path = find_mesh(name)
        act_path = os.path.join(ACT_BASE, f"{name}_actuation.json")
        if mesh_path is None or not os.path.exists(act_path):
            print(f"SKIP {name}")
            continue

        mesh = trimesh.load(mesh_path, force="mesh")
        bounds = mesh.bounds
        offset = np.array([0.0, 0.0, -bounds[0, 2]])
        X_WO = np.eye(4); X_WO[:3, 3] = offset
        obj_center = mesh.centroid + offset
        with open(act_path) as f:
            c = json.load(f)["actuation_contacts"][0]
        actuation_targets = [(np.array(c["pos"]) + offset, np.array(c["dir"]))]

        all_results = []
        n_feas_per_batch = []
        t0 = time.time()
        # Capture per-object stdout (entry-filter breakdown etc.) alongside grasps.
        obj_dir = os.path.join(args.out_dir, name)
        os.makedirs(obj_dir, exist_ok=True)
        log_path = os.path.join(obj_dir, "run.log")
        log_f = open(log_path, "w", buffering=1)

        class _Tee:
            def __init__(self, *streams): self.streams = streams
            def write(self, s):
                for st in self.streams: st.write(s)
            def flush(self):
                for st in self.streams: st.flush()
        tee = _Tee(sys.stdout, log_f)

        for bi in range(args.n_batches):
            print(f"\n{'='*60}\n  {name} — batch {bi+1}/{args.n_batches}\n{'='*60}")
            try:
                with contextlib.redirect_stdout(tee):
                    r = run_batch(name, mesh_path, actuation_targets, X_WO, obj_center,
                                  offset, bi, args.num_envs, args.out_dir,
                                  seed_offset=args.seed_offset)
                for g in r:
                    g["batch_idx"] = bi
                    all_results.append(g)
                n_feas = sum(1 for g in r if g.get("feasible", False))
                n_feas_per_batch.append(n_feas)
                print(f"  batch {bi}: {n_feas} feasible")
            except Exception as e:
                print(f"  batch {bi} ERROR: {e}")
                n_feas_per_batch.append(0)
                continue
        log_f.close()

        elapsed = time.time() - t0
        n_feas_total = sum(1 for g in all_results if g.get("feasible", False))
        summary[name] = {
            "n_batches": args.n_batches, "n_total": len(all_results),
            "n_feas_per_batch": n_feas_per_batch,
            "n_feas_total": n_feas_total,
            "elapsed_s": elapsed,
        }
        # Save pooled grasps
        pool_out = os.path.join(args.out_dir, name, "grasps_pooled.pt")
        torch.save(all_results, pool_out)
        print(f"\n  {name}: {n_feas_total} feasible across {args.n_batches} batches "
              f"({n_feas_per_batch}), {elapsed:.0f}s")

    print(f"\n{'='*80}\n  SUMMARY ({args.num_envs} envs × {args.n_batches} batches)\n{'='*80}")
    for name, s in summary.items():
        print(f"  {name:<30} total_feas={s['n_feas_total']:<4} "
              f"per_batch={s['n_feas_per_batch']} time={s['elapsed_s']:.0f}s")
    torch.save(summary, os.path.join(args.out_dir, "summary.pt"))


if __name__ == "__main__":
    main()
