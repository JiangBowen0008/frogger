#!/usr/bin/env python3
"""Per-stage funnel diagnostic.

For each object, run one batch and report what fraction of envs are SC-clean
at each pipeline stage AND survive entry filter. Answers: is the bottleneck
in init (configs already crashed), support IK (crashes during placement),
or main opt (crashes during FC pursuit)?
"""
import os, sys, json, numpy as np, trimesh, torch, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from frogger.batched_pytorch_solver import BatchedSDF, BatchedGraspOptimizer

_DEXFUN_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
MESH_BASE = os.path.join(_DEXFUN_ROOT, "assets", "mesh_obj")
ACT_BASE = os.path.join(_DEXFUN_ROOT, "assets", "actuation_contacts")
OBJECTS = ["air_blower", "hot_glue_gun", "grinder", "funky_clear_spray_bottle", "flashlight"]

NUM_ENVS = 4000
THRESH_SC = -0.001    # 1mm
THRESH_SURF = 0.020   # 20mm — "support tip near surface"
THRESH_COL = 0.003    # 3mm — "object collision"


def analyze(name):
    mesh_path = os.path.join(MESH_BASE, name, "object.obj")
    act_path = os.path.join(ACT_BASE, f"{name}_actuation.json")
    if not (os.path.exists(mesh_path) and os.path.exists(act_path)):
        print(f"SKIP {name}"); return None

    mesh = trimesh.load(mesh_path, force="mesh")
    bounds = mesh.bounds; offset = np.array([0.0, 0.0, -bounds[0, 2]])
    X_WO = np.eye(4); X_WO[:3, 3] = offset
    obj_c = mesh.centroid + offset
    with open(act_path) as f: c = json.load(f)["actuation_contacts"][0]
    act_t = [(np.array(c["pos"]) + offset, np.array(c["dir"]))]

    torch.manual_seed(42); np.random.seed(42)
    sdf = BatchedSDF(mesh, X_WO, bounds_padding=0.15, resolution=128, device="cuda")
    sdf.add_clearance_volume(act_t[0][0], act_t[0][1], radius=0.020, height=0.03); sdf.add_floor(0.0)
    opt = BatchedGraspOptimizer(sdf, num_envs=NUM_ENVS, device="cuda", hand="rh", hand_type="leap", palm_contact=True)
    opt.optimize(actuation_targets=act_t, object_center=obj_c, steps=300, lr=0.005,
                 save_path=None, opt_sections="ABCD", opt_variant="P")

    ml = opt._metrics_log
    stages = ["S1_after_init", "S2_after_act_ik_palm_slide", "S3_after_support_ik",
              "S4_opt_step0", "S4_opt_step51", "S4_opt_step151", "S4_opt_step300"]
    print(f"\n=== {name} ===")
    print(f"  {'stage':<32} sc>-1mm  surf<20  col<3  ALL")
    for s in stages:
        if s not in ml: continue
        d = ml[s]
        sc = d["sc_worst"]; sf = d["surf_err"]; mc = d["max_col_viol"]
        n = sc.shape[0]
        sc_ok = sc > THRESH_SC
        sf_ok = sf < THRESH_SURF
        mc_ok = mc < THRESH_COL
        clean = sc_ok & sf_ok & mc_ok
        print(f"  {s:<32} {sc_ok.sum():>5}    {sf_ok.sum():>5}  {mc_ok.sum():>4}  {clean.sum():>4}  (of {n})")
    return ml


if __name__ == "__main__":
    objs = sys.argv[1:] if len(sys.argv) > 1 else OBJECTS
    for name in objs:
        analyze(name)
