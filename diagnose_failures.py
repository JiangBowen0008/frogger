#!/usr/bin/env python3
"""Comprehensive diagnosis of why hot_glue_gun and grinder produce 0 feasible grasps."""

import torch
import numpy as np
import os

BASE = "/home/bowenj/Projects/DexFun/third_parties/frogger/output/grasps_target"
ALL_OBJECTS = ["flashlight", "air_blower", "funky_clear_spray_bottle", "hot_glue_gun", "grinder"]

def load_safe(path):
    if os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=False)
    return None

print("=" * 80)
print("PART 1: Per-grasp feasibility analysis for ALL objects")
print("=" * 80)

for obj in ALL_OBJECTS:
    grasps_path = os.path.join(BASE, obj, "grasps.pt")
    grasps = load_safe(grasps_path)
    if grasps is None:
        print(f"\n--- {obj}: NO grasps.pt found ---")
        continue

    print(f"\n{'='*60}")
    print(f"  {obj}: {len(grasps)} saved grasps")
    print(f"{'='*60}")

    if len(grasps) == 0:
        print("  (empty)")
        continue

    # Thresholds
    T_SURF = 0.008      # 8mm
    T_COL = 0.003       # 3mm (max col violation)
    T_SC = 0.001         # 1mm self-collision
    T_SIGMA = 0.01       # force closure
    T_ACT = 0.010        # 10mm actuation distance
    T_PEN = 5.0          # 5% mesh penetration
    T_SC_WORST = 0.0005  # 0.5mm SC worst

    for i, g in enumerate(grasps):
        surf = g.get("surf_err", 999)
        sigma = g.get("sigma_min", 0)
        sc_d = g.get("sc_min_dist", 999)
        act_d = g.get("act_dist", 0)
        l_star = g.get("l_star", -1)
        feas = g.get("feasible", False)
        pen_pct = g.get("mesh_pen_pct", 0)
        sc_worst = g.get("sc_worst", 999)
        min_col = g.get("min_col", 999)
        act_finger = g.get("act_finger", -1)
        env_idx = g.get("env_idx", -1)

        # Check each criterion
        fails = []
        if surf >= T_SURF:
            fails.append(f"SURF={surf*1000:.1f}mm>8mm")
        # min_col is the minimum SDF across all collision points; negative = inside
        # max_col_viol isn't saved directly, but we can check min_col
        if sigma < T_SIGMA:
            fails.append(f"SIGMA={sigma:.4f}<0.01")
        if sc_d < T_SC:
            fails.append(f"SC_MIN={sc_d*1000:.1f}mm<1mm")
        if act_d >= T_ACT:
            fails.append(f"ACT={act_d*1000:.1f}mm>10mm")
        if pen_pct > T_PEN:
            fails.append(f"PEN={pen_pct:.1f}%>5%")
        if sc_worst < T_SC_WORST:
            fails.append(f"SC_WORST={sc_worst*1000:.2f}mm<0.5mm")
        if l_star <= 0:
            fails.append(f"l*={l_star:.4f}<=0")

        status = "FEAS" if feas else "FAIL"
        fail_str = " | ".join(fails) if fails else "ALL PASS"
        print(f"  G{i} [{status}] env={env_idx} act_fi={act_finger} "
              f"surf={surf*1000:.1f}mm σ={sigma:.4f} l*={l_star:.4f} "
              f"act={act_d*1000:.1f}mm pen={pen_pct:.1f}% sc={sc_worst*1000:.1f}mm")
        print(f"        Failures: {fail_str}")

print("\n")
print("=" * 80)
print("PART 2: Actuation finger distribution")
print("=" * 80)

FINGER_NAMES = {0: "IF", 1: "MF", 2: "RF", 3: "TH"}

for obj in ALL_OBJECTS:
    grasps = load_safe(os.path.join(BASE, obj, "grasps.pt"))
    if not grasps:
        continue
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    feas_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for g in grasps:
        fi = g.get("act_finger", -1)
        if fi in counts:
            counts[fi] += 1
            if g.get("feasible", False):
                feas_counts[fi] += 1

    dist_str = " ".join([f"{FINGER_NAMES[fi]}={counts[fi]}({feas_counts[fi]}feas)" for fi in range(4)])
    print(f"  {obj}: {dist_str}")

print("\n")
print("=" * 80)
print("PART 3: Support IK quality (stage_after_support_ik.pt)")
print("=" * 80)

for obj in ALL_OBJECTS:
    stage = load_safe(os.path.join(BASE, obj, "stage_after_support_ik.pt"))
    if stage is None:
        print(f"\n  {obj}: no stage file")
        continue

    print(f"\n  --- {obj}: {len(stage)} grasps in support IK stage ---")

    if len(stage) == 0:
        print("    (empty)")
        continue

    for i, g in enumerate(stage):
        # Stage files have: q_joints, base_pos, base_rot, sigma_min, l_star, feasible
        # They may also have surf_pt, outward_normal, act_finger, env_idx
        sigma = g.get("sigma_min", 0)
        act_fi = g.get("act_finger", -1)
        env_idx = g.get("env_idx", -1)
        print(f"    G{i}: act_fi={act_fi} env={env_idx} σ={sigma:.4f}")

print("\n")
print("=" * 80)
print("PART 4: Meta information")
print("=" * 80)

for obj in ALL_OBJECTS:
    meta = load_safe(os.path.join(BASE, obj, "meta.pt"))
    if meta is None:
        print(f"  {obj}: no meta.pt")
        continue

    if isinstance(meta, dict):
        print(f"\n  --- {obj} ---")
        for k, v in meta.items():
            if isinstance(v, (int, float, str, bool)):
                print(f"    {k}: {v}")
            elif isinstance(v, np.ndarray):
                print(f"    {k}: ndarray shape={v.shape}")
            elif isinstance(v, torch.Tensor):
                print(f"    {k}: tensor shape={v.shape}")
            elif isinstance(v, list):
                print(f"    {k}: list len={len(v)}")
            else:
                print(f"    {k}: {type(v).__name__}")
    else:
        print(f"  {obj}: meta type={type(meta).__name__}")

print("\n")
print("=" * 80)
print("PART 5: Near-miss analysis — relaxed thresholds")
print("=" * 80)

for obj in ["hot_glue_gun", "grinder"]:
    grasps = load_safe(os.path.join(BASE, obj, "grasps.pt"))
    if not grasps:
        continue

    print(f"\n  --- {obj}: relaxed threshold analysis ---")

    for i, g in enumerate(grasps):
        surf = g.get("surf_err", 999)
        sigma = g.get("sigma_min", 0)
        sc_d = g.get("sc_min_dist", 999)
        act_d = g.get("act_dist", 0)
        l_star = g.get("l_star", -1)
        pen_pct = g.get("mesh_pen_pct", 0)
        sc_worst = g.get("sc_worst", 999)

        # Check with original thresholds
        orig_pass = (surf < 0.008 and sigma >= 0.01 and sc_d >= 0.001
                     and act_d < 0.010 and pen_pct <= 5.0 and sc_worst >= 0.0005)

        # Relaxed: surf 10mm
        relax_surf = (surf < 0.010 and sigma >= 0.01 and sc_d >= 0.001
                      and act_d < 0.010 and pen_pct <= 5.0 and sc_worst >= 0.0005)

        # Relaxed: sc 0.5mm
        relax_sc = (surf < 0.008 and sigma >= 0.01 and sc_d >= 0.0005
                    and act_d < 0.010 and pen_pct <= 5.0 and sc_worst >= 0.0005)

        # Relaxed: act 15mm
        relax_act = (surf < 0.008 and sigma >= 0.01 and sc_d >= 0.001
                     and act_d < 0.015 and pen_pct <= 5.0 and sc_worst >= 0.0005)

        # Relaxed: pen 8%
        relax_pen = (surf < 0.008 and sigma >= 0.01 and sc_d >= 0.001
                     and act_d < 0.010 and pen_pct <= 8.0 and sc_worst >= 0.0005)

        # Relaxed: sigma 0.005
        relax_sigma = (surf < 0.008 and sigma >= 0.005 and sc_d >= 0.001
                       and act_d < 0.010 and pen_pct <= 5.0 and sc_worst >= 0.0005)

        # Relaxed: all slightly
        relax_all = (surf < 0.010 and sigma >= 0.005 and sc_d >= 0.0005
                     and act_d < 0.012 and pen_pct <= 8.0 and sc_worst >= 0.0003)

        print(f"    G{i}: orig={'PASS' if orig_pass else 'fail'} "
              f"surf10={'PASS' if relax_surf else 'fail'} "
              f"sc0.5={'PASS' if relax_sc else 'fail'} "
              f"act15={'PASS' if relax_act else 'fail'} "
              f"pen8={'PASS' if relax_pen else 'fail'} "
              f"sig005={'PASS' if relax_sigma else 'fail'} "
              f"allRelax={'PASS' if relax_all else 'fail'}")

        # Distance to nearest feasible for each criterion
        margins = {}
        margins["surf"] = T_SURF - surf   # positive = passes
        margins["sigma"] = sigma - T_SIGMA
        margins["sc_min"] = sc_d - T_SC
        margins["act"] = T_ACT - act_d
        margins["pen"] = T_PEN - pen_pct
        margins["sc_worst"] = sc_worst - T_SC_WORST

        for k, v in margins.items():
            tag = "OK" if v > 0 else f"MISS by {abs(v):.4f}"
            print(f"        {k}: margin={v:.4f} ({tag})")

print("\n")
print("=" * 80)
print("PART 6: Compare init stages across objects")
print("=" * 80)

for obj in ALL_OBJECTS:
    init_stage = load_safe(os.path.join(BASE, obj, "stage_after_init.pt"))
    ik_stage = load_safe(os.path.join(BASE, obj, "stage_after_support_ik.pt"))
    opt_stage = load_safe(os.path.join(BASE, obj, "stage_after_optimization.pt"))
    final = load_safe(os.path.join(BASE, obj, "grasps.pt"))

    counts = {
        "init": len(init_stage) if init_stage else 0,
        "ik": len(ik_stage) if ik_stage else 0,
        "opt": len(opt_stage) if opt_stage else 0,
        "final": len(final) if final else 0,
    }

    # Count feasible in final
    n_feas = 0
    if final:
        n_feas = sum(1 for g in final if g.get("feasible", False))

    print(f"  {obj}: init={counts['init']} → ik={counts['ik']} → opt={counts['opt']} → final={counts['final']} (feas={n_feas})")

print("\n")
print("=" * 80)
print("PART 7: Detailed comparison — flashlight vs grinder vs hot_glue_gun")
print("=" * 80)

for obj in ["flashlight", "hot_glue_gun", "grinder"]:
    grasps = load_safe(os.path.join(BASE, obj, "grasps.pt"))
    if not grasps:
        continue

    print(f"\n  --- {obj} ---")

    surfs = [g.get("surf_err", 999) for g in grasps]
    sigmas = [g.get("sigma_min", 0) for g in grasps]
    acts = [g.get("act_dist", 0) for g in grasps]
    pens = [g.get("mesh_pen_pct", 0) for g in grasps]
    sc_ws = [g.get("sc_worst", 999) for g in grasps]
    lstars = [g.get("l_star", -1) for g in grasps]

    print(f"    surf_err:  min={min(surfs)*1000:.1f}mm  median={np.median(surfs)*1000:.1f}mm  max={max(surfs)*1000:.1f}mm")
    print(f"    sigma:     min={min(sigmas):.4f}  median={np.median(sigmas):.4f}  max={max(sigmas):.4f}")
    print(f"    act_dist:  min={min(acts)*1000:.1f}mm  median={np.median(acts)*1000:.1f}mm  max={max(acts)*1000:.1f}mm")
    print(f"    pen_pct:   min={min(pens):.1f}%  median={np.median(pens):.1f}%  max={max(pens):.1f}%")
    print(f"    sc_worst:  min={min(sc_ws)*1000:.1f}mm  median={np.median(sc_ws)*1000:.1f}mm")
    print(f"    l_star:    n>0={sum(1 for l in lstars if l > 0)}  min={min(lstars):.4f}  max={max(lstars):.4f}")

    # Count how many pass each individual criterion
    n = len(grasps)
    print(f"    --- Individual criterion pass rates ({n} grasps) ---")
    print(f"    surf<8mm:    {sum(1 for s in surfs if s < 0.008)}/{n}")
    print(f"    sigma>0.01:  {sum(1 for s in sigmas if s >= 0.01)}/{n}")
    print(f"    act<10mm:    {sum(1 for a in acts if a < 0.010)}/{n}")
    print(f"    pen<=5%:     {sum(1 for p in pens if p <= 5.0)}/{n}")
    print(f"    sc_w>0.5mm:  {sum(1 for s in sc_ws if s >= 0.0005)}/{n}")
    print(f"    l*>0:        {sum(1 for l in lstars if l > 0)}/{n}")

print("\n")
print("=" * 80)
print("PART 8: Per-grasp joint analysis for failing objects")
print("=" * 80)

for obj in ["hot_glue_gun", "grinder"]:
    grasps = load_safe(os.path.join(BASE, obj, "grasps.pt"))
    if not grasps:
        continue

    print(f"\n  --- {obj}: joint configurations ---")

    for i, g in enumerate(grasps):
        q = g.get("q_joints", None)
        if q is None:
            continue

        # q is 16 joints: 4 per finger (CMC, MCP, PIP, DIP) for IF, MF, RF, TH
        finger_names = ["IF", "MF", "RF", "TH"]
        joint_names = ["CMC", "MCP", "PIP", "DIP"]
        act_fi = g.get("act_finger", -1)

        print(f"  G{i} (act={finger_names[act_fi] if 0<=act_fi<4 else '?'}):")
        for fi in range(4):
            tag = "ACT" if fi == act_fi else "SUP"
            joints = q[fi*4:(fi+1)*4]
            jstr = " ".join([f"{joint_names[j]}={joints[j]:.2f}" for j in range(4)])
            print(f"    {finger_names[fi]}[{tag}]: {jstr}")

        # Base pose
        bp = g.get("base_pos", None)
        br = g.get("base_rot", None)
        if bp is not None:
            print(f"    Base pos: [{bp[0]:.4f}, {bp[1]:.4f}, {bp[2]:.4f}]")
