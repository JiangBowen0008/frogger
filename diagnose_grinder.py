"""Diagnostic script for grinder grasp pipeline analysis."""
import torch
import numpy as np

BASE = "/home/bowenj/Projects/DexFun/third_parties/frogger/output/grasps_target/grinder"

# ============================================================
# 1. Pipeline yield at each stage
# ============================================================
print("=" * 70)
print("  STAGE-BY-STAGE PIPELINE YIELD")
print("=" * 70)

for stage_name in ["stage_after_init", "stage_after_support_ik", "stage_after_optimization", "grasps", "meta"]:
    fpath = f"{BASE}/{stage_name}.pt"
    try:
        data = torch.load(fpath, map_location="cpu", weights_only=False)
        if isinstance(data, list):
            print(f"  {stage_name}: {len(data)} entries (list)")
        elif isinstance(data, dict):
            print(f"  {stage_name}: dict with keys={list(data.keys())[:10]}")
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: shape={v.shape} dtype={v.dtype}")
                elif isinstance(v, np.ndarray):
                    print(f"    {k}: ndarray shape={v.shape}")
                elif isinstance(v, (int, float, str, bool)):
                    print(f"    {k}: {v}")
                elif isinstance(v, list):
                    print(f"    {k}: list len={len(v)}")
        elif isinstance(data, torch.Tensor):
            print(f"  {stage_name}: tensor shape={data.shape}")
        else:
            print(f"  {stage_name}: type={type(data)}")
    except Exception as e:
        print(f"  {stage_name}: LOAD ERROR: {e}")

# ============================================================
# 2. Analyze final grasps — why do they fail?
# ============================================================
print("\n" + "=" * 70)
print("  GRASP-BY-GRASP FAILURE ANALYSIS")
print("=" * 70)

grasps = torch.load(f"{BASE}/grasps.pt", map_location="cpu", weights_only=False)
print(f"  Total grasps in output: {len(grasps)}")

# Feasibility thresholds from solver code
THRESHOLDS = {
    "surf_err": ("< 8mm", lambda v: v < 0.008),
    "min_col": ("> -3mm (margin)", lambda v: True),  # already encoded in max_col_viol
    "sc_min_dist": ("> 1mm", lambda v: v > 0.001),
    "sigma_min": ("> 0.01", lambda v: v > 0.01),
    "act_dist": ("< 10mm", lambda v: v < 0.010),
    "mesh_pen_pct": ("< 5%", lambda v: v < 5.0),
    "sc_worst": ("> 0.5mm", lambda v: v > 0.0005),
    "l_star": ("> 0 (FC)", lambda v: v > 0),
}

finger_names = {0: "IF", 1: "MF", 2: "RF", 3: "TH"}

for i, g in enumerate(grasps):
    feasible = g.get("feasible", False)
    tag = "FEAS" if feasible else "FAIL"
    print(f"\n  --- G{i} [{tag}] env_idx={g.get('env_idx', '?')} ---")

    # Print all metrics
    fails = []
    for metric, (desc, check_fn) in THRESHOLDS.items():
        val = g.get(metric, None)
        if val is None:
            continue
        if metric in ["surf_err", "act_dist", "min_col"]:
            val_str = f"{val*1000:.1f}mm"
        elif metric == "mesh_pen_pct":
            val_str = f"{val:.1f}%"
        elif metric in ["sc_min_dist", "sc_worst"]:
            val_str = f"{val*1000:.2f}mm"
        elif metric == "sigma_min":
            val_str = f"{val:.4f}"
        elif metric == "l_star":
            val_str = f"{val:.4f}"
        else:
            val_str = f"{val}"

        passed = check_fn(val)
        status = "OK" if passed else "FAIL"
        print(f"    {metric:15s} = {val_str:>10s}  threshold {desc:>15s}  [{status}]")
        if not passed:
            fails.append(metric)

    # Finger assignment
    act_fi = g.get("act_finger", None)
    if act_fi is not None:
        print(f"    act_finger     = {act_fi} ({finger_names.get(act_fi, '?')})")

    # Summary
    if fails:
        print(f"    >>> FAILED ON: {', '.join(fails)}")
    else:
        print(f"    >>> ALL CHECKS PASSED")

# ============================================================
# 3. Actuation finger distribution
# ============================================================
print("\n" + "=" * 70)
print("  ACTUATION FINGER ASSIGNMENT")
print("=" * 70)

act_fingers = [g.get("act_finger", -1) for g in grasps]
for fi in range(4):
    count = sum(1 for f in act_fingers if f == fi)
    print(f"  {finger_names.get(fi, '?')} (finger {fi}): {count} grasps")

# The grinder actuation is at top, pressing -z
# For b % 4 assignment with 4000 envs: 1000 of each finger
print(f"\n  With b%4 assignment over 4000 envs:")
for fi in range(4):
    print(f"    {finger_names[fi]}: envs {fi}, {fi+4}, {fi+8}, ... (1000 total)")

# ============================================================
# 4. Palm placement and fingertip positions
# ============================================================
print("\n" + "=" * 70)
print("  PALM & FINGERTIP PLACEMENT")
print("=" * 70)

for i, g in enumerate(grasps):
    bp = g.get("base_pos", None)
    br = g.get("base_rot", None)
    if bp is not None:
        if isinstance(bp, torch.Tensor):
            bp = bp.numpy()
        if isinstance(br, torch.Tensor):
            br = br.numpy()

        palm_inward = br[:, 0] if br is not None else None
        print(f"\n  G{i}: base_pos = [{bp[0]*1000:.1f}, {bp[1]*1000:.1f}, {bp[2]*1000:.1f}] mm")
        if palm_inward is not None:
            print(f"        palm_inward (+x) = [{palm_inward[0]:.3f}, {palm_inward[1]:.3f}, {palm_inward[2]:.3f}]")

        # Distance from cylinder axis (assuming centered at x=0, y=0)
        lateral_dist = np.sqrt(bp[0]**2 + bp[1]**2) * 1000
        print(f"        lateral dist from z-axis = {lateral_dist:.1f}mm")
        print(f"        height (z) = {bp[2]*1000:.1f}mm")

    # Fingertip positions from q_joints
    q = g.get("q_joints", None)
    if q is not None:
        if isinstance(q, torch.Tensor):
            q = q.numpy()
        print(f"        q_joints = [{', '.join(f'{v:.2f}' for v in q)}]")

# ============================================================
# 5. Surface error distribution — the main killer
# ============================================================
print("\n" + "=" * 70)
print("  SURFACE ERROR ANALYSIS (main bottleneck)")
print("=" * 70)

surf_errs = [g.get("surf_err", 999) for g in grasps]
l_stars = [g.get("l_star", -1) for g in grasps]
sc_worsts = [g.get("sc_worst", 999) for g in grasps]
mesh_pens = [g.get("mesh_pen_pct", 0) for g in grasps]

print(f"  Surface errors: {[f'{s*1000:.1f}mm' for s in surf_errs]}")
print(f"  l* values:      {[f'{l:.4f}' for l in l_stars]}")
print(f"  SC worst:       {[f'{s*1000:.2f}mm' for s in sc_worsts]}")
print(f"  Mesh pen %:     {[f'{p:.1f}%' for p in mesh_pens]}")

n_surf_fail = sum(1 for s in surf_errs if s >= 0.008)
n_lstar_fail = sum(1 for l in l_stars if l <= 0)
n_sc_fail = sum(1 for s in sc_worsts if s < 0.0005)
n_pen_fail = sum(1 for p in mesh_pens if p >= 5.0)

print(f"\n  Failure counts (of {len(grasps)}):")
print(f"    surf_err >= 8mm:    {n_surf_fail}")
print(f"    l* <= 0:            {n_lstar_fail}")
print(f"    sc_worst < 0.5mm:   {n_sc_fail}")
print(f"    mesh_pen >= 5%:     {n_pen_fail}")

# ============================================================
# 6. Load stage files for deeper analysis
# ============================================================
print("\n" + "=" * 70)
print("  STAGE FILE ANALYSIS")
print("=" * 70)

for stage_name in ["stage_after_init", "stage_after_support_ik", "stage_after_optimization"]:
    fpath = f"{BASE}/{stage_name}.pt"
    try:
        data = torch.load(fpath, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            print(f"\n  {stage_name}:")
            for k, v in sorted(data.items()):
                if isinstance(v, torch.Tensor):
                    if v.numel() < 20:
                        print(f"    {k}: {v}")
                    else:
                        print(f"    {k}: shape={v.shape} min={v.min():.4f} max={v.max():.4f} mean={v.mean():.4f}")
                elif isinstance(v, np.ndarray):
                    if v.size < 20:
                        print(f"    {k}: {v}")
                    else:
                        print(f"    {k}: ndarray shape={v.shape} min={v.min():.4f} max={v.max():.4f}")
                elif isinstance(v, (list, tuple)):
                    print(f"    {k}: len={len(v)}")
                else:
                    print(f"    {k}: {v}")
    except Exception as e:
        print(f"  {stage_name}: ERROR: {e}")

# ============================================================
# 7. Why is l* = -1 for most grasps?
# ============================================================
print("\n" + "=" * 70)
print("  FORCE CLOSURE ANALYSIS")
print("=" * 70)

for i, g in enumerate(grasps):
    ls = g.get("l_star", -1)
    sm = g.get("sigma_min", 0)
    se = g.get("surf_err", 999)
    print(f"  G{i}: l*={ls:.4f}  sigma_min={sm:.4f}  surf={se*1000:.1f}mm  "
          f"{'FC' if ls > 0 else 'NO FC'}  "
          f"{'on surface' if se < 0.008 else 'OFF surface'}")

print("\n  Key insight: l* = -1 means LP solver returned infeasible.")
print("  If sigma_min > 0.01 but l* = -1, likely due to:")
print("    - Contact normals not opposing (all on same side)")
print("    - Support fingers clustered (not wrapping)")
print("    - Palm not contributing effective wrench")
