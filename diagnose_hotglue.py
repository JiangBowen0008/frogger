"""Hot_glue_gun per-env failure-mode analysis.

Runs pipeline through stage 4 on 4000 envs of hot_glue_gun. For each env
records: act_dist, act_dir_align, act_link_worst, palm_worst. Then tabulates
how many envs fail each filter criterion and the joint distribution of
failures (e.g. envs that pass act_dist but fail palm).
"""
import os, sys, json, numpy as np, trimesh, torch

sys.path.insert(0, "/home/bowenj/Projects/DexFun/third_parties/frogger")
from frogger.batched_pytorch_solver import BatchedSDF, BatchedGraspOptimizer

MESH_BASE = "/home/bowenj/Projects/DexFun/assets/mesh_obj"
MESH_ALT = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg"
ACT_BASE = "/home/bowenj/Projects/DexFun/assets/actuation_contacts"
OBJ = "hot_glue_gun"
NUM_ENVS = 4000


def find_mesh(name):
    p1 = os.path.join(MESH_ALT, name, "object.obj")
    p2 = os.path.join(MESH_BASE, name, "object.obj")
    return p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else None)


def compute_stage4_metrics(opt, ap, ad):
    """After stage 2+2b, compute per-env act/palm/dir/link metrics."""
    B, dev = opt.u.shape[0], opt.u.device
    with torch.no_grad():
        q = opt._u2q(opt.u)
        bT = opt._base_T(opt.pos, opt.rot6d)
        fk = opt.chain.forward_kinematics(q)

        act_dists = torch.zeros(B, device=dev)
        act_dir_align = torch.ones(B, device=dev)
        act_link_worst = torch.zeros(B, device=dev)
        prefixes = ['if', 'mf', 'rf', 'th']
        sfx = [['bs', 'px', 'md']] * 3 + [['mp', 'bs', 'px']]

        for b_fi in range(4):
            mask_fi = (opt.amap_t[:, 0] == b_fi)
            if not mask_fi.any(): continue
            nm = opt.tip_link_names[b_fi]
            wT = bT @ fk[nm].get_matrix()
            off_h = torch.cat([opt.tip_offsets[b_fi], torch.ones(1, device=dev)])
            tp = (wT @ off_h.unsqueeze(-1)).squeeze(-1)[:, :3]
            act_dists += mask_fi.float() * torch.norm(tp - ap[0], dim=-1)
            if ad is not None and ad[0] is not None:
                pad_dir = -wT[:, :3, 0]
                cos_a = (pad_dir * ad[0].unsqueeze(0)).sum(-1)
                act_dir_align = torch.where(mask_fi, cos_a, act_dir_align)
            for suf in sfx[b_fi]:
                lnm = f"leap_rh_{prefixes[b_fi]}_{suf}"
                if lnm not in fk: continue
                for cnm, lp in opt._col_data:
                    if cnm != lnm: continue
                    lwT = bT @ fk[lnm].get_matrix()
                    lwp = (lwT @ lp.T)[:, :3, :].transpose(1, 2)
                    lmin = opt.sdf.query(lwp, include_clearance=False).min(-1).values
                    act_link_worst = torch.where(
                        mask_fi & (lmin < act_link_worst), lmin, act_link_worst)

        palm_worst = torch.zeros(B, device=dev)
        for cnm, lp in opt._col_data:
            if "palm" not in cnm or cnm not in fk: continue
            lwT = bT @ fk[cnm].get_matrix()
            lwp = (lwT @ lp.T)[:, :3, :].transpose(1, 2)
            # Query WITH clearance (palm should not enter either)
            palm_worst = torch.minimum(palm_worst, opt.sdf.query(lwp).min(-1).values)

    return {
        'act_dist': act_dists.cpu().numpy(),
        'act_dir_align': act_dir_align.cpu().numpy(),
        'act_link_worst': act_link_worst.cpu().numpy(),
        'palm_worst': palm_worst.cpu().numpy(),
        'amap': opt.amap[:, 0].copy(),
    }


def main():
    mesh_path = find_mesh(OBJ)
    mesh = trimesh.load(mesh_path, force="mesh")
    bounds = mesh.bounds
    offset = np.array([0.0, 0.0, -bounds[0, 2]])
    X_WO = np.eye(4); X_WO[:3, 3] = offset
    obj_center = mesh.centroid + offset
    with open(f"{ACT_BASE}/{OBJ}_actuation.json") as f:
        c = json.load(f)["actuation_contacts"][0]
    actuation_targets = [(np.array(c["pos"]) + offset, np.array(c["dir"]))]

    sdf = BatchedSDF(mesh, X_WO, bounds_padding=0.15, resolution=128, device="cuda")
    sdf.add_clearance_volume(actuation_targets[0][0], actuation_targets[0][1],
                             radius=0.020, height=0.03)
    sdf.add_floor(0.0)
    opt = BatchedGraspOptimizer(sdf, num_envs=NUM_ENVS, device="cuda",
                                hand="rh", hand_type="leap", palm_contact=True)

    # Drive pipeline up to stage 2+2b only. Easiest: call .optimize() with
    # very few steps and capture state via a hook — but our API runs full
    # pipeline. For a diagnostic just do the full optimize call but stop
    # early by using only 1 opt step. Stage 2 IK and palm slide happen
    # inside optimize() before the main opt.
    # Simpler: call internal builder functions.
    # EASIEST: just run full optimize with save_path=None and capture the
    # metrics from saved stage files.

    # Instead: import _get_points and related. But simplest is to patch:
    # run a mini-optimize that stops after stage 2b. The cleanest path:
    # monkey-patch the main opt to be a no-op after capturing state.
    import frogger.batched_pytorch_solver as bps
    import torch.nn.functional as F
    dev = "cuda"
    n_act = 1

    # Initialize palm pose, etc. (mimic optimize() header)
    ap = torch.stack([torch.tensor(t[0], dtype=torch.float32, device=dev)
                      for t in actuation_targets])
    ad = []
    for t in actuation_targets:
        d = torch.tensor(t[1], dtype=torch.float32, device=dev)
        ad.append(F.normalize(d, dim=0))

    # Directly call the init + IK pipeline by running optimize() with steps=1
    # and capturing state after stage 4.
    # Patch: replace main opt loop with early exit.
    results = opt.optimize(
        actuation_targets=actuation_targets,
        object_center=obj_center,
        steps=1, lr=0.005,
        save_path=None,
        opt_sections="", opt_variant="P",
    )

    # Now opt.u reflects state after whatever ran. Compute stage-4 metrics.
    m = compute_stage4_metrics(opt, ap, ad)

    # Tabulate
    B = NUM_ENVS
    act = m['act_dist']
    dir_a = m['act_dir_align']
    alink = m['act_link_worst']
    palm = m['palm_worst']
    amap = m['amap']

    pos_ok = act < 0.010
    dir_ok = dir_a > 0.80
    acol_ok = alink > -0.003
    palm_ok = palm > -0.003

    print(f"\n=== {OBJ} per-env filter stats (B={B}) ===")
    print(f"  act_dist <10mm:       {pos_ok.sum():>4}/{B} ({pos_ok.mean()*100:.1f}%)")
    print(f"  act_dir_align>0.80:   {dir_ok.sum():>4}/{B} ({dir_ok.mean()*100:.1f}%)")
    print(f"  act_link>-3mm:        {acol_ok.sum():>4}/{B} ({acol_ok.mean()*100:.1f}%)")
    print(f"  palm>-3mm:            {palm_ok.sum():>4}/{B} ({palm_ok.mean()*100:.1f}%)")
    print(f"\n  All pass:             {(pos_ok & dir_ok & acol_ok & palm_ok).sum()}/{B}")

    print(f"\n=== Joint failure modes (envs passing prior stages) ===")
    n = pos_ok.sum()
    print(f"  pos_ok:               {n}/{B}")
    n2 = (pos_ok & dir_ok).sum()
    print(f"  pos+dir ok:           {n2}  (of {n}, lost {n-n2} to direction)")
    n3 = (pos_ok & dir_ok & acol_ok).sum()
    print(f"  pos+dir+actcol ok:    {n3}  (lost {n2-n3} to act-link collision)")
    n4 = (pos_ok & dir_ok & acol_ok & palm_ok).sum()
    print(f"  all pass:             {n4}  (lost {n3-n4} to palm)")

    print(f"\n=== Distributions (of pos_ok envs, n={n}) ===")
    if n > 0:
        mask = pos_ok
        print(f"  act_dir_align:  median={np.median(dir_a[mask]):.3f} "
              f"p10={np.percentile(dir_a[mask],10):.3f} p90={np.percentile(dir_a[mask],90):.3f}")
        print(f"  act_link (mm):  median={np.median(alink[mask])*1000:.1f} "
              f"p10={np.percentile(alink[mask],10)*1000:.1f} p90={np.percentile(alink[mask],90)*1000:.1f}")
        print(f"  palm_worst(mm): median={np.median(palm[mask])*1000:.1f} "
              f"p10={np.percentile(palm[mask],10)*1000:.1f} p90={np.percentile(palm[mask],90)*1000:.1f}")

    print(f"\n=== By chosen amap (finger assignment) ===")
    fingers = ['if', 'mf', 'rf', 'th']
    for fi in range(4):
        sel = amap == fi
        n_sel = sel.sum()
        if n_sel == 0: continue
        pos_sel = (sel & pos_ok).sum()
        all_sel = (sel & pos_ok & dir_ok & acol_ok & palm_ok).sum()
        med_act = np.median(act[sel]) * 1000
        print(f"  {fingers[fi]}: {n_sel:>4} envs, med act_dist={med_act:.0f}mm, "
              f"pos_ok={pos_sel}, all-pass={all_sel}")

    torch.save(m, "output/grasps/hotglue_diag.pt")
    print("\nSaved per-env metrics to output/grasps/hotglue_diag.pt")


if __name__ == "__main__":
    main()
