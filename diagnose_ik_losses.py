"""Track pos/dir/col losses independently over IK steps.

Runs the SAME IK loss as the pipeline (pos + dir + col_non_ds) for 300 steps,
logs each component every 10 steps, and reports which term stagnates.
"""
import os, sys, json, numpy as np, trimesh, torch
import torch.nn.functional as F

sys.path.insert(0, "/home/bowenj/Projects/DexFun/third_parties/frogger")
from frogger.batched_pytorch_solver import BatchedSDF, BatchedGraspOptimizer

MESH_BASE = "/home/bowenj/Projects/DexFun/assets/mesh_obj"
ACT_BASE = "/home/bowenj/Projects/DexFun/assets/actuation_contacts"
OBJ = "hot_glue_gun"
NUM_ENVS = 4000
TRIAL_FI = 3  # thumb


def main():
    mesh_path = os.path.join(MESH_BASE, OBJ, "object.obj")
    mesh = trimesh.load(mesh_path, force="mesh")
    bounds = mesh.bounds
    offset = np.array([0.0, 0.0, -bounds[0, 2]])
    X_WO = np.eye(4); X_WO[:3, 3] = offset
    obj_center = mesh.centroid + offset
    with open(f"{ACT_BASE}/{OBJ}_actuation.json") as f:
        c = json.load(f)["actuation_contacts"][0]
    act_pos_np = np.array(c["pos"]) + offset
    act_dir_np = np.array(c["dir"])

    sdf = BatchedSDF(mesh, X_WO, bounds_padding=0.15, resolution=128, device="cuda")
    sdf.add_clearance_volume(act_pos_np, act_dir_np, radius=0.020, height=0.03)
    sdf.add_floor(0.0)
    opt = BatchedGraspOptimizer(sdf, num_envs=NUM_ENVS, device="cuda",
                                hand="rh", hand_type="leap", palm_contact=True)

    # Bootstrap so self.u / self.pos / self.rot6d are initialised by palm-init
    # ONLY (skip stage-2 IK so we can study it from scratch).
    # Easiest: call init-placement phase only. But optimize() runs it
    # internally. So use the state right after __init__ plus sampling.
    # That's already what opt has — self.u is the init u from sampling.
    dev = "cuda"
    act_pos = torch.tensor(act_pos_np, dtype=torch.float32, device=dev)
    act_dir = F.normalize(torch.tensor(act_dir_np, dtype=torch.float32, device=dev), dim=0)

    # We need to call the init / sample_palm_pose phase that the constructor
    # relies on. Calling optimize() with steps=1 runs full stage-2 IK which
    # we want to OBSERVE, not pre-run. So we call it WITH stage 2, then reset.
    # Alternative: call the internal init method directly.
    actuation_targets = [(act_pos_np, act_dir_np)]
    _ = opt.optimize(actuation_targets=actuation_targets, object_center=obj_center,
                     steps=1, lr=0.005, save_path=None,
                     opt_sections="", opt_variant="P")

    # Now u is post-stage2. Reset act-finger joints to whatever the init curl
    # was — we can store u before optimize() if we re-init. Simpler: just
    # start with the current u (which is already past stage 2 IK), and run
    # the pipeline's IK loss on top for 300 steps. This mirrors "what
    # happens if we bump IK to 300 steps" but from a warm start.
    # For a cold-start run we'd need to expose the init state; skipping that
    # for now since the warm-start trajectory is still informative.

    u_act = opt.u.detach().clone().requires_grad_(True)
    opt_ik = torch.optim.Adam([u_act], lr=0.05)
    act_joint_mask = torch.zeros(NUM_ENVS, 16, device=dev, dtype=torch.bool)
    act_joint_mask[:, TRIAL_FI*4:TRIAL_FI*4+4] = True

    nm = opt.tip_link_names[TRIAL_FI]
    off_h = torch.cat([opt.tip_offsets[TRIAL_FI], torch.ones(1, device=dev)])
    prefixes_ik = ['if', 'mf', 'rf', 'th']
    sfx_ik = [['bs', 'px', 'md']] * 3 + [['mp', 'bs', 'px']]

    log = []
    for step in range(300):
        opt_ik.zero_grad()
        q_ik = opt._u2q(u_act)
        bT_ik = opt._base_T(opt.pos.detach(), opt.rot6d.detach())
        fk_ik = opt.chain.forward_kinematics(q_ik)
        wT = bT_ik @ fk_ik[nm].get_matrix()
        tip = (wT @ off_h.unsqueeze(-1)).squeeze(-1)[:, :3]
        pad_dir = -wT[:, :3, 0]

        # New rebalanced forms matching the solver:
        pos_err = torch.norm(tip - act_pos, dim=-1)  # linear
        cos_align = (pad_dir * act_dir).sum(-1)
        bad_face = F.softplus((0.5 - cos_align) * 10) / 10
        dir_err = bad_face ** 2

        col_loss = torch.zeros(NUM_ENVS, device=dev)
        for suf in sfx_ik[TRIAL_FI]:
            lnm = f"leap_rh_{prefixes_ik[TRIAL_FI]}_{suf}"
            if lnm not in fk_ik: continue
            for cnm, lp in opt._col_data:
                if cnm != lnm: continue
                lwT = bT_ik @ fk_ik[lnm].get_matrix()
                lwp = (lwT @ lp.T)[:, :3, :].transpose(1, 2)
                lsdf = opt.sdf.query(lwp, include_clearance=False)
                col_loss += F.relu(-lsdf).mean(-1)  # mean instead of sum

        total = 500 * pos_err + 50 * dir_err + 100 * col_loss
        non_act_reg = ((u_act - opt.u.detach()) ** 2 * (~act_joint_mask).float()).sum(-1)
        total += 100 * non_act_reg

        total.mean().backward()
        with torch.no_grad():
            u_act.grad[~act_joint_mask] = 0.0
        opt_ik.step()

        # Log every 10 steps
        if step % 10 == 0 or step == 299:
            with torch.no_grad():
                dist = torch.norm(tip - act_pos, dim=-1)
                log.append(dict(
                    step=step,
                    pos_loss=(500 * pos_err).mean().item(),
                    dir_loss=(50 * dir_err).mean().item(),
                    col_loss=(100 * col_loss).mean().item(),
                    total=total.mean().item(),
                    dist_med_mm=dist.median().item() * 1000,
                    dist_p10_mm=torch.quantile(dist, 0.1).item() * 1000,
                    dist_p90_mm=torch.quantile(dist, 0.9).item() * 1000,
                    n_under_10=(dist < 0.010).sum().item(),
                ))

    print(f"\n{'step':>5} {'pos_loss':>10} {'dir_loss':>10} {'col_loss':>10} "
          f"{'total':>10} {'dist_p10':>9} {'med_mm':>8} {'dist_p90':>9} {'<10mm':>6}")
    for r in log:
        print(f"{r['step']:>5} {r['pos_loss']:>10.2f} {r['dir_loss']:>10.3f} "
              f"{r['col_loss']:>10.3f} {r['total']:>10.2f} "
              f"{r['dist_p10_mm']:>9.1f} {r['dist_med_mm']:>8.1f} "
              f"{r['dist_p90_mm']:>9.1f} {r['n_under_10']:>6}")

    torch.save(log, "output/grasps/hotglue_loss_traj.pt")


if __name__ == "__main__":
    main()
