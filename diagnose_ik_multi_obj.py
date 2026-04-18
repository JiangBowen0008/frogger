"""Run IK-loss-trajectory diagnostic on 3 objects to verify IK stagnation is universal.

Each object: 4000 envs, thumb assignment, 300 steps of pipeline IK (pos+dir+col).
Log per-step (pos_loss, dir_loss, col_loss, dist distribution).
Saves per-object logs to output/grasps/ik_traj_{obj}.pt
"""
import os, sys, json, numpy as np, trimesh, torch
import torch.nn.functional as F

sys.path.insert(0, "/home/bowenj/Projects/DexFun/third_parties/frogger")
from frogger.batched_pytorch_solver import BatchedSDF, BatchedGraspOptimizer

MESH_BASE = "/home/bowenj/Projects/DexFun/assets/mesh_obj"
ACT_BASE = "/home/bowenj/Projects/DexFun/assets/actuation_contacts"
NUM_ENVS = 4000
N_STEPS = 300
TRIAL_FI = 3  # thumb
OBJECTS = ["hot_glue_gun", "grinder", "air_blower"]


def run_one(obj_name):
    print(f"\n{'='*60}\n  {obj_name}\n{'='*60}")
    mesh_path = os.path.join(MESH_BASE, obj_name, "object.obj")
    if not os.path.exists(mesh_path):
        print(f"  skip: mesh not found")
        return None
    mesh = trimesh.load(mesh_path, force="mesh")
    bounds = mesh.bounds
    offset = np.array([0.0, 0.0, -bounds[0, 2]])
    X_WO = np.eye(4); X_WO[:3, 3] = offset
    obj_center = mesh.centroid + offset
    with open(f"{ACT_BASE}/{obj_name}_actuation.json") as f:
        c = json.load(f)["actuation_contacts"][0]
    act_pos_np = np.array(c["pos"]) + offset
    act_dir_np = np.array(c["dir"])

    sdf = BatchedSDF(mesh, X_WO, bounds_padding=0.15, resolution=128, device="cuda")
    sdf.add_clearance_volume(act_pos_np, act_dir_np, radius=0.020, height=0.03)
    sdf.add_floor(0.0)
    opt = BatchedGraspOptimizer(sdf, num_envs=NUM_ENVS, device="cuda",
                                hand="rh", hand_type="leap", palm_contact=True)

    dev = "cuda"
    act_pos = torch.tensor(act_pos_np, dtype=torch.float32, device=dev)
    act_dir = F.normalize(torch.tensor(act_dir_np, dtype=torch.float32, device=dev), dim=0)
    actuation_targets = [(act_pos_np, act_dir_np)]

    # Bootstrap the init (palm sampling) by running optimize with steps=1.
    # We'll reset u_act and run IK from scratch to capture fresh trajectory.
    _ = opt.optimize(actuation_targets=actuation_targets, object_center=obj_center,
                     steps=1, lr=0.005, save_path=None,
                     opt_sections="", opt_variant="P")

    u_act = opt.u.detach().clone().requires_grad_(True)
    opt_ik = torch.optim.Adam([u_act], lr=0.05)
    act_joint_mask = torch.zeros(NUM_ENVS, 16, device=dev, dtype=torch.bool)
    act_joint_mask[:, TRIAL_FI*4:TRIAL_FI*4+4] = True

    nm = opt.tip_link_names[TRIAL_FI]
    off_h = torch.cat([opt.tip_offsets[TRIAL_FI], torch.ones(1, device=dev)])
    prefixes_ik = ['if', 'mf', 'rf', 'th']
    sfx_ik = [['bs', 'px', 'md']] * 3 + [['mp', 'bs', 'px']]
    log = []

    for step in range(N_STEPS):
        opt_ik.zero_grad()
        q_ik = opt._u2q(u_act)
        bT_ik = opt._base_T(opt.pos.detach(), opt.rot6d.detach())
        fk_ik = opt.chain.forward_kinematics(q_ik)
        wT = bT_ik @ fk_ik[nm].get_matrix()
        tip = (wT @ off_h.unsqueeze(-1)).squeeze(-1)[:, :3]
        pad_dir = -wT[:, :3, 0]

        pos_err = ((tip - act_pos) ** 2).sum(-1)
        cos_align = (pad_dir * act_dir).sum(-1)
        dir_err = (1.0 - cos_align) ** 2
        col_loss = torch.zeros(NUM_ENVS, device=dev)
        for suf in sfx_ik[TRIAL_FI]:
            lnm = f"leap_rh_{prefixes_ik[TRIAL_FI]}_{suf}"
            if lnm not in fk_ik: continue
            for cnm, lp in opt._col_data:
                if cnm != lnm: continue
                lwT = bT_ik @ fk_ik[lnm].get_matrix()
                lwp = (lwT @ lp.T)[:, :3, :].transpose(1, 2)
                lsdf = opt.sdf.query(lwp, include_clearance=False)
                col_loss += F.relu(-lsdf).sum(-1)
        total = 500 * pos_err + 50 * dir_err + 100 * col_loss
        non_act_reg = ((u_act - opt.u.detach()) ** 2 * (~act_joint_mask).float()).sum(-1)
        total += 100 * non_act_reg
        total.mean().backward()
        with torch.no_grad():
            u_act.grad[~act_joint_mask] = 0.0
        opt_ik.step()

        if step % 15 == 0 or step == N_STEPS - 1:
            with torch.no_grad():
                dist = torch.norm(tip - act_pos, dim=-1)
                log.append(dict(
                    step=step,
                    pos_w=(500*pos_err).mean().item(),
                    dir_w=(50*dir_err).mean().item(),
                    col_w=(100*col_loss).mean().item(),
                    dist_p10=torch.quantile(dist,0.1).item()*1000,
                    dist_med=dist.median().item()*1000,
                    dist_p90=torch.quantile(dist,0.9).item()*1000,
                    n_under_10=(dist<0.010).sum().item(),
                    n_under_5=(dist<0.005).sum().item(),
                ))

    print(f"\n  step  pos_w    dir_w    col_w    p10    med    p90    <10mm  <5mm")
    for r in log:
        print(f"  {r['step']:>4}  {r['pos_w']:>6.2f}  {r['dir_w']:>6.2f}  "
              f"{r['col_w']:>6.2f}  {r['dist_p10']:>5.1f}  {r['dist_med']:>5.1f}  "
              f"{r['dist_p90']:>5.1f}  {r['n_under_10']:>4}   {r['n_under_5']:>4}")

    torch.save(log, f"output/grasps/ik_traj_{obj_name}.pt")
    # Return concise summary for cross-object comparison
    last = log[-1]
    first = log[0]
    return dict(
        obj=obj_name,
        pos_delta=first['pos_w'] - last['pos_w'],
        dir_delta=first['dir_w'] - last['dir_w'],
        col_delta=first['col_w'] - last['col_w'],
        dist_med_start=first['dist_med'],
        dist_med_end=last['dist_med'],
        n10_end=last['n_under_10'],
    )


if __name__ == "__main__":
    summaries = []
    for obj in OBJECTS:
        s = run_one(obj)
        if s: summaries.append(s)
    print(f"\n{'='*60}\n  SUMMARY across objects\n{'='*60}")
    print(f"  {'obj':<20} {'pos_Δ':>8} {'dir_Δ':>8} {'col_Δ':>8} "
          f"{'med_start':>10} {'med_end':>8} {'<10mm':>6}")
    for s in summaries:
        print(f"  {s['obj']:<20} {s['pos_delta']:>8.2f} {s['dir_delta']:>8.2f} "
              f"{s['col_delta']:>8.2f} {s['dist_med_start']:>10.1f} "
              f"{s['dist_med_end']:>8.1f} {s['n10_end']:>6}")
