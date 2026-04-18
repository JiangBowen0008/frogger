"""Is IK actually converging? Track loss over steps; try more iters + higher LR."""
import os, sys, json, numpy as np, trimesh, torch
import torch.nn.functional as F

sys.path.insert(0, "/home/bowenj/Projects/DexFun/third_parties/frogger")
from frogger.batched_pytorch_solver import BatchedSDF, BatchedGraspOptimizer

MESH_BASE = "/home/bowenj/Projects/DexFun/assets/mesh_obj"
ACT_BASE = "/home/bowenj/Projects/DexFun/assets/actuation_contacts"
OBJ = "hot_glue_gun"
NUM_ENVS = 4000


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

    # Bootstrap: run a minimal optimize() so self.u / self.pos / self.rot6d get
    # initialised by the palm-init + stage-2 IK pipeline. Then we restart IK
    # from scratch using only pos_err.
    dev = "cuda"
    act_pos = torch.tensor(act_pos_np, dtype=torch.float32, device=dev)
    actuation_targets = [(act_pos_np, act_dir_np)]
    _ = opt.optimize(actuation_targets=actuation_targets, object_center=obj_center,
                     steps=1, lr=0.005, save_path=None,
                     opt_sections="", opt_variant="P")

    # Reset to the palm-init state (u as stored by init). We keep the same base
    # poses but reinitialise the actuation-finger joints to the init defaults.
    # Easiest: start fresh u from stored init state.
    trial_fi = 3  # thumb — highest individual success rate
    u_act = opt.u.detach().clone().requires_grad_(True)
    # Use higher LR
    opt_ik = torch.optim.Adam([u_act], lr=0.05)
    act_joint_mask = torch.zeros(NUM_ENVS, 16, device=dev, dtype=torch.bool)
    act_joint_mask[:, trial_fi*4:trial_fi*4+4] = True

    nm = opt.tip_link_names[trial_fi]
    off_h = torch.cat([opt.tip_offsets[trial_fi], torch.ones(1, device=dev)])

    trajectory = []
    for step in range(500):  # 500 steps instead of 150
        opt_ik.zero_grad()
        q_ik = opt._u2q(u_act)
        bT_ik = opt._base_T(opt.pos.detach(), opt.rot6d.detach())
        fk_ik = opt.chain.forward_kinematics(q_ik)
        wT = bT_ik @ fk_ik[nm].get_matrix()
        tip = (wT @ off_h.unsqueeze(-1)).squeeze(-1)[:, :3]

        pos_err = ((tip - act_pos) ** 2).sum(-1)
        # ONLY pos_err, no dir, no col — pure positional IK
        loss = 500 * pos_err
        # Joint reg
        non_act_reg = ((u_act - opt.u.detach()) ** 2 * (~act_joint_mask).float()).sum(-1)
        loss += 100 * non_act_reg

        loss.mean().backward()
        with torch.no_grad():
            u_act.grad[~act_joint_mask] = 0.0
        opt_ik.step()

        if step % 50 == 0 or step == 499:
            with torch.no_grad():
                dist = torch.norm(tip - act_pos, dim=-1)
                n5 = (dist < 0.005).sum().item()
                n10 = (dist < 0.010).sum().item()
                n20 = (dist < 0.020).sum().item()
                med = dist.median().item() * 1000
                trajectory.append((step, med, n5, n10, n20))
                print(f"  step {step:3d}: median={med:5.1f}mm  <5mm={n5:>4}  <10mm={n10:>4}  <20mm={n20:>4}")

    # Final analysis
    with torch.no_grad():
        q_ik = opt._u2q(u_act)
        bT_ik = opt._base_T(opt.pos.detach(), opt.rot6d.detach())
        fk_ik = opt.chain.forward_kinematics(q_ik)
        wT = bT_ik @ fk_ik[nm].get_matrix()
        tip = (wT @ off_h.unsqueeze(-1)).squeeze(-1)[:, :3]
        final_dist = torch.norm(tip - act_pos, dim=-1).cpu().numpy()

    print(f"\n=== After 500 steps (pos-only, thumb assignment, all 4000 envs) ===")
    print(f"  <5mm:  {(final_dist < 0.005).sum()}/4000 ({100*(final_dist<0.005).mean():.1f}%)")
    print(f"  <10mm: {(final_dist < 0.010).sum()}/4000 ({100*(final_dist<0.010).mean():.1f}%)")
    print(f"  <20mm: {(final_dist < 0.020).sum()}/4000")
    print(f"  median: {np.median(final_dist)*1000:.1f}mm")
    print(f"  stuck in [20,60]mm: {((final_dist*1000 > 20) & (final_dist*1000 < 60)).sum()}/4000")

    # Compare against original 150-step with multi-assign
    prev = torch.load("output/grasps/hotglue_diag.pt", weights_only=False)
    prev_act = prev['act_dist']
    print(f"\n  (prior 150-step multi-assign: <10mm {(prev_act<0.010).sum()}/4000)")


if __name__ == "__main__":
    main()
