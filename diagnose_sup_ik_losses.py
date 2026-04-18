"""Track support-IK loss components over steps. Same approach as act IK —
 verifies whether surface loss is gradient-starved by pad-alignment / col losses."""
import os, sys, json, numpy as np, trimesh, torch
import torch.nn.functional as F

sys.path.insert(0, "/home/bowenj/Projects/DexFun/third_parties/frogger")
from frogger.batched_pytorch_solver import BatchedSDF, BatchedGraspOptimizer

MESH_BASE = "/home/bowenj/Projects/DexFun/assets/mesh_obj"
ACT_BASE = "/home/bowenj/Projects/DexFun/assets/actuation_contacts"
OBJ = "grinder"  # grinder = works best, clean signal
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
    actuation_targets = [(act_pos_np, act_dir_np)]

    sdf = BatchedSDF(mesh, X_WO, bounds_padding=0.15, resolution=128, device="cuda")
    sdf.add_clearance_volume(act_pos_np, act_dir_np, radius=0.020, height=0.03)
    sdf.add_floor(0.0)
    opt = BatchedGraspOptimizer(sdf, num_envs=NUM_ENVS, device="cuda",
                                hand="rh", hand_type="leap", palm_contact=True)

    # Run full pipeline up to stage 4 filter. This gives us `good` mask + u with
    # act finger placed. Then we copy the solver's support IK code and track
    # per-step loss components.
    # Simplest: monkey-patch the solver to instrument its support IK loop.
    # But that's invasive. Instead, just run optimize() with 1 step so support
    # IK runs, and intercept via logging. Actually the easiest hack: we run
    # optimize() as normal and rely on printed timing + do a manual reruns.
    # For a targeted study, we re-do support IK from scratch here.

    # Bootstrap
    _ = opt.optimize(actuation_targets=actuation_targets, object_center=obj_center,
                     steps=1, lr=0.005, save_path=None,
                     opt_sections="", opt_variant="P")

    # At this point self.u has the state after stage 4 filter + support IK
    # already ran. To observe *starting* support IK state we'd need to
    # intercept it. Quick workaround: re-run support-IK-like loss from the
    # current u (which has support fingers already placed by the pipeline's
    # 400-step run) to see the RESIDUAL loss structure.
    #
    # But that's misleading — post-supIK losses. Better: re-init support
    # fingers to their palm-init state (not the post-supIK state) and watch
    # from step 0.

    # Hack: re-roll support finger joints to a standard curl, then run IK.
    B = NUM_ENVS
    dev = "cuda"

    # Identify which envs passed stage 4 (self._good if set, else assume all)
    # We'll just use all 4000 for the study.
    with torch.no_grad():
        u_cur = opt.u.detach().clone()
        # Reset support fingers to a middle-of-range curl
        for b in range(B):
            act_fi = int(opt.amap[b, 0])
            for fi in range(4):
                if fi == act_fi: continue
                # Set all 4 joints of this finger to middle u ≈ 0.3
                u_cur[b, fi*4:fi*4+4] = 0.0
        u_sup = u_cur.clone().requires_grad_(True)

    opt_sup = torch.optim.Adam([u_sup], lr=0.03)

    sup_joint_mask = torch.zeros(B, 16, device=dev, dtype=torch.bool)
    sup_finger_mask = torch.zeros(B, 4, device=dev, dtype=torch.bool)
    for b in range(B):
        act_fi = int(opt.amap[b, 0])
        for fi in range(4):
            if fi != act_fi:
                sup_joint_mask[b, fi*4:fi*4+4] = True
                sup_finger_mask[b, fi] = True

    obj_c = torch.tensor(obj_center, dtype=torch.float32, device=dev)
    ap = torch.tensor(act_pos_np, dtype=torch.float32, device=dev)

    # Finger targets: a point on surface at each quadrant (simplified)
    finger_targets = {}
    for fi in range(4):
        ang = [1.57, 2.62, 3.14, -1.57][fi]
        search_pt = (obj_c[:2] + 0.1 * torch.tensor([np.cos(ang), np.sin(ang)], dtype=torch.float32, device=dev))
        target_z = torch.full((B, 1), obj_c[2].item(), device=dev)
        search_3d = torch.cat([search_pt.unsqueeze(0).expand(B, -1), target_z], -1)
        tgt_sdf = opt.sdf.query(search_3d.unsqueeze(1)).squeeze(1)
        _, tgt_n = opt.sdf.query_with_normals(search_3d.unsqueeze(1))
        finger_targets[fi] = search_3d - tgt_sdf.unsqueeze(-1) * tgt_n[:, 0]

    log = []
    prefixes = ['if', 'mf', 'rf', 'th']
    suffix_list = [['bs', 'px', 'md', 'ds']] * 3 + [['mp', 'bs', 'px', 'ds']]

    for step in range(400):
        opt_sup.zero_grad()
        q = opt._u2q(u_sup)
        bT = opt._base_T(opt.pos.detach(), opt.rot6d.detach())
        fk = opt.chain.forward_kinematics(q)

        # Track per-loss components
        surf_loss = torch.zeros(B, device=dev)
        pad_loss = torch.zeros(B, device=dev)
        col_loss = torch.zeros(B, device=dev)
        target_loss = torch.zeros(B, device=dev)

        for fi in range(4):
            nm = opt.tip_link_names[fi]
            wT = bT @ fk[nm].get_matrix()
            off_h = torch.cat([opt.tip_offsets[fi], torch.ones(1, device=dev)])
            tp = (wT @ off_h.unsqueeze(-1)).squeeze(-1)[:, :3]
            tp_sdf = opt.sdf.query(tp.unsqueeze(1)).squeeze(1)
            surf_loss += sup_finger_mask[:, fi].float() * 500 * tp_sdf.abs()

            pad_dir = -wT[:, :3, 0]
            _, in_n = opt.sdf.query_with_normals(tp.unsqueeze(1))
            in_n = in_n[:, 0]
            align = (pad_dir * in_n).sum(-1)
            bad_face = F.softplus(-align * 10) / 10
            pad_loss += sup_finger_mask[:, fi].float() * 500 * bad_face ** 2

            target_w = 100 * max(0, 1 - step / 200)
            if target_w > 0 and fi in finger_targets:
                tgt_err = ((tp - finger_targets[fi]) ** 2).sum(-1)
                target_loss += sup_finger_mask[:, fi].float() * target_w * tgt_err

            # Link collision for non-ds links
            for suf in suffix_list[fi][:-1]:  # skip 'ds'
                ln = f"leap_rh_{prefixes[fi]}_{suf}"
                for cnm, clp in opt._col_data:
                    if cnm == ln and ln in fk:
                        lwT = bT @ fk[ln].get_matrix()
                        lwp = (lwT @ clp.T)[:, :3, :].transpose(1, 2)
                        lsdf = opt.sdf.query(lwp)
                        col_loss += sup_finger_mask[:, fi].float() * 500 * F.relu(-lsdf).mean(-1)

        total = surf_loss + pad_loss + col_loss + target_loss
        reg = ((u_sup - u_cur) ** 2 * (~sup_joint_mask).float()).sum(-1)
        total += 200 * reg
        total.mean().backward()
        with torch.no_grad():
            u_sup.grad[~sup_joint_mask] = 0.0
        opt_sup.step()

        if step % 20 == 0 or step == 399:
            with torch.no_grad():
                # Per-env tip-to-surface distance
                q_e = opt._u2q(u_sup)
                fk_e = opt.chain.forward_kinematics(q_e)
                dists = []
                for fi in range(4):
                    if not sup_finger_mask[:, fi].any(): continue
                    wT = bT @ fk_e[opt.tip_link_names[fi]].get_matrix()
                    off_h = torch.cat([opt.tip_offsets[fi], torch.ones(1, device=dev)])
                    tp = (wT @ off_h.unsqueeze(-1)).squeeze(-1)[:, :3]
                    d = opt.sdf.query(tp.unsqueeze(1)).squeeze(1).abs()
                    # Only count support-assigned envs for this finger
                    d_masked = d[sup_finger_mask[:, fi]]
                    dists.append(d_masked)
                all_d = torch.cat(dists)
                log.append(dict(
                    step=step,
                    surf=surf_loss.mean().item(),
                    pad=pad_loss.mean().item(),
                    col=col_loss.mean().item(),
                    target=target_loss.mean().item(),
                    d_med=all_d.median().item()*1000,
                    d_p10=torch.quantile(all_d, 0.1).item()*1000,
                    d_p90=torch.quantile(all_d, 0.9).item()*1000,
                    n_under_5=(all_d < 0.005).sum().item(),
                ))

    print(f"\n{'step':>4} {'surf':>8} {'pad':>8} {'col':>8} {'target':>8} "
          f"{'p10':>6} {'med':>6} {'p90':>6} {'<5mm':>6}")
    for r in log:
        print(f"{r['step']:>4} {r['surf']:>8.3f} {r['pad']:>8.2f} {r['col']:>8.2f} "
              f"{r['target']:>8.3f} {r['d_p10']:>6.1f} {r['d_med']:>6.1f} "
              f"{r['d_p90']:>6.1f} {r['n_under_5']:>6}")


if __name__ == "__main__":
    main()
