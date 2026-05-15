"""Check WHY l* fails on grasps where sigma>0.01 and geometry looks fine.
Hypothesis: support tips clustered on one side of object -> contact normals
point same way -> non-negative combo can't oppose all external wrenches.
"""
import os, sys, json, numpy as np, trimesh, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from frogger.batched_pytorch_solver import BatchedSDF, BatchedGraspOptimizer

MESH_BASE = os.path.normpath(os.path.join(HERE, "..", "..", "assets", "mesh_obj"))
RUN_TAG = os.environ.get("FROGGER_DIAG_RUN", "final_3batch_v12")
N_BATCHES = int(os.environ.get("FROGGER_DIAG_BATCHES", "3"))


def compute_tips_and_normals(opt, q_joints, base_pos, base_rot):
    """FK + surface inward normal at each tip. Returns (tips [4,3], normals [4,3])."""
    dev = opt.device
    q = torch.tensor(q_joints, dtype=torch.float32, device=dev).unsqueeze(0)
    R = torch.tensor(base_rot, dtype=torch.float32, device=dev).unsqueeze(0)
    bT = torch.eye(4, device=dev).unsqueeze(0)
    bT[0, :3, :3] = R[0]
    bT[0, :3, 3] = torch.tensor(base_pos, dtype=torch.float32, device=dev)
    fk = opt.chain.forward_kinematics(q)
    tips = []
    for fi in range(4):
        nm = opt.tip_link_names[fi]
        wT = bT @ fk[nm].get_matrix()
        off_h = torch.cat([opt.tip_offsets[fi], torch.ones(1, device=dev)])
        tp = (wT @ off_h.unsqueeze(-1)).squeeze(-1)[0, :3]
        tips.append(tp)
    tips_t = torch.stack(tips, dim=0).unsqueeze(0)  # [1, 4, 3]
    _, inward = opt.sdf.query_with_normals(tips_t)   # [1, 4, 3]
    return tips_t[0].cpu().numpy(), inward[0].cpu().numpy()


def main():
    for obj in ['funky_clear_spray_bottle', 'hot_glue_gun', 'air_blower', 'flashlight', 'grinder']:
        mp = f"{MESH_BASE}/{obj}/object.obj"
        if not os.path.exists(mp): continue
        mesh = trimesh.load(mp, force='mesh')
        offset = np.array([0, 0, -mesh.bounds[0, 2]])
        obj_center = mesh.centroid + offset
        X_WO = np.eye(4); X_WO[:3, 3] = offset

        sdf = BatchedSDF(mesh, X_WO, bounds_padding=0.15, resolution=64, device='cuda')
        opt = BatchedGraspOptimizer(sdf, num_envs=1, device='cuda', hand='rh', hand_type='leap', palm_contact=True)

        feas_stats = []
        lfail_stats = []

        for b in range(N_BATCHES):
            p = f'output/{RUN_TAG}/{obj}/batch_{b}/grasps.pt'
            if not os.path.exists(p): continue
            for g in torch.load(p, weights_only=False):
                tips, normals = compute_tips_and_normals(opt, g['q_joints'], g['base_pos'], g['base_rot'])
                act_fi = g['act_finger']
                sup_idx = [i for i in range(4) if i != act_fi]
                sup_tips = tips[sup_idx]
                sup_norms = normals[sup_idx]
                sup_norms = sup_norms / (np.linalg.norm(sup_norms, axis=1, keepdims=True) + 1e-9)

                # Position-based cluster (radial direction from object center)
                v = sup_tips - obj_center
                vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
                cluster = float(np.linalg.norm(vn.sum(0)))  # 0 = well-spread, 3 = all same

                # Normal-based metrics (the FC-relevant geometry).
                # n_sum: sum of inward normals. Magnitude ~3 means all 3 contacts
                # push the object the same way → no opposing wrench → FC fails.
                # Magnitude ~0 means contacts span balanced directions.
                n_sum = float(np.linalg.norm(sup_norms.sum(0)))
                # Pairwise normal dots: high positive = normals same direction.
                n_pair = [np.dot(sup_norms[0], sup_norms[1]),
                          np.dot(sup_norms[0], sup_norms[2]),
                          np.dot(sup_norms[1], sup_norms[2])]
                n_max_cos = float(max(n_pair))
                n_min_cos = float(min(n_pair))

                stats = {
                    'cluster': cluster,
                    'n_sum': n_sum,
                    'n_max_cos': n_max_cos,
                    'n_min_cos': n_min_cos,
                    'sigma': g['sigma_min'],
                    'l_star': g['l_star'],
                    'surf': g['surf_err'] * 1000,
                    'dspd': g['ds_pad_worst'] * 1000,
                }
                if g['feasible']:
                    feas_stats.append(stats)
                elif (g['sigma_min'] > 0.01 and g['l_star'] <= 0
                      and g['surf_err'] < 0.008 and g['max_col_viol'] < 0.003
                      and g['ds_pad_worst'] > -0.005):
                    lfail_stats.append(stats)

        if not feas_stats and not lfail_stats:
            print(f'\n=== {obj}: no data ===')
            continue
        print(f'\n=== {obj} ===')

        def summ(label, s):
            if not s:
                print(f'  {label}: (none)')
                return
            def stat(k):
                vals = [x[k] for x in s]
                return f'med={np.median(vals):+.2f}  p10={np.percentile(vals, 10):+.2f}  p90={np.percentile(vals, 90):+.2f}'
            print(f'  {label} ({len(s)}):')
            print(f'    pos_cluster (0=spread, 3=same side):  {stat("cluster")}')
            print(f'    normal_sum  (0=opposing, 3=same dir):  {stat("n_sum")}')
            print(f'    normal_max_cos (>0 means co-aligned):  {stat("n_max_cos")}')
            print(f'    normal_min_cos (<0 means an opposed pair exists): {stat("n_min_cos")}')

        summ('FEASIBLE', feas_stats)
        summ('l*_fail ', lfail_stats)


if __name__ == "__main__":
    main()
