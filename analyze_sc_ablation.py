#!/usr/bin/env python3
"""Compare SC trajectory + feasibility across uniform_viable baseline and
SC-loss-strengthening ablations, on air_blower (and any other available object).

Reads grasps_pooled.pt + batch_0/grasps_metrics.pt for each run dir under
output/, prints per-stage SC pass rates and per-env trajectories for l*>0
grasps. Quote-ready output.
"""
import os, sys, torch, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {0: 'IF', 1: 'MF', 2: 'RF', 3: 'TH'}

def load_run(root, obj):
    gp = os.path.join(root, obj, 'grasps_pooled.pt')
    mp = os.path.join(root, obj, 'batch_0', 'grasps_metrics.pt')
    if not os.path.exists(gp): return None
    g = torch.load(gp, weights_only=False)
    m = torch.load(mp, weights_only=False) if os.path.exists(mp) else None
    return g, m


def main():
    runs = sys.argv[1:] if len(sys.argv) > 1 else [
        'output/ablation_uniform_v1',
        'output/ablation_sc_strong',
    ]
    objects = ['air_blower', 'hot_glue_gun', 'flashlight',
               'funky_clear_spray_bottle', 'grinder']

    for obj in objects:
        print(f'\n{"="*70}\n  {obj}\n{"="*70}')
        rows = []
        for r in runs:
            data = load_run(os.path.join(HERE, r), obj)
            if data is None:
                rows.append((r, None, None)); continue
            rows.append((r, *data))

        # Pooled feasibility summary
        print(f'  {"run":<30} N    l*>0 feas  TH/IF/MF/RF feas-act')
        for r, g, m in rows:
            if g is None: print(f'  {r:<30} (missing)'); continue
            n_lpos = sum(1 for x in g if x.get('l_star', -1) > 0)
            n_feas = sum(1 for x in g if x.get('feasible'))
            from collections import Counter
            c = Counter(NAMES[x['act_finger']] for x in g if x.get('feasible'))
            print(f'  {r.split("/")[-1]:<30} {len(g):<4} {n_lpos:<4} {n_feas:<4} ' +
                  '/'.join(str(c.get(k, 0)) for k in ['TH','IF','MF','RF']))

        # Per-stage SC pass rates (population-level)
        print(f'  Per-stage SC pass rate (sc>-1mm), 4000 envs:')
        stages = ['S1_after_init', 'S3_after_support_ik',
                  'S4_opt_step0', 'S4_opt_step151', 'S4_opt_step300']
        for r, g, m in rows:
            if m is None: continue
            pr = []
            for s in stages:
                if s not in m: pr.append('-'); continue
                a = np.asarray(m[s]['sc_worst']) * 1000
                pr.append(f'{(a>-1).mean()*100:5.1f}%')
            print(f'    {r.split("/")[-1]:<28} ' + ' '.join(f'{s[-6:]:>7}' for s in stages))
            print(f'    {"":<28} ' + ' '.join(f'{p:>7}' for p in pr))

        # Per-env SC trajectories for l*>0
        for r, g, m in rows:
            if g is None or m is None: continue
            lpos = [x for x in g if x.get('l_star', -1) > 0]
            if not lpos: continue
            print(f'  [{r.split("/")[-1]}] l*>0 envs trajectory (mm):')
            sc = {s: np.asarray(m[s]['sc_worst']) * 1000 for s in stages if s in m}
            hdr = 'env    act ' + ' '.join(f'{s[-6:]:>7}' for s in stages if s in m)
            print('    ' + hdr)
            for x in lpos:
                e = x['env_idx']
                row = [sc[s][e] for s in stages if s in m]
                feas = 'F' if x['feasible'] else 'x'
                print(f'    {e:<5} {NAMES[x["act_finger"]]:<3} [{feas}] ' +
                      ' '.join(f'{v:7.2f}' for v in row) + f'  l*={x["l_star"]:.3f}')


if __name__ == '__main__':
    main()
