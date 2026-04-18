#!/usr/bin/env python3
"""Diagnose feasibility failures by recomputing all metrics on saved grasps."""
import os, sys, torch, numpy as np, trimesh, json
sys.path.insert(0, os.path.dirname(__file__))
from frogger.batched_pytorch_solver import BatchedSDF, BatchedGraspOptimizer

MESH_BASE = "/home/bowenj/Projects/DexFun/assets/mesh_obj"
MESH_ALT = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg"
ACT_BASE = "/home/bowenj/Projects/DexFun/assets/actuation_contacts"
OUT_DIR = "output/grasps_target"

# Test one object
OBJ = sys.argv[1] if len(sys.argv) > 1 else "funky_clear_spray_bottle"

def find_mesh(name):
    p1 = os.path.join(MESH_ALT, name, "object.obj")
    p2 = os.path.join(MESH_BASE, name, "object.obj")
    return p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else None)

mesh_path = find_mesh(OBJ)
mesh = trimesh.load(mesh_path, force="mesh")
bounds = mesh.bounds
offset = np.array([0.0, 0.0, -bounds[0, 2]])
X_WO = np.eye(4); X_WO[:3, 3] = offset

sdf = BatchedSDF(mesh, X_WO, bounds_padding=0.15, resolution=128, device="cuda")
act_path = os.path.join(ACT_BASE, f"{OBJ}_actuation.json")
with open(act_path) as f:
    act_data = json.load(f)
c = act_data["actuation_contacts"][0]
actuation_targets = [(np.array(c["pos"]) + offset, np.array(c["dir"]))]
sdf.add_clearance_volume(actuation_targets[0][0], actuation_targets[0][1], radius=0.015, height=0.05)
sdf.add_floor(0.0)

opt = BatchedGraspOptimizer(sdf, num_envs=10, device="cuda", hand="rh", hand_type="leap", palm_contact=True)

# Load saved grasps
grasps = torch.load(f"{OUT_DIR}/{OBJ}/grasps.pt", weights_only=False)
print(f"\n{'='*80}")
print(f"  FEASIBILITY DIAGNOSIS: {OBJ} ({len(grasps)} grasps)")
print(f"{'='*80}")

dev = torch.device("cuda")
for i, r in enumerate(grasps):
    q = torch.tensor(r["q_joints"], dtype=torch.float32, device=dev).unsqueeze(0)
    fk = opt.chain.forward_kinematics(q)
    bT = np.eye(4)
    bT[:3, :3] = r["base_rot"]; bT[:3, 3] = r["base_pos"]
    bT_t = torch.tensor(bT, dtype=torch.float32, device=dev).unsqueeze(0)

    # Compute all collision SDF values
    non_ds_worst = 999.0
    ds_worst = 0.0
    ds_pad_worst = 0.0
    ds_back_worst = 0.0
    for li, (nm, pts) in enumerate(opt._col_data):
        if nm not in fk: continue
        lwT = bT_t @ fk[nm].get_matrix()
        ph = pts[:, :4].float()  # [N, 4] homogeneous
        lwp = (lwT @ ph.T)[:, :3, :].transpose(1, 2)  # [1, N, 3]
        lsdf = sdf.query(lwp)  # [1, N]
        
        if "_ds" in nm:
            ds_sdf = lsdf[0].min().item()
            if ds_sdf < ds_worst:
                ds_worst = ds_sdf
            
            # Separate pad-facing vs back
            # Pad faces -y direction in link frame. 
            # Points with local y < -0.02 (near pad) vs y > -0.01 (back/sides)
            local_y = pts[:, 1].cpu().numpy()
            pad_mask = local_y < -0.02
            back_mask = local_y > -0.01
            if pad_mask.any():
                pad_sdf = lsdf[0][torch.tensor(pad_mask, device=dev)].min().item()
                if pad_sdf < ds_pad_worst:
                    ds_pad_worst = pad_sdf
            if back_mask.any():
                back_sdf = lsdf[0][torch.tensor(back_mask, device=dev)].min().item()
                if back_sdf < ds_back_worst:
                    ds_back_worst = back_sdf
        else:
            si, ei = opt._col_link_ranges[li]
            margin = opt._col_margins[si].item()
            viol = max(0, margin - lsdf[0].min().item())
            if viol > 0 and lsdf[0].min().item() < non_ds_worst:
                non_ds_worst = lsdf[0].min().item()

    feas = r.get("feasible", False)
    tag = "FEAS" if feas else "FAIL"
    print(f"\n  G{i} [{tag}] l*={r.get('l_star',-1):.4f} sigma={r.get('sigma_min',0):.4f}")
    print(f"    surf={r.get('surf_err',0)*1000:.1f}mm  non_ds_worst={non_ds_worst*1000:.1f}mm  "
          f"ds_worst={ds_worst*1000:.1f}mm")
    print(f"    ds_pad={ds_pad_worst*1000:.1f}mm  ds_back={ds_back_worst*1000:.1f}mm  "
          f"sc_sdf={r.get('sc_min_dist',0)*1000:.1f}mm")
    print(f"    pen_pct={r.get('mesh_pen_pct',0):.1f}%  sc_pts={r.get('sc_worst',999)*1000:.1f}mm")

    # Check each feasibility criterion
    fails = []
    if r.get('surf_err', 0) > 0.008: fails.append(f"surf>{8}mm")
    if non_ds_worst < -0.003: fails.append(f"non_ds_col<-3mm ({non_ds_worst*1000:.1f}mm)")
    if ds_worst < -0.015: fails.append(f"ds_pen<-15mm ({ds_worst*1000:.1f}mm)")
    if r.get('sc_min_dist', 0) < -0.001: fails.append(f"sc_sdf<-1mm ({r.get('sc_min_dist',0)*1000:.1f}mm)")
    if r.get('sigma_min', 0) < 0.01: fails.append(f"sigma<0.01")
    if r.get('mesh_pen_pct', 0) > 5.0: fails.append(f"verify_pen>{5}%")
    if r.get('sc_worst', 999) < 0.0005: fails.append(f"verify_sc<0.5mm")
    if fails:
        print(f"    FAILS: {', '.join(fails)}")
    else:
        print(f"    All checks pass")
