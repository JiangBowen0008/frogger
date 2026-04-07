"""Compare FK output between SDF and URDF models."""
import pytorch_kinematics as pk
import torch
import numpy as np

hand = "lh"

# Load SDF chain
with open(f"models/allegro/allegro_{hand}_clean.sdf") as f:
    chain_sdf = pk.build_chain_from_sdf(f.read())

# Load URDF chain
with open(f"models/allegro/allegro_{hand}.urdf") as f:
    chain_urdf = pk.build_chain_from_urdf(f.read())

print(f"SDF joints:  {chain_sdf.get_joint_parameter_names()}")
print(f"URDF joints: {chain_urdf.get_joint_parameter_names()}")

# Test 1: zero configuration
print("\n=== q = 0 ===")
q0 = torch.zeros(1, 16)
fk_sdf = chain_sdf.forward_kinematics(q0)
fk_urdf = chain_urdf.forward_kinematics(q0)

for name in sorted(fk_urdf.keys()):
    if name not in fk_sdf:
        continue
    pos_s = fk_sdf[name].get_matrix()[0, :3, 3].numpy()
    pos_u = fk_urdf[name].get_matrix()[0, :3, 3].numpy()
    diff = np.linalg.norm(pos_s - pos_u)
    flag = " *** MISMATCH ***" if diff > 1e-4 else ""
    print(f"  {name:30s}: SDF={pos_s}  URDF={pos_u}  diff={diff:.6f}{flag}")

# Test 2: some thumb actuation
print("\n=== thumb q=[1.0, 0.5, 0.5, 0.5] ===")
q1 = torch.zeros(1, 16)
q1[0, 12] = 1.0  # th_cmc
q1[0, 13] = 0.5  # th_axl
q1[0, 14] = 0.5  # th_mcp
q1[0, 15] = 0.5  # th_ipl
fk_sdf_1 = chain_sdf.forward_kinematics(q1)
fk_urdf_1 = chain_urdf.forward_kinematics(q1)

for name in ["algr_lh_th_mp", "algr_lh_th_bs", "algr_lh_th_px", "algr_lh_th_ds"]:
    pos_s = fk_sdf_1[name].get_matrix()[0, :3, 3].numpy()
    pos_u = fk_urdf_1[name].get_matrix()[0, :3, 3].numpy()
    diff = np.linalg.norm(pos_s - pos_u)
    # Also check rotation
    R_s = fk_sdf_1[name].get_matrix()[0, :3, :3].numpy()
    R_u = fk_urdf_1[name].get_matrix()[0, :3, :3].numpy()
    rdiff = np.linalg.norm(R_s - R_u)
    flag = " *** MISMATCH ***" if diff > 1e-4 or rdiff > 1e-3 else ""
    print(f"  {name:30s}: SDF={pos_s}  URDF={pos_u}  pos_diff={diff:.6f} rot_diff={rdiff:.6f}{flag}")

# Test 3: some finger actuation
print("\n=== fingers q=[0.2, 0.8, 0.7, 0.6, ...] ===")
q2 = torch.tensor([[
    0.2, 0.8, 0.7, 0.6,   # if
    0.1, 0.9, 0.5, 0.4,   # mf
    -0.1, 0.7, 0.8, 0.5,  # rf
    1.2, 0.5, 0.8, 0.7,   # th
]])
fk_sdf_2 = chain_sdf.forward_kinematics(q2)
fk_urdf_2 = chain_urdf.forward_kinematics(q2)

max_pos_diff = 0.0
max_rot_diff = 0.0
for name in sorted(fk_urdf_2.keys()):
    if name not in fk_sdf_2:
        continue
    if "FROGGER" in name:
        continue
    pos_s = fk_sdf_2[name].get_matrix()[0, :3, 3].numpy()
    pos_u = fk_urdf_2[name].get_matrix()[0, :3, 3].numpy()
    R_s = fk_sdf_2[name].get_matrix()[0, :3, :3].numpy()
    R_u = fk_urdf_2[name].get_matrix()[0, :3, :3].numpy()
    pd = np.linalg.norm(pos_s - pos_u)
    rd = np.linalg.norm(R_s - R_u)
    max_pos_diff = max(max_pos_diff, pd)
    max_rot_diff = max(max_rot_diff, rd)
    flag = " *** MISMATCH ***" if pd > 1e-4 or rd > 1e-3 else ""
    print(f"  {name:30s}: pos_diff={pd:.6f} rot_diff={rd:.6f}{flag}")

print(f"\n  Max pos diff: {max_pos_diff:.8f}")
print(f"  Max rot diff: {max_rot_diff:.8f}")
if max_pos_diff < 1e-4 and max_rot_diff < 1e-3:
    print("  >>> FK MATCH: SDF and URDF produce identical results <<<")
else:
    print("  >>> FK MISMATCH: SDF and URDF differ! <<<")
