#!/usr/bin/env python3
"""
Visualize grasps at different optimization stages in viser.
Shows: init (before any optimization), after P0, after P1, final.
Saves snapshots as .pt files and displays them all.
"""
import os, sys, numpy as np, trimesh, torch, time, viser
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(__file__))
from frogger.batched_pytorch_solver import BatchedSDF, BatchedGraspOptimizer, _visual_meshes

MESH_PATH = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg/hot_glue_gun/object.obj"
URDF_PATH = os.path.join(os.path.dirname(__file__), "models/leap_rh/leap.urdf")
MESH_DIR = os.path.join(os.path.dirname(__file__), "models/leap_rh")

# Add snapshot method to the optimizer to capture init state
mesh = trimesh.load(MESH_PATH, force="mesh")
bounds = mesh.bounds
offset = np.array([0.0, 0.0, -bounds[0, 2]])
X_WO = np.eye(4); X_WO[:3, 3] = offset
obj_center = mesh.centroid + offset

verts_W = (X_WO[:3, :3] @ np.asarray(mesh.vertices).T).T + X_WO[:3, 3]
mesh_W = trimesh.Trimesh(vertices=verts_W, faces=mesh.faces)
candidate = np.array([[0.0, 0.0, offset[2] + (bounds[1, 2] - bounds[0, 2]) * 0.8]])
closest_pts, _, _ = trimesh.proximity.closest_point(mesh_W, candidate)

sdf = BatchedSDF(mesh, X_WO, bounds_padding=0.15, resolution=128, device="cuda")
opt = BatchedGraspOptimizer(sdf, num_envs=50, device="cuda", hand="rh", hand_type="leap", palm_contact=True)

# Monkey-patch _snapshot to save grasp files
_orig_snapshot = opt._snapshot
def _patched_snapshot(tag, idx=0):
    _orig_snapshot(tag, idx)
    with torch.no_grad():
        q = opt._u2q(opt.u)
        R = opt._rot6d_to_matrix(opt.rot6d)
        grasps = []
        for i in range(min(10, opt.num_envs)):
            grasps.append({
                "q_joints": q[i].cpu().numpy(),
                "base_pos": opt.pos[i].detach().cpu().numpy(),
                "base_rot": R[i].cpu().numpy(),
                "sigma_min": 0.0, "l_star": 0.0, "feasible": False,
            })
        torch.save(grasps, f"output/grasps/stage_{tag}.pt")

        # Palm orientation check
        palm_inward = -R[:10, :, 2]
        obj_c = torch.tensor(obj_center, device="cuda")
        to_center = obj_c.unsqueeze(0) - opt.pos[:10].detach()
        to_center = to_center / to_center.norm(dim=-1, keepdim=True)
        dots = (palm_inward * to_center).sum(-1)
        print(f"    PALM ORIENTATION [{tag}]: facing={int((dots>0.3).sum())} away={int((dots<-0.3).sum())} mean={dots.mean().item():.2f}")

opt._snapshot = _patched_snapshot

# Re-enable snapshots in the optimizer temporarily
import frogger.batched_pytorch_solver as _solver
# We need the snapshot calls back — add them via the _phase_snapshots attribute
opt._phase_snapshots = []

results = opt.optimize(
    actuation_targets=[(closest_pts[0], None)],
    object_center=obj_center,
    steps=400, lr=0.005,
    save_path="output/grasps/stage_final.pt",
)
final_path = "output/grasps/stage_final.pt"
init_path = "output/grasps/stage_after_init.pt" if os.path.exists("output/grasps/stage_after_init.pt") else final_path

# Visualize in viser
server = viser.ViserServer(host="0.0.0.0", port=8090)
print(f"\nStage viewer -> http://localhost:8090")

chain = pk.build_chain_from_urdf(open(URDF_PATH).read())
vis = _visual_meshes("rh", "leap")
link_mesh_cache = {}
for link_name, mesh_list in vis.items():
    for mi, (mf, vp) in enumerate(mesh_list):
        path = os.path.join(MESH_DIR, mf)
        if not os.path.exists(path): continue
        lm = trimesh.load(path, force="mesh")
        link_mesh_cache[(link_name, mi)] = (
            np.asarray(lm.vertices, dtype=np.float32),
            np.asarray(lm.faces, dtype=np.int32), vp)

obj_verts = verts_W.astype(np.float32)
obj_faces = np.asarray(mesh.faces, dtype=np.int32)

stages = {
    "00_init": init_path,
    "01_final": final_path,
}

def show(stage_name, grasp_idx):
    data = torch.load(stages[stage_name], weights_only=False)
    if grasp_idx >= len(data): grasp_idx = 0
    g = data[grasp_idx]

    server.scene.add_mesh_simple("/object", vertices=obj_verts, faces=obj_faces,
                                  color=(180, 180, 180), opacity=0.7)

    q = torch.tensor(g["q_joints"], dtype=torch.float32).unsqueeze(0)
    fk = chain.forward_kinematics(q)
    T_base = np.eye(4)
    T_base[:3, :3] = g["base_rot"]
    T_base[:3, 3] = g["base_pos"]

    for (link_name, mi), (lv, lf, vp) in link_mesh_cache.items():
        if link_name not in fk: continue
        wT = T_base @ fk[link_name].get_matrix()[0].numpy()
        if vp is not None:
            vpa = np.array(vp)
            Rv = Rotation.from_euler("xyz", vpa[3:]).as_matrix()
            Tv = np.eye(4); Tv[:3, :3] = Rv; Tv[:3, 3] = vpa[:3]
            wT = wT @ Tv
        vw = (wT[:3, :3] @ lv.T).T + wT[:3, 3]
        is_palm = "palm" in link_name
        color = (50, 100, 255) if is_palm else (255, 200, 100)
        server.scene.add_mesh_simple(f"/hand/{link_name}_{mi}",
            vertices=vw.astype(np.float32), faces=lf, color=color, opacity=0.85)

stage_names = list(stages.keys())
with server.gui.add_folder("Stage Browser"):
    dd = server.gui.add_dropdown("Stage", options=stage_names, initial_value=stage_names[0])
    gi = server.gui.add_slider("Grasp idx", min=0, max=9, step=1, initial_value=0)

@dd.on_update
def _(_): show(dd.value, int(gi.value))
@gi.on_update
def _(_): show(dd.value, int(gi.value))

show(stage_names[0], 0)
print("Blue = palm, Orange = fingers")
print("Switch between 00_init and 01_final to see the change")
try:
    while True: time.sleep(1)
except KeyboardInterrupt: pass
