#!/usr/bin/env python3
"""Multi-object grasp visualizer with stage comparison.

Shows grasps from run_target_objects.py output with:
- Object dropdown (switch between objects)
- Stage dropdown (init → support_ik → optimized)
- Grasp slider (browse top 10 grasps per stage)
- Metrics overlay (σ_min, l*, surface, collision, etc.)
"""
import os, sys, numpy as np, torch, trimesh, viser, time, json
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(__file__))
from frogger.batched_pytorch_solver import _visual_meshes, BatchedSDF, BatchedGraspOptimizer

URDF = os.path.join(os.path.dirname(__file__), "models/leap_rh/leap.urdf")
MDIR = os.path.join(os.path.dirname(__file__), "models/leap_rh")
RESULTS_DIR = "output/grasps_target"
PORT = 8090

import subprocess, signal
try:
    out = subprocess.check_output(["lsof", "-ti", f":{PORT}"]).decode().strip()
    for pid in out.split("\n"):
        if pid: os.kill(int(pid), signal.SIGKILL)
    time.sleep(1)
except (subprocess.CalledProcessError, ProcessLookupError):
    pass

server = viser.ViserServer(host="0.0.0.0", port=PORT)

# FK chain + visual meshes
chain = pk.build_chain_from_urdf(open(URDF).read())
vis = _visual_meshes("rh", "leap")
link_cache = {}
for ln, ml in vis.items():
    for mi, (mf, vp) in enumerate(ml):
        p = os.path.join(MDIR, mf)
        if not os.path.exists(p): continue
        lm = trimesh.load(p, force="mesh")
        link_cache[(ln, mi)] = (np.asarray(lm.vertices, np.float32),
                                np.asarray(lm.faces, np.int32), vp)

# Discover available objects
available_objects = []
if os.path.isdir(RESULTS_DIR):
    for d in sorted(os.listdir(RESULTS_DIR)):
        obj_dir = os.path.join(RESULTS_DIR, d)
        if os.path.isdir(obj_dir) and os.path.exists(os.path.join(obj_dir, "grasps.pt")):
            available_objects.append(d)

if not available_objects:
    print("No results found. Run run_target_objects.py first.")
    sys.exit(1)

print(f"Found {len(available_objects)} objects: {available_objects}")

# State
FINGER_LABELS = ["IF", "MF", "RF", "TH"]
contact_base = np.array([0.023, 0, 0.048])

# Per-object cached data
obj_cache = {}


def load_object(name):
    """Load all stages for an object."""
    if name in obj_cache:
        return obj_cache[name]

    obj_dir = os.path.join(RESULTS_DIR, name)
    data = {"name": name, "stages": {}, "sdf": None, "opt": None}

    # Load stages
    stage_files = {
        "1_init": "stage_after_init.pt",
        "2_support_ik": "stage_after_support_ik.pt",
        "3_optimized": "stage_after_optimization.pt",
    }
    for sname, fname in stage_files.items():
        path = os.path.join(obj_dir, fname)
        if os.path.exists(path):
            data["stages"][sname] = torch.load(path, weights_only=False, map_location="cpu")

    # Load final grasps — show feasible first, then infeasible
    grasps_path = os.path.join(obj_dir, "grasps.pt")
    if os.path.exists(grasps_path):
        grasps = torch.load(grasps_path, weights_only=False, map_location="cpu")
        if grasps:
            feas = [g for g in grasps if g.get("feasible", False)]
            infeas = [g for g in grasps if not g.get("feasible", False)]
            data["stages"]["3_optimized"] = feas + infeas  # feasible first

    # Load metadata
    meta_path = os.path.join(obj_dir, "meta.pt")
    if os.path.exists(meta_path):
        meta = torch.load(meta_path, weights_only=False, map_location="cpu")
        data["mesh_path"] = meta["mesh_path"]
        data["offset"] = meta["offset"]
        data["act_path"] = meta.get("act_path")
    else:
        # Fallback: try to find mesh
        from frogger.batched_pytorch_solver import BatchedSDF
        mesh_raw = f"/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg/{name}/object.obj"
        mesh_obj = f"/home/bowenj/Projects/DexFun/assets/mesh_obj/{name}/object.obj"
        data["mesh_path"] = mesh_raw if os.path.exists(mesh_raw) else mesh_obj
        act_path = f"/home/bowenj/Projects/DexFun/assets/actuation_contacts/{name}_actuation.json"
        data["act_path"] = act_path if os.path.exists(act_path) else None
        obj_mesh = trimesh.load(data["mesh_path"], force="mesh")
        data["offset"] = np.array([0.0, 0.0, -obj_mesh.bounds[0, 2]])

    # Load mesh
    obj_mesh = trimesh.load(data["mesh_path"], force="mesh")
    obj_mesh.apply_translation(data["offset"])
    data["mesh"] = obj_mesh

    # Load actuation
    if data.get("act_path") and os.path.exists(data["act_path"]):
        with open(data["act_path"]) as f:
            act_data = json.load(f)
        c = act_data["actuation_contacts"][0]
        data["act_pos"] = np.array(c["pos"], dtype=np.float32) + data["offset"]
        data["act_dir"] = np.array(c["dir"], dtype=np.float32)

    # Build SDF + optimizer for metrics
    obj_raw = trimesh.load(data["mesh_path"], force="mesh")
    X_WO = np.eye(4); X_WO[:3, 3] = data["offset"]
    try:
        data["sdf"] = BatchedSDF(obj_raw, X_WO, resolution=128, device="cuda")
        data["opt"] = BatchedGraspOptimizer(data["sdf"], num_envs=1, device="cuda",
                                            hand="rh", hand_type="leap", palm_contact=True)
    except Exception as e:
        print(f"  Warning: SDF/optimizer failed for {name}: {e}")

    obj_cache[name] = data
    return data


current_obj = [available_objects[0]]
current_stage = ["3_optimized"]
current_grasp = [0]


def show(obj_name=None, stage=None, grasp_idx=None):
    """Display a grasp for the current object/stage/index."""
    if obj_name is None: obj_name = current_obj[0]
    if stage is None: stage = current_stage[0]
    if grasp_idx is None: grasp_idx = current_grasp[0]

    data = load_object(obj_name)
    if not data["stages"]:
        info_md.content = f"No stages found for {obj_name}"
        return

    # Show object mesh
    obj = data["mesh"]
    server.scene.add_mesh_simple("/object",
        vertices=np.asarray(obj.vertices, dtype=np.float32),
        faces=np.asarray(obj.faces, dtype=np.int32),
        color=(180, 180, 180), opacity=0.6)

    # Get stage data
    stage_names = list(data["stages"].keys())
    if stage not in data["stages"]:
        stage = stage_names[-1]
    # Use latest stage for env_idx matching
    latest = data["stages"][stage_names[-1]]
    if grasp_idx >= len(latest):
        grasp_idx = 0
    target_env = latest[grasp_idx].get("env_idx", grasp_idx)

    # Find matching grasp in requested stage
    stage_data = data["stages"][stage]
    g = None
    for entry in stage_data:
        if entry.get("env_idx", -1) == target_env:
            g = entry
            break
    if g is None:
        g = stage_data[min(grasp_idx, len(stage_data)-1)]

    R = g["base_rot"]; pos = g["base_pos"]
    q = torch.tensor(g["q_joints"], dtype=torch.float32).unsqueeze(0)
    fk = chain.forward_kinematics(q)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pos

    # Draw hand
    for (ln, mi), (v, f, vp) in link_cache.items():
        if ln not in fk: continue
        wT = T @ fk[ln].get_matrix()[0].numpy()
        if vp is not None:
            vpa = np.array(vp)
            Rv = Rotation.from_euler("xyz", vpa[3:]).as_matrix()
            Tv = np.eye(4); Tv[:3, :3] = Rv; Tv[:3, 3] = vpa[:3]
            wT = wT @ Tv
        vw = (wT[:3, :3] @ v.T).T + wT[:3, 3]
        c = (50, 100, 255) if "palm" in ln else (255, 200, 100)
        server.scene.add_mesh_simple(f"/hand/{ln}_{mi}",
            vertices=vw.astype(np.float32), faces=f, color=c, opacity=0.85)

    # Actuation target
    if "act_pos" in data:
        server.scene.add_icosphere("/markers/act_target", radius=0.008,
            color=(255, 0, 0), position=data["act_pos"])
        act_tip = data["act_pos"] + 0.03 * data["act_dir"]
        server.scene.add_icosphere("/markers/act_dir", radius=0.004,
            color=(200, 0, 0), position=act_tip)

    # Palm contact center
    palm_center = R @ contact_base + pos
    server.scene.add_icosphere("/markers/palm_center", radius=0.006,
        color=(255, 0, 255), position=palm_center.astype(np.float32))

    # Surface point + normal
    surf = g.get("surf_pt")
    outward = g.get("outward_normal")
    if surf is not None:
        server.scene.add_icosphere("/markers/surf_pt", radius=0.006,
            color=(255, 255, 0), position=surf.astype(np.float32))
        if outward is not None:
            normal_tip = surf + 0.04 * outward
            server.scene.add_icosphere("/markers/normal_tip", radius=0.004,
                color=(255, 165, 0), position=normal_tip.astype(np.float32))
    else:
        # Clear markers
        server.scene.add_icosphere("/markers/surf_pt", radius=0.001,
            color=(0, 0, 0), position=(0, 0, 0))
        server.scene.add_icosphere("/markers/normal_tip", radius=0.001,
            color=(0, 0, 0), position=(0, 0, 0))

    # Frame axes
    x_tip = palm_center + 0.04 * R[:, 0]
    server.scene.add_icosphere("/markers/x_tip", radius=0.004,
        color=(255, 0, 0), position=x_tip.astype(np.float32))

    # Build metrics text
    lines = [f"**{obj_name} | Grasp {grasp_idx} (env {target_env}) | {stage}**\n"]

    # Key metrics from saved data
    sigma = g.get("sigma_min", 0)
    lstar = g.get("l_star", -1)
    feasible = g.get("feasible", "?")
    surf_err = g.get("surf_err", 0)
    act_dist = g.get("act_dist", 0)

    lines.append(f"**σ_min:** {sigma:.4f}  |  **l*:** {lstar:.4f}")
    lines.append(f"**feasible:** {feasible}")
    if surf_err:
        lines.append(f"**surf_err:** {surf_err*1000:.1f}mm")
    if act_dist:
        lines.append(f"**act_dist:** {act_dist*1000:.1f}mm")

    pen = g.get("mesh_pen_pct", None)
    if pen is not None:
        lines.append(f"**penetration:** {pen:.1f}%")
    sc = g.get("sc_worst", None)
    if sc is not None and sc < 999:
        lines.append(f"**SC worst:** {sc*1000:.1f}mm")

    # Act finger
    act_fi = g.get("act_finger", 0)
    lines.append(f"**act finger:** {FINGER_LABELS[act_fi] if act_fi < 4 else f'F{act_fi}'}")

    # Stage comparison if available
    lines.append("\n**Stage comparison:**")
    for sn in stage_names:
        sd = data["stages"][sn]
        match = None
        for entry in sd:
            if entry.get("env_idx", -1) == target_env:
                match = entry
                break
        if match:
            s = match.get("sigma_min", 0)
            l = match.get("l_star", -1)
            se = match.get("surf_err", 0)
            f_tag = "FEAS" if match.get("feasible") else ""
            lines.append(f"  {sn}: σ={s:.4f} l*={l:.4f} surf={se*1000:.1f}mm {f_tag}")

    info_md.content = "\n".join(lines)


# GUI
with server.gui.add_folder("Grasp Browser"):
    info_md = server.gui.add_markdown("")
    obj_dd = server.gui.add_dropdown("Object", options=available_objects, initial_value=available_objects[0])

    # Stage options (will update when object changes)
    data0 = load_object(available_objects[0])
    stage_opts = list(data0["stages"].keys())
    stage_dd = server.gui.add_dropdown("Stage", options=stage_opts if stage_opts else ["none"],
                                        initial_value=stage_opts[-1] if stage_opts else "none")

    gi_slider = server.gui.add_slider("Grasp", min=0, max=9, step=1, initial_value=0)
    btn_prev = server.gui.add_button("< Prev")
    btn_next = server.gui.add_button("Next >")


@obj_dd.on_update
def _(_):
    current_obj[0] = obj_dd.value
    d = load_object(obj_dd.value)
    sn = list(d["stages"].keys())
    if sn:
        stage_dd.options = sn
        stage_dd.value = sn[-1]
        current_stage[0] = sn[-1]
    current_grasp[0] = 0
    gi_slider.value = 0
    show()


@stage_dd.on_update
def _(_):
    current_stage[0] = stage_dd.value
    show()


@gi_slider.on_update
def _(_):
    current_grasp[0] = int(gi_slider.value)
    show()


@btn_prev.on_click
def _(_):
    d = load_object(current_obj[0])
    n = len(d["stages"].get(current_stage[0], []))
    current_grasp[0] = (current_grasp[0] - 1) % max(n, 1)
    gi_slider.value = current_grasp[0]


@btn_next.on_click
def _(_):
    d = load_object(current_obj[0])
    n = len(d["stages"].get(current_stage[0], []))
    current_grasp[0] = (current_grasp[0] + 1) % max(n, 1)
    gi_slider.value = current_grasp[0]


# Initial display
show()
print(f"http://localhost:{PORT}")
print(f"Objects: {', '.join(available_objects)}")
try:
    while True: time.sleep(1)
except KeyboardInterrupt:
    pass
