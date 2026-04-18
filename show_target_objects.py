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
    # NOTE: stage names must NOT start with digits — viser's markdown CSS fails
    # with "Unexpected character `1/2/3` before name" errors otherwise.
    stage_files = {
        "a_init": "stage_after_init.pt",
        "b_support_ik": "stage_after_support_ik.pt",
        "c_optimized": "stage_after_optimization.pt",
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
            data["stages"]["c_optimized"] = feas + infeas  # feasible first

    # Load metadata
    meta_path = os.path.join(obj_dir, "meta.pt")
    if os.path.exists(meta_path):
        meta = torch.load(meta_path, weights_only=False, map_location="cpu")
        data["mesh_path"] = meta["mesh_path"]
        data["offset"] = meta["offset"]
        data["act_path"] = meta.get("act_path")
    else:
        # Fallback: try to find mesh
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

    # Build SDF + optimizer for metrics (no try/except — let errors surface)
    obj_raw = trimesh.load(data["mesh_path"], force="mesh")
    X_WO = np.eye(4); X_WO[:3, 3] = data["offset"]
    data["sdf"] = BatchedSDF(obj_raw, X_WO, resolution=128, device="cuda")
    # Match solver: 20mm clearance cylinder (must match run_target_objects.py)
    if "act_pos" in data and "act_dir" in data:
        data["sdf"].add_clearance_volume(data["act_pos"], data["act_dir"],
                                         radius=0.020, height=0.03)
    data["sdf"].add_floor(0.0)
    data["opt"] = BatchedGraspOptimizer(data["sdf"], num_envs=1, device="cuda",
                                        hand="rh", hand_type="leap", palm_contact=True)

    obj_cache[name] = data
    return data


current_obj = [available_objects[0]]
current_stage = ["c_optimized"]
current_grasp = [0]
show_col_pts = [False]  # toggleable debug overlay


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
        # Floor plane (table) — visible reference at z=0
    floor_verts = np.array([
        [-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]
    ], dtype=np.float32)
    floor_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    server.scene.add_mesh_simple("/floor", vertices=floor_verts, faces=floor_faces,
        color=(200, 200, 200), opacity=0.3)

    server.scene.add_mesh_simple("/object",
        vertices=np.asarray(obj.vertices, dtype=np.float32),
        faces=np.asarray(obj.faces, dtype=np.int32),
        color=(180, 180, 180), opacity=0.6)

    # Clearance cylinder (where support fingers must stay out of).
    # Built as a triangulated open cylinder, 32 segments.
    if "act_pos" in data and "act_dir" in data:
        center = np.asarray(data["act_pos"], dtype=np.float32)
        axis = np.asarray(data["act_dir"], dtype=np.float32)
        axis /= (np.linalg.norm(axis) + 1e-9)
        radius = 0.020   # must match solver + runner
        height = 0.030
        # Find two vectors perpendicular to axis
        ref = np.array([1, 0, 0], dtype=np.float32) if abs(axis[0]) < 0.9 else np.array([0, 1, 0], dtype=np.float32)
        u_vec = np.cross(axis, ref); u_vec /= np.linalg.norm(u_vec) + 1e-9
        v_vec = np.cross(axis, u_vec)
        segments = 32
        ring_bottom = []
        ring_top = []
        for i in range(segments):
            a = 2 * np.pi * i / segments
            p_bot = center + radius * (np.cos(a) * u_vec + np.sin(a) * v_vec)
            p_top = p_bot + height * axis
            ring_bottom.append(p_bot); ring_top.append(p_top)
        verts = np.array(ring_bottom + ring_top, dtype=np.float32)
        # Side triangles
        faces = []
        for i in range(segments):
            j = (i + 1) % segments
            a, b, c, d = i, j, segments + j, segments + i   # bot_i, bot_j, top_j, top_i
            faces.append([a, b, c])
            faces.append([a, c, d])
        # Cap the top and bottom (so it's a filled cylinder when rendered)
        top_center_idx = len(verts)
        bot_center_idx = top_center_idx + 1
        verts = np.vstack([verts, (center + height * axis).reshape(1, 3), center.reshape(1, 3)])
        for i in range(segments):
            j = (i + 1) % segments
            faces.append([segments + i, segments + j, top_center_idx])
            faces.append([j, i, bot_center_idx])
        faces = np.array(faces, dtype=np.int32)
        server.scene.add_mesh_simple("/clearance", vertices=verts, faces=faces,
                                     color=(240, 180, 60), opacity=0.25)

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

    # Per-link collision check (for red highlighting)
    # Each link is red if any collision point violates threshold:
    #   - Object SDF: worst point < -3mm (3mm penetration margin)
    #   - Clearance SDF: worst point < 0 (no entry, except actuation finger)
    # link_name -> (worst_obj_mm, worst_cl_mm, worst_floor_mm, is_red, reason)
    link_status = {}
    act_fi = g.get("act_finger", 0)
    act_prefix = ['if','mf','rf','th'][act_fi] if act_fi < 4 else None
    sdf = data.get("sdf")
    opt = data.get("opt")
    if sdf is not None and opt is not None:
        dev = torch.device("cuda")
        T_t = torch.tensor(T, dtype=torch.float32, device=dev).unsqueeze(0)
        for li, (nm, pts) in enumerate(opt._col_data):
            if nm not in fk: continue
            with torch.no_grad():
                lwT = T_t @ fk[nm].get_matrix().to(dev)
                ph = pts[:, :4].float()
                lwp = (lwT @ ph.T)[:, :3, :].transpose(1, 2)
                # Use include_clearance=False to get object+floor only.
                # Then compute clearance separately, and floor separately.
                obj_floor_sdf = sdf.query(lwp, include_clearance=False)  # includes floor
                cl_sdf = sdf._clearance_sdf(lwp) if hasattr(sdf, '_clearance_center') else torch.full_like(obj_floor_sdf, float('inf'))
                floor_sdf = sdf._floor_sdf(lwp) if hasattr(sdf, '_floor_z') else torch.full_like(obj_floor_sdf, float('inf'))
                worst_obj_fl = obj_floor_sdf[0].min().item()
                worst_cl = cl_sdf[0].min().item()
                worst_floor = floor_sdf[0].min().item()
            # Separate floor from object: if worst_obj_fl is very close to worst_floor,
            # it's floor penetration (not object). Otherwise object collision.
            floor_dominant = worst_floor < worst_obj_fl + 0.001 and worst_floor < 0
            is_act_link = act_prefix is not None and f"_{act_prefix}_" in nm
            # Determine violation reasons
            reasons = []
            if worst_floor < -0.003:  # 3mm into floor
                reasons.append(f"floor={worst_floor*1000:+.1f}mm")
            if not floor_dominant:
                obj_limit = -0.005 if "_ds" in nm else -0.003
                if "_ds" in nm and is_act_link:
                    obj_limit = -999  # act ds can be anywhere near trigger
                if worst_obj_fl < obj_limit:
                    reasons.append(f"obj={worst_obj_fl*1000:+.1f}mm")
            if (not is_act_link) and worst_cl < 0:
                reasons.append(f"clear={worst_cl*1000:+.1f}mm")
            is_red = len(reasons) > 0
            link_status[nm] = (worst_obj_fl*1000, worst_cl*1000, worst_floor*1000, is_red, reasons)
        reds = [(nm, link_status[nm][4]) for nm in link_status if link_status[nm][3]]
        print(f"[VISER] grasp {grasp_idx} stage={stage} act={act_prefix} RED: {reds}")

    # Draw hand with per-link colors
    for (ln, mi), (v, f, vp) in link_cache.items():
        if ln not in fk: continue
        wT = T @ fk[ln].get_matrix()[0].numpy()
        if vp is not None:
            vpa = np.array(vp)
            Rv = Rotation.from_euler("xyz", vpa[3:]).as_matrix()
            Tv = np.eye(4); Tv[:3, :3] = Rv; Tv[:3, 3] = vpa[:3]
            wT = wT @ Tv
        vw = (wT[:3, :3] @ v.T).T + wT[:3, 3]
        # Color: red if violating, else default (blue palm / orange fingers)
        ls = link_status.get(ln)
        is_red = ls[3] if ls is not None else False
        if is_red:
            c = (255, 30, 30)  # red for bad
        elif "palm" in ln:
            c = (50, 100, 255)
        else:
            c = (255, 200, 100)
        server.scene.add_mesh_simple(f"/hand/{ln}_{mi}",
            vertices=vw.astype(np.float32), faces=f, color=c, opacity=0.85)

    # Debug: collision-check points overlay (toggleable). Shows where the SDF
    # is being queried — should align with visual mesh. Colored by violation type:
    # green = outside everything; red = in object; blue = below floor; yellow = in clearance.
    if show_col_pts[0] and sdf is not None and opt is not None:
        dev = torch.device("cuda")
        T_t = torch.tensor(T, dtype=torch.float32, device=dev).unsqueeze(0)
        all_pts_world = []
        all_colors = []
        for li, (nm, pts) in enumerate(opt._col_data):
            if nm not in fk: continue
            with torch.no_grad():
                lwT = T_t @ fk[nm].get_matrix().to(dev)
                ph = pts[:, :4].float()
                lwp = (lwT @ ph.T)[:, :3, :].transpose(1, 2)
                # query returns min(obj, clearance, floor) when include_clearance=True.
                # We want to separate — so query each component.
                obj_floor_sdf = sdf.query(lwp, include_clearance=False)[0].cpu().numpy()
                cl_sdf = sdf._clearance_sdf(lwp)[0].cpu().numpy() if hasattr(sdf, '_clearance_center') else np.full(lwp.shape[1], float('inf'))
                floor_sdf = sdf._floor_sdf(lwp)[0].cpu().numpy() if hasattr(sdf, '_floor_z') else np.full(lwp.shape[1], float('inf'))
                world_pts = lwp[0].cpu().numpy()
            is_act_link = act_prefix is not None and f"_{act_prefix}_" in nm
            for pi in range(len(world_pts)):
                in_floor = floor_sdf[pi] < 0
                # object: if obj_floor_sdf < 0 and not explained by floor
                in_obj = obj_floor_sdf[pi] < 0 and not (in_floor and abs(floor_sdf[pi] - obj_floor_sdf[pi]) < 0.001)
                in_cl = (not is_act_link) and cl_sdf[pi] < 0
                all_pts_world.append(world_pts[pi])
                if in_obj:
                    all_colors.append([255, 0, 0])       # red: in object
                elif in_floor:
                    all_colors.append([30, 30, 255])     # blue: below floor
                elif in_cl:
                    all_colors.append([255, 220, 0])     # yellow: in clearance
                else:
                    all_colors.append([0, 200, 0])       # green: OK
        if all_pts_world:
            all_pts_world = np.array(all_pts_world, dtype=np.float32)
            all_colors = np.array(all_colors, dtype=np.uint8)
            server.scene.add_point_cloud("/col_pts", points=all_pts_world,
                colors=all_colors, point_size=0.002, point_shape="circle")
    else:
        # Clear point cloud if toggled off
        server.scene.add_point_cloud("/col_pts", points=np.zeros((1,3), dtype=np.float32),
            colors=np.array([[0,0,0]], dtype=np.uint8), point_size=0.0001)

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

    # ---------------------------------------------------------------------
    # Diagnostic panel for a dexterous grasp.
    # Aesthetic: engineering telemetry — verdict up top, prose diagnosis for
    # failures, aligned code blocks for scanning numbers, no redundancy.
    # MDX-safe: no "<digit" / ">digit" patterns (parsed as JSX tags).
    # ---------------------------------------------------------------------

    # Thresholds (must match solver's feasibility check)
    TH = {
        'surf':   (0.008,  '<'),   # fingertip distance from surface
        'col':    (0.003,  '<'),   # hand body object penetration
        'back':   (-0.003, '>'),   # fingertip back-side penetration
        'pad':    (-0.005, '>'),   # fingertip pad-side penetration
        'sc':     (-0.001, '>'),   # box-box inter-finger overlap
        'sigma':  (0.01,   '>'),   # min singular value of wrench matrix
        'act':    (0.010,  '<'),   # actuation tip distance to target
        'pen':    (5.0,    '<'),   # verification penetration percentage
        'sc_pts': (0.0005, '>'),   # point-cloud self-collision distance
    }

    # Raw fields
    sigma     = g.get("sigma_min", 0)
    lstar     = g.get("l_star", -1)
    feasible  = g.get("feasible", False)
    surf_err  = g.get("surf_err", 0)
    act_dist  = g.get("act_dist", None)
    col_viol  = g.get("max_col_viol", None)
    ds_back   = g.get("ds_back_worst", None)
    ds_pad    = g.get("ds_pad_worst", None)
    sc        = g.get("sc_min_dist", None)
    pen       = g.get("mesh_pen_pct", None)
    sc_pts    = g.get("sc_worst", None)
    act_fi    = g.get("act_finger", 0)
    act_short = FINGER_LABELS[act_fi] if act_fi < 4 else f'F{act_fi}'
    act_full  = {'IF': 'index', 'MF': 'middle', 'RF': 'ring', 'TH': 'thumb'}.get(act_short, '?')

    def passes(name, val):
        if val is None: return True
        thresh, op = TH[name]
        return (val < thresh) if op == '<' else (val > thresh)

    # Terse fragment describing a failing criterion — just the numeric fact.
    def reason_for(name, val):
        if name == 'surf':
            return f"fingertips {val*1000:.1f} mm from surface (limit 8)"
        if name == 'col':
            return f"body links {val*1000:.1f} mm into object (limit 3)"
        if name == 'back':
            return f"fingertip back {abs(val)*1000:.1f} mm into object (limit 3)"
        if name == 'pad':
            return f"fingertip pad {abs(val)*1000:.1f} mm into object (limit 5)"
        if name == 'sc':
            return f"inter-finger overlap {abs(val)*1000:.1f} mm (limit 1)"
        if name == 'sigma':
            return f"σ_min {val:.4f} (limit 0.01)"
        if name == 'act':
            return f"actuation tip {val*1000:.1f} mm from trigger (limit 10)"
        if name == 'pen':
            return f"{val:.1f}% URDF points inside object (limit 5)"
        if name == 'sc_pts':
            return f"point self-col {val*1000:.2f} mm (limit 0.5)"
        return f"{name} = {val}"

    # Human-readable link name
    def friendly_link(short_name):
        segments = {'bs': 'base link', 'px': 'proximal', 'md': 'middle',
                    'ds': 'fingertip', 'mp': 'MCP link'}
        parts = {'if': 'index', 'mf': 'middle', 'rf': 'ring', 'th': 'thumb'}
        if short_name == 'palm':
            return 'palm'
        f, s = short_name.split('_', 1)
        return f"{parts.get(f, f)} {segments.get(s, s)}"

    # Split each link's reasons into object-vs-env and self-collision buckets
    def split_reasons(reasons):
        obj_entries = []   # (link, detail) — collision with object/floor
        self_entries = []  # (link, detail) — in clearance / inter-finger
        for r in reasons:
            k, v = r.split("=")
            mm = abs(float(v.replace("mm", "")))
            if k == "obj":
                obj_entries.append(f"{mm:.1f} mm into object")
            elif k == "floor":
                obj_entries.append(f"{mm:.1f} mm below table")
            elif k == "clear":
                self_entries.append(f"{mm:.1f} mm into actuation clearance")
        return obj_entries, self_entries

    # --- Compose panel ---------------------------------------------------
    # Header (title + verdict) goes ABOVE the controls in a separate widget.
    tag = '✅ FEASIBLE' if feasible else '❌ INFEASIBLE'
    header_lines = [
        f"### {obj_name} · grasp {grasp_idx}",
        "",
        f"**{tag}**",
    ]
    header_md.content = "\n".join(header_lines)

    # Rest of the panel (below the controls)
    lines = []

    # 3 — Diagnosis (prose, only if something is wrong)
    # Failure severity order — most impactful first
    order = ['pad', 'back', 'col', 'sc', 'surf', 'sigma', 'act', 'pen', 'sc_pts']
    checks_map = {'pad': ds_pad, 'back': ds_back, 'col': col_viol, 'sc': sc,
                  'surf': surf_err, 'sigma': sigma, 'act': act_dist,
                  'pen': pen, 'sc_pts': sc_pts}
    fails = [n for n in order if n in checks_map and checks_map[n] is not None
             and not passes(n, checks_map[n])]

    # Treat ℓ* ≤ 0 as a semantic failure (no true force closure) even when
    # the feasibility check only requires σ_min > 0.01. This matters for the
    # reader — a grasp can pass σ while still not force-closing.
    lstar_fail = lstar <= 0

    if not feasible or lstar_fail:
        lines.append("**Diagnosis**")
        lines.append("")
        for nm in fails:
            lines.append(f"- {reason_for(nm, checks_map[nm])}")
        if lstar_fail and 'sigma' not in fails:
            lines.append(f"- ℓ\\* = {lstar:+.2f}")
        lines.append("")

    # 4 — Key metrics block (monospace, aligned, pass/fail tag, all values)
    def row(name, val, label, unit='mm'):
        if val is None:
            return f"{label:<20}    —"
        ok = passes(name, val)
        disp = val * (1 if unit == '%' else 1000)
        flag = 'ok  ' if ok else 'FAIL'
        return f"{label:<20}   {disp:>+7.2f} {unit:<2}   [{flag}]"

    lines.append("**Metrics**")
    lines.append("```")
    # Always show: target contact + stability
    lines.append(f"{'actuation finger':<20}   {act_full:>10}")
    lines.append(f"{'actuation target':<20}   {act_dist*1000 if act_dist is not None else 0:>+7.2f} mm   " +
                 (f"[ok  ]" if passes('act', act_dist) else "[FAIL]"))
    lines.append(f"{'sigma (stability)':<20}   {sigma:>7.4f}      " +
                 (f"[ok  ]" if passes('sigma', sigma) else "[FAIL]"))
    lines.append(f"{'l_star (force cl.)':<20}   {lstar:>+7.4f}      " +
                 ("[ok  ]" if lstar > 0 else "[FAIL]"))
    lines.append("")
    lines.append(f"{'surface reach':<20}   {surf_err*1000:>+7.2f} mm   " +
                 (f"[ok  ]" if passes('surf', surf_err) else "[FAIL]"))
    if col_viol is not None:
        lines.append(row('col', col_viol, 'body object col'))
    if ds_back is not None:
        lines.append(row('back', ds_back, 'fingertip back'))
    if ds_pad is not None:
        lines.append(row('pad', ds_pad, 'fingertip pad'))
    if sc is not None:
        lines.append(row('sc', sc, 'inter-finger gap'))
    lines.append("")
    if pen is not None:
        lines.append(row('pen', pen, 'verify penetration', unit='%'))
    if sc_pts is not None:
        lines.append(row('sc_pts', sc_pts, 'point self-col'))
    lines.append("```")
    lines.append("")

    # 5 — Per-link collisions split into object vs self buckets
    bad_links = [(nm, ls) for nm, ls in link_status.items() if ls[3]]
    obj_rows = []   # (name, details) — vs object/floor
    self_rows = []  # (name, details) — vs other hand parts / clearance zone
    for nm, (obj, cl, fl, _, reasons) in bad_links:
        short = nm.replace("leap_rh_", "")
        nice = friendly_link(short)
        obj_r, self_r = split_reasons(reasons)
        if obj_r:
            obj_rows.append((nice, ', '.join(obj_r)))
        if self_r:
            self_rows.append((nice, ', '.join(self_r)))

    # Inter-finger box-box overlap (sc metric) is conceptually self-collision
    # but does NOT tie to a single link; report it separately if it fails.
    if sc is not None and not passes('sc', sc):
        self_rows.append(('inter-finger (box-box)', f"{abs(sc)*1000:.1f} mm overlap"))

    if obj_rows:
        lines.append("**Object collisions**")
        lines.append("```")
        longest = max(len(n) for n, _ in obj_rows) + 2
        for name, detail in obj_rows:
            lines.append(f"{name:<{longest}} {detail}")
        lines.append("```")
        lines.append("")

    if self_rows:
        lines.append("**Self-collisions**")
        lines.append("```")
        longest = max(len(n) for n, _ in self_rows) + 2
        for name, detail in self_rows:
            lines.append(f"{name:<{longest}} {detail}")
        lines.append("```")
        lines.append("")

    # 6 — Stage evolution (code block, aligned)
    stage_map = {'a_init': 'init', 'b_support_ik': 'support IK', 'c_optimized': 'optimized'}
    stage_match = []
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
            label = stage_map.get(sn, sn)
            stage_match.append(f"{label:<14}   σ={s:.4f}   l*={l:>+7.4f}   tip={se*1000:>5.1f} mm")
    if stage_match:
        lines.append("**Stage evolution**")
        lines.append("```")
        for s in stage_match:
            lines.append(s)
        lines.append("```")

    content = "\n".join(lines)
    info_md.content = content
    open("/tmp/viser_current.md","w").write(content)
    print(f"[VISER] info_md updated, len={len(content)}")


# GUI
with server.gui.add_folder("Grasp Browser"):
    # Top: title + verdict (always a quick glance)
    header_md = server.gui.add_markdown("")
    # Controls directly under the verdict
    obj_dd = server.gui.add_dropdown("Object", options=available_objects, initial_value=available_objects[0])
    # Stage options (will update when object changes)
    data0 = load_object(available_objects[0])
    stage_opts = list(data0["stages"].keys())
    stage_dd = server.gui.add_dropdown("Stage", options=stage_opts if stage_opts else ["none"],
                                        initial_value=stage_opts[-1] if stage_opts else "none")
    gi_slider = server.gui.add_slider("Grasp", min=0, max=9, step=1, initial_value=0)
    btn_prev = server.gui.add_button("< Prev")
    btn_next = server.gui.add_button("Next >")
    show_col_cb = server.gui.add_checkbox("Show collision points", initial_value=False)
    # Below controls: the detailed diagnosis / metrics / collisions / stages
    info_md = server.gui.add_markdown("")

@show_col_cb.on_update
def _(_):
    show_col_pts[0] = show_col_cb.value
    show()


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
while True: time.sleep(1)
