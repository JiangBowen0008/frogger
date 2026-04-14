#!/usr/bin/env python3
"""Plot object + palm grid with labeled points and frames in 2D projections."""
import os, sys, numpy as np, torch, trimesh
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as ScipyR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from frogger.batched_pytorch_solver import BatchedSDF
import pytorch_kinematics as pk

URDF = os.path.join(os.path.dirname(__file__), "models/leap_rh/leap.urdf")
OUT = "output/diagnostics/init_debug"

def plot_init(grasp_path, mesh_path, idx=0):
    os.makedirs(OUT, exist_ok=True)

    # Load
    results = torch.load(grasp_path, weights_only=False, map_location="cpu")
    g = results[idx]
    R = g["base_rot"]
    pos = g["base_pos"]

    obj = trimesh.load(mesh_path, force="mesh")
    offset = np.array([0.0, 0.0, -obj.bounds[0, 2]])
    X_WO = np.eye(4); X_WO[:3, 3] = offset
    obj_verts = (X_WO[:3, :3] @ np.asarray(obj.vertices).T).T + X_WO[:3, 3]

    sdf = BatchedSDF(obj, X_WO, resolution=128, device="cuda")

    # Palm grid points in world frame
    tree = ET.parse(URDF)
    chain = pk.build_chain_from_urdf(open(URDF).read())
    q = torch.tensor(g["q_joints"], dtype=torch.float32).unsqueeze(0)
    fk = chain.forward_kinematics(q)
    T_base = np.eye(4); T_base[:3, :3] = R; T_base[:3, 3] = pos

    palm_pts_world = []
    for e in tree.getroot().findall("link"):
        if e.get("name") != "leap_rh_palm": continue
        link_T = fk["leap_rh_palm"].get_matrix()[0].numpy()
        wT = T_base @ link_T
        for col in e.findall("collision"):
            ge = col.find("geometry")
            b = ge.find("box") if ge is not None else None
            if b is None: continue
            sz = [float(x) for x in b.get("size").split()]
            o = col.find("origin")
            xyz = np.array([float(x) for x in o.get("xyz", "0 0 0").split()])
            rpy = np.array([float(x) for x in o.get("rpy", "0 0 0").split()])
            Rl = ScipyR.from_euler("xyz", rpy).as_matrix() if np.any(np.abs(rpy) > 1e-6) else np.eye(3)
            hx, hy, hz = sz[0]/2, sz[1]/2, sz[2]/2
            for gx in np.arange(-hx, hx+0.003, 0.005):
                for gy in np.arange(-hy, hy+0.003, 0.005):
                    for gz in np.arange(-hz, hz+0.003, 0.005):
                        local = Rl @ np.array([gx, gy, gz]) + xyz
                        world = wT[:3, :3] @ local + wT[:3, 3]
                        palm_pts_world.append(world)
    palm_pts = np.array(palm_pts_world)

    # Key points
    # Palm contact center = R @ [0.023, 0, 0.048] + pos
    palm_contact_base = np.array([0.023, -0.000, 0.048])
    palm_center = R @ palm_contact_base + pos
    x_hat = R[:, 0]
    y_hat = R[:, 1]
    z_hat = R[:, 2]

    # Use saved surface point and normal from init (if available)
    surf_pt = g.get("surf_pt", None)
    outward_normal_saved = g.get("outward_normal", None)
    z_hat_init = g.get("z_hat_init", None)

    # If not saved, try to find surface along +z
    if surf_pt is None:
        for t in np.arange(0, 0.1, 0.001):
            pt = palm_center + t * x_hat
            sv = sdf.query(torch.tensor(pt, dtype=torch.float32, device="cuda").reshape(1,1,3)).item()
            if sv <= 0:
                surf_pt = pt
                break

    # Surface normal at surf_pt
    surf_normal = outward_normal_saved  # outward normal from init
    if surf_normal is None and surf_pt is not None:
        _, n = sdf.query_with_normals(torch.tensor(surf_pt, dtype=torch.float32, device="cuda").reshape(1,1,3))
        surf_normal = -n[0, 0].cpu().numpy()

    # Plot
    ax_len = 0.04
    projs = [("XY", 0, 1, "X", "Y"), ("YZ", 1, 2, "Y", "Z"), ("XZ", 0, 2, "X", "Z")]

    for pname, a0, a1, l0, l1 in projs:
        fig, ax = plt.subplots(figsize=(12, 10))

        # Object outline
        ax.scatter(obj_verts[:, a0], obj_verts[:, a1], c="lightgray", s=0.5, alpha=0.3, zorder=1)

        # Palm grid points
        ax.scatter(palm_pts[:, a0], palm_pts[:, a1], c="blue", s=2, alpha=0.4, zorder=2, label="palm grid")

        # Palm center (magenta)
        ax.plot(palm_center[a0], palm_center[a1], 'o', c="magenta", ms=12, zorder=10, label="palm center")

        # Surface point (yellow)
        if surf_pt is not None:
            ax.plot(surf_pt[a0], surf_pt[a1], 's', c="gold", ms=12, zorder=10, label="surface point")
            # Line palm center → surface point
            ax.plot([palm_center[a0], surf_pt[a0]], [palm_center[a1], surf_pt[a1]],
                    c="magenta", lw=2, ls="--", zorder=9)

        # Surface normal at surf_pt (orange arrow)
        if surf_pt is not None and surf_normal is not None:
            ax.annotate("", xy=(surf_pt[a0] + surf_normal[a0]*ax_len, surf_pt[a1] + surf_normal[a1]*ax_len),
                        xytext=(surf_pt[a0], surf_pt[a1]),
                        arrowprops=dict(arrowstyle="->", color="orange", lw=2), zorder=11)
            ax.text(surf_pt[a0] + surf_normal[a0]*ax_len*1.1, surf_pt[a1] + surf_normal[a1]*ax_len*1.1,
                    "obj normal", fontsize=8, color="orange")

        # Frame axes at palm center
        colors = {"red": "+X (palm normal)", "green": "+Y", "blue": "+Z"}
        for ai, (c, label) in enumerate(zip(["red", "green", "blue"], ["+X", "+Y", "+Z"])):
            axis = [x_hat, y_hat, z_hat][ai]
            ax.annotate("", xy=(palm_center[a0] + axis[a0]*ax_len, palm_center[a1] + axis[a1]*ax_len),
                        xytext=(palm_center[a0], palm_center[a1]),
                        arrowprops=dict(arrowstyle="->", color=c, lw=2.5), zorder=11)
            ax.text(palm_center[a0] + axis[a0]*ax_len*1.15, palm_center[a1] + axis[a1]*ax_len*1.15,
                    label, fontsize=9, color=c, fontweight="bold")

        ax.set_xlabel(l0); ax.set_ylabel(l1)
        ax.set_title(f"Init G{idx} — {pname}")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        path = os.path.join(OUT, f"g{idx}_{pname}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--grasp", required=True, dest="grasp_path")
    p.add_argument("--mesh", required=True, dest="mesh_path")
    p.add_argument("--idx", type=int, default=0)
    plot_init(**vars(p.parse_args()))
