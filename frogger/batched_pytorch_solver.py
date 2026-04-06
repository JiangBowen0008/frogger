"""
Batched differentiable grasp optimizer for dexterous hands using PyTorch.

Implements the FRoGGeR formulation (Li et al., IROS 2023):
  - Optimises the min-weight metric l* for force-closure quality.
  - Surface contact equality constraints (fingertips on object surface).
  - Collision avoidance inequality constraints.
  - Joint-limit enforcement via sigmoid parameterisation.
  - Contact normals from differentiable SDF gradients.
  - Grasp map G and wrench matrix W computed in PyTorch on GPU.
  - Min-weight LP solved per candidate; analytic gradient ∇l*.
"""

import torch
import torch.nn.functional as F
import pytorch_kinematics as pk
import numpy as np
import os
import trimesh
import open3d as o3d
from typing import Optional, List, Tuple
from scipy.optimize import linprog as scipy_linprog
import time


# ---------------------------------------------------------------------------
# SDF grid
# ---------------------------------------------------------------------------
class BatchedSDF:
    """Pre-computed signed-distance field on a regular 3-D grid."""

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        X_WO_matrix: np.ndarray,
        bounds_padding: float = 0.15,
        resolution: int = 128,
        device: str = "cuda",
    ):
        self.device = device
        self.resolution = resolution

        verts_O = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        R_WO = X_WO_matrix[:3, :3].astype(np.float64)
        t_WO = X_WO_matrix[:3, 3].astype(np.float64)
        verts_W = (R_WO @ verts_O.astype(np.float64).T).T + t_WO
        self._verts_W = verts_W  # store for OBB computation

        self.bbox_min = np.min(verts_W, axis=0) - bounds_padding
        self.bbox_max = np.max(verts_W, axis=0) + bounds_padding

        print(f"  Computing {resolution}^3 SDF grid ...")
        lin = [
            np.linspace(self.bbox_min[i], self.bbox_max[i], resolution)
            for i in range(3)
        ]
        gx, gy, gz = np.meshgrid(*lin, indexing="ij")
        pts_W = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)

        # Object-frame query via Open3D
        R_OW = R_WO.T
        pts_O = ((pts_W - t_WO) @ R_OW.T).astype(np.float32)
        mesh_o3d = o3d.t.geometry.TriangleMesh()
        mesh_o3d.vertex.positions = o3d.core.Tensor(verts_O)
        mesh_o3d.triangle.indices = o3d.core.Tensor(faces)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_o3d)
        sdf_vals = scene.compute_signed_distance(
            o3d.core.Tensor(pts_O, dtype=o3d.core.float32)
        ).numpy()
        print(f"  SDF range [{sdf_vals.min():.4f}, {sdf_vals.max():.4f}]")

        self.sdf_tensor = (
            torch.tensor(
                sdf_vals.reshape(resolution, resolution, resolution),
                dtype=torch.float32,
                device=device,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )  # [1,1,D,H,W]
        self.bbox_min_t = torch.tensor(
            self.bbox_min, dtype=torch.float32, device=device
        )
        self.bbox_max_t = torch.tensor(
            self.bbox_max, dtype=torch.float32, device=device
        )
        self.range_t = self.bbox_max_t - self.bbox_min_t

        # Tight object bounding box (world frame, no padding)
        verts_f32 = verts_W.astype(np.float32)
        self.obj_bbox_min = torch.tensor(
            verts_f32.min(axis=0), dtype=torch.float32, device=device
        )
        self.obj_bbox_max = torch.tensor(
            verts_f32.max(axis=0), dtype=torch.float32, device=device
        )
        self.obj_center = (self.obj_bbox_min + self.obj_bbox_max) / 2

    def query(self, points: torch.Tensor) -> torch.Tensor:
        """Differentiable SDF look-up.  points [B,N,3] -> [B,N]."""
        B, N, _ = points.shape
        norm = 2.0 * (points - self.bbox_min_t) / self.range_t - 1.0
        # grid_sample expects (w, h, d) = (Z, Y, X) for tensor stored as [D=X, H=Y, W=Z]
        norm_grid = norm[..., [2, 1, 0]]
        out = F.grid_sample(
            self.sdf_tensor.expand(B, -1, -1, -1, -1),
            norm_grid.view(B, N, 1, 1, 3),
            align_corners=True,
            padding_mode="border",
        )
        return out.view(B, N)

    def query_with_normals(self, points: torch.Tensor):
        """SDF look-up returning (sdf_values [B,N], inward_normals [B,N,3]).

        Inward normals = -∇s / |∇s| (pointing INTO the object).
        Works both inside and outside torch.no_grad() context.
        """
        needs_grad = torch.is_grad_enabled() or points.requires_grad

        if needs_grad:
            pts = points.detach().requires_grad_(True)
            sdf = self.query(pts)
            grad_outputs = torch.ones_like(sdf)
            (grad,) = torch.autograd.grad(sdf, pts, grad_outputs, create_graph=False)
        else:
            # Finite differences for gradient when autograd is not available
            eps = 1e-4
            grad = torch.zeros_like(points)
            for d in range(3):
                pts_p = points.clone()
                pts_m = points.clone()
                pts_p[..., d] += eps
                pts_m[..., d] -= eps
                grad[..., d] = (self.query(pts_p) - self.query(pts_m)) / (2 * eps)
            sdf = self.query(points)

        norms = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        inward = -grad / norms
        return sdf, inward


# ---------------------------------------------------------------------------
# Grasp wrench space helpers (FRoGGeR formulation)
# ---------------------------------------------------------------------------

def compute_primitive_forces_torch(ns: int, mu: float, device="cuda"):
    """Pyramidal friction cone approximation.  Returns F [3, ns]."""
    nums = torch.arange(ns, dtype=torch.float32, device=device)
    scale = 1.0 / (1.0 + mu ** 2) ** 0.5
    fx = mu * scale * torch.cos(2 * np.pi * nums / ns)
    fy = mu * scale * torch.sin(2 * np.pi * nums / ns)
    fz = scale * torch.ones(ns, device=device)
    return torch.stack([fx, fy, fz], dim=0)  # [3, ns]


def compute_contact_frames(positions, inward_normals):
    """Build contact-to-object-frame transforms from positions and normals.

    Args:
        positions: [B, nc, 3] contact positions in object frame
        inward_normals: [B, nc, 3] inward-pointing normals in object frame

    Returns:
        g_OCs: [B, nc, 4, 4] homogeneous transforms
    """
    B, nc, _ = positions.shape
    dev = positions.device
    n = inward_normals  # [B, nc, 3]

    # Build tangent vectors via cross product with arbitrary axis
    zeta = torch.tensor([1.2, 2.3, 3.4], device=dev)
    zeta_p = torch.tensor([3.4, 2.3, 1.2], device=dev)

    # tx = zeta - (n·zeta)*n, handle degenerate case
    # Check if normal is too close to zeta
    diff = (n - zeta.view(1, 1, 3)).norm(dim=-1)  # [B, nc]
    arb = zeta.view(1, 1, 3).expand(B, nc, 3).clone()
    arb[diff <= 1e-6] = zeta_p

    dot_nt = (n * arb).sum(-1, keepdim=True)  # [B, nc, 1]
    tx = arb - n * dot_nt
    tx = tx / tx.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    ty = torch.cross(n, tx, dim=-1)
    ty = ty / ty.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # g_OC: rotation columns = [tx, ty, n], translation = position
    R = torch.stack([tx, ty, n], dim=-1)  # [B, nc, 3, 3]
    g = torch.zeros(B, nc, 4, 4, device=dev)
    g[:, :, :3, :3] = R
    g[:, :, :3, 3] = positions
    g[:, :, 3, 3] = 1.0
    return g


def compute_grasp_matrix_torch(g_OCs):
    """Compute grasp matrix G from contact frames.

    Args:
        g_OCs: [B, nc, 4, 4] contact frame transforms

    Returns:
        G: [B, 6, nc*3] grasp matrix (hard contact model)
    """
    B, nc = g_OCs.shape[:2]
    dev = g_OCs.device

    # Following the original FRoGGeR (grasping.py):
    # 1. g_inv = g_OC^{-1} = [[R^T, -R^T p], [0, 1]]
    # 2. Ad(g_inv) = [[R^T, [(-R^T p)]× @ R^T], [0, R^T]]
    #    where [a]× is the skew-symmetric matrix of a
    # 3. G_i = Ad(g_inv)^T @ Bc
    R = g_OCs[:, :, :3, :3]  # [B, nc, 3, 3]
    p = g_OCs[:, :, :3, 3]   # [B, nc, 3]
    Rt = R.transpose(-1, -2)  # [B, nc, 3, 3]

    # p_inv = -R^T @ p (translation of g_inv)
    p_inv = -(Rt @ p.unsqueeze(-1)).squeeze(-1)  # [B, nc, 3]

    # wedge(p_inv) = skew-symmetric matrix of p_inv
    a1, a2, a3 = p_inv[..., 0], p_inv[..., 1], p_inv[..., 2]
    skew_p = torch.zeros(B, nc, 3, 3, device=dev)
    skew_p[..., 0, 1] = -a3; skew_p[..., 0, 2] = a2
    skew_p[..., 1, 0] = a3;  skew_p[..., 1, 2] = -a1
    skew_p[..., 2, 0] = -a2; skew_p[..., 2, 1] = a1

    # Ad(g_inv) = [[Rt, skew_p @ Rt], [0, Rt]]
    Ad_ginv = torch.zeros(B, nc, 6, 6, device=dev)
    Ad_ginv[:, :, :3, :3] = Rt
    Ad_ginv[:, :, 3:, 3:] = Rt
    Ad_ginv[:, :, :3, 3:] = skew_p @ Rt

    # Ad(g_inv)^T
    Ad_ginv_T = Ad_ginv.transpose(-1, -2)

    # Hard contact basis: Bc [6, 3] = [[I_3], [0_3]]
    Bc = torch.zeros(6, 3, device=dev)
    Bc[:3, :3] = torch.eye(3, device=dev)

    # G_i = Ad_ginv_T @ Bc → [B, nc, 6, 3]
    G_parts = Ad_ginv_T @ Bc.unsqueeze(0).unsqueeze(0)

    # Stack: G = [G_1, G_2, ..., G_nc] → [B, 6, nc*3]
    G = G_parts.permute(0, 2, 1, 3).reshape(B, 6, nc * 3)
    return G


def compute_wrench_matrix(G, F_prim, nc, ns):
    """Compute wrench matrix W = G @ kron(I_nc, F).

    Args:
        G: [B, 6, nc*3] grasp matrix
        F_prim: [3, ns] primitive forces
        nc: number of contacts
        ns: number of friction cone sides

    Returns:
        W: [B, 6, nc*ns] wrench matrix
    """
    B = G.shape[0]
    dev = G.device
    # kron(I_nc, F) is block diagonal: [nc*3, nc*ns]
    kron = torch.zeros(nc * 3, nc * ns, device=dev)
    for i in range(nc):
        kron[i*3:(i+1)*3, i*ns:(i+1)*ns] = F_prim
    W = G @ kron  # [B, 6, nc*ns]
    return W


def solve_min_weight_lp_batch(W_batch_np):
    """Solve min-weight LP for a batch of wrench matrices.

    For each W [6, m], solves:
        max l  s.t. W @ alpha = 0, 1^T alpha = 1, alpha >= l

    Args:
        W_batch_np: numpy array [B, 6, m]

    Returns:
        l_stars: [B] optimal min-weight values
        alphas: [B, m] optimal weights
        lambs: [B, m] dual variables for inequality constraints
        nus: [B, 7] dual variables for equality constraints
    """
    B, _, m = W_batch_np.shape
    l_stars = np.full(B, -1.0)
    alphas = np.zeros((B, m))
    lambs = np.zeros((B, m))
    nus = np.zeros((B, 7))

    # LP in standard form (scipy minimises):
    # min -l  s.t. W @ alpha = 0, 1^T alpha = 1, alpha - l >= 0
    # Decision variable: x = [alpha (m), l (1)]
    c = np.zeros(m + 1)
    c[-1] = -1.0  # minimise -l = maximise l

    # Inequality: -alpha_i + l <= 0  ↔  alpha_i >= l
    A_ub = np.zeros((m, m + 1))
    A_ub[:, :m] = -np.eye(m)
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(m)

    for b in range(B):
        W = W_batch_np[b]
        A_eq = np.zeros((7, m + 1))
        A_eq[:6, :m] = W
        A_eq[6, :m] = 1.0

        try:
            res = scipy_linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.array([0.,0.,0.,0.,0.,0.,1.]),
                                method='highs', options={'presolve': False})
            if res.success:
                l_stars[b] = res.x[-1]
                alphas[b] = res.x[:m]
                # Extract dual variables for KKT gradient computation
                lambs[b] = -res.ineqlin.marginals  # sign convention: scipy uses negative
                nus[b] = -res.eqlin.marginals
        except Exception:
            pass

    return l_stars, alphas, lambs, nus


def min_weight_gradient_batch(W_np, l_stars, alphas, lambs, nus, device="cuda"):
    """Compute ∂l*/∂W for a batch using the KKT sensitivity exploit (Prop. 1).

    Follows the original FRoGGeR implementation in metrics.py exactly:
    the gradient dl/dW is computed by solving C @ dz = -dH/dW for the last
    component of dz (which corresponds to dl), where C encodes the KKT system.

    Args:
        W_np: [B, 6, m] wrench matrices
        l_stars: [B] optimal min-weight values
        alphas: [B, m] primal LP solutions
        lambs: [B, m] dual variables for inequality constraints
        nus: [B, 7] dual variables for equality constraints

    Returns:
        dl_dW: [B, 6, m] gradient of l* w.r.t. W elements
    """
    B = W_np.shape[0]
    m = W_np.shape[2]
    dl_dW = np.zeros((B, 6, m), dtype=np.float32)

    for b in range(B):
        if l_stars[b] <= -0.99:  # LP failed entirely
            continue

        W = W_np[b]            # [6, m]
        alpha = alphas[b]      # [m]
        nu = nus[b]            # [7]
        lamb = lambs[b]        # [m]

        Im = np.eye(m)
        ones_m = np.ones((m, 1))

        # Constraint matrices in the original (non-standard) LP form:
        #   Ain = [-I_m | 1_m]  (m x m+1)   -->  -alpha_i + l <= 0
        #   Aeq = [[W | 0], [1^T | 0]]  (7 x m+1)
        Ain = np.hstack((-Im, ones_m))
        Aeq = np.zeros((7, m + 1))
        Aeq[:6, :m] = W
        Aeq[6, :m] = 1.0

        # C matrix from KKT: C = [[diag(lamb) @ Ain], [Aeq]]
        # Shape: (m + 7) x (m + 1)
        C = np.vstack((Ain * lamb[:, None], Aeq))

        # dH/dW: the KKT conditions depend on W only through the equality
        # constraint H3: Aeq @ x - beq = 0.  Specifically H3[:6] = W @ alpha.
        # For the full KKT vector (H2 complementarity, H3 stationarity):
        #   - H2 block (m rows): diag(lamb) @ (Ain @ x - bin) -- no W dependence
        #   - H3 block (7 rows): Aeq @ x - beq -- depends on W through W @ alpha
        #
        # dH3_i/dW_{ij} = alpha_j (for i<6), so the Jacobian tensor DH_W has
        # nonzero only in the H3 block (bottom 7 rows of the 2m+7 system).
        # However, our C is (m+7) x (m+1) [H2 has m rows, H3 has 7 rows].
        #
        # Following metrics.py: build the RHS matrix for the bottom block only,
        # then solve C @ Dl_W_row = -RHS for the last row of the solution.

        # RHS for the H3 block: DH3/dW has shape (7, 6*m)
        # DH3[i, i*m+j] = alpha[j] for i<6, row 6 is zero
        DH3_W = np.zeros((7, 6 * m))
        for i in range(6):
            DH3_W[i, i * m: i * m + m] = alpha

        # Full RHS: top m rows are zero (H2 has no W dependence)
        RHS = np.vstack((np.zeros((m, 6 * m)), DH3_W))  # (m+7, 6m)

        try:
            # Solve C @ dz = -RHS for dz, extract last row (dl component)
            # C is (m+7) x (m+1), overdetermined -> lstsq
            sol = np.linalg.lstsq(C, -RHS, rcond=None)[0]  # (m+1, 6m)
            dl_dW[b] = sol[-1].reshape(6, m)  # last row = dl/dW
        except Exception:
            pass

    return torch.tensor(dl_dW, dtype=torch.float32, device=device)


def compute_fc_proxy_loss(tip_positions, tip_normals, object_center):
    """Differentiable force-closure proxy loss.

    Encourages opposing contact normals and contact spread around the object.

    Args:
        tip_positions: [B, nc, 3] fingertip positions
        tip_normals: [B, nc, 3] inward-pointing normals
        object_center: [3] object center

    Returns:
        loss: [B] per-environment loss (lower = better force closure)
    """
    B, nc, _ = tip_positions.shape
    dev = tip_positions.device

    # 1) Normal opposition: penalise parallel contact normals
    dots = torch.bmm(tip_normals, tip_normals.transpose(1, 2))  # [B, nc, nc]
    triu_mask = torch.triu(torch.ones(nc, nc, device=dev), diagonal=1).bool()
    pair_dots = dots[:, triu_mask]  # [B, nc*(nc-1)/2]
    L_oppose = F.relu(pair_dots + 0.3).mean(-1)  # want dot < -0.3

    # 2) Contact spread: fingers should surround the object
    r = tip_positions - object_center.unsqueeze(0).unsqueeze(0)  # [B, nc, 3]
    r_norm = F.normalize(r, dim=-1)
    r_dots = torch.bmm(r_norm, r_norm.transpose(1, 2))  # [B, nc, nc]
    r_pair_dots = r_dots[:, triu_mask]
    # Want diverse approach directions (negative dots = opposing sides)
    L_spread = r_pair_dots.mean(-1)  # smaller = more spread

    return 2.0 * L_oppose + 1.0 * L_spread


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# -- Allegro hand ----------------------------------------------------------
_JOINT_LOWER = np.array(
    [
        -0.47, -0.196, -0.174, -0.227,
        -0.47, -0.196, -0.174, -0.227,
        -0.47, -0.196, -0.174, -0.227,
        0.263, -0.105, -0.189, -0.162,
    ],
    dtype=np.float32,
)
_JOINT_UPPER = np.array(
    [
        0.47, 1.61, 1.709, 1.618,
        0.47, 1.61, 1.709, 1.618,
        0.47, 1.61, 1.709, 1.618,
        1.396, 1.163, 1.644, 1.719,
    ],
    dtype=np.float32,
)

# -- LEAP hand (PK joint order: axl, mcp, pip, dip) ------------------------
_LEAP_JOINT_LOWER = np.array(
    [
        -0.314, -1.047, -0.506, -0.366,
        -0.314, -1.047, -0.506, -0.366,
        -0.314, -1.047, -0.506, -0.366,
        -0.349, -0.47, -1.2, -1.34,
    ],
    dtype=np.float32,
)
_LEAP_JOINT_UPPER = np.array(
    [
        2.23, 1.047, 1.885, 2.042,
        2.23, 1.047, 1.885, 2.042,
        2.23, 1.047, 1.885, 2.042,
        2.094, 2.443, 1.9, 1.88,
    ],
    dtype=np.float32,
)


def _link_names(hand, hand_type="allegro"):
    if hand_type == "leap":
        # LEAP URDFs always use 'rh' prefix in link names, regardless of hand variant
        h = "rh"
        tip = [f"leap_{h}_if_ds", f"leap_{h}_mf_ds", f"leap_{h}_rf_ds", f"leap_{h}_th_ds"]
        col = [
            f"leap_{h}_palm",
            f"leap_{h}_if_bs", f"leap_{h}_if_px", f"leap_{h}_if_md", f"leap_{h}_if_ds",
            f"leap_{h}_mf_bs", f"leap_{h}_mf_px", f"leap_{h}_mf_md", f"leap_{h}_mf_ds",
            f"leap_{h}_rf_bs", f"leap_{h}_rf_px", f"leap_{h}_rf_md", f"leap_{h}_rf_ds",
            f"leap_{h}_th_mp", f"leap_{h}_th_bs", f"leap_{h}_th_px", f"leap_{h}_th_ds",
        ]
        return tip, col
    # Allegro
    h = hand
    tip = [f"algr_{h}_if_ds", f"algr_{h}_mf_ds", f"algr_{h}_rf_ds", f"algr_{h}_th_ds"]
    col = [
        f"algr_{h}_palm",
        f"algr_{h}_if_bs", f"algr_{h}_if_px", f"algr_{h}_if_md", f"algr_{h}_if_ds",
        f"algr_{h}_mf_bs", f"algr_{h}_mf_px", f"algr_{h}_mf_md", f"algr_{h}_mf_ds",
        f"algr_{h}_rf_bs", f"algr_{h}_rf_px", f"algr_{h}_rf_md", f"algr_{h}_rf_ds",
        f"algr_{h}_th_mp", f"algr_{h}_th_bs", f"algr_{h}_th_px", f"algr_{h}_th_ds",
    ]
    return tip, col


# Visual mesh files per link (relative to models/<hand_type_dir>/)
def _visual_meshes(hand, hand_type="allegro"):
    if hand_type == "leap":
        return _leap_visual_meshes(hand)
    return _allegro_visual_meshes(hand)


def _allegro_visual_meshes(hand):
    h = hand
    base_mesh = "meshes/base_link_left.obj" if h == "lh" else "meshes/base_link_right.obj"
    base_pose = [0, 0, 0.095, -np.pi/2, 0, 0] if h == "lh" else [0, 0, 0.095, 0, 0, 0]
    th_mp_mesh = f"meshes/link_12.0_{'left' if h == 'lh' else 'right'}.obj"
    th_mp_pose = [0, 0, 0, np.pi, 0, 0] if h == "lh" else None
    finger_tip_pose = [0, 0, 0.0267, 0, 0, 0]
    thumb_tip_pose = [0, 0, 0.0423, 0, 0, 0]
    return {
        f"algr_{h}_palm":  [(base_mesh, base_pose)],
        f"algr_{h}_if_bs": [("meshes/link_0.0.obj", None)],
        f"algr_{h}_if_px": [("meshes/link_1.0.obj", None)],
        f"algr_{h}_if_md": [("meshes/link_2.0.obj", None)],
        f"algr_{h}_if_ds": [("meshes/link_3.0.obj", None), ("meshes/link_3.0_tip.obj", finger_tip_pose)],
        f"algr_{h}_mf_bs": [("meshes/link_0.0.obj", None)],
        f"algr_{h}_mf_px": [("meshes/link_1.0.obj", None)],
        f"algr_{h}_mf_md": [("meshes/link_2.0.obj", None)],
        f"algr_{h}_mf_ds": [("meshes/link_3.0.obj", None), ("meshes/link_3.0_tip.obj", finger_tip_pose)],
        f"algr_{h}_rf_bs": [("meshes/link_0.0.obj", None)],
        f"algr_{h}_rf_px": [("meshes/link_1.0.obj", None)],
        f"algr_{h}_rf_md": [("meshes/link_2.0.obj", None)],
        f"algr_{h}_rf_ds": [("meshes/link_3.0.obj", None), ("meshes/link_3.0_tip.obj", finger_tip_pose)],
        f"algr_{h}_th_mp": [(th_mp_mesh, th_mp_pose)],
        f"algr_{h}_th_bs": [("meshes/link_13.0.obj", None)],
        f"algr_{h}_th_px": [("meshes/link_14.0.obj", None)],
        f"algr_{h}_th_ds": [("meshes/link_15.0.obj", None), ("meshes/link_15.0_tip.obj", thumb_tip_pose)],
    }


def _leap_visual_meshes(hand):
    # LEAP URDFs always use 'rh' prefix in link names
    h = "rh"
    P = np.pi
    H = np.pi / 2
    # Visual origins differ between LH and RH for palm and th_mp
    if hand == "lh":
        palm_mesh = "meshes_obj/palm_lower_left.obj"
        palm_pose = [-0.0956, -0.1170, 0.0208, H, 0, P]
        th_mp_mesh = "meshes_obj/thumb_left_temp_base.obj"
        th_mp_pose = [0.0439, 0.0068, 0.0215, H, 0, 0]
    else:
        palm_mesh = "meshes_obj/palm_lower.obj"
        palm_pose = [-0.0201, 0.0258, -0.0347, 0, 0, 0]
        th_mp_mesh = "meshes_obj/pip.obj"
        th_mp_pose = [-0.0054, 0.0003, 0.0008, -H, -H, 0]
    bs_pose = [0.0084, 0.0078, 0.0147, 0, 0, 0]
    px_pose = [0.0096, 0.0003, 0.0008, -H, -H, 0]
    md_pose = [0.0211, -0.0084, 0.0098, -P, 0, 0]
    ds_pose = [0.0133, -0.0061, 0.0145, P, 0, 0]
    th_bs_pose = [0.0120, 0, -0.0159, H, 0, 0]
    th_px_pose = [0.0440, 0.0580, -0.0086, 0, 0, 0]
    th_ds_pose = [0.0626, 0.0785, 0.0490, 0, 0, 0]
    return {
        f"leap_{h}_palm":  [(palm_mesh, palm_pose)],
        f"leap_{h}_if_bs": [("meshes_obj/mcp_joint.obj", bs_pose)],
        f"leap_{h}_if_px": [("meshes_obj/pip.obj", px_pose)],
        f"leap_{h}_if_md": [("meshes_obj/dip.obj", md_pose)],
        f"leap_{h}_if_ds": [("meshes_obj/fingertip.obj", ds_pose)],
        f"leap_{h}_mf_bs": [("meshes_obj/mcp_joint.obj", bs_pose)],
        f"leap_{h}_mf_px": [("meshes_obj/pip.obj", px_pose)],
        f"leap_{h}_mf_md": [("meshes_obj/dip.obj", md_pose)],
        f"leap_{h}_mf_ds": [("meshes_obj/fingertip.obj", ds_pose)],
        f"leap_{h}_rf_bs": [("meshes_obj/mcp_joint.obj", bs_pose)],
        f"leap_{h}_rf_px": [("meshes_obj/pip.obj", px_pose)],
        f"leap_{h}_rf_md": [("meshes_obj/dip.obj", md_pose)],
        f"leap_{h}_rf_ds": [("meshes_obj/fingertip.obj", ds_pose)],
        f"leap_{h}_th_mp": [(th_mp_mesh, th_mp_pose)],
        f"leap_{h}_th_bs": [("meshes_obj/thumb_pip.obj", th_bs_pose)],
        f"leap_{h}_th_px": [("meshes_obj/thumb_dip.obj", th_px_pose)],
        f"leap_{h}_th_ds": [("meshes_obj/thumb_fingertip.obj", th_ds_pose)],
    }


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------
class BatchedGraspOptimizer:
    """GPU-batched differentiable grasp optimiser for dexterous hands."""

    def __init__(
        self,
        sdf: BatchedSDF,
        num_envs: int = 16000,
        device: str = "cuda",
        hand: str = "lh",
        hand_type: str = "allegro",
        palm_contact: bool = False,
    ):
        self.num_envs = num_envs
        self.device = device
        self.sdf = sdf
        self.hand = hand
        self.hand_type = hand_type
        self.palm_contact = palm_contact
        self.tip_link_names, self.collision_link_names = _link_names(hand, hand_type)

        # Fingertip contact offsets in body frame
        if hand_type == "leap":
            f_off = [-0.0025, -0.0449, 0.0143]
            t_off = [-0.0020, -0.0558, -0.0144]
            palm_off = [-0.004, -0.035, -0.015]  # inner palm surface center
            self.palm_link = f"leap_{hand}_palm"
        else:
            th = np.pi / 4.0
            r = 0.012
            f_off = [r * np.sin(th), 0.0, 0.0267 + r * np.cos(th)]
            t_off = [r * np.sin(th), 0.0, 0.0423 + r * np.cos(th)]
            palm_off = [0.0, 0.0, 0.0]  # Allegro palm center
            self.palm_link = f"algr_{hand}_palm"
        offsets = [f_off, f_off, f_off, t_off]
        if palm_contact:
            offsets.append(palm_off)
        self.tip_offsets = torch.tensor(offsets, dtype=torch.float32, device=device)
        self.palm_offset = torch.tensor(palm_off, dtype=torch.float32, device=device)

        # FK chain (URDF for correct joint-axis frames)
        if hand_type == "leap":
            urdf_path = os.path.join(
                os.path.dirname(__file__),
                f"../models/leap_{hand}/leap.urdf",
            )
        else:
            urdf_path = os.path.join(
                os.path.dirname(__file__),
                f"../models/allegro/allegro_{hand}.urdf",
            )
        with open(urdf_path) as fh:
            self.chain = pk.build_chain_from_urdf(fh.read()).to(device=device)
        assert len(self.chain.get_joint_parameter_names()) == 16

        if hand_type == "leap":
            self.q_lo = torch.tensor(_LEAP_JOINT_LOWER, dtype=torch.float32, device=device)
            self.q_hi = torch.tensor(_LEAP_JOINT_UPPER, dtype=torch.float32, device=device)
        else:
            self.q_lo = torch.tensor(_JOINT_LOWER, dtype=torch.float32, device=device)
            self.q_hi = torch.tensor(_JOINT_UPPER, dtype=torch.float32, device=device)
        self.q_range = self.q_hi - self.q_lo

        # Precompute dense collision-check points on each link mesh
        self._precompute_collision_points()

    # -- parameterisation -------------------------------------------------
    def _u2q(self, u):
        return self.q_lo + torch.sigmoid(u) * self.q_range

    def _q2u(self, q):
        n = ((q - self.q_lo) / self.q_range).clamp(0.01, 0.99)
        return torch.log(n / (1.0 - n))

    @staticmethod
    def _rot6d_to_matrix(r6d):
        """6D rotation -> 3x3 matrix via Gram-Schmidt."""
        a1, a2 = r6d[:, :3], r6d[:, 3:]
        b1 = F.normalize(a1, dim=-1)
        dot = (b1 * a2).sum(-1, keepdim=True)
        b2 = F.normalize(a2 - dot * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)  # [B,3,3] columns

    def _base_T(self, pos, r6d):
        B = pos.shape[0]
        R = self._rot6d_to_matrix(r6d)
        T = torch.eye(4, device=pos.device).unsqueeze(0).expand(B, -1, -1).clone()
        T[:, :3, :3] = R
        T[:, :3, 3] = pos
        return T

    # -- collision point pre-computation ----------------------------------
    @staticmethod
    def _fps(verts, n):
        """Farthest-point sampling on *verts* (N×3), returning *n* points."""
        n = min(n, len(verts))
        idx = [0]
        min_d = np.full(len(verts), np.inf)
        for _ in range(n - 1):
            d = np.linalg.norm(verts - verts[idx[-1]], axis=1)
            min_d = np.minimum(min_d, d)
            idx.append(int(np.argmax(min_d)))
        return verts[idx]

    def _precompute_collision_points(self):
        """Sample collision-check points on each link visual mesh.

        For non-distal links we sample surface points via farthest-point
        sampling.  For distal (tip) links we keep only the origin so as not
        to conflict with the surface/actuation losses.

        For non-palm links the vertices are first filtered to those within a
        lateral radius of the link's principal axis (PCA).  This removes
        motor-housing protrusions that cause spurious collision violations on
        hands with integrated motors (e.g. LEAP) while preserving coverage
        of the actual link body.
        """
        from scipy.spatial.transform import Rotation as ScipyR

        if self.hand_type == "leap":
            mesh_dir = os.path.join(os.path.dirname(__file__), f"../models/leap_{self.hand}")
        else:
            mesh_dir = os.path.join(os.path.dirname(__file__), "../models/allegro")
        vis = _visual_meshes(self.hand, self.hand_type)
        tip_set = set(self.tip_link_names)

        # Max lateral distance (from the link's principal axis) for collision
        # point candidates.  Trims motor-housing protrusions so that the
        # collision cross-section matches the functional link body.
        # Allegro's SDF defines collision boxes ~19.6×27.5 mm (max ~14 mm
        # lateral).  LEAP's visual meshes are ~35 mm across (including
        # integrated motor housings); 10 mm keeps just the inner finger body.
        if self.hand_type == "leap":
            max_lateral = 0.010  # 10 mm → ~20 mm cross-section
        else:
            max_lateral = 0.019  # 19 mm — slightly above Allegro's max

        # Tip offsets in local link frame (for excluding tip region from
        # distal link collision points).
        tip_off_np = self.tip_offsets.cpu().numpy()  # [4, 3]
        tip_off_map = dict(zip(self.tip_link_names, tip_off_np))
        tip_excl_radius = 0.010  # 10mm sphere around tip to exclude

        col_data = []  # [(link_name, local_pts [N,4] homogeneous)]
        for nm in self.collision_link_names:
            is_tip = nm in tip_set
            if nm not in vis:
                pts = np.array([[0, 0, 0]], dtype=np.float32)
            elif is_tip:
                # Distal (tip) links: sample collision points on the link
                # body, EXCLUDING the region near the tip offset where the
                # surface constraint operates. This protects the link housing
                # from penetrating the object.
                all_verts = []
                for mesh_file, vis_pose in vis[nm]:
                    path = os.path.join(mesh_dir, mesh_file)
                    if not os.path.exists(path):
                        continue
                    lm = trimesh.load(path, force="mesh")
                    verts = np.asarray(lm.vertices, dtype=np.float64)
                    if vis_pose is not None:
                        vp = np.array(vis_pose, dtype=np.float64)
                        Rv = ScipyR.from_euler("xyz", vp[3:]).as_matrix()
                        verts = (Rv @ verts.T).T + vp[:3]
                    all_verts.append(verts)
                if all_verts:
                    all_verts = np.vstack(all_verts)
                    # Exclude sphere around tip offset
                    t_off = tip_off_map[nm]
                    dists = np.linalg.norm(all_verts - t_off, axis=1)
                    keep = dists > tip_excl_radius
                    body_verts = all_verts[keep] if keep.sum() >= 32 else all_verts
                    # PCA lateral filter on body
                    if len(body_verts) > 64:
                        centered = body_verts - body_verts.mean(axis=0)
                        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                        axis = Vt[0]
                        proj = np.outer(centered @ axis, axis)
                        lat_dist = np.linalg.norm(centered - proj, axis=1)
                        mask = lat_dist < max_lateral
                        if mask.sum() >= 64:
                            body_verts = body_verts[mask]
                    pts = self._fps(body_verts, 64).astype(np.float32)
                else:
                    pts = np.array([[0, 0, 0]], dtype=np.float32)
            else:
                mesh_file, vis_pose = vis[nm][0]  # use first (body) mesh
                path = os.path.join(mesh_dir, mesh_file)
                if not os.path.exists(path):
                    pts = np.array([[0, 0, 0]], dtype=np.float32)
                else:
                    lm = trimesh.load(path, force="mesh")
                    verts = np.asarray(lm.vertices, dtype=np.float64)

                    # Transform to link frame first so filtering is in the
                    # correct coordinate system.
                    if vis_pose is not None:
                        vp = np.array(vis_pose, dtype=np.float64)
                        Rv = ScipyR.from_euler("xyz", vp[3:]).as_matrix()
                        verts = (Rv @ verts.T).T + vp[:3]
                        vis_pose = None  # already applied

                    is_palm = "palm" in nm
                    n_pts = 256 if is_palm else 48

                    if not is_palm and len(verts) > n_pts:
                        # Filter to vertices within max_lateral of the
                        # link's principal axis (PCA first component).
                        centered = verts - verts.mean(axis=0)
                        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                        axis = Vt[0]
                        proj = np.outer(centered @ axis, axis)
                        lat_dist = np.linalg.norm(centered - proj, axis=1)
                        mask = lat_dist < max_lateral
                        if mask.sum() >= n_pts:
                            verts = verts[mask]

                    pts = self._fps(verts, n_pts)

                    # vis_pose already applied above
                    pts = pts.astype(np.float32)

            pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
            col_data.append(
                (nm, torch.tensor(pts_h, device=self.device))
            )
        self._col_data = col_data

        # Per-collision-point clearance margin: collision points must stay at
        # least this far outside the object surface (sdf >= margin).
        margins = []
        for nm, pts in col_data:
            if "palm" in nm:
                m = 0.003   # 3mm clearance for palm
            elif nm in tip_set:
                m = 0.005   # 5mm clearance for distal link body
            else:
                m = 0.002   # 2mm clearance for other finger links
            margins.extend([m] * pts.shape[0])
        self._col_margins = torch.tensor(margins, dtype=torch.float32,
                                         device=self.device)

        # Group collision-point indices by finger for self-collision avoidance.
        # To keep cdist memory manageable at high batch sizes, cap each
        # finger's index list at _SC_MAX points by subsampling.
        _SC_MAX = 80
        finger_keys = ['if', 'mf', 'rf', 'th']
        _fcol = {}
        offset = 0
        for nm, pts in col_data:
            n = pts.shape[0]
            for fk in finger_keys:
                if f'_{fk}_' in nm:
                    _fcol.setdefault(fk, []).extend(range(offset, offset + n))
                    break
            offset += n
        self._self_col_pairs = []
        fk_list = [k for k in finger_keys if k in _fcol]
        for i in range(len(fk_list)):
            for j in range(i + 1, len(fk_list)):
                idx1 = _fcol[fk_list[i]]
                idx2 = _fcol[fk_list[j]]
                # Subsample if too many points (take evenly spaced)
                if len(idx1) > _SC_MAX:
                    step = len(idx1) / _SC_MAX
                    idx1 = [idx1[int(k * step)] for k in range(_SC_MAX)]
                if len(idx2) > _SC_MAX:
                    step = len(idx2) / _SC_MAX
                    idx2 = [idx2[int(k * step)] for k in range(_SC_MAX)]
                self._self_col_pairs.append((
                    torch.tensor(idx1, dtype=torch.long, device=self.device),
                    torch.tensor(idx2, dtype=torch.long, device=self.device),
                ))

    # -- FK point extraction ----------------------------------------------
    def _get_points(self, fk, bT):
        dev = self.device
        tips = []
        tip_x_axes = []  # x-axis of each fingertip frame (push direction)
        for i, nm in enumerate(self.tip_link_names):
            wT = bT @ fk[nm].get_matrix()  # [B, 4, 4]
            oh = torch.cat([self.tip_offsets[i], torch.ones(1, device=dev)])
            tips.append((wT @ oh.unsqueeze(-1)).squeeze(-1)[:, :3])
            tip_x_axes.append(wT[:, :3, 0])  # x-axis column of rotation
        # Add palm contact point if enabled
        if self.palm_contact and self.palm_link in fk:
            wT_palm = bT @ fk[self.palm_link].get_matrix()
            palm_h = torch.cat([self.palm_offset, torch.ones(1, device=dev)])
            tips.append((wT_palm @ palm_h.unsqueeze(-1)).squeeze(-1)[:, :3])
            tip_x_axes.append(wT_palm[:, :3, 0])  # palm "push" direction
        cols = []
        for nm, local_pts in self._col_data:
            if nm in fk:
                wT = bT @ fk[nm].get_matrix()          # [B, 4, 4]
                wp = (wT @ local_pts.T)[:, :3, :].transpose(1, 2)  # [B, N_i, 3]
                cols.append(wp)
        nc = len(tips)  # 4 or 5
        return (torch.stack(tips, 1), torch.cat(cols, dim=1),
                torch.stack(tip_x_axes, 1))  # [B, nc, 3]

    # -- initialisation ---------------------------------------------------
    def _init(self, center, n_act, act_positions=None, act_directions=None):
        """Initialise by sampling palm contact points ON the object surface.

        Strategy: pick random surface points on the object body, compute the
        base pose that places the palm inner surface at each point with the
        correct orientation (palm tangent to surface). This guarantees the
        palm starts in contact with the object.
        """
        B, dev = self.num_envs, self.device

        if self.hand_type == "leap":
            dq = torch.tensor(
                [0.0, 0.0, 0.5, 0.5,
                 0.0, 0.0, 0.5, 0.5,
                 0.0, 0.0, 0.5, 0.5,
                 1.0, 0.5, 0.5, 0.3],
                device=dev,
            )
        else:
            dq = torch.tensor(
                [0.0, 0.5, 0.5, 0.5,
                 0.0, 0.5, 0.5, 0.5,
                 0.0, 0.5, 0.5, 0.5,
                 1.2, 0.5, 0.5, 0.4],
                device=dev,
            )
        du = self._q2u(dq)
        self.u = (du + 0.3 * torch.randn(B, 16, device=dev)).detach().requires_grad_(True)

        c = torch.tensor(center, dtype=torch.float32, device=dev)

        # --- Surface-based palm placement ---
        # 1. Sample random surface points on the object
        # 2. Get surface normals at those points
        # 3. Compute base pose that places palm inner surface AT each point
        with torch.no_grad():
            # Sample random points near the object surface using the SDF
            # Start from random directions around center, march to surface
            verts_W = torch.tensor(self.sdf._verts_W, dtype=torch.float32, device=dev)
            n_verts = verts_W.shape[0]
            # Randomly sample vertex indices (surface points)
            idx = torch.randint(0, n_verts, (B,), device=dev)
            surf_pts = verts_W[idx]  # [B, 3] points on surface

            # Get outward normals at surface points via SDF gradient
            surf_pts_q = surf_pts.unsqueeze(1)  # [B, 1, 3]
            _, surf_normals = self.sdf.query_with_normals(surf_pts_q)
            # query_with_normals returns INWARD normals, negate for outward
            outward_normals = -surf_normals[:, 0, :]  # [B, 3]
            outward_normals = F.normalize(outward_normals, dim=-1)

            # Palm geometry (verified by FK at q=0):
            #   Inner surface center in BASE frame: [0, 0.005, 0.05]
            #   Palm inward direction in BASE frame: [0, 0, -1] (base -z)
            #   So: base -z must point toward object (= inward normal = -outward)
            #   Equivalently: base +z = outward normal (away from object)
            # Contact point near finger bases (NOT deep in palm).
            # The object rests where fingers can reach around it.
            # In base frame at q=0: finger bases are at z≈0.09, palm inner at z≈0.05.
            # Contact should be between them, closer to finger bases.
            palm_contact_base = torch.tensor([0.0, 0.005, 0.085], device=dev)

            # Orientation: base z = outward normal (so palm inner face = -z faces object)
            z_hat = outward_normals  # [B, 3]

            # y_hat: prefer OBB longest axis for finger spread along object
            obb_long = self._obb_axes[:, 0]
            y_hat = obb_long.unsqueeze(0).expand(B, -1).clone()
            y_sign = 2.0 * (torch.rand(B, device=dev) > 0.5).float() - 1.0
            y_hat = y_hat * y_sign.unsqueeze(-1)
            y_hat = y_hat + 0.15 * torch.randn(B, 3, device=dev)
            y_hat = y_hat - (y_hat * z_hat).sum(-1, keepdim=True) * z_hat
            y_hat = F.normalize(y_hat, dim=-1)
            x_hat = torch.cross(y_hat, z_hat, dim=-1)
            x_hat = F.normalize(x_hat, dim=-1)

            R_base = torch.stack([x_hat, y_hat, z_hat], dim=-1)  # [B, 3, 3]

            # Place base so that palm INNER SURFACE contact point is at the surface:
            # contact_world = R_base @ palm_contact_base + base_pos = surf_pt + margin * outward
            margin = 0.002 + 0.008 * torch.rand(B, device=dev)  # 2-10mm outside surface
            palm_target = surf_pts + margin.unsqueeze(-1) * outward_normals
            contact_in_world = (R_base @ palm_contact_base.unsqueeze(-1)).squeeze(-1)
            base_pos = palm_target - contact_in_world

        palm_pos = base_pos + 0.005 * torch.randn(B, 3, device=dev)
        self.pos = palm_pos.detach().requires_grad_(True)

        # 5) 6D rotation = [x_column, y_column] from the computed R_base
        r6d = torch.cat([x_hat, y_hat], dim=-1)
        r6d = r6d + 0.05 * torch.randn_like(r6d)  # small noise to preserve palm orientation
        self.rot6d = r6d.detach().requires_grad_(True)

        # Actuation-finger assignment
        self.amap = np.zeros((B, max(n_act, 1)), dtype=np.int64)
        if n_act:
            for b in range(B):
                self.amap[b] = [(b + i) % 4 for i in range(n_act)]
        else:
            for b in range(B):
                self.amap[b] = [0]
        self.amap_t = torch.tensor(self.amap, dtype=torch.long, device=dev)

    # -- main loop --------------------------------------------------------
    def optimize(
        self,
        actuation_targets: List[Tuple[np.ndarray, Optional[np.ndarray]]],
        object_center: np.ndarray,
        steps: int = 800,
        lr: float = 0.005,
        mu: float = 0.5,
        ns: int = 4,
        save_path: Optional[str] = None,
    ):
        """Optimise grasps using the FRoGGeR formulation.

        Phase 1: Get fingertips onto the object surface (SDF-based).
        Phase 2: Maximise the min-weight metric l* for force closure while
                 keeping fingertips on the surface and avoiding collisions.

        Args:
            actuation_targets: list of (position, direction) pairs
            object_center: object center in world frame
            steps: total optimisation steps
            lr: learning rate
            mu: friction coefficient for friction cone
            ns: number of friction cone pyramid sides
            save_path: if given, save results to this .pt file
        """
        n_act = len(actuation_targets)
        B, dev = self.num_envs, self.device
        nc = 5 if self.palm_contact else 4  # number of contacts
        m = nc * ns  # total basis wrenches

        # Compute OBB of object mesh for initialization
        verts_W = self.sdf._verts_W if hasattr(self.sdf, '_verts_W') else None
        if verts_W is None:
            # Fallback: use identity OBB axes
            self._obb_axes = torch.eye(3, device=dev)
            self._obb_lengths = torch.ones(3, device=dev)
        else:
            cov = np.cov(verts_W.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = eigvals.argsort()[::-1]
            eigvecs = eigvecs[:, order]
            proj = verts_W @ eigvecs
            lengths = proj.max(axis=0) - proj.min(axis=0)
            self._obb_axes = torch.tensor(eigvecs, dtype=torch.float32, device=dev)
            self._obb_lengths = torch.tensor(lengths, dtype=torch.float32, device=dev)

        # Extract actuation positions and directions for biased init
        act_pos_list = [t[0] for t in actuation_targets] if n_act else None
        act_dir_list = [t[1] for t in actuation_targets] if n_act else None
        self._init(object_center, n_act=n_act,
                   act_positions=act_pos_list, act_directions=act_dir_list)

        # Precompute friction cone primitive forces
        F_prim = compute_primitive_forces_torch(ns, mu, device=dev)  # [3, ns]

        ap = torch.stack([torch.tensor(t[0], dtype=torch.float32, device=dev)
                          for t in actuation_targets]) if n_act else None
        # Actuation directions (normalised push direction per target)
        ad = None
        if n_act:
            ad_list = []
            for t in actuation_targets:
                if t[1] is not None:
                    d = torch.tensor(t[1], dtype=torch.float32, device=dev)
                    ad_list.append(F.normalize(d, dim=0))
                else:
                    ad_list.append(None)
            ad = ad_list  # list of (3,) tensors or None per target

        obj_c = torch.tensor(object_center, dtype=torch.float32, device=dev)

        # -- Phase 0: Two-stage warm-start ---
        # Stage A: FREEZE base, optimize JOINTS only (batched IK: curl fingers)
        # Stage B: Optimize BASE POSITION only (approach object, bring palm near)
        p0_steps = steps // 5
        p0a = p0_steps * 2 // 3  # joints
        p0b = p0_steps - p0a      # base approach

        # Stage A: joints only
        opt0a = torch.optim.Adam([self.u], lr=lr * 3.0)
        for s in range(p0a):
            opt0a.zero_grad()
            q = self._u2q(self.u)
            bT = self._base_T(self.pos, self.rot6d)
            fk = self.chain.forward_kinematics(q)
            tp, _, _ = self._get_points(fk, bT)
            ts = self.sdf.query(tp)
            ts_abs = ts.abs()
            loss0 = 500 * ((ts ** 2).sum(-1) + 5 * ts_abs.sum(-1)
                           + 20 * ts_abs.max(dim=-1).values ** 2
                           + 10 * ts_abs.max(dim=-1).values)
            if n_act and ap is not None:
                for j in range(n_act):
                    fi = self.amap_t[:, j]
                    loss0 += 200 * ((tp[torch.arange(B, device=dev), fi] - ap[j]) ** 2).sum(-1)
            loss0.mean().backward()
            opt0a.step()

        # Stage B: base position only (keep joints and rotation frozen)
        opt0b = torch.optim.Adam([self.pos], lr=lr * 2.0)
        for s in range(p0b):
            opt0b.zero_grad()
            q = self._u2q(self.u.detach())
            bT = self._base_T(self.pos, self.rot6d.detach())
            fk = self.chain.forward_kinematics(q)
            tp, _, _ = self._get_points(fk, bT)
            ts = self.sdf.query(tp)
            ts_abs = ts.abs()
            loss0 = 500 * ((ts ** 2).sum(-1) + 5 * ts_abs.sum(-1)
                           + 20 * ts_abs.max(dim=-1).values ** 2
                           + 10 * ts_abs.max(dim=-1).values)
            loss0.mean().backward()
            opt0b.step()

        with torch.no_grad():
            q0 = self._u2q(self.u)
            bT0 = self._base_T(self.pos, self.rot6d)
            fk0 = self.chain.forward_kinematics(q0)
            tp0, _, _ = self._get_points(fk0, bT0)
            ts0 = self.sdf.query(tp0)
            print(f"  P0 done ({p0_steps} steps). Mean tip SDF: {ts0.abs().mean():.4f}")

        # -- Phase 1: Get fingertips onto surface -------------------------
        p1_steps = steps * 2 // 5
        # FREEZE rotation: palm orientation (facing object) must be preserved.
        # Only optimize joints (finger curling) and base position (palm approach).
        opt1 = torch.optim.Adam([self.u, self.pos], lr=lr)
        sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, p1_steps, lr * 0.1)
        aB = torch.arange(B, device=dev)

        t0 = time.time()
        for s in range(p1_steps):
            opt1.zero_grad()
            q = self._u2q(self.u)
            bT = self._base_T(self.pos, self.rot6d)
            fk = self.chain.forward_kinematics(q)
            tp, cp, tip_x_p1 = self._get_points(fk, bT)
            ts = self.sdf.query(tp)
            cs = self.sdf.query(cp)

            # Surface: fingertips should be ON the surface (sdf = 0)
            # L2 + L1 for strong gradient even when far from surface
            ts_abs_p1 = ts.abs()
            Ls = ((ts ** 2).sum(-1)
                  + 5.0 * ts_abs_p1.sum(-1)
                  + 20.0 * ts_abs_p1.max(dim=-1).values ** 2
                  + 10.0 * ts_abs_p1.max(dim=-1).values)
            # Collision: non-tip links must stay outside (sdf >= margin)
            pen_link = F.relu(self._col_margins - cs)
            pen_tip = F.relu(-ts - 0.0005)
            Lp = (pen_link ** 2).sum(-1) + (pen_tip ** 2).sum(-1)
            Lp = Lp + 0.1 * F.relu(self._col_margins + 0.004 - cs).sum(-1)
            # Attraction: bring tips close to surface (focus on worst finger)
            Lat = ts.abs().sum(-1) + 3 * ts.abs().max(dim=-1).values
            # Spread: encourage fingers to spread out
            pw = torch.cdist(tp, tp)
            triu = torch.triu(torch.ones(nc, nc, device=dev), diagonal=1).bool()
            Ld = -pw[:, triu].mean(-1)

            # Actuation loss (P1): guide assigned fingers toward actuation targets
            La_p1 = torch.zeros(B, device=dev)
            if n_act and ap is not None:
                for j in range(n_act):
                    fi = self.amap_t[:, j]  # [B]
                    La_p1 += ((tp[aB, fi] - ap[j]) ** 2).sum(-1)
                    # Direction: fingertip pad should face the push direction
                    if ad is not None and ad[j] is not None:
                        finger_push = tip_x_p1[aB, fi]
                        cos_align = (finger_push * ad[j]).sum(-1)
                        La_p1 += 0.2 * (1.0 - cos_align) ** 2

            # Self-collision: penalize inter-finger overlap
            L_sc = torch.zeros(B, device=dev)
            for sc_i1, sc_i2 in self._self_col_pairs:
                d = torch.cdist(cp[:, sc_i1, :], cp[:, sc_i2, :])
                min_d = d.reshape(B, -1).min(dim=-1).values
                L_sc = L_sc + F.relu(0.010 - min_d) ** 2

            # Wrapping: encourage contacts to surround the object
            wrap_dirs = F.normalize(tp - obj_c.unsqueeze(0).unsqueeze(0), dim=-1)
            L_wrap = wrap_dirs.sum(dim=1).norm(dim=-1) ** 2

            total = (100 * La_p1
                     + 500 * Ls + 500 * Lp + 300 * L_sc
                     + 60 * Lat + 3 * Ld + 10 * L_wrap)
            total.mean().backward()
            opt1.step(); sch1.step()

            if s % (p1_steps // 4) == 0:
                act_str = f" act={La_p1.mean():.3e}" if n_act else ""
                print(f"  P1 step {s:3d} | mean={total.mean():.4f} "
                      f"surf={Ls.mean():.3e} attr={Lat.mean():.3e}{act_str}")

        # -- Select top-K from Phase 1 -----------------------------------
        with torch.no_grad():
            q_final = self._u2q(self.u)
            bT_final = self._base_T(self.pos, self.rot6d)
            fk_final = self.chain.forward_kinematics(q_final)
            tp_final, _, _ = self._get_points(fk_final, bT_final)
            ts_final = self.sdf.query(tp_final)
            wrap_score = ts_final.abs().sum(-1)
            K = min(2000, B // 4)
            top_idx = wrap_score.argsort()[:K]
            print(f"  P1 done. Top-{K} mean tip SDF: {ts_final[top_idx].abs().mean():.4f}")

        # -- Phase 2: FRoGGeR refinement ----------------------------------
        M = max(4, B // K)
        B2 = K * M
        with torch.no_grad():
            u2 = self.u[top_idx].repeat(M, 1) + 0.02 * torch.randn(K * M, 16, device=dev)
            p2 = self.pos[top_idx].repeat(M, 1) + 0.002 * torch.randn(K * M, 3, device=dev)
            r2 = self.rot6d[top_idx].repeat(M, 1) + 0.01 * torch.randn(K * M, 6, device=dev)
        self.u = u2.detach().requires_grad_(True)
        self.pos = p2.detach().requires_grad_(True)
        self.rot6d = r2.detach().requires_grad_(True)

        # Re-assign actuation fingers
        self.amap = np.zeros((B2, max(n_act, 1)), dtype=np.int64)
        if n_act:
            with torch.no_grad():
                q_init = self._u2q(self.u)
                bT_init = self._base_T(self.pos, self.rot6d)
                fk_init = self.chain.forward_kinematics(q_init)
                tp_init, _, _ = self._get_points(fk_init, bT_init)
                for j in range(n_act):
                    dists = torch.norm(tp_init - ap[j].unsqueeze(0).unsqueeze(0), dim=-1)
                    closest = dists.argmin(dim=-1).cpu().numpy()
                    self.amap[:, j] = closest
                n_diverse = B2 // 4
                for b in range(n_diverse):
                    self.amap[b] = [(b + i) % 4 for i in range(n_act)]
        else:
            self.amap[:] = 0
        self.amap_t = torch.tensor(self.amap, dtype=torch.long, device=dev)
        aB2 = torch.arange(B2, device=dev)

        # ================================================================
        # Phase 2: Projected Gradient Descent
        # ================================================================
        # 1. Adam step on objective: maximise σ_min(W) for force closure
        #    + light actuation/spread/palm losses
        # 2. Project onto constraints (torch.no_grad):
        #    a) Surface: move tips onto object surface via gradient of sdf^2
        #    b) Collision: push collision points outside via SDF gradient
        #    c) Self-collision: push overlapping finger pairs apart
        # The projection enforces constraints that soft penalties cannot.

        p2_steps = steps - p1_steps
        # FREEZE rotation in Phase 2 as well — preserve palm-on-object orientation
        opt2 = torch.optim.Adam([self.u, self.pos], lr=lr * 0.5)
        sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, p2_steps, lr * 0.05)
        best_sigma = torch.full((B2,), -1.0, device=dev)
        best_u = self.u.clone().detach()
        best_p = self.pos.clone().detach()
        best_r = self.rot6d.clone().detach()
        best_score = torch.full((B2,), -float("inf"), device=dev)

        # Palm grid for proximity
        palm_grid_h = None
        if self.palm_link in self.chain.get_link_names():
            if self.hand_type == "leap":
                gy = torch.linspace(-0.055, -0.015, 3, device=dev)
                gz = torch.linspace(-0.015, 0.010, 3, device=dev)
                yy, zz = torch.meshgrid(gy, gz, indexing='ij')
                pp = torch.stack([torch.full_like(yy, -0.005), yy, zz], dim=-1).reshape(-1, 3)
            else:
                gy = torch.linspace(-0.04, 0.04, 3, device=dev)
                gz = torch.linspace(0.01, 0.05, 3, device=dev)
                yy, zz = torch.meshgrid(gy, gz, indexing='ij')
                pp = torch.stack([torch.full_like(yy, 0.005), yy, zz], dim=-1).reshape(-1, 3)
            palm_grid_h = torch.cat([pp, torch.ones(pp.shape[0], 1, device=dev)], -1)

        triu_mask = torch.triu(torch.ones(nc, nc, device=dev), diagonal=1).bool()
        eps_fd = 5e-4
        fd_offsets = torch.zeros(3, 3, device=dev)
        fd_offsets[0, 0] = eps_fd; fd_offsets[1, 1] = eps_fd; fd_offsets[2, 2] = eps_fd

        # Projection hyperparameters — tuned for stability
        # Large step sizes approximate a true projection (move directly
        # toward constraint surface). Clamped internally by autograd.
        proj_alpha_surf = 3.0    # surface projection step size (very aggressive)
        proj_alpha_col = 1.0     # collision projection step size
        proj_alpha_sc = 0.8      # self-collision projection step size
        proj_iters = 3           # projection sub-iterations per step
        sc_clearance = 0.012     # 12mm self-collision clearance

        for s in range(p2_steps):
            t_frac = s / max(p2_steps - 1, 1)

            # ==============================================================
            # ALTERNATING MINIMIZATION:
            # Even steps: constraint satisfaction (surface + collision + SC)
            # Odd steps: objective optimization (FC + actuation + spread)
            # This prevents the tug-of-war between constraints and objective
            # ==============================================================
            opt2.zero_grad()
            q = self._u2q(self.u)
            bT = self._base_T(self.pos, self.rot6d)
            fk = self.chain.forward_kinematics(q)
            tp, cp, tip_x = self._get_points(fk, bT)
            ts = self.sdf.query(tp)
            cs = self.sdf.query(cp)

            if s % 2 == 0:
                # --- CONSTRAINT STEP: surface + collision + self-collision ---
                ts_abs = ts.abs()
                L_surf = ((ts ** 2).sum(-1)
                          + 5.0 * ts_abs.sum(-1)
                          + 20.0 * ts_abs.max(-1).values ** 2
                          + 10.0 * ts_abs.max(-1).values)
                pen = F.relu(self._col_margins - cs)
                L_col = ((pen ** 2).sum(-1)
                         + 2.0 * pen.sum(-1)
                         + 5.0 * pen.max(-1).values)
                L_sc = torch.zeros(B2, device=dev)
                for sc_i1, sc_i2 in self._self_col_pairs:
                    d = torch.cdist(cp[:, sc_i1, :], cp[:, sc_i2, :])
                    min_d = d.reshape(B2, -1).min(-1).values
                    L_sc += F.relu(0.012 - min_d) ** 2 + 2.0 * F.relu(0.012 - min_d)
                # Palm proximity
                L_palm = torch.zeros(B2, device=dev)
                if palm_grid_h is not None and self.palm_link in fk:
                    wT_p = bT @ fk[self.palm_link].get_matrix()
                    pw = (wT_p @ palm_grid_h.T)[:, :3, :].transpose(1, 2)
                    ps = self.sdf.query(pw)
                    L_palm = F.relu(ps - 0.003).mean(-1)
                total = 1500 * L_surf + 1000 * L_col + 800 * L_sc + 200 * L_palm
                sigma_min = torch.zeros(B2, device=dev)  # not computed on constraint steps
            else:
                # --- OBJECTIVE STEP: FC + actuation + spread ---
                # Differentiable normals via FD
                gx_n = (self.sdf.query(tp + fd_offsets[0]) - self.sdf.query(tp - fd_offsets[0])) / (2*eps_fd)
                gy_n = (self.sdf.query(tp + fd_offsets[1]) - self.sdf.query(tp - fd_offsets[1])) / (2*eps_fd)
                gz_n = (self.sdf.query(tp + fd_offsets[2]) - self.sdf.query(tp - fd_offsets[2])) / (2*eps_fd)
                sdf_grad = torch.stack([gx_n, gy_n, gz_n], dim=-1)
                tip_normals = -sdf_grad / sdf_grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)

                g_OCs = compute_contact_frames(tp, tip_normals)
                G = compute_grasp_matrix_torch(g_OCs)
                W = compute_wrench_matrix(G, F_prim, nc, ns)
                sigma_min = torch.linalg.svdvals(W)[:, -1]
                L_fc = -sigma_min

                L_proxy = compute_fc_proxy_loss(tp, tip_normals, obj_c)
                La = torch.zeros(B2, device=dev)
                if n_act:
                    for j in range(n_act):
                        fi = self.amap_t[:, j]
                        d_sq = ((tp[aB2, fi] - ap[j]) ** 2).sum(-1)
                        d = d_sq.sqrt()
                        La += torch.where(d < 0.015, d_sq, 0.015 * d - 0.015**2 / 2)
                        if ad is not None and ad[j] is not None:
                            cos_al = (tip_x[aB2, fi] * ad[j]).sum(-1)
                            La += 0.5 * (1.0 - cos_al) ** 2
                pw_d = torch.cdist(tp, tp)
                Ld = -pw_d[:, triu_mask].mean(-1)

                fc_w = min(1.0, t_frac / 0.30) * 200.0
                proxy_w = max(0.0, 1.0 - t_frac / 0.50) * 80.0
                total = fc_w * L_fc + proxy_w * L_proxy + 600 * La + 3 * Ld

            total.mean().backward()
            opt2.step()
            sch2.step()

            # ==============================================================
            # Track best solutions
            # ==============================================================
            with torch.no_grad():
                q_eval = self._u2q(self.u)
                bT_eval = self._base_T(self.pos, self.rot6d)
                fk_eval = self.chain.forward_kinematics(q_eval)
                tp_eval, cp_eval, _ = self._get_points(fk_eval, bT_eval)
                ts_eval = self.sdf.query(tp_eval)
                cs_eval = self.sdf.query(cp_eval)

                se = ts_eval.abs().max(-1).values
                mc = cs_eval.min(-1).values

                # Check self-collision
                sc_min_d = torch.full((B2,), 1.0, device=dev)
                for sc_i1, sc_i2 in self._self_col_pairs:
                    d = torch.cdist(cp_eval[:, sc_i1, :], cp_eval[:, sc_i2, :])
                    pair_min = d.reshape(B2, -1).min(-1).values
                    sc_min_d = torch.minimum(sc_min_d, pair_min)

                feas = (se < 0.002) & (mc > -0.002) & (sc_min_d > 0.008)
                # Only update best on objective steps (odd) when sigma_min is computed
                if s % 2 == 0:
                    # Constraint step: update best based on feasibility alone
                    score = torch.where(feas, best_score + 0.001, best_score - 1.0)
                else:
                    # Objective step: score by σ_min, penalize infeasible
                    score = torch.where(
                        feas,
                        sigma_min,
                        sigma_min - 5.0 * se - 5.0 * F.relu(-mc) - 5.0 * F.relu(0.008 - sc_min_d)
                    )
                improved = score > best_score
                if improved.any():
                    best_score[improved] = score[improved]
                    best_sigma[improved] = sigma_min[improved]
                    best_u[improved] = self.u[improved]
                    best_p[improved] = self.pos[improved]
                    best_r[improved] = self.rot6d[improved]

                if s % 100 == 0 or s == p2_steps - 1:
                    n_ok = feas.sum().item()
                    n_fc = (sigma_min > 0.01).sum().item()
                    n_surf = (se < 0.002).sum().item()
                    n_nocol = (mc > -0.002).sum().item()
                    n_nosc = (sc_min_d > 0.008).sum().item()
                    bi = best_score.argmax().item()
                    print(f"  P2 {s:3d} | σ_min_best={best_sigma[bi]:.4f} "
                          f"surf<2mm={n_surf}/{B2} nocol={n_nocol}/{B2} "
                          f"noSC={n_nosc}/{B2} feas={n_ok}/{B2} fc={n_fc}/{B2}")

        elapsed = time.time() - t0
        print(f"  Done ({elapsed:.1f}s, {B} P1 -> {K} top -> {B2} P2)")

        # -- Final evaluation: rank by σ_min (force closure) + LP l* ---
        with torch.no_grad():
            qb = self._u2q(best_u)
            bT_all = self._base_T(best_p, best_r)
            fk_all = self.chain.forward_kinematics(qb)
            tp_all, cp_all, tip_x_all = self._get_points(fk_all, bT_all)
            ts_all = self.sdf.query(tp_all)
            cs_all = self.sdf.query(cp_all)
            _, tip_normals_all = self.sdf.query_with_normals(tp_all)

            g_OCs_all = compute_contact_frames(tp_all, tip_normals_all)
            G_all = compute_grasp_matrix_torch(g_OCs_all)
            W_all = compute_wrench_matrix(G_all, F_prim, nc, ns)

            # σ_min for all candidates (fast, on GPU)
            sigma_all = torch.linalg.svdvals(W_all)[:, -1]

            # LP l* for top candidates (slower, on CPU)
            W_all_np = W_all.cpu().numpy()
            final_lstars, _, _, _ = solve_min_weight_lp_batch(W_all_np)
            final_lstars_t = torch.tensor(final_lstars, dtype=torch.float32, device=dev)
            final_lbar = m * final_lstars

            surf_err = ts_all.abs().max(dim=-1).values
            min_col = cs_all.min(dim=-1).values

            act_dist = torch.zeros(B2, device=dev)
            act_dir_score = torch.zeros(B2, device=dev)
            if n_act and ap is not None:
                amap_t_all = torch.tensor(self.amap, dtype=torch.long, device=dev)
                for j in range(n_act):
                    fi = amap_t_all[:, j]
                    act_dist += torch.norm(tp_all[torch.arange(B2, device=dev), fi] - ap[j], dim=-1)
                    if ad is not None and ad[j] is not None:
                        finger_push = tip_x_all[torch.arange(B2, device=dev), fi]
                        act_dir_score += (finger_push * ad[j]).sum(-1)
                act_dist /= n_act

            feasible = (surf_err < 0.002) & (min_col > -0.002)
            if n_act:
                feasible = feasible & (act_dist < 0.008)

            # Self-collision check for all candidates
            sc_min_d_all = torch.full((B2,), 1.0, device=dev)
            for sc_i1, sc_i2 in self._self_col_pairs:
                d = torch.cdist(cp_all[:, sc_i1, :], cp_all[:, sc_i2, :])
                pair_min = d.reshape(B2, -1).min(-1).values
                sc_min_d_all = torch.minimum(sc_min_d_all, pair_min)

            feasible = feasible & (sc_min_d_all > 0.008)

            # Rank: feasible first, then by σ_min (force closure quality)
            if n_act:
                rank_score = torch.where(
                    feasible,
                    sigma_all - 3.0 * act_dist,
                    torch.tensor(-10.0, device=dev) + sigma_all - 10.0 * act_dist,
                )
            else:
                rank_score = torch.where(feasible, sigma_all, sigma_all - 10.0)
            order = rank_score.argsort(descending=True)

            bi = order[0].item()
            q1 = qb[bi:bi+1]
            bT1 = self._base_T(best_p[bi:bi+1], best_r[bi:bi+1])
            fk1 = self.chain.forward_kinematics(q1)
            tp1, cp1, _ = self._get_points(fk1, bT1)
            ts1 = self.sdf.query(tp1)
            cs1 = self.sdf.query(cp1)
            _, tn1 = self.sdf.query_with_normals(tp1)
            R_best = self._rot6d_to_matrix(best_r)

            n_feasible = feasible.sum().item()
            n_fc = (sigma_all > 0.01).sum().item()
            n_act_ok = (act_dist < 0.010).sum().item() if n_act else B2
            print(f"")
            print(f"  === Results: {n_feasible}/{B2} feasible, {n_fc}/{B2} σ_min>0.01 ===")
            if n_act:
                print(f"  === Actuation: {n_act_ok}/{B2} within 10mm, "
                      f"median dist={act_dist.median():.4f}m ===")
            print(f"  === Best (idx {bi}, σ_min={sigma_all[bi]:.4f}, "
                  f"l*={final_lstars[bi]:.4f}) ===")
            tip_names = self.tip_link_names
            if self.palm_contact:
                tip_names = list(tip_names) + [self.palm_link]
            for k, nm in enumerate(tip_names[:tp1.shape[1]]):
                print(f"    {nm}: pos={tp1[0,k].cpu().numpy()}, sdf={ts1[0,k]:.4f}")
            if n_act and ap is not None:
                for j in range(n_act):
                    fi = self.amap[bi, j]
                    dist = torch.norm(tp1[0, fi] - ap[j]).item()
                    dir_str = ""
                    if ad is not None and ad[j] is not None:
                        cos_ang = (tip_x_all[bi, fi] * ad[j]).sum().item()
                        dir_str = f", cos_align={cos_ang:.3f}"
                    print(f"    actuation[{j}]->finger{fi}: dist={dist:.4f}{dir_str}")
            print(f"    min link SDF = {cs1.min():.4f}")
            print(f"    max tip |SDF| = {surf_err[bi]:.4f}")
            print(f"    min inter-finger dist = {sc_min_d_all[bi]:.4f}")

            # --- Detailed verification for top 5 grasps ---
            print(f"\n  === VERIFICATION (top 5) ===")
            for rank in range(min(5, B2)):
                ix = order[rank].item()
                q_v = qb[ix:ix+1]
                bT_v = self._base_T(best_p[ix:ix+1], best_r[ix:ix+1])
                fk_v = self.chain.forward_kinematics(q_v)
                tp_v, cp_v, _ = self._get_points(fk_v, bT_v)
                ts_v = self.sdf.query(tp_v)
                _, tn_v = self.sdf.query_with_normals(tp_v)

                # Pairwise normal dot products
                ndots = (tn_v[0] @ tn_v[0].T)  # [nc, nc]
                triu_v = torch.triu(torch.ones(nc, nc, device=dev), diagonal=1).bool()
                pair_ndots = ndots[triu_v]  # [nc*(nc-1)/2]
                min_ndot = pair_ndots.min().item()
                has_opposing = (pair_ndots < -0.3).any().item()

                # Pairwise tip distances
                tip_dists = torch.cdist(tp_v, tp_v)[0]
                pair_tdists = tip_dists[triu_v]
                min_tdist = pair_tdists.min().item()

                # Self-collision
                sc_md = 1.0
                for sc_i1, sc_i2 in self._self_col_pairs:
                    d_v = torch.cdist(cp_v[:, sc_i1, :], cp_v[:, sc_i2, :])
                    sc_md = min(sc_md, d_v.reshape(-1).min().item())

                # Palm SDF
                palm_sdf_str = "N/A"
                if palm_grid_h is not None and self.palm_link in fk_v:
                    wT_pv = bT_v @ fk_v[self.palm_link].get_matrix()
                    pw_v = (wT_pv @ palm_grid_h.T)[:, :3, :].transpose(1, 2)
                    ps_v = self.sdf.query(pw_v)
                    palm_sdf_str = f"min={ps_v.min():.4f} mean={ps_v.mean():.4f}"

                f_tag = "FEAS" if feasible[ix] else "----"
                opp_tag = "OPP" if has_opposing else "---"
                print(f"  #{rank+1} [{f_tag}] [{opp_tag}] idx={ix} "
                      f"σ_min={sigma_all[ix]:.4f} l*={final_lstars[ix]:.4f}")
                print(f"    tip SDF: {[f'{v:.4f}' for v in ts_v[0].tolist()]}")
                print(f"    tip |SDF| max: {ts_v[0].abs().max():.4f}")
                print(f"    min tip-pair dist: {min_tdist:.4f}")
                print(f"    min inter-finger dist: {sc_md:.4f}")
                print(f"    normal dots: {[f'{v:.2f}' for v in pair_ndots.tolist()]} "
                      f"min={min_ndot:.3f}")
                print(f"    palm SDF: {palm_sdf_str}")

            res = []
            for i in range(min(10, B2)):
                ix = order[i].item()
                res.append({
                    "q_joints": qb[ix].cpu().numpy(),
                    "base_pos": best_p[ix].cpu().numpy(),
                    "base_rot": R_best[ix].cpu().numpy(),
                    "score": float(best_score[ix]),
                    "l_star": float(final_lstars[ix]),
                    "l_bar": float(final_lbar[ix]),
                    "feasible": bool(feasible[ix]),
                    "act_assignment": self.amap[ix].tolist(),
                    "act_dist": float(act_dist[ix]) if n_act else 0.0,
                    "surf_err": float(surf_err[ix]),
                    "min_col": float(min_col[ix]),
                    "sigma_min": float(sigma_all[ix]),
                    "sc_min_dist": float(sc_min_d_all[ix]),
                })
        if save_path is not None:
            import torch as _torch
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            _torch.save(res, save_path)
            print(f"  Results saved to {save_path}")
        return res
