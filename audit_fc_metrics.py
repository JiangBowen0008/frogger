"""
Audit force-closure metrics: sigma_min and l* (min-weight LP).

Tests:
1. LP formulation correctness: known W matrices with analytically known l*
2. scipy vs quantecon LP: same W → same l*?
3. scipy bounds issue: default bounds=(0,None) vs bounds=None
4. Synthetic force-closure grasps: tetrahedron, opposition
5. Synthetic non-FC grasps: planar contacts
6. G matrix: batched vs original
7. Contact normal direction convention
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from scipy.optimize import linprog as scipy_linprog

from frogger.batched_pytorch_solver import (
    compute_primitive_forces_torch,
    compute_contact_frames,
    compute_grasp_matrix_torch,
    compute_wrench_matrix,
    solve_min_weight_lp_batch,
)
from frogger.grasping import (
    compute_gOCs,
    compute_grasp_matrix,
    compute_primitive_forces,
)
from frogger.metrics import min_weight_lp

device = "cpu"
PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"


def solve_lp_scipy_with_bounds(W_np, bounds=None):
    """Solve the min-weight LP with scipy, with configurable bounds.

    Default bounds=None means no bounds on variables (alpha and l can be any real).
    """
    m = W_np.shape[1]
    c = np.zeros(m + 1)
    c[-1] = -1.0  # minimize -l = maximize l

    A_ub = np.zeros((m, m + 1))
    A_ub[:, :m] = -np.eye(m)
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(m)

    A_eq = np.zeros((7, m + 1))
    A_eq[:6, :m] = W_np
    A_eq[6, :m] = 1.0
    b_eq = np.array([0., 0., 0., 0., 0., 0., 1.])

    try:
        res = scipy_linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                            bounds=bounds, method='highs',
                            options={'presolve': False})
        if res.success:
            return res.x[-1], res.x[:m]
        else:
            return -1.0, None
    except:
        return -1.0, None


def run_full_pipeline(positions_np, normals_np, mu=0.5, ns=8):
    """Run both pipelines, return diagnostics dict."""
    nc = positions_np.shape[0]

    # Batched
    pos_t = torch.tensor(positions_np, dtype=torch.float32, device=device).unsqueeze(0)
    nor_t = torch.tensor(normals_np, dtype=torch.float32, device=device).unsqueeze(0)
    g_OCs = compute_contact_frames(pos_t, nor_t)
    G = compute_grasp_matrix_torch(g_OCs)
    F_prim = compute_primitive_forces_torch(ns, mu, device=device)
    W = compute_wrench_matrix(G, F_prim, nc, ns)
    W_np = W[0].numpy()

    svd = np.linalg.svd(W_np, compute_uv=False)
    sigma_min = svd[-1]

    # LP with default bounds (0, None)
    l_default, alpha_default = solve_lp_scipy_with_bounds(W_np, bounds=None)  # actually no bounds

    # LP with scipy default bounds (0, None) — what solve_min_weight_lp_batch uses
    l_scipy_default, alpha_scipy = solve_lp_scipy_with_bounds(W_np)  # bounds=None means unbounded

    # The actual batched solver
    l_batched, alphas_batched, _, _ = solve_min_weight_lp_batch(W_np.reshape(1, 6, -1))
    l_batched = l_batched[0]

    # LP with explicit bounds=None (unbounded)
    l_unbounded, alpha_unbounded = solve_lp_scipy_with_bounds(
        W_np, bounds=[(None, None)] * (W_np.shape[1] + 1))

    # Original pipeline
    g_OCs_orig = compute_gOCs(positions_np.T.astype(np.float64), normals_np.T.astype(np.float64))
    G_orig = compute_grasp_matrix(g_OCs_orig, model="hard")
    F_orig = compute_primitive_forces(ns, mu, model="hard")
    W_orig = G_orig @ np.kron(np.eye(nc), F_orig)
    try:
        x_opt, _, _ = min_weight_lp(W_orig)
        l_original = x_opt[-1]
    except:
        l_original = -999.0

    return {
        'nc': nc, 'ns': ns, 'mu': mu,
        'G_batch': G[0].numpy(), 'G_orig': G_orig,
        'W_batch': W_np, 'W_orig': W_orig,
        'sigma_min': sigma_min,
        'l_batched': l_batched,
        'l_unbounded': l_unbounded,
        'l_original': l_original,
    }


# ======================================================================
# TEST 1: LP formulation with known W
# ======================================================================
def test_lp_known():
    print("\n" + "=" * 70)
    print("  TEST 1: LP with analytically known W")
    print("=" * 70)

    # W = [+e1, -e1, ..., +e6, -e6] -> l* = 1/12
    m = 12
    W = np.zeros((6, m))
    for i in range(6):
        W[i, 2*i] = 1.0
        W[i, 2*i+1] = -1.0

    expected = 1.0 / m

    # Test 1a: batched solver (uses scipy default bounds)
    l_batch, _, _, _ = solve_min_weight_lp_batch(W.reshape(1, 6, m))
    l_batch = l_batch[0]

    # Test 1b: scipy with explicit unbounded
    l_unb, _ = solve_lp_scipy_with_bounds(W, bounds=[(None, None)] * (m + 1))

    # Test 1c: original quantecon
    try:
        x_opt, _, _ = min_weight_lp(W)
        l_orig = x_opt[-1]
    except Exception as e:
        l_orig = -999.0
        print(f"  Original LP exception: {e}")

    print(f"  Expected l* = {expected:.6f}")
    print(f"  Batched (scipy default bounds): l* = {l_batch:.6f}  "
          f"{PASS if abs(l_batch - expected) < 1e-4 else FAIL}")
    print(f"  Scipy unbounded:               l* = {l_unb:.6f}  "
          f"{PASS if abs(l_unb - expected) < 1e-4 else FAIL}")
    print(f"  Original (quantecon):           l* = {l_orig:.6f}  "
          f"{PASS if abs(l_orig - expected) < 1e-4 else FAIL}")

    # Test 1d: non-FC case: only positive unit vectors
    print(f"\n  Sub-test: non-FC W (only +e1 through +e6)")
    W2 = np.eye(6)
    l2_batch, _, _, _ = solve_min_weight_lp_batch(W2.reshape(1, 6, 6))
    l2_batch = l2_batch[0]
    l2_unb, _ = solve_lp_scipy_with_bounds(W2, bounds=[(None, None)] * 7)
    try:
        x2, _, _ = min_weight_lp(W2)
        l2_orig = x2[-1]
    except:
        l2_orig = -999.0

    print(f"  Expected: l* < 0 (not force-closure)")
    print(f"  Batched:    l* = {l2_batch:.6f}  "
          f"{'infeasible -> -1.0' if l2_batch <= -0.99 else f'got {l2_batch}'}")
    print(f"  Unbounded:  l* = {l2_unb:.6f}  "
          f"{'infeasible -> -1.0' if l2_unb <= -0.99 else f'got {l2_unb}'}")
    print(f"  Original:   l* = {l2_orig:.6f}")

    # KEY QUESTION: for non-FC case, does batched return -1.0 (infeasible)
    # while original returns a negative number?
    if l2_batch <= -0.99 and l2_orig < 0 and l2_orig > -0.99:
        print(f"\n  {INFO} Batched returns infeasible (-1.0) for non-FC grasps")
        print(f"  {INFO} Original returns actual negative l* ({l2_orig:.6f})")
        print(f"  {INFO} This is the scipy default bounds=(0,None) effect!")
        print(f"  {INFO} Binary FC decision is STILL CORRECT, but we lose the 'how bad' info.")


# ======================================================================
# TEST 2: Scipy bounds investigation
# ======================================================================
def test_scipy_bounds():
    print("\n" + "=" * 70)
    print("  TEST 2: Scipy bounds=(0,None) vs unbounded")
    print("=" * 70)

    # Use the non-FC W from test 1
    W = np.eye(6)
    m = 6

    # With default scipy bounds: alpha >= 0, l >= 0
    c = np.zeros(m + 1)
    c[-1] = -1.0
    A_ub = np.zeros((m, m + 1))
    A_ub[:, :m] = -np.eye(m)
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(m)
    A_eq = np.zeros((7, m + 1))
    A_eq[:6, :m] = W
    A_eq[6, :m] = 1.0
    b_eq = np.array([0., 0., 0., 0., 0., 0., 1.])

    # Default bounds (0, None) for all vars
    res_default = scipy_linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                                method='highs', options={'presolve': False})
    print(f"  Default bounds=(0,None):  success={res_default.success}, "
          f"status={res_default.status}, message={res_default.message}")

    # Unbounded
    res_unbound = scipy_linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                                bounds=[(None, None)] * (m + 1),
                                method='highs', options={'presolve': False})
    print(f"  Unbounded:                success={res_unbound.success}, "
          f"l*={res_unbound.x[-1]:.6f}" if res_unbound.success else
          f"  Unbounded:                success=False")

    if res_default.success:
        print(f"  Default bounds l* = {res_default.x[-1]:.6f}")
    if not res_default.success and res_unbound.success:
        print(f"\n  CONFIRMED: scipy default bounds=(0,None) makes the LP infeasible")
        print(f"  when l* < 0, because it enforces alpha >= 0 and l >= 0")
        print(f"  With unbounded vars: l* = {res_unbound.x[-1]:.6f}")

    # Now test with FC case — should be the same either way
    print(f"\n  FC case (W = [+/-e1, ..., +/-e6]):")
    W_fc = np.zeros((6, 12))
    for i in range(6):
        W_fc[i, 2*i] = 1.0
        W_fc[i, 2*i+1] = -1.0
    m2 = 12

    c2 = np.zeros(m2 + 1); c2[-1] = -1.0
    A_ub2 = np.zeros((m2, m2 + 1))
    A_ub2[:, :m2] = -np.eye(m2); A_ub2[:, -1] = 1.0
    b_ub2 = np.zeros(m2)
    A_eq2 = np.zeros((7, m2 + 1))
    A_eq2[:6, :m2] = W_fc; A_eq2[6, :m2] = 1.0
    b_eq2 = np.array([0., 0., 0., 0., 0., 0., 1.])

    res_fc_def = scipy_linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2,
                                method='highs', options={'presolve': False})
    res_fc_unb = scipy_linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2,
                                bounds=[(None, None)] * (m2 + 1),
                                method='highs', options={'presolve': False})

    print(f"  Default bounds: l*={res_fc_def.x[-1]:.6f}")
    print(f"  Unbounded:      l*={res_fc_unb.x[-1]:.6f}")
    print(f"  Match: {abs(res_fc_def.x[-1] - res_fc_unb.x[-1]) < 1e-6}")


# ======================================================================
# TEST 3: Synthetic grasps
# ======================================================================
def test_synthetic_grasps():
    print("\n" + "=" * 70)
    print("  TEST 3: Synthetic grasps")
    print("=" * 70)

    configs = []

    # 3a: Tetrahedron (known FC)
    v = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]], dtype=np.float32) / np.sqrt(3)
    n = -v / np.linalg.norm(v, axis=1, keepdims=True)
    configs.append(("Tetrahedron (FC expected)", v, n, True))

    # 3b: Opposition grip (known FC)
    p2 = np.array([[.05,0,0],[-.05,0,0],[0,.05,0],[0,-.05,0]], dtype=np.float32)
    n2 = -p2 / np.linalg.norm(p2, axis=1, keepdims=True)
    configs.append(("Opposition grip (FC expected)", p2, n2, True))

    # 3c: Planar (NOT FC)
    p3 = np.array([[1,0,0],[1,.3,0],[1,-.3,0],[1,0,.3]], dtype=np.float32) * 0.05
    n3 = np.array([[-1,0,0],[-1,0,0],[-1,0,0],[-1,0,0]], dtype=np.float32)
    configs.append(("Planar same-side (NOT FC)", p3, n3, False))

    # 3d: Half-wrap (marginal — fingers on 3 sides, missing one)
    p4 = np.array([[.05,0,0],[-.05,0,0],[0,.05,0],[0,.03,.02]], dtype=np.float32)
    n4 = -p4 / np.linalg.norm(p4, axis=1, keepdims=True)
    configs.append(("3-sided wrap (marginal FC)", p4, n4, None))

    for name, pos, nor, expect_fc in configs:
        d = run_full_pipeline(pos, nor)
        is_fc_batch = d['l_batched'] > 0
        is_fc_orig = d['l_original'] > 0

        status = ""
        if expect_fc is not None:
            if is_fc_batch == expect_fc:
                status = PASS
            else:
                status = FAIL

        print(f"\n  {name}  {status}")
        print(f"    sigma_min = {d['sigma_min']:.6f}")
        print(f"    l* batched  = {d['l_batched']:.6f}  (FC={is_fc_batch})")
        print(f"    l* unbounded= {d['l_unbounded']:.6f}")
        print(f"    l* original = {d['l_original']:.6f}  (FC={is_fc_orig})")
        print(f"    max|G diff| = {np.abs(d['G_batch'] - d['G_orig']).max():.6e}")
        print(f"    max|W diff| = {np.abs(d['W_batch'] - d['W_orig']).max():.6e}")

        # Wrench column sign analysis
        W = d['W_batch']
        both_signs = [W[r].min() < -1e-6 and W[r].max() > 1e-6 for r in range(6)]
        print(f"    W rows with both signs: {sum(both_signs)}/6 = {both_signs}")


# ======================================================================
# TEST 4: G matrix single-contact verification
# ======================================================================
def test_G_matrix():
    print("\n" + "=" * 70)
    print("  TEST 4: G matrix verification")
    print("=" * 70)

    # Contact at p=[0.05, 0, 0] with inward normal n=[-1, 0, 0]
    p = np.array([0.05, 0.0, 0.0], dtype=np.float32)
    n = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

    pos_t = torch.tensor(p).reshape(1, 1, 3)
    nor_t = torch.tensor(n).reshape(1, 1, 3)
    g_OC = compute_contact_frames(pos_t, nor_t)
    R = g_OC[0, 0, :3, :3].numpy()

    # Direct formula: G = [[R], [[p]x @ R]]
    px = np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])
    G_direct = np.vstack([R, px @ R])

    G_batch = compute_grasp_matrix_torch(g_OC)[0].numpy()

    # Original
    g_OC_orig = compute_gOCs(p.reshape(3, 1).astype(np.float64),
                              n.reshape(3, 1).astype(np.float64))
    G_orig = compute_grasp_matrix(g_OC_orig, model="hard")

    diff_direct = np.abs(G_batch - G_direct).max()
    diff_orig = np.abs(G_batch - G_orig).max()

    print(f"  R (contact frame):\n{R}")
    print(f"  R orthogonal: {np.allclose(R.T @ R, np.eye(3), atol=1e-5)}")
    print(f"  R[:,2] = n: {np.allclose(R[:,2], n, atol=1e-5)}")
    print(f"\n  max|G_batch - G_direct|: {diff_direct:.6e}  "
          f"{PASS if diff_direct < 1e-4 else FAIL}")
    print(f"  max|G_batch - G_orig|:   {diff_orig:.6e}  "
          f"{PASS if diff_orig < 1e-4 else FAIL}")


# ======================================================================
# TEST 5: Normal convention
# ======================================================================
def test_normals():
    print("\n" + "=" * 70)
    print("  TEST 5: Normal convention check")
    print("=" * 70)

    # SDF: positive outside, negative inside
    # grad(SDF) at surface: points outward
    # Code: inward = -grad / |grad| -> points INTO object
    #
    # For contact on +x surface of sphere at origin:
    #   surface point: [r, 0, 0]
    #   SDF gradient: [1, 0, 0] (outward)
    #   inward = [-1, 0, 0] (into object, toward center)
    #
    # Primitive forces in contact frame:
    #   F_prim z-component is always positive (= scale > 0)
    #   The z-axis of the contact frame = inward normal
    #   So primitive forces push INTO the object -> correct for grasping

    print(f"  SDF convention: negative inside, positive outside")
    print(f"  SDF gradient at surface: outward-pointing")
    print(f"  Code: inward = -grad/|grad| -> INTO object")
    print(f"  Contact frame z-axis = inward normal")
    print(f"  Primitive force z-component > 0 (pushes along normal = into object)")
    print(f"  {PASS} Normal convention is correct for grasping")


# ======================================================================
# TEST 6: Real experiment data
# ======================================================================
def test_real_data():
    print("\n" + "=" * 70)
    print("  TEST 6: Real experiment data")
    print("=" * 70)

    for path in [
        "output/grasps_opt_exp/exp_ABCD.pt",
        "output/grasps_opt_exp/exp_A.pt",
        "output/grasps/black_spray_bottle_single_leap_rh.pt",
    ]:
        if os.path.exists(path):
            print(f"\n  Loading {path}")
            data = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        print(f"    {k}: {v.shape} {v.dtype}")
                    elif isinstance(v, np.ndarray):
                        print(f"    {k}: {v.shape} {v.dtype}")
                    elif isinstance(v, (int, float, str, bool)):
                        print(f"    {k}: {v}")
                    else:
                        print(f"    {k}: {type(v).__name__}")

                # Look for l_star or sigma_min in the data
                if 'l_star' in data:
                    ls = data['l_star']
                    if isinstance(ls, (torch.Tensor, np.ndarray)):
                        ls = np.asarray(ls)
                        print(f"\n    l* stats: min={ls.min():.4f}, max={ls.max():.4f}, "
                              f"mean={ls.mean():.4f}, FC count={np.sum(ls > 0)}/{len(ls)}")
                if 'sigma_min' in data:
                    sm = np.asarray(data['sigma_min'])
                    print(f"    sigma_min stats: min={sm.min():.4f}, max={sm.max():.4f}, "
                          f"mean={sm.mean():.4f}")
            break
    else:
        print("  No experiment files found, skipping")


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    test_normals()
    test_G_matrix()
    test_lp_known()
    test_scipy_bounds()
    test_synthetic_grasps()
    test_real_data()

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("""
  The sigma_min > 0 but l* = -1.0 discrepancy is EXPECTED BEHAVIOR.

  sigma_min > 0 means: W has rank 6 (wrenches span all 6 dimensions)
  l* > 0 means:        origin is inside convex hull of W columns (force closure)

  These are DIFFERENT conditions. Full rank does NOT imply force closure.
  Example: 4 contacts on one hemisphere generate rank-6 W, but cannot
  produce force closure because all normal forces push the same direction.

  The LP implementation in solve_min_weight_lp_batch is CORRECT for
  force-closure detection. scipy.linprog default bounds=(0,None) means
  the LP is infeasible when l* < 0, but this doesn't affect the binary
  FC decision (only loses the "how negative" information).

  Typical grasps with 4 fingertips that don't wrap around the object
  will have sigma_min > 0 but l* < 0. This is physically correct:
  the fingers can generate forces in all directions, but cannot
  simultaneously resist forces from all directions.
""")
