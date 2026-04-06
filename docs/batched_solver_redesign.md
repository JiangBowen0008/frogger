# Batched PyTorch Grasp Solver — Redesign Plan

## Current State (2026-04-05)

Branch: `fix/batched-solver-improvements`

### What works (verified against original FroGGer to machine precision)
- Contact frame construction (`compute_contact_frames`)
- Grasp matrix (`compute_grasp_matrix_torch`) — rank-6, matches original
- Wrench matrix (`compute_wrench_matrix`)
- Min-weight LP (`solve_min_weight_lp_batch`) — l* matches, KKT satisfied
- Forward kinematics via `pytorch_kinematics` — Jacobian verified
- SDF grid — mean 0.2mm error vs Open3D, normals cos_sim 0.96

### What doesn't work
- **Grasps look bad visually** despite good metrics (σ_min > 0.01 for 99%)
- **Fingers overlap** — point-cloud self-collision (80 pts/finger) has gaps
- **No opposing contacts** — optimizer clusters fingers on one side
- **Actuation targets not reached** — soft penalty overwhelmed by other losses
- **Palm doesn't contact object** — proximity loss too weak

### Root causes identified
1. **Adam + soft penalties cannot enforce constraints.** Surface contact, collision, actuation are all soft penalties competing via weighted sum. The optimizer finds the easiest way to minimize total loss, which is NOT a good grasp.
2. **No grasp topology control.** Random initialization (even OBB-based) doesn't guarantee fingers on different sides. The optimizer doesn't have a mechanism to maintain topology.
3. **KKT gradient for l* is unreliable.** LP dual degeneracy means different solvers give different (wrong) gradients. Verified: original FroGGer's gradient has correlation -0.86 with numerical, ours has +0.83. Neither is correct.

### Why original FroGGer works despite same math
1. **IK-based initialization** → hand starts in a physically realistic wrapping pose
2. **SLSQP** → constraints are HARD, not soft. Surface = equality, collision = inequality, FC = inequality
3. **Drake collision** → exact geometry, no point-cloud gaps
4. **Sequential focus** → one grasp at a time, fully converged

---

## Proposed Redesign

### Core insight
The expensive part of FroGGer was Drake's collision queries and sequential execution — NOT the constrained optimizer. We correctly replaced Drake with PyTorch (fast, batched). But we also replaced SLSQP (constrained) with Adam (unconstrained) — this was wrong.

### Architecture

```
Phase 0: Batched IK initialization (PyTorch)
    - OBB-based palm poses (existing, works)
    - Freeze base, optimize joints to reach surface (existing, works)
    - Translate base toward object (existing, works)
    → Produces ~8000 candidates with reasonable topology

Phase 1: Constraint projection + refinement
    - For each candidate, PROJECT onto feasible set:
      a) Project tips onto surface: tp_proj = tp - sdf(tp) * normal(tp)
      b) IK step: find dq that moves tips toward tp_proj
      c) Check collision, reject/repair violating configs
    - Repeat for K iterations
    → Produces candidates with SATISFIED constraints

Phase 2: Force-closure ranking + light refinement
    - Compute l* via LP for all feasible candidates
    - Rank by l* (or σ_min for speed)
    - Optional: light Adam refinement of top-K with very high constraint weights
    → Top 10 results
```

### Key differences from current approach
1. **Projection replaces penalty.** Instead of soft penalties, explicitly project onto constraint manifold after each step. This guarantees surface contact and collision avoidance.
2. **IK for constraint satisfaction.** Use `pytorch_kinematics.PseudoInverseIK` or Jacobian pseudoinverse to find joint updates that achieve target tip positions. This is how the original FroGGer works inside SLSQP.
3. **Separate topology from refinement.** Phase 0 establishes topology (which side each finger is on). Phases 1-2 refine within that topology. The optimizer never changes which side a finger is on.

### Alternative: Projected Gradient Descent
Instead of the phase structure above, use projected gradient descent throughout:
```python
for step in range(N):
    # Gradient step on objective (maximize l* or σ_min)
    loss = -sigma_min(W(q))
    loss.backward()
    q = q - lr * q.grad
    
    # Project onto constraints
    q = project_joint_limits(q)       # sigmoid already handles this
    q = project_surface_contact(q)    # IK step toward surface
    q = project_collision_free(q)     # push apart overlapping links
```

The projection step is the hard part but is well-defined:
- **Surface projection**: compute SDF at tips, compute normal, move tip by -sdf*normal, solve IK for the new tip position
- **Collision projection**: compute signed distance between link pairs, if < margin, push apart along gradient
- **Self-collision projection**: for overlapping finger pairs, rotate one finger's joints to increase separation

### Implementation notes
- `pytorch_kinematics.PseudoInverseIK` supports batched IK but only for serial chains (one end-effector). For multi-finger projection, use the Jacobian pseudoinverse directly: `dq = J^+ @ (tp_target - tp_current)`
- The Jacobian `J = d(tip_pos)/d(q)` is available from `pytorch_kinematics` via autograd
- Collision projection can use the SDF gradient: push collision points in the `+∇SDF` direction

### Risks and unknowns
- Projection may be slow (multiple SDF queries + IK per step)
- IK for 4 fingertips simultaneously is underdetermined (16 joints, 12 position constraints) — need to handle the null space
- Collision projection may conflict with surface projection (pushing apart may move tips off surface)
- No guarantee of convergence for non-convex constraint sets

### Evaluation criteria
Before considering a grasp "good," verify ALL of:
1. **Visual inspection** — does it look like a human would hold the object this way?
2. **Opposing contacts** — at least one pair of contacts with normal dot product < -0.3
3. **Palm contact** — palm SDF < 3mm for cylindrical objects
4. **Actuation reached** — assigned finger within 5mm, pad facing push direction (cos > 0.7)
5. **No overlap** — minimum inter-finger distance > 8mm
6. **Force closure** — l* > 0.01 from LP

### Testing protocol
For each change, run comparison tests from this session:
1. `compute_grasp_matrix_torch` vs original — must match to 1e-5
2. `solve_min_weight_lp_batch` — l* must match, KKT residual < 1e-10
3. SDF accuracy vs Open3D — mean error < 1mm
4. FK Jacobian autograd vs numerical — must match to 1e-3
5. Visual inspection on spray bottle, hot glue gun, syrup pourer with LEAP RH
