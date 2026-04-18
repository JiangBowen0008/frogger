# Pipeline Architecture & Audit

## Decision variables
- `self.pos` [B,3] — palm base position (world)
- `self.rot6d` [B,6] — palm base orientation (6D rep)
- `self.u` [B,16] — joint angles via sigmoid: `q = q_lo + sigmoid(u) * range`
  - 4 fingers × 4 joints (CMC, MCP, PIP, DIP for if/mf/rf; CMC/MCP/IPL/DIP for thumb)
- `self.amap` [B,1] — which finger is actuation (varies per env)

## Stage-by-stage

### STAGE 1 — Palm placement (`_init`, lines 1548-1649)
- **Optimizes**: NONE (sampling only)
- **Outputs**: `self.pos` (position s.t. palm inner surface is at random surface point),
  `self.rot6d` (palm +X faces object via normal), `self.u` (moderate curl seed).
- **Strategy**: 4000 samples: 50% uniform on body, 50% biased near actuation target.
  Y-axis chosen from 4 canonical OBB projections.
- **Frozen after**: `self.rot6d` frozen (`requires_grad_(False)` line 1643).

### STAGE 2 — Actuation finger IK (lines 1688-1754)
- **Optimizes**: `u_act` with Adam lr=0.05, **only actuation finger's 4 joints** (masked).
- **Loss** (150 steps):
  - Tip position error: `500 * (tip - act_pos)²`
  - Pad direction align: `50 * (1 - cos(pad_dir, act_dir))²`
  - Actuation finger body collision (non-ds, `include_clearance=False`): `100 * relu(-sdf).sum()`
  - Non-act joint regularization: `100 * (u - u_init)²`
- **Frozen**: palm pose (pos, rot6d), other finger joints.
- **Reports**: act_dists to trigger; stores sort order.

### STAGE 3 — Palm slide + joint act-finger IK (lines 1763-1916)
- **Part A (palm slide, 30 iterations)**: no_grad, discrete search over 4 tangent directions,
  2mm steps (max 50mm total). **Optimizes** `self.pos` (base position). Tries to clear
  palm-back + act-finger collision.
- **Part B (joint act IK, 30 steps)**: `opt_act_ik` Adam lr=0.02, **only actuation joints**.
  - Tip target: `1000 * (tip - act_pos)²`
  - Pad align: `100 * (1 - cos)²`
  - Act body collision: `500 * relu(-sdf).sum()` with `include_clearance=False`
- **Frozen**: rot6d.

### STAGE 4 — Actuation filter (lines 2002-2019)
- Keeps envs with actuation tip within 10mm of target.

### STAGE 5 — CMC avoidance init (lines 2022-2064)
- **Writes directly** to `self.u` (no optimizer). Sets support finger CMCs to specific
  values spaced 0.7 rad apart, starting 1.0 rad from the act finger's CMC.
- **Also randomizes MCP/PIP/DIP** within hand-coded ranges.

### STAGE 6 — Support IK (lines 2067-2326, 400 steps)
- **Optimizes**: `u_sup` Adam lr=0.03, **all 4 joints of support fingers** (CMC+MCP+PIP+DIP).
  CMC is free here (frozen later).
- **Loss**:
  - Surface contact: `500 * sdf(tip)²` (single point `tip_offsets[fi]`)
  - Below object penalty: `2000 * relu(obj_z_min - tip_z)²`
  - Actuation repulsion (additive): `500 * relu(0.040 - dist_to_act)²` when within 40mm
  - Spread target (fades 100→0 over 200 steps): `target_w * (tip - target)²`
  - Support finger link collision (non-ds): `50 * relu(-sdf).sum()`
  - **Back-side ds collision** (for `ds` links, back-side only): `200 * relu(-sdf - 3mm).max()²`
  - Non-support joint regularization: `(u - u_init)² * ~sup_mask`
- At step 100: re-randomizes curl if support still overlapping actuation.

### STAGE 7 — Entry filter (`opt_mask`, lines 2404-2408)
- Keep envs with: `act_dist<10mm AND tip_sdf_mean<15mm AND worst_sup_link>-5mm AND worst_act_link>-10mm`

### STAGE 8 — Main optimization (lines 2479-2791, 300 steps, direct-q)
- **Optimizes**: `q_opt` Adam lr=0.003 directly on joint angles (clamp after each step).
- **Mask**: `opt_joint_mask` = MCP+PIP+DIP for SUPPORT fingers (CMC and actuation are frozen).
- **Loss**:
  - Section A (surface): `1000 * Huber(sdf(tip_offset))` — single-point pad center contact
  - Section A' (pad corners, 8 points): `2000 * relu(-sdf - 3mm)²` — only beyond 3mm inside
  - Section B1 (non-ds links, support only): `500 * relu(-sdf).sum()`
  - Section B2 (ds back): `1000 * relu(-sdf_back - 1mm).max()²`
  - Section B2 (ds pad): `500 * relu(-sdf_pad - 3mm).max()²`
  - Section C (FC): σ_min grad via SVD + l* grad via LP (every 5-20 steps). weight=1, boost=3
  - Section D (SC box-box SDF): `5000 * relu(-sd - 1mm)²` for non-adjacent cross-finger link pairs. Every 5 steps. opt_mask envs only. Distant pairs skipped.
  - Section E (act exclusion): `200 * relu(0.035 - dist_to_act_finger)²` when support near actuation.

### STAGE 9 — Feasibility check + verification (lines 2998+)
- **Feasibility** (computed with default `include_clearance=True` for support fingers):
  - surf<8, col<3, ds_back>-3, ds_pad>-5 (act finger excluded), sc_sdf>-1, σ>0.01, act_dist<10
- **Verification** (5mm box-grid, URDF boxes + visual pad samples):
  - mesh_pen_pct<5%, sc_worst>0.5mm via KDTree

---

## KEY QUESTIONS TO INVESTIGATE

### Q1: Does the pad lie flat after Support IK (before main optimization)?
Support IK uses SINGLE-POINT surface contact (`sdf(tip_offset)²`). Same bug as v5 — pad
can be angled into object after Support IK. Then main optimization inherits this state.

**Experiment**: snapshot ds_pad before and after main optimization. If Support IK already
leaves pad=-20mm, that's where we need to fix it first.

### Q2: CMC is FROZEN during main optimization. Is the orientation achievable?
During support IK, CMC is free. During main optimization, CMC is frozen (line 2422:
`opt_joint_mask[b, fi*4+1:fi*4+4] = True` — skips index 0 = CMC).

With only MCP/PIP/DIP free (3 DOF), the pad orientation has limited degrees of freedom.
Pad alignment is fundamentally a 3D rotation problem; 3 DOF is tight even if perfectly
oriented initially.

**Experiment**: unfreeze CMC in main optimization and see if pad can reach flat contact.

### Q3: What fails per object?
- `pad>-5: 2500/4000` means 63% have good pad (so the loss CAN work)
- But no single env passes ALL criteria simultaneously.
- **Experiment**: per-env failure histogram. Is it always the same combination (e.g., pad+sc)?
  Or different envs fail different criteria? If the LATTER, we're close — just need to find
  envs at the intersection.

### Q4: Section D (SC box-box) every 5 steps vs every step
The SC computation is the slowest loss term. If we can afford every step, the gradient
is much stronger. Alternative: always compute a cheap SC proxy every step, full box-box
every 5.

### Q5: Section A pulls CENTER to surface. Is the center at the right place?
`tip_offsets[fi] = [-10mm, -32mm, +15mm]` — at y=-32mm, which is past the URDF body box.
This point is on the rounded tip extension, not the main contact pad face.
- On cylinder: pad center on surface = rounded tip on cylinder → body pad (y=-17) ABOVE surface.
- So Section A may be pulling the WRONG point to the surface.

**Experiment**: query `sdf(body_pad_center)` at y=-17mm for the best grasps. If these are
ABOVE surface, we're contacting with the rounded tip not the pad.

---

## TARGETED EXPERIMENTS (in order)

1. **Per-stage ds_pad snapshot**: Measure pad penetration AFTER each stage.
   This will tell us if pad angling is introduced in Support IK, Main Opt, or both.

2. **Per-env failure pattern**: Among the 2500 envs with pad>-5mm, how many pass surface?
   How many pass SC? The intersection size tells us how far we are.

3. **Freeze comparison**: Run two experiments in parallel:
   - CMC frozen (current): see current baseline
   - CMC free: see if unfrozen CMC gives better pad alignment

4. **Contact point relocation**: move `tip_offsets` to the main pad center (y=-17mm).
   See if grasps change quality.

The goal is to identify the ONE critical bottleneck, not patch symptoms.
