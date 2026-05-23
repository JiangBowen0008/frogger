# FroGGer grasp optimizer — research code rules

## Project layout

```
frogger/
  batched_pytorch_solver.py   # Main GPU solver (palm sampling → IK → opt → FC check)
  robot_models/               # LEAP URDF, collision geometry, kinematics
  sdf/                        # Batched SDF, clearance SDF
run_target_objects_batched.py # Multi-batch runner for 5 target objects
diagnose_lstar_fails.py       # Contact spread analysis for l*-fail grasps
output/                       # gitignored: grasp .pt files + per-stage metrics
```

<!-- principles:start -->
## Goal

A grasp synthesis pipeline that produces physically valid, high-quality grasps on arbitrary objects. Real force closure, no penetration, fingertips on the surface, sensible palm wrapping, actuation finger pointing the right way. The five test objects (grinder, spray, flashlight, air_blower, hot_glue_gun) are a benchmark, not the target.

## Principles

- **Generalization is the point.** No per-object tuning. No hacks specific to one shape. A fix that only helps because we knew the object beforehand does not count.

- **Simplicity over patches.** The pipeline should get shorter when we improve it, not longer. Remove rather than gate behind default-off env vars. Each loss term, each stage, each flag has to earn its place. Audit holistically; don't keep patching.

- **Measure, don't guess.** "X is the bottleneck" needs a measurement of X — an isolated experiment varying only X, or a direct read. Inferences from indirect evidence are speculation; label them so and don't act on them as facts.

- **Fix the hard cases, not the easy ones.** Per-object balance matters more than total count. Grinder going from 10 to 12 is wasted effort while air_blower is at 1.

- **Small experiments before big ones.** If a 20-minute targeted test on broken objects can falsify an idea, run that first. Abort early when there's no signal.

- **Evidence has strength levels.** *Observed* = one run, one seed. *Consistent* = 3+ runs the same direction. *Verified* = multiple independent setups. Do not silently promote.

- **No shortcuts.** Don't loosen feasibility criteria to make more grasps pass. Don't add fallbacks that mask invalid state. Don't add backward-compat shims. The pipeline has to actually be correct.

- **Confirm before tuning.** Before changing a loss weight, verify the loss is actually firing on the cases we care about.

- **Look at the grasps.** Visual plausibility matters. Numbers alone aren't enough — render, look at the geometry, confirm it makes physical sense.
<!-- principles:end -->

## Repo conventions

- Do not create, edit, move, or delete files under `/tmp` or `/var/tmp`.
- Entry-point scripts: verb-noun (e.g., `run_batched.py`, `diagnose_lstar.py`). No permanent `*_v2.py`, `*_new.py`, `*_tmp.py` filenames.
- Grasp outputs → `output/<run_name>/<object>/batch_<N>/`. `output/` is gitignored — don't commit output files.

## Env vars

Two ablation toggles for offline debugging:
- `FROGGER_NO_MULTI=1` — force single-assignment IK (skip multi-assign trials)
- `FROGGER_NO_BASE=1` — freeze base pose during main opt

Two interventions with single-batch evidence, awaiting multi-batch confirmation
before becoming default behavior:
- `FROGGER_ACT_SELECT=uniform_viable` — multi-assign topology diversity (vs `argmin` default)
- `FROGGER_IK_SUP_SC=1` — support↔support SC point-repulsion in support IK
  (margin defaults to 5 mm, the proven setting; the original 20 mm was falsified)

Falsified, untested-with-no-evidence, and orphaned-debug knobs have been
deleted from the solver. See git history at commits prior to this point
(`e61c281` was the last commit with the full set) and memory files
`project_sc_proj_falsified.md`, `project_sc_loss_inert.md`,
`project_support_ik_no_sup_sc.md`, `project_cdist5mm_winner.md`,
`project_sigma_to_feas_gap.md`, `project_full_bottleneck_picture.md`
for the evidence behind each decision.
