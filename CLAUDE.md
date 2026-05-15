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

## Core principles

- Write clean, concise, principled research code.
- Prefer explicit, fail-fast behavior.
- No backward-compatibility layers, no per-object tuning knobs.
- Do not create, edit, move, or delete files under `/tmp` or `/var/tmp`.

## Error-handling standard

- Surface bugs instead of hiding them.
- Do not use broad or silent error handling.
- Do not use fallback patterns that quietly mask invalid state.

## Debugging methodology

- Prioritize root-cause analysis over quick fixes.
- Do not speculate about causes and then treat the speculation as fact. State uncertainty explicitly ("I suspect X because Y, but we should verify by Z").
- "X is the bottleneck" requires either an isolated experiment varying only X, or a direct measurement of X. Inferences from indirect evidence do not count.
- When multiple anomalies coexist, treat them as DISTINCT until you can prove a shared mechanism.
- Label each result: **verified** (multi-run, controlled), **observed** (single run, uncontrolled), **inferred** (not directly measured). Never promote inferences to verified facts.

## Working style

- Before editing, identify the minimal file set that must change.
- Before asserting "fixed" or "ruled out", run the command that would falsify the claim and quote its output.
- Before tuning a loss weight, verify the loss is actually firing (not zero or negligible) for the targeted envs.
- Test fixes on the targeted/broken objects first. Abort early if no signal rather than waiting for the full 5-object × 3-batch run.
- Per-object balance over total count. 4 objects × 5 grasps > 1 × 20 + 3 × 0.

## Experiment workflow

- Before launching a run: state the hypothesis and define the expected direction of change per object.
- After completing: check per-object counts, not just total. Grinder is capped at 10 — don't overweight it.
- Run single-object 1-batch sanity checks before full 3-batch runs.
- Do not present results without verifying there were no silent failures (check log for errors, NaN, zero-grad).
- Intermediate result labels:
  - **Observed** = single batch, one seed
  - **Consistent** = 3+ batches same direction
  - **Verified** = multiple independent runs, consistent direction

## Naming

- Entry-point scripts: verb-noun (e.g., `run_batched.py`, `diagnose_lstar.py`)
- No permanent `*_v2.py`, `*_new.py`, `*_tmp.py` filenames.

## Output / data paths

- Grasp outputs → `output/<run_name>/<object>/batch_<N>/`
- Do not commit output files. `output/` is gitignored.
