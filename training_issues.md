# Training issues — P1 anti-interference (`P1-anti-interference`)

Log of problems found while getting the task to train, with cause, evidence and fix.
Machine for local checks: RTX 5080 Laptop GPU (16 GB), Windows 11, mjlab 1.6.0.

## 1. Training \~75× too slow at 4096 envs — fixed

**Found:** 2026-11-05, first local check (300 iterations planned, stopped at 11).

**Symptom:** 131 s per iteration (128 s of it collecting experience), ETA 10 h 22 min for
300 iterations. GPU at 100 % with 15.9 of 16.3 GB memory in use.

**Cause:** the solver buffers inherited from the template were far larger than this task
needs: `njmax` 1500 (mjlab default), `contact_sensor_maxmatch` 500, `ccd_iterations` 500.
GPU memory grows with the number of envs: 9.1 GB at 2048 envs, so about 18 GB at 4096,
more than the 16 GB card. Windows then spills GPU memory into system RAM, and every
step slows down about 30×.

Not the cause: self-collision between robot parts (turning it off changed nothing) or
the 500 Hz physics rate.

**Evidence** (simulation only, random actions, no learning):

| Settings | Envs | env-steps/s | s per iteration (4096 × 24 steps) | Peak GPU memory |
|----|----|----|----|----|
| template (training run) | 4096 | \~770 | 128 | 15.9 GB (full) |
| template | 1024 | 23,876 | 4.1 | – |
| template | 2048 | 29,890 | 3.3 | 9.1 GB |
| patched | 1024 | 33,680 | 2.9 | – |
| patched, no self-collision | 1024 | 32,203 | 3.1 | – |
| patched | 4096 | 57,726 | 1.7 | 3.9 GB |

Buffer check with the patched values (512 envs, random actions with pushes, 600 steps):
at most 46 of 300 constraint rows per env used, no overflow warnings.

**Patch:** `p1_velocity/src/p1_velocity/env_cfgs.py`, in `p1_anti_interference_cfg()`.
The values are the ones mjlab uses in its own flat-ground Go1 config.

```diff
-  cfg.sim.mujoco.ccd_iterations = 500
-  cfg.sim.contact_sensor_maxmatch = 500
-  cfg.sim.nconmax = 50
+  # Solver buffers sized for this flat-ground task (mjlab Go1-flat values). The
+  # template's 500 / 500 / default 1500 needed ~18 GB of GPU memory at 4096 envs;
+  # see training_issues.md in the main folder.
+  cfg.sim.mujoco.ccd_iterations = 50
+  cfg.sim.contact_sensor_maxmatch = 64
+  cfg.sim.nconmax = 50
+  cfg.sim.njmax = 300
```

**Learning before the patch** (the 11 iterations that ran): failures fell from 48.8 to 1.5
per log interval and time-outs rose from 14.1 to 28.8; mean episode length reached 239
of 250 steps. Early failures were the arms touching the ground: random rear-leg motion
lowers the arm mount, which sits only 17.6 mm above the ground.

**Rerun after the patch:** see the section below.

## 2. `rl_cfg.py` crashed `uv run train` — fixed

`RslRlModelCfg(stochastic=True)` is not accepted by mjlab 1.6.0, so importing the package
failed. Replaced with an explicit Gaussian actor:
`distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}`.
Checked with one PPO iteration.

## 3. Local venv had CPU-only torch — worked around

On Windows, `uv sync` installs `torch 2.10.0+cpu` (PyPI's Windows build), so training
could not use the GPU. Installed `torch 2.10.0+cu128` into `p1_velocity/.venv` only; the
lock file is unchanged. Plain `uv run` re-syncs back to the CPU build, so run locally with:

```sh
cd p1_velocity
uv run --no-sync train P1-anti-interference --agent.logger tensorboard
```

On vast.ai (Linux), the lock already installs CUDA 12.8 torch. Choose a machine whose
driver supports CUDA 13 or newer: warp 1.19 reports being built with CUDA 13.4.

## Rerun after patch 1 — ready for longer training

300 iterations, 4096 envs, same laptop. **9 min 13 s in total, 1.84 s per iteration**
(1.7 s collecting, 0.13 s learning), compared with 131 s before the patch. No errors.

| Iteration | Mean reward | Mean episode length (of 250) | Time-outs | Failures (non-tire contact) |
|----|----|----|----|----|
| 0 | −0.67 | 21 (first rollout only) | 14.1 | 48.9 |
| 50 | 4.28 | 242 | 10.0 | 0.88 |
| 100 | 5.27 | 239 | 11.5 | 0.50 |
| 150 | 5.79 | 248 | 14.6 | 0.46 |
| 200 | 6.25 | 250 | 18.7 | 0.46 |
| 250 | 6.73 | 248 | 22.5 | 0.42 |
| 299 | 7.10 | 249.5 | 14.5 | 0.33 |

Per-term reward per episode at iteration 299 (the maximum for a positive term is about
1 per second; episodes are 5 s, and the log reports each term averaged per second):
upright 0.99, CoM height 0.96, leg pose 0.82, wheel air time −0.72, action rate −0.68,
failure −0.004, late recovery 0 (never fires with 5 s episodes; see the open items).
Action std fell from 0.70 (iteration 50) to 0.56, so the policy is settling.

**Verdict:** the setup trains. Reward rises steadily, failures stay rare, there are no
errors, and memory and speed are fine. A 10,000-iteration run would take about 5 h on
this laptop, so a short vast.ai run is optional rather than needed.

Logs and checkpoints (`model_0.pt` … `model_299.pt`, ONNX export):
`p1_velocity/logs/rsl_rl/p1_velocity/2026-11-05_05-42-38_local_check_patched/`.

## Open items (not blocking)

* **5 s episodes:** a push at 1–2 s plus the 5 s recovery window runs past the episode
  end, so `late_recovery` never fires. Use 8–10 s episodes to activate it.
* **Wheel lift after pushes:** `wheel_air_time` went from about −0.94 (iteration 11) to
  −0.72 (iteration 299). It is now the largest penalty: wheels still stay off the ground
  for more than 1 s after some pushes. Measured with `model_299` (deterministic, 256
  envs, one episode with pushes), compared with zero actions:

|    | All 4 tires down, before push | All 4 tires down, after push |
|----|----|----|
| zero actions (passive) | 97.6 % | 99.2 % |
| trained policy | 92.4 % | 67.6 % |

  Tire loads stay at the static values (front about 43 N, rear about 68 N per tire), so
  this is real lifting by the policy, not contact flicker. The policy has no wheel
  action, so it counters pushes with leg motion that lifts a wheel. Watch whether this
  falls in a long run; if not, raise the `wheel_air_time` weight.
* **Arm clearance:** the arm mount sits 17.6 mm above the ground. Sideways pushes of
  150 N for 0.15 s put an arm on the ground with a passive robot (push limits are now
  100 N sideways, 150 N front/back).



## 4. Changes before the vast.ai run (2026-11-05, not trained yet)

These change the task, so the 300-iteration numbers above are not comparable with the next run. Details are in `specification/env_cfg.md` and `specification/mdp.md` (Implementation mode).

| Open item above | Change |
|----|----|
| 5 s episodes, `late_recovery` never fires | Episodes are 10 s with about 3 pushes each (first after 1–2 s, then 2–4 s after each push ends). `late_recovery` is removed: returning to the pose is already rewarded by `leg_pose`. |
| Wheel lift after pushes | `wheel_air_time` is replaced by `tire_lift`: −1 per tire per second off the ground after a 0.1 s grace, at all times. New logged metric `Episode_Metrics/all_tires_down`. Compare it with the 92.4 % / 67.6 % above (trained policy, before / after pushes). |
| Arm clearance | Arm contact (arm, wrist, hand, arm mount) costs −5 per second, and longer than 0.5 s counts as failure. Other non-tire contact is still failure right away. Pushes are now up to 150 N in every direction. Arm friction against the ground is 0.5 (was MuJoCo's default 1.0). |

Also changed:

* **Policy inputs:** no body linear velocity (the critic keeps it) and no wheel angles.
* **Play mode:** keeps the pushes.
* **Training length:** `max_iterations` 3,000 (was 10,000).

To check first on vast.ai:

* `uv run --with pytest pytest tests -s`.
* The solver-buffer test, because arm contacts now continue instead of ending the episode.

## 5. vast.ai run with the section 4 changes (2026-11-05)

RTX 5090 (32 GB, driver 595.91, CUDA 13.2), 4096 envs, 3,000 iterations.

* **Time:** 36 min 44 s, 0.74 s per iteration (132k steps/s).
* **Resources:** GPU 81 % busy, 3.8 GB memory. No errors.
* **Tests:** all 26 passed before training. The solver-buffer test peaked at 50 of 300 constraint rows, with no overflow.

| Iteration | Mean reward (per 10 s episode) | Episode length (of 500) | Failures | `all_tires_down` | `tire_lift` /s | `arm_contact` /s | Action std |
|----|----|----|----|----|----|----|----|
| 0 | −0.03 | 18 | 0.46 | 4 % | −0.003 | −0.002 | 0.99 |
| 100 | 28.12 | 500 | 0 | 53 % | −0.008 | −0.002 | 0.25 |
| 500 | 29.23 | 500 | 0 | 85 % | −0.003 | −0.0006 | 0.16 |
| 1000 | 29.56 | 500 | 0 | 96 % | −0.0005 | 0.0000 | 0.10 |
| 2000 | 29.61 | 500 | 0 | 96 % | −0.0008 | 0.0000 | 0.09 |
| 2999 | 29.57 | 500 | 0 | 94 % | −0.0006 | 0.0000 | 0.11 |

"Failures" counts body/leg contact plus arm contact longer than 0.5 s. All values are measured during training, with the policy's action noise.

**Verdict:**

* **Robot stays up:** after about iteration 100, no episode ends early, and the arm stops touching the ground.
* **Tires:** stay down about 95 % of the time. The rest is mostly short bounces: `tire_lift` only counts lifts longer than 0.1 s and stays near 0.
* **Leveled off:** progress flattened after about iteration 1,000.
* **The pushes are too easy:** pushes up to 150 N never cause a failure. For more anti-interference, use stronger pushes or raise them during training.
* **Not done yet:** a deterministic play/eval with pushes, which would compare with the 92.4 % / 67.6 % in the open items.

**Files:** logs and checkpoints (`model_0.pt` … `model_2999.pt`, ONNX export, TensorBoard, `train.log`) are in `p1_velocity/logs/rsl_rl/p1_velocity/2026-11-05_22-58-10_vast_5090/`.
