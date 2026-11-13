# Experiment log

A running lab notebook: what was tried, what happened, and what it means. Numbers are copied from
the `eval_*.json` files that `rlagent evaluate` writes into each run directory.

## Phase 4: training an agent on Warehouse-v0

**Target (exit criterion):** a PPO agent solves held-out *medium* layouts (12x12, 3 items,
capacity 2) with at least 90% success and efficiency of at least 0.85 (efficiency = optimal steps /
agent steps, 0 for failed episodes). Reference on the same layouts: greedy 100% success / 0.89
efficiency, optimal 1.00.

**Protocol.** Held-out = layouts from the `test` split (seed range disjoint from training), 100
episodes, deterministic actions, one training seed per row unless stated. "Final" is the policy at
the last training step; "best" is the checkpoint with the highest 20-episode eval return.

### What did not work (small setting, 8x8, 2 items)

| Variant | Train / held-out success |
|---|---|
| Plain PPO, vector observation, invalid actions penalised | 0% |
| PPO, vector MLP, 1M steps | 50% / 66% |
| MaskablePPO, vector MLP, sparse reward, 300k steps | 19% / 24% |
| MaskablePPO, vector MLP, PBRS shaping, 300k steps | 27% / 35% |
| **MaskablePPO, grid CNN, PBRS shaping, 600k steps** | **87% / 96%** |

Three separate problems were found and fixed in turn:

1. **Wasted exploration on invalid actions.** With an invalid-action penalty, PPO's entropy
   collapsed onto "safe" moves and it never learned to pick. Masking invalid actions (MaskablePPO)
   removes the problem; the policy can no longer choose them, so `mean_invalid_actions` is exactly 0.
2. **A local view is not enough to navigate.** The vector observation only contains a 7x7 patch of
   the map. The grid observation (6 channels, whole map) with a small CNN needed less than half
   the steps and reached far higher success.
3. **Shaping drift.** An early PBRS potential gave a large negative reward on the final step
   (potential forced to 0 at termination). The potential is now "negative remaining work", which
   is naturally 0 when the order is complete.

### Medium setting (12x12, 3 items), MaskablePPO + grid CNN + PBRS

All rows below use the grid CNN with masking, seed 0 unless noted. "Curr" = curriculum
8x8/1 item -> 10x10/2 -> 12x12/3 (promote at 85% training success over 200 episodes).

| Run | Steps | Change | Best ckpt: success / eff | **Final: success / eff** |
|---|---|---|---|---|
| g_med_pbrs | 2M | baseline, no curriculum | 0.78 / 0.54 | 0.76 / 0.53 |
| g_med_curr | 2M | + curriculum | 0.88 / 0.64 | 0.89 / 0.69 |
| g_med_sparse | 2M | no shaping | 0.64 / 0.15 | 0.58 / 0.14 |
| g_med_step05 | 2M | step penalty -0.05 | 0.01 / 0.01 | 0.07 / 0.05 |
| g_c4_base | 4M | curriculum, longer | 0.93 / 0.78 | 0.97 / 0.88 |
| g_c4_lr | 4M | + linear LR decay, ent 0.005 | 0.97 / 0.75 | 0.98 / 0.83 |
| g_c4_step02 | 4M | step penalty -0.02 | 0.92 / 0.69 | 0.97 / 0.89 |
| g_c4_g995 | 4M | gamma = shaping gamma = 0.995 | 0.99 / 0.77 | 0.93 / 0.87 |
| **g_c5_a (seed 0)** | 5M | gamma 0.995 + LR decay + ent 0.005 | 0.90 / 0.61 | **0.99 / 0.93** |
| **g_c5_a (seed 1)** | 5M | same, seed 1 | 0.88 / 0.65 | **1.00 / 0.95** |
| g_c5_b | 5M | gamma 0.999, GAE lambda 0.97 | 0.90 / 0.68 | 0.87 / 0.64 |
| g_c5_c | 5M | a + rollout 2048, batch 512, ent 0.003 | 0.99 / 0.82 | 1.00 / 0.85 |

**Result: the exit criterion is met.** The chosen configuration (`configs/warehouse/maskable_grid.yaml`,
row g_c5_a) reaches 99-100% held-out success and 0.93-0.95 efficiency on two seeds. That is above the
greedy baseline's 0.89 efficiency: the agent plans item order better than "go to the nearest item".
Curated model: `models/warehouse-maskable-ppo/` (seed 1), GIF `docs/figures/warehouse_maskable_ppo.gif`.

### What the runs say

- **Curriculum helped at equal budget** (0.76 -> 0.89 success at 2M steps, 0.53 -> 0.69 efficiency),
  single seed. It is a candidate for the E5 ablation.
- **Shaping matters for efficiency, not just success.** Without shaping the agent still reaches 58-64%
  success, but episodes average about 255 steps against about 55 for the optimal planner
  (efficiency 0.14, with failed episodes counted as 0).
- **A big step penalty is harmful, a moderate one is neutral.** -0.05 collapsed training (not
  diagnosed further); -0.02 was within noise of -0.01. The Phase 3 worry that -0.01 is too weak an
  efficiency signal did not materialise once the agent was trained long enough with PBRS.
- **Final policy beats best checkpoint on this task.** In the two g_c5_a runs the "best" checkpoint
  scored 0.61-0.65 efficiency while the final policy scored 0.93-0.95. The best checkpoint is picked
  by a 20-episode mean return; once success is near 100% that return barely distinguishes a fast
  policy from a slow one, so an early checkpoint can win on noise. I did not investigate further. The
  warehouse protocol therefore reports the **final policy** (`eval_model: final`).

### Caveats (read before quoting numbers)

- **Selection effect.** The configuration was chosen by looking at the held-out numbers of about 12
  runs, so the 0.93-0.95 is a mildly optimistic estimate. Phase 5 re-runs the chosen configuration on
  seeds 2-4 (never used for selection), and hyperparameter tuning selects on *train-distribution*
  layouts only.
- **One or two seeds per row.** Differences of a few points between rows (for example c4_base vs
  c4_lr) are within seed noise and should not be read as effects.
- **The agent was trained and tested on 12x12 / 3 items only.** The 16x16 / 5-item "hard" setting is
  not attempted yet.
- Layouts vary the shelves; item positions are re-sampled on every reset, so "10 layouts" in E4
  means 10 shelf arrangements with random item placements.

### Cost of these runs

Four grid-CNN runs in parallel on the RTX 5060 each train at about 470-590 env steps/s, so a 5M-step
run takes about 2h45m and a 2M-step run about 1h05m.

## Phase 5 plan (tooling built, experiments not yet run)

Everything below is driven by `configs/experiments/phase5.yaml`:

```bash
uv run rlagent matrix configs/experiments/phase5.yaml --dry-run    # plan + estimate
uv run rlagent matrix configs/experiments/phase5.yaml --tier core  # train + evaluate (resumable)
uv run rlagent aggregate configs/experiments/phase5.yaml           # IQM + 95% CI tables, figures
uv run rlagent tune configs/sweeps/ppo_warehouse.yaml --workers 4  # Optuna sweep
uv run rlagent watch runs                                           # live progress bars
```

Each group is trained for several seeds, then the **final** policy is evaluated for 100 episodes on
the training layouts and on the held-out test layouts. `aggregate` reports the mean, standard
deviation and interquartile mean with a 95% percentile-bootstrap interval over seeds.

| ID | Question | Arms (all 3 seeds, 3M steps) |
|---|---|---|
| headline | Does the final agent hold up on fresh seeds? | PBRS + curriculum + masking, **5 seeds, 5M steps** |
| E1 | Does shaping help, and can it hurt? | sparse, naive (hackable), PBRS (= `ref3m`) |
| E2 | PPO vs DQN | plain PPO + invalid penalty, DQN + invalid penalty (`ref3m` for reference) |
| E3 | Observation design | vector MLP vs grid CNN |
| E4 | Generalisation | 10 / 100 / unlimited training layouts, tested on unseen layouts |
| E6 | Penalty vs masking | plain PPO with penalty vs MaskablePPO |
| E5 (optional) | Curriculum on vs off | no curriculum vs `ref3m` |
| tuning | Better hyperparameters? | Optuna, 24 trials of 1M steps on 10x10 / 2 items, 8 parameters |

E7 (from-scratch PPO on the warehouse) is not in the matrix: the scratch agents have no action
masking, so the comparison would be confounded. Adding masking to the scratch PPO is the
prerequisite.

### Time estimate (4 workers, measured throughput)

Measured with `rlagent bench` on the RTX 5060 (`results/throughput.csv`, 40k-step runs):
1 worker 603 steps/s, 4 workers 516 steps/s each (2,063 total). The estimates below use
520 steps/s per grid-CNN worker, 1,500 for the CPU vector MLP (E3) and a guessed 300 for DQN (E2,
**not measured yet**) plus 90 s of evaluation per run.

| Part | Runs | Compute | Wall time (4 workers) |
|---|---|---|---|
| headline, seeds 2-4 only (seeds 0-1 reused from g_c5_a) | 3 x 5M | 8h | about 2h45m |
| `ref3m` + E1 + E4 + E6 (6 groups) | 18 x 3M | 29h | about 7.5h |
| E3 (vector MLP, CPU) | 3 x 3M | 1.7h | about 35m |
| E2 DQN (unmeasured speed) | 3 x 3M | 8.4h | about 2h50m |
| **Core tier, total** | 27 | about 47h | **about 12-13h** |
| E5 curriculum off (optional) | 3 x 3M | 4.9h | about 1h40m |
| Optuna sweep | 24 x 1M, pruned | about 10h | about 2.5-3.5h |

So the whole of Phase 5 is roughly 17-19 hours of wall time. It is resumable and can be split over
several nights (`--only`, `--tier`, `--seeds`). A 16-env / 64-step rollout shape measured about 40%
faster (2,885 vs 2,063 steps/s total with 4 workers), which would bring the total to about 12-13
hours, but it changes PPO's rollout and has not been validated for learning quality.
