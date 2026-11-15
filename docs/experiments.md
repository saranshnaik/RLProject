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

## Phase 5 results

All numbers: **final policy**, 100 episodes on **held-out** 12x12 / 3-item layouts, capacity 2. "IQM" is
the interquartile mean over seeds with a 95% percentile-bootstrap interval; with 3 seeds that interval
is close to the min-max range and understates the uncertainty, so non-overlap is not significance.
Tables: `results/phase5_summary.md`; per-run numbers: `results/phase5_runs.csv`; figures
`results/phase5_*.png`. Ablation arms train for 3M steps; the headline for 5M.

### Headline: the 5M-step agent on five seeds

| Seed | Success | Efficiency |
|---|---|---|
| 0 (reused run g_c5_a_s0) | 99% | 0.927 |
| 1 (reused run g_c5_a_s1) | 100% | 0.947 |
| 2 | 100% | 0.961 |
| 3 | 99% | 0.916 |
| 4 | 100% | 0.921 |
| **IQM [95% CI]** | 1.00 [0.99, 1.00] | **0.93 [0.92, 0.96]** |

The target (>= 90% success, >= 0.85 efficiency) is met on every seed, and every seed beats the greedy
baseline (0.89). Seeds 2-4 were not used to choose the configuration. Seeds 0-1 were, and the 5M budget
and hyperparameters were chosen after seeing them, so the earlier selection caveat is reduced rather
than removed.

### Ablations (3M steps, 3 seeds each unless stated)

| Experiment | Arm | Success | Efficiency |
|---|---|---|---|
| reference (`ref3m`) | PBRS + curriculum + masking, unlimited layouts | 0.99 [0.97, 1.00] | 0.78 [0.77, 0.79] |
| E1 shaping | sparse | 0.99 [0.98, 0.99] | 0.81 [0.80, 0.82] |
| | naive (hackable), bonus 0.1 | 1.00 [0.99, 1.00] | 0.83 [0.81, 0.87] |
| | naive (hackable), bonus 1.0 (2 seeds) | **0.00** | **0.00** |
| E2 PPO vs DQN | DQN, penalty, no masking | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| E3 observation | vector MLP, local 7x7 patch | 0.24 [0.20, 0.30] | 0.16 [0.12, 0.21] |
| E4 layouts | 10 training layouts | 0.94 [0.93, 0.96] | 0.80 [0.76, 0.87] |
| | 100 training layouts | 1.00 [0.99, 1.00] | 0.80 [0.78, 0.81] |
| E5 curriculum | off | 0.57 [0.53, 0.63] | 0.39 [0.33, 0.45] |
| E6 masking | plain PPO + invalid-action penalty | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |

Budget matters more than most of these choices: the same configuration reaches 0.78 efficiency at 3M
steps and 0.93 at 5M.

What each experiment says, and what it does not:

- **E1, shaping (surprise).** With the curriculum and action masking in place, sparse, naive and PBRS
  reward end up within about 0.05 efficiency of each other, and with 3 seeds I cannot rank them. This
  differs from the single-seed Phase 4 result without the curriculum at 2M steps, where sparse reward
  reached 58% success and 0.14 efficiency against 76% and 0.53 with PBRS. The likeliest explanation is that the curriculum and masking already solve
  the exploration problem, so shaping adds little; E5 below supports this but is one seed.
- **E1, reward hacking: not at the default bonus, clearly at 10x.** The naive bonus is +0.1 per step that
  gets closer to the goal. By my arithmetic oscillating earns about +0.04 per step, roughly +10 over a
  full-length episode against about +28 for finishing, so the exploit is weak at that setting, and the
  default-bonus agents (100% success) did not find it. With `naive_bonus: 1.0` (`e1_naive_big`, 2 seeds,
  3M steps) both agents hack: 0% held-out success, **0 items delivered**, every episode runs the full 308
  steps, yet the mean return is about +150 (146 and 149). In a replay of an in-training checkpoint of
  this arm, 97% of the moves undid the previous move. `docs/figures/warehouse_reward_hacking.gif` shows the final policy of seed 0. The bonus
  pays +1 for each step towards the subgoal and nothing for stepping away, so a step back and forth
  nets more than any delivery bonus. That is the mechanism the potential-based form is designed to rule
  out; I did not run PBRS at bonus 1.0, so the cure is shown by the Phase 4/5 PBRS runs only at the
  default scale. Eval returns are not comparable across reward arms, so compare success and efficiency
  only (and note that here a high return coincides with total task failure).
- **E3, observation design.** The vector MLP reaches only about a quarter of the success of the
  grid CNN. Observation type and network changed together, and the vector observation only sees a 7x7 local
  patch, so this shows that "whole-map grid + CNN beats local patch + MLP", not which of the two matters.
- **E4, layout diversity.** Training on 10 layouts shows a consistent train/held-out gap in all three
  seeds (success 98-100% on training layouts vs 93-96% held-out; efficiency lower by 0.04-0.055). With 100
  layouts the gap is gone: held-out success and efficiency match the unlimited-layout reference (and are
  not worse than on the training layouts). So on this layout family somewhere between 10 and 100 layouts is
  enough to generalise. The held-out efficiency level varies a lot between seeds (0.76-0.87 with 10
  layouts), so do not rank the arms on efficiency.
- **E5, curriculum (3 seeds).** Without the curriculum the agent reaches 57% held-out success
  (53, 57, 63%) and 0.39 efficiency (0.33-0.45) at 3M steps, against 97-100% and 0.77-0.79 with it. All three
  seeds are below the lowest curriculum seed, and the learning curves climb late and noisily
  (`results/phase5_E5_curriculum_curves.png`). The 2M-step result in Phase 4 points the same way. Seed 0
  came first as a single run (53%); seeds 1-2 were run afterwards and agree. With more steps the gap could
  narrow: this measures speed of learning at 3M steps, not the final ceiling.
- **E6, masking.** Plain PPO with a -0.1 invalid-action penalty and no masking never learned to move: all
  three seeds have 0% success and about 300 invalid actions per episode (the whole time limit). The
  comparison changes the algorithm class (PPO vs MaskablePPO) as well as masking, and it does not isolate the
  penalty size, so it shows that this penalty setup fails, not that no penalty setup could work.
- **E2, PPO vs DQN: both fail without masking.** SB3 DQN (grid CNN, same curriculum, invalid actions
  penalised like the PPO arm of E6) gets 0% success on all three seeds with 122-242 invalid actions per
  episode, against about 300 for penalty-only PPO. So neither algorithm learns the task without masking,
  and this experiment cannot rank them: it shows the masking, not the algorithm, decides the outcome.
  A fair DQN comparison would need masked DQN (SB3 has none), a longer run, or a tuned exploration
  schedule. The DQN hyperparameters were set once, untuned, and 3M steps is a modest budget for
  off-policy learning here, so a stronger DQN is not ruled out.

### Hyperparameter sweep (Optuna)

`rlagent tune configs/sweeps/ppo_warehouse.yaml`: TPE sampler with a median pruner on a cheaper problem
(10x10, 2 items, no curriculum, 1M steps, seed 0). The objective is mean efficiency of the final policy
on 50 fresh *training-distribution* layouts, so the held-out layouts were never used. 27 trials ran
(17 completed, 10 pruned; the target was 24, and parallel workers overshoot it). Results:
`results/tuning/ppo_warehouse_trials.csv`, `results/tuning/ppo_warehouse_best.yaml`.

- **Best trial (13):** efficiency 0.935, success 98%: learning rate 9.8e-4 (linear decay), entropy
  coefficient 0.0011, gamma 0.995, GAE lambda 0.9, 128 steps, batch 256, 9 epochs, clip 0.2.
- **Pattern in the top 5:** all use GAE lambda 0.9, gamma 0.995 and batch size 256, learning rates of
  4e-4 to 1e-3 (default 3e-4), entropy 0.001-0.002 (default 0.005) and 8-10 epochs (default 4). Gamma
  0.995 was already the default; lambda 0.9 and batch 256 shift the other way.
- **Not shown:** that the tuned configuration beats the default. The sweep never evaluated the default
  hyperparameters on this problem, and the best of 27 single-seed trials is optimistically biased (the
  trial scores are noisy: 50 episodes, one seed). Neither the headline nor any ablation uses the tuned
  values, which stay a candidate. `configs/experiments/tuning_check.yaml` is the fair check (default
  vs tuned, 3 fresh seeds each, about 35 minutes); it has not been run.
- **Operational note:** three of the four parallel workers crashed at start-up on a database race
  (study creation); the sweep ran on one worker for about 5 hours before I added three workers to the same
  study. The result is unaffected, but the run took longer than the 3 hour estimate.

### Not run

- The **tuned-vs-default check** above, the **16x16 / 5-item** setting, the from-scratch PPO on the
  warehouse (E7), and PBRS at a large bonus.

### Cost and an interruption

The core runs took about 6h40m of wall time on 4 workers (21 jobs, started 22:09, finished 04:47), then
about 1h40m for E4 (100 layouts) and E5 seed 0. Runs came in faster than estimated (a 3M-step run took
1h10m-1h30m against the 1h37m estimate). A forced Windows Update restart killed the first attempt at the
second batch at 15%; it was re-run from zero, so no partial run is in the results.

A second night (started 23:34) ran DQN x3, E5 seeds 1-2 and the two large-bonus runs as 7 jobs on 4
workers (finished 05:09, 5h35m against a 4h59m estimate; DQN ran at about 264 steps/s per worker with 3
workers, `results/throughput_dqn.csv`), followed by the Optuna sweep (finished 10:20).

## Phase 5 plan (as designed)

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
