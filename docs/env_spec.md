# Warehouse-v0 specification

A robot works in a grid warehouse. Items sit on aisle cells beside shelves. The robot collects them and delivers them to a single packing station. The task is episodic with a time limit.

```python
import gymnasium as gym, rlagent

env = gym.make("Warehouse-v0", height=12, width=12, n_items=3, capacity=2)
```

## World

- A `height x width` grid. Cells are either **shelf** (blocked) or **aisle** (walkable).
- Generated procedurally from a *layout seed*: shelf rows two cells thick with every third row left as an aisle, each shelf cell kept with probability `shelf_density`. Row 0, the bottom row and column 0 are always aisles. Any aisle cell cut off from the station is converted to shelf, so **every walkable cell is reachable**.
- The **station** is at the bottom-left corner. The robot starts there.
- `n_items` items are placed on distinct reachable aisle cells, preferring cells next to a shelf. Items are drawn fresh every episode; only the shelves come from the layout seed.

## Actions: `Discrete(6)`

| id | action | valid when |
|---|---|---|
| 0-3 | up, down, left, right | target cell is inside the grid and not a shelf |
| 4 | pick | standing on an available item and carrying fewer than `capacity` items |
| 5 | drop | at the station carrying at least one item; delivers **all** carried items |

An invalid action changes nothing except the clock and incurs the invalid-action penalty. `env.action_masks()` returns the valid set (used for MaskablePPO).

## Observations

`obs_type="vector"` (default), float32 in [-1, 1], length `56 + 3 * max_items`:

| part | size | content |
|---|---|---|
| robot position | 2 | (y, x) / (max_size - 1) |
| load | 1 | carried / capacity |
| station offset | 2 | (dy, dx) / max_size |
| items | 3 each | (dy, dx, available) / max_size; zeros once picked, delivered, or padding |
| local patch | 49 | 7x7 occupancy around the robot (1 = shelf or outside the grid) |
| progress | 1 | fraction of items not yet delivered |
| time left | 1 | 1 - t / max_steps |

`obs_type="grid"`: float32 planes `(6, max_size, max_size)`: shelves (padding counts as shelf), robot, available items, station, load (constant), time left (constant).

Offsets are divided by `max_size`, and `max_size` / `max_items` pad the observation, so **the shape stays fixed while grid size and item count change** (needed for curriculum learning). The env never tells the agent which item to fetch next.

## Rewards (`RewardConfig`)

| component | default | when |
|---|---|---|
| step | -0.01 | every step |
| pick | +1 | successful pick |
| delivery | +5 | per item delivered |
| complete | +10 | all items delivered |
| invalid | -0.1 | invalid action |
| shaping | 0 | see below |

Each step's `info["reward_components"]` gives the breakdown, and the reward is exactly its sum. At the end of an episode `info["episode_reward_components"]` holds the totals.

### Shaping modes (`reward: {shaping: ...}`)

- **`none`**: sparse task rewards only.
- **`pbrs`**: potential-based shaping `F = gamma * phi(s') - phi(s)` (Ng, Harada & Russell, 1999) scaled by `shaping_scale`. `phi(s) = -(2 * items_to_pick + items_to_deliver) - dist_to_subgoal / D`, where the subgoal is the nearest available item (or the station when full or finished) and `D` upper-bounds any path. `phi` is 0 at completion, and a potential that depends only on the state leaves the optimal policy unchanged. `shaping_gamma` must equal the algorithm's discount (default 0.99). Tests verify the shaping total over an episode equals `-phi(s0)` and that picking and delivering never lower `phi`.
- **`naive`**: `+naive_bonus` whenever a move reduces the distance to the subgoal, nothing for moving away. Not potential-based. Stepping away and back earns a net gain, so the optimum under this reward is to oscillate. This mode exists to demonstrate **reward hacking**.

## Episode end

- `terminated`: every item delivered.
- `truncated`: `max_steps` reached (default `4 * n_items * (height + width) + 20`).
- At the end, `info` carries `is_success`, `optimal_steps`, `efficiency` (optimal / actual steps on success, else 0), `invalid_actions`, `n_delivered`.

## Layout pools and held-out evaluation

| kwarg | meaning |
|---|---|
| `n_layouts` | pool size: `1` = one fixed layout, `N` = N layouts, `None` = a fresh layout every episode |
| `split` | `train` draws layout seeds below 10,000,000; `test` draws seeds from 10,000,000 up, so the two **never overlap** |

`reset(options={"layout_seed": s, "items": [(y, x), ...]})` forces a specific layout and item placement.

## Baselines and the efficiency metric

`rlagent.agents.baselines`: `random`, `greedy` (walk to the nearest available item, return when full or finished), and `optimal` (exact plan from `envs/warehouse/planner.py`).

The planner solves a small capacity-constrained TSP by DP over item subsets (exact for up to 10 items). With `capacity=1` the total cost is the same for every pickup order, so capacity above 1 is what makes routing non-trivial. **Efficiency** = optimal steps / agent steps, so 1.0 is perfect.

Held-out layouts (`split=test`), 200 episodes each (`results/warehouse_baselines.csv`):

| setting | agent | success | steps | efficiency | invalid / ep |
|---|---|---|---|---|---|
| small (8x8, 2 items, cap 2) | random | 0.5% | 147.5 | 0.001 | 81.8 |
| | greedy | 100% | 26.2 | 1.000 | 0 |
| | optimal | 100% | 26.2 | 1.000 | 0 |
| medium (12x12, 3 items, cap 2) | random | 0% | 308.0 | 0.000 | 186.4 |
| | greedy | 100% | 62.6 | 0.890 | 0 |
| | optimal | 100% | 55.1 | 1.000 | 0 |
| hard (16x16, 5 items, cap 3) | random | 0% | 660.0 | 0.000 | 407.3 |
| | greedy | 100% | 105.0 | 0.910 | 0 |
| | optimal | 100% | 95.2 | 1.000 | 0 |

## Known limitations and notes

- **The step penalty is a weak efficiency signal.** On the medium setting greedy takes 14% more steps than optimal but its return is only about 0.2 lower (35.5 vs 35.7), because the -0.01 step cost is small next to the +5 / +10 bonuses. An agent can succeed without being efficient. Phase 4 checked this: with PBRS and enough training the default -0.01 reaches 0.93-0.95 efficiency, -0.02 was within noise, and -0.05 prevented learning (see [experiments.md](experiments.md)).
- **Return is not a success proxy.** A random agent on the hard setting has a *positive* mean return (+18.8) with 0% success, because lucky picks and drops earn +1 and +5 each. Always report success rate and efficiency alongside return.
- Items are placed near shelves but are otherwise random; there are no time windows, priorities or moving obstacles.
