# RLProject

A config-driven reinforcement learning system: Stable Baselines3 and from-scratch DQN/PPO on Gymnasium benchmarks, with a custom **Warehouse robot** environment as the headline problem.

> Work in progress, built phase by phase. See the status table below.

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Scaffold, tooling, CI | Done |
| 1 | SB3 benchmark pipeline (train / evaluate / record) | Done |
| 2 | From-scratch DQN and PPO | Planned |
| 3 | Warehouse-v0 environment and baselines | Planned |
| 4 | Warehouse training and reward engineering | Planned |
| 5 | Experiments, tuning, statistics | Planned |
| 6 | Streamlit dashboard | Planned |
| 7 | Polish | Planned |

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # installs CUDA 12.8 PyTorch on Windows
uv sync --extra box2d   # adds Box2D for LunarLander
```

## Usage

```bash
# Train from a YAML config; any field can be overridden with --set
uv run rlagent train configs/benchmarks/cartpole_ppo.yaml --seed 0
uv run rlagent train configs/benchmarks/cartpole_dqn.yaml --set train.total_timesteps=20000

# Evaluate and record a finished run
uv run rlagent evaluate runs/CartPole-v1/ppo-sb3/zoo/seed_0 --episodes 100
uv run rlagent record runs/CartPole-v1/ppo-sb3/zoo/seed_0 --out docs/figures/cartpole_ppo.gif

# Watch learning curves
uv run tensorboard --logdir runs
```

Each run writes to `runs/<env>/<algo>-<impl>/<name>/seed_<k>/` with the resolved `config.yaml`, `progress.csv`, TensorBoard logs, and the best and final models (with `VecNormalize` statistics when used).

## Benchmark results (Phase 1)

Hyperparameters start from the RL Baselines3 Zoo. Evaluation uses the best checkpoint, deterministic actions, 100 episodes.

| Env | Algo | Seed | Mean return |
|---|---|---|---|
| CartPole-v1 | PPO | 0 | 500.0 |
| CartPole-v1 | DQN | 0 | 500.0 |
| LunarLander-v3 | PPO (1M steps) | 0 | 255.9 ± 20.2 |
| LunarLander-v3 | PPO (1M steps) | 1 | 258.9 ± 18.9 |
| LunarLander-v3 | PPO (1M steps) | 2 | 260.2 ± 21.9 |

LunarLander is considered solved at 200. DQN on CartPole is unstable: the best checkpoint scores 500, but its final checkpoint scored 104, so best-checkpoint selection matters. DQN on LunarLander and multi-seed CartPole runs come with the aggregated statistics in Phase 5.

| CartPole PPO | LunarLander PPO |
|---|---|
| ![CartPole PPO](docs/figures/cartpole_ppo.gif) | ![LunarLander PPO](docs/figures/lunarlander_ppo.gif) |

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest -m "not slow"
```
