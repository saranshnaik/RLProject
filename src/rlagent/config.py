"""Typed experiment configuration (YAML -> Pydantic) with CLI-style dot overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EnvConfig(_Strict):
    id: str
    kwargs: dict[str, Any] = Field(default_factory=dict)
    n_envs: int = Field(1, ge=1)
    subproc: bool = False  # SubprocVecEnv (spawn on Windows); default DummyVecEnv
    vec_normalize: bool = False
    norm_obs: bool = True
    norm_reward: bool = True


class AlgoConfig(_Strict):
    name: Literal["ppo", "dqn", "maskable_ppo"]
    impl: Literal["sb3", "scratch"] = "sb3"
    policy: str = "MlpPolicy"
    device: str = "cpu"  # MLP policies are faster on CPU; use "cuda" for CNN/DQN
    policy_kwargs: dict[str, Any] = Field(default_factory=dict)
    # Passed straight to the algorithm. Strings like "lin_0.001" become linear schedules.
    hyperparams: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_scratch_support(self) -> AlgoConfig:
        if self.impl == "scratch":
            if self.name == "maskable_ppo":
                raise ValueError("maskable_ppo is only available with impl: sb3")
            if self.policy != "MlpPolicy":
                raise ValueError("scratch agents only support policy: MlpPolicy")
        return self


class TrainConfig(_Strict):
    total_timesteps: int = Field(gt=0)
    seed: int = 0
    eval_freq: int = Field(10_000, gt=0)  # in total env steps
    n_eval_episodes: int = Field(10, ge=1)
    checkpoint_freq: int = Field(0, ge=0)  # in total env steps; 0 disables


class CurriculumConfig(_Strict):
    """Warehouse difficulty stages. Training starts at ``stages[0]``; the evaluation env always
    uses the full target difficulty from ``env.kwargs``. The learner is promoted to the next stage
    when the success rate over the last ``window`` training episodes reaches ``threshold``."""

    stages: list[dict[str, Any]] = Field(min_length=2)
    threshold: float = Field(0.85, ge=0, le=1)  # 0 = promote after `window` episodes regardless
    window: int = Field(200, ge=10)


class ExperimentConfig(_Strict):
    name: str = "default"
    env: EnvConfig
    algo: AlgoConfig
    train: TrainConfig
    curriculum: CurriculumConfig | None = None

    @model_validator(mode="after")
    def _check_curriculum_support(self) -> ExperimentConfig:
        if self.curriculum is not None:
            if self.env.id != "Warehouse-v0":
                raise ValueError("curriculum is only supported for Warehouse-v0")
            if self.algo.impl == "scratch":
                raise ValueError("curriculum is only supported with impl: sb3")
        return self


def _set_nested(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = data
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"Cannot override '{dotted_key}': '{key}' is not a mapping")
    node[keys[-1]] = value


def apply_overrides(data: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``key.sub=value`` overrides; values are parsed as YAML (``3``, ``true``, ``[1,2]``)."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like key.sub=value")
        key, raw = item.split("=", 1)
        _set_nested(data, key.strip(), yaml.safe_load(raw))
    return data


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    apply_overrides(data, overrides)
    return ExperimentConfig.model_validate(data)


def dump_config(cfg: ExperimentConfig, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.model_dump(mode="json"), f, sort_keys=False)
