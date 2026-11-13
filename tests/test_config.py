from pathlib import Path

import pytest
from pydantic import ValidationError

from rlagent.agents.sb3_factory import LinearSchedule, parse_hyperparams
from rlagent.config import apply_overrides, load_config

CONFIGS = sorted(Path(__file__).parent.parent.glob("configs/**/*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_all_shipped_configs_validate(path):
    if "sweeps" in path.parts:  # tuning specs have their own schema
        from rlagent.training.tune import load_tune_spec

        spec = load_tune_spec(path)
        load_config(spec.base)
    elif "experiments" in path.parts:  # experiment matrices, validated and fully expanded
        from rlagent.training.matrix import expand_jobs, load_spec

        assert expand_jobs(load_spec(path))
    else:
        load_config(path)


def test_overrides_are_parsed_as_yaml():
    data = apply_overrides(
        {}, ["train.seed=3", "env.vec_normalize=true", "algo.policy_kwargs.net_arch=[64,64]"]
    )
    assert data["train"]["seed"] == 3
    assert data["env"]["vec_normalize"] is True
    assert data["algo"]["policy_kwargs"]["net_arch"] == [64, 64]


def test_override_requires_equals():
    with pytest.raises(ValueError):
        apply_overrides({}, ["train.seed"])


def test_unknown_field_is_rejected():
    path = Path(__file__).parent.parent / "configs/benchmarks/cartpole_ppo.yaml"
    with pytest.raises(ValidationError):
        load_config(path, ["train.total_timestep=10"])


def test_linear_schedule():
    parsed = parse_hyperparams({"learning_rate": "lin_0.001", "gamma": 0.9})
    assert isinstance(parsed["learning_rate"], LinearSchedule)
    assert parsed["learning_rate"](0.5) == pytest.approx(0.0005)
    assert parsed["gamma"] == 0.9
