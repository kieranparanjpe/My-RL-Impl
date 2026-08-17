from datetime import datetime

import torch

from rl_commons.config.run_info import RLRunInfo
from rl_commons.mdp.mdp_config import MdpConfig
from rl_commons.policies.policy import Policy
from my_rl_impl.algorithms.policies.policy_configs import BetaPolicyConfig, CategoricalPolicyConfig
from my_rl_impl.algorithms.ppo_config import PPOHyperparams
from my_rl_impl.algorithms.value_function_config import ValueFunctionConfig
from my_rl_impl.config import RunConfig
from my_rl_impl.train import Trainer


def _tiny_discrete_run_config():
    return RunConfig(
        algorithm=PPOHyperparams(n_timesteps=40, buffer_size=8, batch_size=4, gradient_epochs=1),
        policy=CategoricalPolicyConfig(hidden_sizes=[8]),
        value_fn=ValueFunctionConfig(hidden_sizes=[8]),
        mdp=MdpConfig(normalise_obs=False, normalise_reward=False),
    )


def _tiny_discrete_run_info():
    return RLRunInfo(task_id="CartPole-v1", algorithm_id="ppo", policy_id="categorical",
                     grid_index=None, time=datetime(2026, 1, 1, 12, 0, 0))


def _tiny_continuous_run_config():
    return RunConfig(
        algorithm=PPOHyperparams(n_timesteps=12, buffer_size=6, batch_size=3, gradient_epochs=1),
        policy=BetaPolicyConfig(hidden_sizes=[8]),
        value_fn=ValueFunctionConfig(hidden_sizes=[8]),
        mdp=MdpConfig(normalise_obs=False, normalise_reward=False),
    )


def _tiny_continuous_run_info():
    return RLRunInfo(task_id="HalfCheetah-v5", algorithm_id="ppo", policy_id="single_beta",
                     grid_index=None, time=datetime(2026, 1, 1, 12, 0, 0))


def test_trainer_runs_end_to_end_discrete_categorical_policy_on_cartpole():
    trainer = Trainer(_tiny_discrete_run_info(), _tiny_discrete_run_config(),
                      logging=False, save_policy=False, record=False)
    initial_params = [p.clone() for p in trainer.policy.parameters()]

    trainer.run()

    assert any(not torch.equal(p0, p1) for p0, p1 in zip(initial_params, trainer.policy.parameters()))


def test_trainer_runs_end_to_end_continuous_beta_policy_on_half_cheetah():
    trainer = Trainer(_tiny_continuous_run_info(), _tiny_continuous_run_config(),
                      logging=False, save_policy=False, record=False)
    initial_params = [p.clone() for p in trainer.policy.parameters()]

    trainer.run()

    assert any(not torch.equal(p0, p1) for p0, p1 in zip(initial_params, trainer.policy.parameters()))


def test_trainer_saves_and_reloads_policy_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trainer = Trainer(_tiny_discrete_run_info(), _tiny_discrete_run_config(),
                      logging=False, save_policy=True, record=False)

    trainer.run()

    checkpoints = list((tmp_path / "saved_policies" / "CartPole-v1").rglob("*.pth"))
    assert len(checkpoints) >= 1

    loaded = Policy.load(str(checkpoints[-1]), policy_id="categorical")
    assert loaded.obs_norm_stats.mean == 0.0
    assert loaded.obs_norm_stats.var == 1.0
    assert loaded.input_size == 4
    assert loaded._number_actions == 2
