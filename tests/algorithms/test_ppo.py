import pytest
import torch

from rl_commons.log import NullLogger
from rl_commons.mdp.mdp_termination_state import MdpTerminationState
from src.algorithms.policies.categorical_policy import CategoricalPolicy
from src.algorithms.policies.policy_configs import CategoricalPolicyConfig
from src.algorithms.ppo import PPO
from src.algorithms.ppo_config import PPOHyperparams
from src.algorithms.value_function_config import ValueFunctionConfig


def _make_ppo(**hp_overrides):
    hyperparameters = PPOHyperparams(buffer_size=4, batch_size=2, gradient_epochs=1, **hp_overrides)
    policy = CategoricalPolicy(2, 2, CategoricalPolicyConfig(hidden_sizes=[4]))
    return PPO(hyperparameters, policy, obs_dimension=2, action_dimension=2, discrete=True,
              logger=NullLogger(), value_fn_config=ValueFunctionConfig(hidden_sizes=[4]))


def test_loss_clips_ratio_and_takes_min_for_positive_advantage():
    ppo = _make_ppo(importance_ratio_clip=0.2, entropy_loss_weight=0.0)

    loss = ppo._loss(new_policy_log_prob=torch.tensor([0.0]), old_policy_log_prob=torch.tensor([-1.0]),
                     advantage=torch.tensor([1.0]), entropy=torch.tensor([0.0]))

    ratio = torch.exp(torch.tensor(1.0)).item()
    expected = -min(ratio * 1.0, 1.2 * 1.0)
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-5)


def test_loss_clips_ratio_and_takes_min_for_negative_advantage():
    ppo = _make_ppo(importance_ratio_clip=0.2, entropy_loss_weight=0.0)

    loss = ppo._loss(new_policy_log_prob=torch.tensor([0.0]), old_policy_log_prob=torch.tensor([-1.0]),
                     advantage=torch.tensor([-1.0]), entropy=torch.tensor([0.0]))

    ratio = torch.exp(torch.tensor(1.0)).item()
    expected = -min(ratio * -1.0, 1.2 * -1.0)
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-5)


def test_loss_adds_entropy_bonus_when_weighted():
    ppo = _make_ppo(importance_ratio_clip=0.2, entropy_loss_weight=0.5)

    loss = ppo._loss(new_policy_log_prob=torch.tensor([0.0]), old_policy_log_prob=torch.tensor([0.0]),
                     advantage=torch.tensor([1.0]), entropy=torch.tensor([2.0]))

    # ratio=1 -> clipped objective=1.0; entropy bonus adds 0.5*2.0=1.0 -> objective=2.0 -> loss=-2.0
    assert torch.allclose(loss, torch.tensor(-2.0), atol=1e-5)


def test_gae_backwards_matches_hand_computed_values():
    ppo = _make_ppo(gamma=1.0, lamda=1.0)
    ppo._value = lambda obs: obs[..., :1]  # deterministic stub: value(obs) = obs[0]

    buffer = ppo._replay_buffer
    obs0, obs1, obs2 = torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0]), torch.tensor([3.0, 0.0])
    buffer.append(obs0, torch.tensor([0]), torch.tensor(1.0), torch.tensor(0.0), obs1,
                 torch.tensor(MdpTerminationState.IN_PROGRESS.value))
    buffer.append(obs1, torch.tensor([0]), torch.tensor(2.0), torch.tensor(0.0), obs2,
                 torch.tensor(MdpTerminationState.TERMINATED.value))

    ppo._gae_backwards()

    advantages = [buffer[i][5].item() for i in range(2)]
    value_targets = [buffer[i][6].item() for i in range(2)]

    assert advantages == pytest.approx([2.0, 0.0])
    assert value_targets == pytest.approx([3.0, 2.0])


def test_update_and_observe_triggers_update_at_capacity_and_resets():
    ppo = _make_ppo()  # buffer_size=4, batch_size=2, gradient_epochs=1

    initial_value_params = [p.clone() for p in ppo._value.parameters()]
    initial_policy_params = [p.clone() for p in ppo.policy.parameters()]

    obs, next_obs = torch.zeros(2), torch.zeros(2)
    action, log_prob = ppo.sample_action(obs)

    updated_flags = [
        ppo.update_and_observe(obs, next_obs, action, log_prob, 1.0, MdpTerminationState.IN_PROGRESS, t)
        for t in range(4)
    ]

    assert updated_flags == [False, False, False, True]
    assert ppo._replay_buffer.size == 0

    assert any(not torch.equal(p0, p1) for p0, p1 in zip(initial_value_params, ppo._value.parameters()))
    assert any(not torch.equal(p0, p1) for p0, p1 in zip(initial_policy_params, ppo.policy.parameters()))
