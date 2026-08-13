import pytest
import torch

from src.algorithms.replay_buffer import ReplayBuffer


def test_append_until_full_then_rejects():
    buf = ReplayBuffer(capacity=2, obs_size=3, action_size=1)

    assert buf.append(torch.zeros(3), torch.zeros(1), torch.tensor(1.0), torch.tensor(0.0),
                      torch.zeros(3), torch.tensor(0)) is True
    assert buf.is_full() is False
    assert len(buf) == 1

    assert buf.append(torch.zeros(3), torch.zeros(1), torch.tensor(1.0), torch.tensor(0.0),
                      torch.zeros(3), torch.tensor(0)) is True
    assert buf.is_full() is True
    assert len(buf) == 2

    assert buf.append(torch.zeros(3), torch.zeros(1), torch.tensor(1.0), torch.tensor(0.0),
                      torch.zeros(3), torch.tensor(0)) is False
    assert len(buf) == 2


def test_getitem_returns_stored_values():
    buf = ReplayBuffer(capacity=2, obs_size=2, action_size=1)
    obs = torch.tensor([1.0, 2.0])
    next_obs = torch.tensor([3.0, 4.0])
    buf.append(obs, torch.tensor([5.0]), torch.tensor(1.5), torch.tensor(0.25), next_obs, torch.tensor(1))
    buf.append(obs, torch.tensor([5.0]), torch.tensor(1.5), torch.tensor(0.25), next_obs, torch.tensor(1))

    stored_obs, action, reward, log_prob, stored_next_obs, advantage, value_target, terminal = buf[0]

    assert torch.equal(stored_obs, obs)
    assert torch.equal(stored_next_obs, next_obs)
    assert action.item() == 5.0
    assert reward.item() == 1.5
    assert log_prob.item() == 0.25
    assert terminal.item() == 1


def test_insert_advantage_and_value_target():
    buf = ReplayBuffer(capacity=2, obs_size=2, action_size=1)
    buf.append(torch.zeros(2), torch.zeros(1), torch.tensor(0.0), torch.tensor(0.0), torch.zeros(2), torch.tensor(0))
    buf.append(torch.zeros(2), torch.zeros(1), torch.tensor(0.0), torch.tensor(0.0), torch.zeros(2), torch.tensor(0))

    buf.insert_advantage(0, torch.tensor(1.5))
    buf.insert_value_target(1, torch.tensor(2.5))

    assert buf[0][5].item() == 1.5
    assert buf[1][6].item() == 2.5


def test_standardize_advantages():
    buf = ReplayBuffer(capacity=3, obs_size=1, action_size=1)
    for _ in range(3):
        buf.append(torch.zeros(1), torch.zeros(1), torch.tensor(0.0), torch.tensor(0.0),
                   torch.zeros(1), torch.tensor(0))
    buf.insert_advantage(0, torch.tensor(1.0))
    buf.insert_advantage(1, torch.tensor(2.0))
    buf.insert_advantage(2, torch.tensor(3.0))

    buf.standardize_advantages()

    advantages = buf._advantages.squeeze()
    assert torch.allclose(advantages.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(advantages.std(), torch.tensor(1.0), atol=1e-3)


def test_reset_clears_size_but_keeps_capacity():
    buf = ReplayBuffer(capacity=2, obs_size=1, action_size=1)
    buf.append(torch.zeros(1), torch.zeros(1), torch.tensor(0.0), torch.tensor(0.0), torch.zeros(1), torch.tensor(0))

    buf.reset()

    assert len(buf) == 0
    assert buf.is_full() is False


def test_capacity_below_two_warns():
    with pytest.warns(UserWarning, match="capacity is 1"):
        ReplayBuffer(capacity=1, obs_size=1, action_size=1)


def test_getitem_before_full_warns():
    buf = ReplayBuffer(capacity=2, obs_size=1, action_size=1)
    buf.append(torch.zeros(1), torch.zeros(1), torch.tensor(0.0), torch.tensor(0.0), torch.zeros(1), torch.tensor(0))

    with pytest.warns(UserWarning, match="unfinished"):
        _ = buf[0]
