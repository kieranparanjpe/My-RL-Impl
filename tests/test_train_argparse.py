import sys

from src.train import parse_args


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py"])

    args = parse_args()

    assert args.environment == "CartPole-v1"
    assert args.algorithm == "ppo"
    assert args.policy == "categorical"
    assert args.hyperparameters is None
    assert args.grid is None
    assert args.log is False
    assert args.save is False
    assert args.record is False


def test_parse_args_overrides(monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["train.py", "-e", "Pendulum-v1", "-a", "ppo", "-p", "single_gaussian", "--log"])

    args = parse_args()

    assert args.environment == "Pendulum-v1"
    assert args.policy == "single_gaussian"
    assert args.log is True
