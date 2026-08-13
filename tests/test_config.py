from pathlib import Path

from src.algorithms.ppo_config import PPOHyperparams
from src.config import RunConfig, load_config, load_grid_configs

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_load_config_parses_single_run_file():
    config = load_config(str(FIXTURES_DIR / "test_hyperparams.json"),
                         algorithm_id="ppo", policy_id="single_gaussian")

    assert isinstance(config, RunConfig)
    assert isinstance(config.algorithm, PPOHyperparams)
    assert config.algorithm.n_timesteps == 2000000
    assert config.algorithm.batch_size == 512
    assert config.policy.hidden_sizes == [64, 64]


def test_load_grid_configs_parses_grid_file():
    configs = load_grid_configs(str(FIXTURES_DIR / "test_hyperparams_grid.json"),
                                algorithm_id="ppo", policy_id="single_gaussian")

    assert len(configs) == 2  # only entropy_loss_weight has more than one grid value
    entropy_weights = {c.algorithm.entropy_loss_weight for c in configs}
    assert entropy_weights == {0.0, 0.005}
    for c in configs:
        assert c.algorithm.n_timesteps == 5000000
        assert c.policy.hidden_sizes == [64, 64]
