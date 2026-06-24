import argparse

import torch
from tqdm.auto import tqdm
from datetime import datetime

from src.log import WandBLogger, NullLogger, NullRecorder, Recorder
from src.config import RunHyperparams, RunInfo, ConfigLoader
from src.algorithms import PPO
from src.algorithms.policies import PolicyFactory
from src.mdp import MdpGym, MdpTerminationState
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import os


class Trainer:

    def __init__(self, run_info: RunInfo, run_config: RunHyperparams,
                 logging=True, save_policy=False, record=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._run_config = run_config
        self._run_info = run_info

        print(f"\nTraining: {self._run_info.run_name()} with algorithm: [{self._run_info.algorithm_id}] and policy: "
              f"[{self._run_info.policy_id}]")
        print(f"Using config: {self._run_config!r}")

        self._logger = WandBLogger(self._run_info, run_config.__dict__, {
            "charts/episodic_return": 0.0,
            "charts/episode_length": 0,
            "global_step": 0,
        }) if logging else NullLogger()

        self._should_save_policy = save_policy
        if self._should_save_policy:
            self._create_policy_folder()

        self._recorder = Recorder(self._run_info.local_folder_path("saved_videos"),
                                  5, run_config.algorithm.n_timesteps) if record else NullRecorder()

        self._mdp = MdpGym(self._run_info.environment_id, self.device, render_mode=None,
                           recorder=self._recorder, mdp_config=run_config.mdp)

        self.policy = PolicyFactory.build_policy(
            self._run_info.policy_id,
            self._mdp.obs_dimension,
            self._mdp.action_dimension,
            run_config.policy,
        ).to(self.device)

        if self._run_info.algorithm_id == 'ppo':
            self.algorithm = PPO(
                run_config.algorithm, self.policy,
                self._mdp.obs_dimension, self._mdp.action_dimension, self._mdp.discrete,
                logger=self._logger, device=self.device,
                value_fn_config=run_config.value_fn,
            )

    def _create_policy_folder(self):
        directory_path = self._run_info.local_folder_path("saved_policies")
        os.makedirs(directory_path, exist_ok=True)
        return directory_path

    def _save_policy(self, timestep):
        n_timesteps = self._run_config.algorithm.n_timesteps
        width = len(str(n_timesteps))
        save_dict = {"policy": self.policy.state_dict()}
        if (stats := self._mdp.obs_rms_stats) is not None:
            save_dict["norm_stats"] = {
                "obs_mean": torch.tensor(stats[0]),
                "obs_var": torch.tensor(stats[1]),
            }
        torch.save(save_dict, f'{self._run_info.local_folder_path("saved_policies")}/policy_{timestep:0{width}d}.pth')

    def train(self):
        last_observation = self._mdp.reset()
        episode_number = 0
        n_timesteps = self._run_config.algorithm.n_timesteps
        for timestep in tqdm(range(n_timesteps)):
            action, log_prob_action = self.algorithm.sample_action(last_observation)

            next_observation, reward, termination_state = self._mdp.step(action)

            updated_policy = self.algorithm.update_and_observe(last_observation, next_observation, action,
                                                               log_prob_action, reward,
                                                               termination_state, timestep)

            if ((updated_policy and self._should_save_policy and episode_number % 500 == 0) or
                    timestep == n_timesteps - 1):
                self._save_policy(timestep)

            self._logger.sum_log_data({
                "charts/episodic_return": reward,
                "charts/episode_length": 1,
            })
            if termination_state is not MdpTerminationState.IN_PROGRESS:
                last_observation = self._mdp.reset()

                self._logger.set_log_data({"global_step": timestep})
                self._logger.log_data("charts/episodic_return", "charts/episode_length", "global_step")
                self._logger.reset("charts/episodic_return", "charts/episode_length")

                self._recorder.new_episode = True
                episode_number += 1

            else:
                last_observation = next_observation

                self._recorder.new_episode = False

        self._mdp.close()
        self._logger.upload_videos(self._recorder)
        self._logger.finish()


def run_one(args, run_config: RunHyperparams, index, now):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    run_info = RunInfo(
        environment_id=args.environment,
        algorithm_id=args.algorithm,
        policy_id=args.policy,
        grid_index=index,
        time=now,
    )

    trainer = Trainer(run_info, run_config, logging=args.log,
                      save_policy=args.save, record=args.record)
    trainer.train()
    return True


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--environment", "-e", help="Environment Id to run", default="CartPole-v1")
    parser.add_argument("--algorithm", "-a", help="Algorithm to use", default="ppo")
    parser.add_argument("--policy", "-p", help="Policy Id to use", default="categorical")
    parser.add_argument("--hyperparameters", help="Path to hyperparameter json file", default=None)
    parser.add_argument("--grid", help="Path to hyperparameter grid json file", default=None)

    parser.add_argument("--log", "-l", help="Enable log to wandb", action="store_true")
    parser.add_argument("--save", "-s", help="Enable policy saving after each update", action="store_true")
    parser.add_argument("--record", "-r", help="Enable episode recording", action="store_true")

    return parser.parse_args()


def gridsearch(args, configs: list[RunHyperparams], now):
    max_parallel = min(os.cpu_count() or 1, 8)
    config_index = 0

    with ProcessPoolExecutor(max_workers=max_parallel) as pool:
        in_flight = set()

        for _ in range(min(max_parallel, len(configs))):
            in_flight.add(pool.submit(run_one, args, configs[config_index], config_index, now))
            config_index += 1

        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)

            for fut in done:
                fut.result()
                if config_index < len(configs):
                    in_flight.add(pool.submit(run_one, args, configs[config_index], config_index, now))
                    config_index += 1


def main():
    args = parse_args()
    now = datetime.now()

    if args.grid is not None:
        configs = ConfigLoader.load_grid(args.grid, args.algorithm, args.policy)
        gridsearch(args, configs, now)
    elif args.hyperparameters is not None:
        run_config = ConfigLoader.load_single(args.hyperparameters, args.algorithm, args.policy)
        run_one(args, run_config, None, now)
    else:
        run_one(args, RunHyperparams(), None, now)


if __name__ == "__main__":
    main()
