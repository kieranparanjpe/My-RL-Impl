import argparse
import torch
from tqdm.auto import tqdm
from datetime import datetime

from src.log import WandBLogger, NullLogger, NullRecorder, Recorder
from src.algorithms import Hyperparameters, PPOHyperparams, PPO, HyperparameterLoader
from src.algorithms.policies import PolicyFactory
from src.mdp.mdp_gym import MdpGym
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import os


class Trainer:

    def __init__(self, environment_id, algorithm_id : str, policy_id : str, hyperparameters : Hyperparameters,
                 logging=True, save_policy=False, record=False, index=0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hyperparameters = hyperparameters

        self.environment_id = environment_id

        now = datetime.now()
        self.run_name = f"{environment_id}@{now:%Y-%m-%d-%H-%M-%S}_RUN-{index}"

        print(f"\nTraining: {self.run_name} with algorithm: [{algorithm_id}] and policy: [{policy_id}]")
        print(f"Using hyperparameters: {self.hyperparameters.__repr__()}")


        self.logger = WandBLogger(self.run_name, environment_id, algorithm_id, policy_id, hyperparameters.__dict__, {
            "charts/episodic_return": 0.0,
            "charts/episode_length": 0,
            "global_step": 0,
        }) if logging else NullLogger()

        self.save_policy = save_policy
        if self.save_policy:
            self.save_policy_folder = self.create_policy_folder()

        self.recorder = Recorder(f"saved_videos/{self.environment_id}/{self.run_name}",
5, self.hyperparameters.n_timesteps) if record else NullRecorder()

        self.mdp = MdpGym(environment_id, self.device, render_mode=None, recorder=self.recorder)

        self.policy = PolicyFactory.build_policy(policy_id, self.mdp.obs_dimension, self.mdp.action_dimension).to(
            self.device)

        if algorithm_id == 'ppo':
            self.algorithm = PPO(self.hyperparameters, self.policy, self.mdp.obs_dimension,
                                 self.mdp.action_dimension, self.mdp.discrete,
                                 logger=self.logger, device=self.device)


    def create_policy_folder(self):
        directory_path = f"saved_policies/{self.environment_id}/{self.run_name}"

        os.makedirs(directory_path, exist_ok=True)
        return directory_path

    def _save_policy(self, timestep):
        torch.save(self.policy.state_dict(), f'{self.save_policy_folder}/policy_{timestep}.pth')

    def train(self):
        last_observation = self.mdp.reset()
        for timestep in tqdm(range(self.hyperparameters.n_timesteps)):
            action, log_prob_action = self.algorithm.sample_action(last_observation)

            next_observation, reward, done = self.mdp.step(action)

            updated_policy = self.algorithm.update_and_observe(last_observation, next_observation, action, log_prob_action, reward,
                                              done, timestep)

            if updated_policy and self.save_policy:
                self._save_policy(timestep)

            self.logger.sum_log_data({
                "charts/episodic_return": reward,
                "charts/episode_length": 1,
            })
            if done:
                last_observation = self.mdp.reset()

                self.logger.set_log_data({'global_step': timestep})
                self.logger.log_data()
                self.logger.reset("charts/episodic_return", "charts/episode_length")

                self.recorder.new_episode = True

            else:
                last_observation = next_observation

                self.recorder.new_episode = False

        self.mdp.close()
        self.logger.upload_videos(self.recorder)
        self.logger.finish()


def run_one(args, hyperparameters, index):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    trainer = Trainer(args.environment, args.algorithm, args.policy, hyperparameters, logging=args.log,
                      save_policy=args.save, record=args.record, index=index)
    trainer.train()
    return True

def parse_args() :
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

def gridsearch(args, hyperparameters_grid):
    max_parallel = min(os.cpu_count() or 1, 8)  # pick your cap
    hyperparam_index = 0

    with ProcessPoolExecutor(max_workers=max_parallel) as pool:
        in_flight = set()

        # start initial batch
        for _ in range(min(max_parallel, len(hyperparameters_grid))):
            hp = hyperparameters_grid[hyperparam_index]
            in_flight.add(pool.submit(run_one, args, hp, hyperparam_index))
            hyperparam_index += 1

        # keep launching next combo when one finishes
        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)

            for fut in done:
                fut.result()
                if hyperparam_index < len(hyperparameters_grid):
                    hp = hyperparameters_grid[hyperparam_index]
                    in_flight.add(pool.submit(run_one, args, hp, hyperparam_index))
                    hyperparam_index += 1

def main():
    args = parse_args()

    if args.grid is not None:
        hyperparameters_grid = HyperparameterLoader.load_grid(args.grid, args.algorithm)
        gridsearch(args, hyperparameters_grid)
    elif args.hyperparameters is not None:
        run_one(args, HyperparameterLoader.load_single(args.hyperparameters, args.algorithm), 0)
    else:
        run_one(args, PPOHyperparams(), 0)

if __name__ == "__main__":
    main()
