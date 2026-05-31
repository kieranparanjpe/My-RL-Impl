import queue
import threading

import torch
import argparse
from src.algorithms.policies import PolicyFactory
from src.mdp import MdpGym


class Evaluator:

    def __init__(self, environment_id, policy_id, policy_weights_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._mdp = MdpGym(environment_id, self.device, render_mode='human')

        self.policy = PolicyFactory.build_policy(policy_id, self._mdp.obs_dimension, self._mdp.action_dimension).to(self.device)

        state_dict = torch.load(policy_weights_path, weights_only=True)
        self.policy.load_state_dict(state_dict)

        self._stop = threading.Event()

    def _listen_for_commands(self):
        commands = queue.Queue()
        while not self._stop.is_set():
            cmd = input().strip().lower()
            commands.put(cmd)
            if cmd in {"x", "close", "quit", "exit"}:
                self._stop.set()

    def evaluate(self):
        print("Type 'x' or 'close' + Enter to stop.")
        threading.Thread(target=self._listen_for_commands, daemon=True).start()

        last_observation = self._mdp.reset()

        while not self._stop.is_set(): # should write graceful closing for this
            distribution = self.policy.forward(last_observation)
            action = self.policy.sample_action(distribution)

            next_observation, reward, done = self._mdp.step(action)

            if done:
                last_observation = self._mdp.reset()
            else:
                last_observation = next_observation

        self._mdp.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--environment", "-e", help="Environment Id to run", default="CartPole-v1")
    parser.add_argument("--policy", "-p", help="Policy Id to use", default="categorical")
    parser.add_argument("--weights", "-w", help="Path to weights to load into policy")

    args = parser.parse_args()

    evaluator = Evaluator(args.environment, args.policy,args.weights)
    evaluator.evaluate()
