import argparse
import os
import torch
from pathlib import Path

from rl_commons.execution import BaseEvaluator
from rl_commons.mdp import MdpTerminationState, MdpConfig
from src.algorithms.policies import PolicyFactory

class PolicyExporter(BaseEvaluator):

    def load_checkpoint(self, path):
        import torch
        return torch.load(path, weights_only=False) if path else {}

    def __init__(self, environment_id, policy_id, policy_weights_path):
        self.original_checkpoint = self.load_checkpoint(policy_weights_path)
        policy_obj_or_dict = self.original_checkpoint.get("policy")

        norm_stats_raw = self.original_checkpoint.get("norm_stats")
        obs_rms_stats = (
            norm_stats_raw["obs_mean"].numpy(),
            norm_stats_raw["obs_var"].numpy(),
        ) if norm_stats_raw is not None else None

        mdp_config = MdpConfig(
            normalise_obs=(obs_rms_stats is not None),
            normalise_reward=False,
        )

        super().__init__(task_id=environment_id, mdp_config=mdp_config, obs_rms_stats=obs_rms_stats)

        if isinstance(policy_obj_or_dict, dict):
            self.policy = PolicyFactory.build_policy(policy_id, int(self._mdp.obs_dimension),
                                                     int(self._mdp.action_dimension)).to(self.device)
            if policy_obj_or_dict:
                self.policy.load_state_dict(policy_obj_or_dict)
        elif policy_obj_or_dict is not None:
            self.policy = policy_obj_or_dict.to(self.device)

        self.policy_weights_path = policy_weights_path

    def export(self):
        original_path = Path(self.policy_weights_path)
        
        try:
            parts = list(original_path.parts)
            if 'saved_policies' in parts:
                idx = parts.index('saved_policies')
                parts.insert(idx + 1, 'exported')
                new_path = Path(*parts)
            else:
                new_path = original_path.parent / ('exported_' + original_path.name)
        except Exception:
            new_path = original_path.parent / ('exported_' + original_path.name)
            
        new_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {"policy": self.policy}
        if "norm_stats" in self.original_checkpoint:
            save_dict["norm_stats"] = self.original_checkpoint["norm_stats"]
            
        print(f"Saving fully instantiated policy object to {new_path}...")
        torch.save(save_dict, str(new_path))
        print("Done. You can now load this via: checkpoint = torch.load('{}')".format(new_path))

    def _run(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--environment", "-e", help="Environment Id to run", default="CartPole-v1")
    parser.add_argument("--policy", "-p", help="Policy Id to use", default="categorical")
    parser.add_argument("--weights", "-w", help="Path to weights to load into policy")

    args = parser.parse_args()

    exporter = PolicyExporter(args.environment, args.policy, args.weights)
    exporter.export()
