from abc import ABC, abstractmethod
import torch

class Policy(ABC, torch.nn.Module):

    def __init__(self, input_size : int, number_actions : int):
        super().__init__()
        self.input_size = input_size
        self.number_actions = number_actions

    @abstractmethod
    def forward(self, observation : torch.Tensor) -> torch.distributions.Distribution:
        pass

    def sample(self, obs : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass, sample the policy to get an action. Returns the action and its log probability"""
        distribution = self.forward(obs)
        actions = distribution.sample()
        log_probs = distribution.log_prob(actions).sum(-1) # we need to sum the log probabilities together because
        # for an n dimensional action a with independent elements, p(a) = p(a1) * p(a2) * ... * p(an) => log(p(a)) =
        # log(p(a1)) + ... + log(p(an))

        return actions, log_probs

    def log_probability_of_action(self, obs : torch.Tensor, action : torch.Tensor) -> torch.Tensor:
        """Run forward pass, find log prob of the given action"""
        distribution = self.forward(obs)
        log_probs = distribution.log_prob(action).sum(-1)  # we need to sum the log probabilities together because
        # for an n dimensional action a with independent elements, p(a) = p(a1) * p(a2) * ... * p(an) => log(p(a)) =
        # log(p(a1)) + ... + log(p(an))

        return log_probs
