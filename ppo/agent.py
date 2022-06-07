import torch
import torch.nn as nn

from typing import Tuple
from torch import Tensor

from ppo.utils import init_linear, RunningMeanStd
from torch.distributions import Normal


class Agent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.actor_mu = nn.Sequential(
            init_linear(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            init_linear(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            init_linear(nn.Linear(hidden_dim, action_dim), std=0.01)
        )
        self.actor_log_sigma = nn.Parameter(torch.zeros(1, action_dim, requires_grad=True))

        self.critic = nn.Sequential(
            init_linear(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            init_linear(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            init_linear(nn.Linear(hidden_dim, 1), std=1.0)
        )
        self.state_rms = RunningMeanStd(shape=state_dim)

    def normalize_state(self, state: Tensor) -> Tensor:
        state = (state - self.state_rms.mean) / torch.sqrt(self.state_rms.var + 1e-8)
        return state

    def update_state_rms(self, state: Tensor):
        if state.dim() == 1:
            state = state.reshape(1, -1)
        self.state_rms.update(state)

    def action_dist(self, state: Tensor) -> Normal:
        state = self.normalize_state(state)
        return Normal(self.actor_mu(state), self.actor_log_sigma.exp())

    def get_action(self, state: torch.Tensor, greedy: bool = False) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        dist = self.action_dist(state)
        if greedy:
            action = dist.loc
        else:
            action = dist.sample()

        log_prob, entropy = dist.log_prob(action).sum(-1), dist.entropy().sum(-1)

        return action, (log_prob, entropy)

    def get_value(self, state: Tensor) -> Tensor:
        return self.critic(self.normalize_state(state))
