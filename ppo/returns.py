import torch

from torch import Tensor


# For now all return computations are aware about done's, but I will remove it in the future, as
# in average reward setting there is no dones (well at least in theory)
def average_one_step_returns(
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
        reward_rate: float
) -> Tensor:
    assert len(rewards) + 1 == len(values), "values should contain 1 more estimate for final state"
    returns = rewards - reward_rate + (1 - dones) * values[1:]

    return returns


def average_n_step_returns(
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
        reward_rate: float
) -> Tensor:
    """
        rewards: Tensor[num_steps, num_envs]
        values: Tensor[num_steps + 1, num_envs]
        dones: Tensor[num_steps, num_envs]
    """
    assert len(rewards) + 1 == len(values), "values should contain 1 more estimate for final state"
    returns = torch.zeros_like(rewards)

    last_value = values[-1]
    for t in reversed(range(len(rewards))):
        last_value = rewards[t] - reward_rate + (1 - dones[t]) * last_value
        returns[t] = last_value

    return returns


def average_gae_returns(
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
        reward_rate: float,
        gae_lambda: float = 0.95
) -> Tensor:
    """
        rewards: Tensor[num_steps, num_envs]
        values: Tensor[num_steps + 1, num_envs]
        dones: Tensor[num_steps, num_envs]
    """
    assert len(rewards) + 1 == len(values), "values should contain 1 more estimate for final state"
    returns, gae = torch.zeros_like(rewards), 0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] - reward_rate + (1 - dones[t]) * values[t + 1] - values[t]
        gae = delta + gae_lambda * (1 - dones[t]) * gae
        # compute returns as return = advantage + value
        returns[t] = gae + values[t]

    return returns
