import os
import gym
import math
import torch
import random

import envpool
import numpy as np
import torch.nn as nn

from typing import Optional


def init_linear(
        linear: torch.nn.Linear,
        std: float = math.sqrt(2),
        bias: float = 0.0
) -> torch.nn.Linear:
    torch.nn.init.orthogonal_(linear.weight, std)
    torch.nn.init.constant_(linear.bias, bias)
    return linear


def set_seed(seed: int, env: Optional[gym.Env] = None):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)


def make_vec_env(
        env_name: str,
        num_envs: int = 4,
        normalize_reward: bool = False,
        seed: Optional[float] = None
):
    vec_env = envpool.make(env_name, env_type="gym", num_envs=num_envs, seed=seed)
    vec_env.num_envs = num_envs
    vec_env.is_vector_env = True
    vec_env.single_action_space = vec_env.action_space
    vec_env.single_observation_space = vec_env.observation_space

    if normalize_reward:
        vec_env = gym.wrappers.NormalizeReward(vec_env, gamma=1.0)
        vec_env = gym.wrappers.TransformReward(vec_env, lambda reward: np.clip(reward, -10, 10))

    vec_env = gym.wrappers.ClipAction(vec_env)

    return vec_env


class RunningMeanStd(nn.Module):
    """Tracks the mean, variance and count of values."""
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        super().__init__()
        """Tracks the mean, variance and count of values."""
        self.mean = torch.zeros(shape, dtype=torch.float, requires_grad=False)
        self.var = torch.ones(shape, dtype=torch.float, requires_grad=False)
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        if batch_count == 1:
            # as in numpy version for batch_size == 1
            batch_var = torch.zeros_like(batch_mean)

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = self.update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    @staticmethod
    def update_mean_var_count_from_moments(
        mean, var, count, batch_mean, batch_var, batch_count
    ):
        """Updates the mean, var and count using the previous mean, var, count and batch values."""
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count


def rollout(env, agent, greedy: bool = False, device: str = "cpu"):
    total_reward, total_steps = 0.0, 0.0
    state, done = env.reset(), [False]
    while not done[0]:
        state = torch.tensor(state, dtype=torch.float, device=device).reshape(1, -1)

        action, _ = agent.get_action(state, greedy=greedy)
        state, reward, done, info = env.step(action.cpu().numpy().reshape(1, -1))

        total_reward += reward[0]
        total_steps += 1

    return total_reward, total_steps
