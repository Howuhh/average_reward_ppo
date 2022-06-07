import wandb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm, trange

from ppo.utils import make_vec_env, set_seed, rollout
from ppo.returns import average_n_step_returns, average_gae_returns


class PPOTrainer:
    def __init__(
            self,
            env_name,
            num_envs=2,
            reward_tau=0.1,
            value_tau=0.001,
            value_constraint=0.0,
            norm_adv=True,
            clip_eps=0.2,
            num_epochs=10,
            num_steps=2048,
            batch_size=64,
            gae_lambda=0.95,
            value_loss_coef=0.5,
            entropy_loss_coef=0.0,
            learning_rate=3e-4,
            linear_decay_lr=True,
            adam_eps=1e-5,
            clip_grad=0.5,
            target_kl=None,
            device="cpu"
    ):
        # Make it as factory
        self.train_env_f = lambda seed: make_vec_env(env_name, num_envs, normalize_reward=True, seed=seed)
        self.eval_env_f = lambda seed: make_vec_env(env_name, num_envs=1, seed=seed)
        # self.train_env = make_vec_env(env_name, num_envs=num_envs, normalize_reward=True)
        # self.eval_env = make_vec_env(env_name, num_envs=1)

        self.learning_rate = learning_rate
        self.decay_lr = linear_decay_lr
        self.adam_eps = adam_eps
        self.clip_grad = clip_grad

        self.reward_tau = reward_tau
        self.value_tau = value_tau
        self.gae_lambda = gae_lambda
        self.value_constraint = value_constraint

        self.num_epochs = num_epochs
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.clip_eps = clip_eps
        self.norm_adv = norm_adv
        self.value_loss_coef = value_loss_coef
        self.entropy_loss_coef = entropy_loss_coef
        self.target_kl = target_kl

        self.device = device

        self._reward_rate = 0.0
        self._value_rate = 0.0

    def _reset_rates(self):
        self._reward_rate = 0.0
        self._value_rate = 0.0

    def _actor_loss(self, agent, states, actions, log_probs, advantages):
        act_dist = agent.action_dist(states)
        new_log_probs = act_dist.log_prob(actions).sum(-1)

        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        assert new_log_probs.shape == log_probs.shape
        log_probs_ratio = new_log_probs - log_probs
        probs_ratio = log_probs_ratio.exp()

        assert advantages.shape == probs_ratio.shape
        actor_loss_1 = advantages * probs_ratio
        actor_loss_2 = advantages * torch.clip(probs_ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)

        assert actor_loss_1.shape == actor_loss_2.shape
        actor_loss = -torch.minimum(actor_loss_1, actor_loss_2).mean()
        entropy_loss = act_dist.entropy().sum(-1).mean()

        return actor_loss, entropy_loss, log_probs_ratio

    def _critic_loss(self, agent, states, target_returns):
        values = agent.get_value(states).view(-1)

        # update value rate estimate
        mean_value = values.mean()
        self._value_rate = (1 - self.value_tau) * self._value_rate + self.value_tau * mean_value.item()

        assert values.shape == target_returns.shape
        with torch.no_grad():
            target_returns = target_returns - self.value_constraint * self._value_rate

        critic_loss = F.mse_loss(values, target_returns)

        return critic_loss, mean_value

    def _update(self, agent, optimizer, scheduler, states, actions, log_probs, returns, advantages):
        states = states.flatten(0, 1)
        actions = actions.flatten(0, 1)
        log_probs = log_probs.flatten()
        returns = returns.flatten()
        advantages = advantages.flatten()

        actor_loss_epoch, critic_loss_epoch, entropy_loss_epoch, mean_value_epoch = 0, 0, 0, 0
        kl_divs_epoch = 0.0

        kl_div_check = True
        total_idxs = np.arange(self.num_envs * self.num_steps)
        for epoch in trange(1, self.num_epochs + 1, desc="Epochs", leave=False):
            np.random.shuffle(total_idxs)
            for start in range(0, len(total_idxs), self.batch_size):
                batch_idxs = total_idxs[start:start + self.batch_size]

                actor_loss, entropy_loss, log_probs_ratio = self._actor_loss(
                    agent,
                    states[batch_idxs],
                    actions[batch_idxs],
                    log_probs[batch_idxs],
                    advantages[batch_idxs]
                )
                critic_loss, mean_value = self._critic_loss(agent, states[batch_idxs], returns[batch_idxs])

                loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_loss_coef * entropy_loss

                with torch.no_grad():
                    approx_kl_div = torch.mean((torch.exp(log_probs_ratio) - 1) - log_probs_ratio).cpu().numpy()
                    kl_divs_epoch += approx_kl_div

                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        kl_div_check = False
                        tqdm.write(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        break

                optimizer.zero_grad()
                loss.backward()
                if self.clip_grad is not None:
                    nn.utils.clip_grad_norm_(agent.parameters(), self.clip_grad)
                optimizer.step()

                actor_loss_epoch += actor_loss.item()
                critic_loss_epoch += critic_loss.item()
                entropy_loss_epoch += entropy_loss.item()
                mean_value_epoch += mean_value.item()

            if not kl_div_check:
                break

        if self.decay_lr:
            scheduler.step()
        total_updates = epoch * (len(total_idxs) // self.batch_size)

        return {
            "actor_loss_epoch": actor_loss_epoch / total_updates,
            "critic_loss_epoch": critic_loss_epoch / total_updates,
            "entropy_loss_epoch": entropy_loss_epoch / total_updates,
            "kl_div_epoch": kl_divs_epoch / total_updates,
            "mean_value_epoch": mean_value_epoch / total_updates,
        }

    @torch.no_grad()
    def evaluate(self, agent, num_evals, seed):
        set_seed(seed)
        eval_env = self.eval_env_f(seed=seed)

        returns, lens = [], []
        for _ in trange(num_evals, desc="Evaluation", leave=False):
            ep_return, ep_len = rollout(eval_env, agent, greedy=True, device=self.device)
            returns.append(ep_return)
            lens.append(ep_len)

        return np.array(returns), np.array(lens)

    def train(self, agent, total_steps, eval_every=10, num_evals=10, seed=42, eval_seed=42):
        set_seed(seed)
        self._reset_rates()

        train_env = self.train_env_f(seed=seed)

        num_steps, num_envs, device = self.num_steps, self.num_envs, self.device
        num_updates = round(total_steps / (num_envs * num_steps))

        optim = torch.optim.Adam(agent.parameters(), lr=self.learning_rate, eps=self.adam_eps)
        scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.0, total_iters=num_updates)

        # buffers for on-policy data from vec envs
        states = torch.zeros((num_steps, num_envs) + train_env.single_observation_space.shape, dtype=torch.float, device=device)
        actions = torch.zeros((num_steps, num_envs) + train_env.single_action_space.shape, dtype=torch.float, device=device)
        rewards = torch.zeros(num_steps, num_envs, device=device)
        dones = torch.zeros(num_steps, num_envs, device=device)
        log_probs = torch.zeros(num_steps, num_envs, device=device)
        values = torch.zeros(num_steps + 1, num_envs, device=device)

        state = torch.tensor(train_env.reset(), dtype=torch.float, device=device)
        for update in trange(num_updates):
            agent.train()
            # gather batch of on-policy experience from vectorized environment
            for step in trange(num_steps, leave=False, desc="Env step"):
                with torch.no_grad():
                    # update state moments for state normalization
                    agent.update_state_rms(state)

                    action, (log_prob, entropy) = agent.get_action(state)
                    next_state, reward, done, info = train_env.step(action.cpu().numpy())

                    states[step] = state
                    actions[step] = action
                    log_probs[step] = log_prob
                    rewards[step] = torch.tensor(reward, dtype=torch.float, device=device)
                    values[step] = agent.get_value(state).squeeze()
                    dones[step] = torch.tensor(done, device=device)

                    state = torch.tensor(next_state, dtype=torch.float, device=device)
            else:
                # this is value estimate of the trajectory after num_steps
                with torch.no_grad():
                    values[-1] = agent.get_value(state).squeeze()

            # compute returns and advantages from rollout data
            # returns = average_n_step_returns(
            returns = average_gae_returns(
                rewards, values, dones,
                reward_rate=self._reward_rate,
                gae_lambda=self.gae_lambda
            )
            advantages = returns - values[:-1]

            # update on-policy reward rate estimate
            with torch.no_grad():
                self._reward_rate = (1 - self.reward_tau) * self._reward_rate + self.reward_tau * rewards.mean(1).mean(0).item()

            # update networks for number of epochs
            update_info = self._update(agent, optim, scheduler, states, actions, log_probs, returns, advantages)

            total_transitions = (update + 1) * (num_envs * num_steps)
            wandb.log({
                "step": total_transitions,
                "train/lr": scheduler.get_last_lr()[0],
                "train/value_rate": self._value_rate,
                "train/reward_rate": self._reward_rate,
                **{f"train/{k}": v for k, v in update_info.items()}
            })

            # evaluate agent
            if update % eval_every == 0 or update == num_updates - 1:
                agent.eval()
                eval_returns, eval_lens = self.evaluate(agent, num_evals=num_evals, seed=eval_seed)
                wandb.log({
                    "eval/reward_mean": eval_returns.mean(),
                    "eval/reward_std": eval_returns.std(),
                    "step": total_transitions
                })
                # print out some metrics for debugging
                tqdm.write(
                    f"STEPS: {total_transitions} "
                    f"LR: {scheduler.get_last_lr()[0]:.8f} "
                    f"Actor loss: {update_info['actor_loss_epoch']:.3f} "
                    f"Critic_loss: {update_info['critic_loss_epoch']:.3f} "
                    f"Entropy loss: {update_info['entropy_loss_epoch']:.3f} "
                    f"KL div: {update_info['kl_div_epoch']:.3f} "
                    f"MEAN REWARD: {np.mean(eval_returns):.3f} "
                )

        return agent