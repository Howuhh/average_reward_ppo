import wandb
import torch

from ppo.agent import Agent
from ppo.utils import set_seed
from ppo.trainer import PPOTrainer


def main():
    wandb.init(
        project="PPO",
        entity="Howuhh",
        group="cheetah_average"
    )

    set_seed(32)
    agent = Agent(state_dim=17, action_dim=6, hidden_dim=64)
    trainer = PPOTrainer(
        env_name="HalfCheetah-v3",
        # num_envs=1024,
        # num_steps=32,
        num_envs=4,
        num_steps=100,
        num_epochs=10,
        # batch_size=256,
        batch_size=32,
        learning_rate=3e-4,
        linear_decay_lr=False,
        entropy_loss_coef=0.0,
        value_loss_coef=0.5,
        reward_tau=0.3,
        value_tau=0.3,  # WHY IS LOSS RISING?
        value_constraint=1.0,
        # gae_lambda=0.95,
        clip_eps=0.2,
        clip_grad=0.5,
        # target_kl=0.075
    )
    trainer.train(
        agent=agent,
        total_steps=3_000_000,
        eval_every=5
    )
    torch.save(agent, "agent.pt")


if __name__ == "__main__":
    main()