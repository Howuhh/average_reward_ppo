import torch

from ppo.agent import Agent
from ppo.utils import set_seed, WandbLogger, make_vec_env_gym, make_vec_env_envpool, rollout
from ppo.trainer import PPOTrainer


def main():
    # logger = WandbLogger(
    #     project="PPO",
    #     entity="Howuhh",
    #     group="cheetah_average",
    #     mode="disabled"
    # )
    logger = None

    set_seed(32)
    agent = Agent(state_dim=17, action_dim=6, hidden_dim=64)
    # agent = Agent(state_dim=8, action_dim=2, hidden_dim=64)
    trainer = PPOTrainer(
        env_name="HalfCheetah-v3",
        checkpoints_path="checkpoints",
        num_envs=64,
        num_steps=256,
        num_epochs=10,
        batch_size=256,
        learning_rate=3e-4,
        linear_decay_lr=False,
        entropy_loss_coef=0.0,
        value_loss_coef=0.5,
        reward_tau=0.1,
        value_tau=0.1,
        value_constraint=1.0,
        gae_lambda=0.95,
        clip_eps=0.2,
        clip_grad=0.5
    )
    # trainer.train(
    #     agent=agent,
    #     logger=logger,
    #     total_steps=5_000_000,
    #     eval_every=50,
    #     seed=42,
    #     eval_seed=42
    # )

    env = make_vec_env_gym("HalfCheetah-v3", num_envs=1, seed=42)
    agent.load_state_dict(torch.load("checkpoints/62a47fc5-7d2d-4f1b-94f8-fb89461a5166/agent_304.pt"))
    # print(torch.load("checkpoints/a2f827ea-3ef0-40bc-a7e1-b8a234565f06/agent_60.pt"))
    #
    print(rollout(env, agent, greedy=True, render_path="rollout.mp4"))


if __name__ == "__main__":
    main()