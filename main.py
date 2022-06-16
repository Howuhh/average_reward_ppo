import wandb

from ppo.agent import Agent
from ppo.utils import set_seed, WandbLogger
from ppo.trainer import PPOTrainer


def main():
    # logger = WandbLogger(
    #     project="PPO",
    #     entity="Howuhh",
    #     group="cheetah_average",
    #     mode="disabled"
    # )
    logger = None

    # Default: 64 envs, 256 steps, 15 epochs, 256 batch size

    set_seed(32)
    agent = Agent(state_dim=17, action_dim=6, hidden_dim=64)
    # agent = Agent(state_dim=8, action_dim=2, hidden_dim=64)
    trainer = PPOTrainer(
        env_name="HalfCheetah-v3",
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
    trainer.train(
        agent=agent,
        logger=logger,
        total_steps=1_000_000,
        eval_every=10,
        seed=42,
        eval_seed=42
    )
    # torch.save(agent, "agent.pt")


if __name__ == "__main__":
    main()