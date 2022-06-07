import gym
import wandb
import argparse

from ppo.agent import Agent
from ppo.utils import set_seed
from ppo.trainer import PPOTrainer
from distutils.util import strtobool


def create_argparser():
    parser = argparse.ArgumentParser(description="APO training script.")
    # wandb configuration
    parser.add_argument("--project", type=str, default="PPO")
    parser.add_argument("--entity", type=str, default="Howuhh")
    parser.add_argument("--group", type=str, default="APO")
    parser.add_argument("--name", type=str, default="apo")
    # gym configuration
    parser.add_argument("--env_name", type=str, default="Swimmer-v4")
    parser.add_argument("--num_envs", type=int, default=64)
    # ppo configuration
    parser.add_argument("--num_steps", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_decay", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--ent_loss_coef", type=float, default=0.0)
    parser.add_argument("--value_loss_coef", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--value_constraint", type=float, default=0.5)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--clip_grad", type=float, default=0.5)
    parser.add_argument("--train_steps", type=int, default=1_000_000)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument('--eval_every', type=int, default=5)

    parser.add_argument("--train_seed", type=str, default=42)
    parser.add_argument("--eval_seed", type=str, default=10)
    parser.add_argument("--device", type=str, default="cpu")

    return parser


def run_experiment(config):
    set_seed(config.train_seed)
    run = wandb.init(
        project=config.project,
        entity=config.entity,
        group=f"{config.group}_{config.env_name}",
        name=f"{config.name}_{config.env_name}_{config.train_seed}",
        config=config, reinit=True
    )

    tmp_env = gym.make(config.env_name)
    agent = Agent(
        state_dim=tmp_env.observation_space.shape[0],
        action_dim=tmp_env.action_space.shape[0],
        hidden_dim=config.hidden_dim
    ).to(config.device)

    trainer = PPOTrainer(
        env_name=config.env_name,
        num_envs=config.num_envs,
        num_steps=config.num_steps,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        learning_rate=config.lr,
        linear_decay_lr=config.lr_decay,
        entropy_loss_coef=config.ent_loss_coef,
        value_loss_coef=config.value_loss_coef,
        reward_tau=config.tau,
        value_tau=config.tau,
        value_constraint=config.value_constraint,
        gae_lambda=config.gae_lambda,
        clip_eps=config.clip_eps,
        clip_grad=config.clip_grad,
        device=config.device
    )
    trainer.train(
        agent=agent,
        total_steps=config.train_steps,
        eval_every=config.eval_every,
        seed=config.train_seed,
        eval_seed=config.eval_seed
    )

    run.finish()


if __name__ == "__main__":
    args = create_argparser().parse_args()
    run_experiment(args)
