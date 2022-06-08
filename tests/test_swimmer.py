import unittest

from ppo.agent import Agent
from ppo.trainer import PPOTrainer
from ppo.utils import set_seed


class TestSwimmerV3(unittest.TestCase):
    def setUp(self):
        set_seed(seed=42)

        self.agent = Agent(state_dim=8, action_dim=2, hidden_dim=64)
        self.trainer = PPOTrainer(
            env_name="Swimmer-v3",
            num_envs=64,
            num_steps=256,
            num_epochs=15,
            batch_size=256,
            learning_rate=3e-4,
            linear_decay_lr=False,
            entropy_loss_coef=0.0,
            value_loss_coef=1.0,
            reward_tau=0.1,
            value_tau=0.1,
            value_constraint=1.0,
            gae_lambda=0.99,
            clip_eps=0.2,
            clip_grad=10.0,
        )

    def test_train(self):
        self.trainer.train(
            agent=self.agent,
            total_steps=1_000_000,
            eval_every=10,
            seed=42, 
            eval_seed=42
        )
        returns, lens = self.trainer.evaluate(self.agent, 10, seed=42)

        self.assertGreaterEqual(returns.mean(), 350.0)


if __name__ == "__main__":
    unittest.main()