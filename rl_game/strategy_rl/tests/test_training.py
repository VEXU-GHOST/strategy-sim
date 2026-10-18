import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from vex_rl.config import load_config
from vex_rl.model import MaskedActorCritic, load_checkpoint, save_checkpoint
from vex_rl.simulator import StrategyEnv
from vex_rl.trainer import PPOParams, time_aware_gae, train_policy
from vex_rl.evaluate import evaluate_policy


class TestLearning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()

    def test_time_discount_zero_interval(self):
        a, returns = time_aware_gae(np.array([1, 2], dtype=np.float32),
            np.array([0, 0], dtype=np.float32), np.array([False, True]),
            np.array([0.0, 2.0]), 0.0, gamma_per_second=0.9, lambda_per_second=1.0)
        self.assertAlmostEqual(float(a[1]), 2.0, places=5)
        self.assertAlmostEqual(float(a[0]), 3.0, places=5)

    def test_masked_policy_never_invalid(self):
        env = StrategyEnv(self.cfg, seed=1)
        model = MaskedActorCritic(env.encoder.obs_dim, len(env.codec))
        mask = env.action_mask()
        obs = env.observation()
        for _ in range(100):
            selected = model.choose(obs, mask)
            self.assertTrue(mask[selected])

    def test_save_reload_inference(self):
        env = StrategyEnv(self.cfg, seed=1)
        model = MaskedActorCritic(env.encoder.obs_dim, len(env.codec))
        with tempfile.TemporaryDirectory() as path:
            save_checkpoint(model, Path(path) / "model.pt", self.cfg, {"test": True})
            loaded = load_checkpoint(Path(path) / "model.pt", env.encoder.obs_dim,
                                     len(env.codec), self.cfg)
            obs = env.observation()
            self.assertEqual(model.choose(obs, env.action_mask(), deterministic=True),
                             loaded.choose(obs, env.action_mask(), deterministic=True))

    def test_short_ppo_and_eval(self):
        with tempfile.TemporaryDirectory() as path:
            config = load_config()
            params = PPOParams.from_config(config)
            params.rollout, params.minibatch, params.epochs = 32, 16, 1
            filename = str(Path(path) / "model.pt")
            model = train_policy(config, opponent="random", steps=64, seed=5,
                                 save=filename, params=params)
            s = evaluate_policy(config, "ppo", "random", 2, model=model)
            self.assertEqual(s["episodes"], 2)
            self.assertTrue(np.isfinite(s["our_score"]))
            self.assertTrue(Path(filename).exists())


if __name__ == "__main__":
    unittest.main()
