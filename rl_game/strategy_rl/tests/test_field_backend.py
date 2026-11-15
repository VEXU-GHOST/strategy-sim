"""PPO contract + Override rules on top of rl_game/env/vex_field_environment."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from vex_rl.actions import Action
from vex_rl.config import FIELD_CONFIG, load_config
from vex_rl.contract import check_contract
from vex_rl.envs import make_env
from vex_rl.evaluate import evaluate_policy
from vex_rl.field_backend import FieldStrategyEnv, ITEM_TYPES  # also puts rl_game/env on sys.path
from vex_rl.policies import choose_action
from vex_field_environment import Pose
from vex_rl.trainer import PPOParams, train_policy


def run_until_idle(env, robot="A0", limit=200):
    for _ in range(limit):
        if env.pending[robot] is None or env.done:
            return
        env.step(env.codec.encode(Action("WAIT")))
    raise AssertionError("robot never became idle")


class TestFieldBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(FIELD_CONFIG)

    def env(self, **kwargs):
        return make_env(self.cfg, **{"seed": 1, **kwargs})

    def test_factory_contract_and_dimensions(self):
        env = self.env(opponent="greedy")
        self.assertIsInstance(env, FieldStrategyEnv)
        check_contract(env)
        self.assertEqual(len(env.codec), 64)  # 9x3 collect + 9x3 score + 4 toggles + midfield + wait + 4 block
        self.assertEqual(env.encoder.obs_dim, 302)
        self.assertEqual(env.total_time, 120)
        self.assertEqual(env.engine.settings["episode_seconds"], 120)

    def test_derived_layout_matches_override(self):
        owners = sorted(g["owner"] for g in self.cfg["goals"])
        self.assertEqual(owners.count("red"), 2)
        self.assertEqual(owners.count("blue"), 2)
        self.assertEqual(owners.count("neutral"), 5)
        self.assertEqual([g["quadrant"] for g in self.cfg["goals"]].count(-1), 1)  # one Midfield Goal
        self.assertEqual(len(self.cfg["toggles"]), 4)

    def test_preload_and_protected_alliance_goals(self):
        env = self.env()
        for rid, robot in env.field.robots.items():
            color = ("red", "blue")[robot.team]
            self.assertEqual(robot.held_items[ITEM_TYPES.index(f"{color}_yellow")], 1)
        mask = env.action_mask("A0")
        enc = env.codec.encode
        self.assertTrue(mask[enc(Action("SCORE", "G_RN", "alliance_pin"))])
        self.assertTrue(mask[enc(Action("SCORE", "G_RN", "yellow_pin"))])  # red/yellow: either face
        self.assertTrue(mask[enc(Action("SCORE", "G_NW", "alliance_pin"))])
        for goal in ("G_BN", "G_BS"):  # SG9
            self.assertFalse(mask[enc(Action("SCORE", goal, "alliance_pin"))])
        self.assertFalse(mask[enc(Action("SCORE", "G_RN", "cup"))])  # no cup carried
        self.assertFalse(mask[enc(Action("COLLECT", "Z_W", "alliance_pin"))])  # one Pin max (SG6)
        self.assertTrue(mask[enc(Action("COLLECT", "Z_W", "cup"))])

    def test_match_loads_driver_only_and_own_alliance(self):
        env = self.env()
        enc = env.codec.encode
        env.field.robots["A0"].held_items[:] = 0
        env._version += 1
        self.assertFalse(env.action_mask("A0")[enc(Action("COLLECT", "L_RN", "alliance_pin"))])
        env.field.time = 31.0
        env._version += 1
        mask = env.action_mask("A0")
        self.assertTrue(mask[enc(Action("COLLECT", "L_RN", "alliance_pin"))])
        self.assertTrue(mask[enc(Action("COLLECT", "L_RN", "yellow_pin"))])
        self.assertFalse(mask[enc(Action("COLLECT", "L_RS", "yellow_pin"))])  # no yellow stock there
        self.assertFalse(mask[enc(Action("COLLECT", "L_BN", "alliance_pin"))])  # opponent loader

    def test_native_score_through_engine(self):
        env = self.env(opponent="none")
        env.step(env.codec.encode(Action("SCORE", "G_RN", "alliance_pin")))
        self.assertEqual(env.active_robot, "A1")
        env.step(env.codec.encode(Action("WAIT")))
        run_until_idle(env)
        self.assertEqual(env.field.goals["G_RN"].stack, [(ITEM_TYPES.index("red_yellow"), "red")])
        self.assertEqual(env.scores(), [5.0, 0.0])
        self.assertFalse(env.goals["G_RN"].can_add("red_pin"))  # needs a Cup next
        self.assertGreater(env.time, 0.0)

    def test_toggle_owns_quadrant_yellow_halves(self):
        env = self.env()
        env.field.goals["G_NW"].stack.append((ITEM_TYPES.index("yellow_yellow"), "yellow"))
        self.assertEqual(env.scores(), [0.0, 0.0])
        env.field.zone_colors[1] = 1  # T_N set to red
        self.assertEqual(env.scores(), [10.0, 0.0])
        env.field.zone_colors[1] = 2
        self.assertEqual(env.scores(), [0.0, 10.0])

    def test_midfield_goal_ownership_and_robot_points(self):
        env = self.env()
        env.field.goals["G_MID"].stack.append((ITEM_TYPES.index("yellow_yellow"), "yellow"))
        env.field.set_pose("A0", Pose(-0.8, 0.0))
        self.assertTrue(env.in_midfield("A0"))
        self.assertEqual(env.scores(), [0.0, 0.0])  # evaluated only at match end (SC1)
        final = env.scores(include_midfield=True)
        self.assertEqual(final, [10.0 + 8.0, 0.0])
        env.field.set_pose("B0", Pose(0.8, 0.0))
        self.assertEqual(env.scores(include_midfield=True), [8.0, 8.0])  # tie: nobody owns

    def test_blue_view_is_symmetric_at_reset(self):
        env = self.env()
        red = env.observation(team=0, active_robot="A0")
        blue = env.observation(team=1, active_robot="B0")
        np.testing.assert_allclose(red[75:], blue[75:], atol=1e-6)

    def test_full_episode_each_baseline(self):
        for style in ("random", "greedy", "coordination"):
            env = self.env(opponent="greedy", seed=20)
            env.reset(seed=21)
            finished, decisions = False, 0
            while not finished:
                mask = env.action_mask()
                choice = choose_action(env, env.active_robot, style, env.rng)
                self.assertTrue(mask[choice])
                _, reward, finished, _, info = env.step(choice)
                self.assertTrue(np.isfinite(reward))
                decisions += 1
                self.assertLess(decisions, 600, "too many decisions: collision livelock?")
            self.assertEqual(info["game_time"], 120)
            self.assertIn(info["win"], (0.0, 0.5, 1.0))
            self.assertIsNone(env.active_robot)

    def test_short_ppo_on_field_backend(self):
        with tempfile.TemporaryDirectory() as path:
            params = PPOParams.from_config(self.cfg)
            params.rollout, params.minibatch, params.epochs = 32, 16, 1
            model = train_policy(self.cfg, opponent="greedy", steps=64, seed=5,
                                 save=str(Path(path) / "field.pt"), params=params)
            summary = evaluate_policy(self.cfg, "ppo", "greedy", 1, model=model)
            self.assertTrue(np.isfinite(summary["our_score"]))


if __name__ == "__main__":
    unittest.main()
