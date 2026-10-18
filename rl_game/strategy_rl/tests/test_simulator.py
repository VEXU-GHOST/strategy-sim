import unittest
import numpy as np
from vex_rl.actions import Action
from vex_rl.config import load_config
from vex_rl.contract import check_contract
from vex_rl.policies import choose_action
from vex_rl.simulator import StrategyEnv


class TestSimulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()

    def test_contract_and_dimensions(self):
        check_contract(StrategyEnv(self.config, seed=3))

    def test_exact_reset_seed(self):
        a, b = StrategyEnv(self.config, seed=5), StrategyEnv(self.config, seed=5)
        self.assertTrue(np.array_equal(a.observation(), b.observation()))
        self.assertEqual(a.state()["phase"], "autonomous")
        self.assertEqual(a.total_time, 120)
        self.assertEqual(a.auto_time, 30)

    def test_valid_masks_and_block_disabled(self):
        env = StrategyEnv(self.config, seed=1)
        for rid in ("A0", "A1", "B0", "B1"):
            mask = env.action_mask(rid)
            self.assertTrue(mask[env.codec.encode(Action("WAIT"))])
            for action_id in range(len(mask)):
                if env.codec.decode(action_id).verb == "BLOCK":
                    self.assertFalse(mask[action_id])

    def test_collect_score_and_stack(self):
        env = StrategyEnv(self.config, opponent="none", seed=2)
        # A0 starts with an alliance-colored preload, so it can score immediately.
        target = Action("SCORE", "G0", "alliance_pin")
        self.assertTrue(env.action_mask("A0")[env.codec.encode(target)])
        env.step(env.codec.encode(target))  # dispatch A0
        env.step(env.codec.encode(Action("WAIT")))  # dispatch A1
        for _ in range(8):
            if env.goals["G0"].stack:
                break
            env.step(env.codec.encode(Action("WAIT")))
        self.assertEqual(len(env.goals["G0"].stack), 1)
        self.assertEqual(env.goals["G0"].stack[0].kind, "red_pin")
        self.assertGreaterEqual(env.scores()[0], 5)
        # It is not possible to score another Pin on this goal until placing a Cup.
        self.assertFalse(env.goals["G0"].can_add("red_pin"))
        self.assertTrue(env.goals["G0"].can_add("cup"))

    def test_symmetry_action_id_mapping(self):
        env = StrategyEnv(self.config, opponent="none", seed=4)
        for item_type, lhs, rhs in (("SCORE", "G0", "G5"),
                                    ("COLLECT", "S0", "S5"),
                                    ("SET_TOGGLE", "Q0", "Q2")):
            kind = "alliance_pin" if item_type != "SET_TOGGLE" else ""
            can = env.codec.encode(Action(item_type, lhs, kind))
            actual = env.codec.encode(Action(item_type, rhs, kind))
            self.assertEqual(env.codec.physical_index(can, team=1), actual)
            self.assertEqual(env.codec.physical_index(actual, team=1), can)
            mask = env.action_mask("B0")
            self.assertEqual(env.codec.canonical_mask(mask, team=1)[can], mask[actual])
        obs_blue = env.observation(team=1, active_robot="B0")
        obs_red = env.observation(team=0, active_robot="A0")
        self.assertEqual(len(obs_blue), env.encoder.obs_dim)
        # Exact symmetry of initial goal/source/toggle features is essential
        # for using a trained red policy as a frozen blue opponent.
        np.testing.assert_allclose(obs_red[75:], obs_blue[75:])

    def test_toggle_yellow_pin_ownership_swing(self):
        env = StrategyEnv(self.config, opponent="none", seed=1)
        goal = env.goals["G1"]
        self.assertEqual(goal.stack[0].kind, "yellow_pin")
        self.assertEqual(env.scores(), [0., 0.])
        env.toggles["Q0"] = 0
        self.assertEqual(env.scores()[0], 10.0)
        env.toggles["Q0"] = 1
        self.assertEqual(env.scores()[1], 10.0)
        self.assertEqual(env.scores()[0], 0.0)

    def test_variable_action_duration(self):
        env = StrategyEnv(self.config, seed=1)
        act = Action("SCORE", "G0", "alliance_pin")
        duration_1 = env.action_duration("A0", act)
        modified = load_config()
        modified["robot_speed_units_per_second"] /= 2
        env_slow = StrategyEnv(modified, seed=1)
        self.assertGreater(env_slow.action_duration("A0", act), duration_1)

    def test_full_episode_each_baseline(self):
        for style in ("random", "greedy", "coordination"):
            env = StrategyEnv(self.config, opponent="greedy", seed=20)
            obs, _ = env.reset(seed=21)
            finished = False
            iterations = 0
            while not finished:
                mask = env.action_mask()
                choice = choose_action(env, env.active_robot, style, env.rng)
                self.assertTrue(mask[choice])
                obs, rew, finished, _, info = env.step(choice)
                self.assertTrue(np.isfinite(rew))
                iterations += 1
                self.assertLess(iterations, 1000, "environment did not terminate")
            self.assertEqual(info["game_time"], 120)
            self.assertEqual(obs.shape[0], env.encoder.obs_dim)
            self.assertIsNone(env.active_robot)

    def test_awp_not_added_to_match_score(self):
        env = StrategyEnv(self.config, seed=2)
        env.awp_met[0] = True
        s1 = env.scores()
        env.awp_met[0] = False
        self.assertEqual(env.scores(), s1)

    def test_source_reservation_and_collect_completes(self):
        env = StrategyEnv(self.config, opponent="none", seed=4)
        # A0 holds its preload, but may pick up a Cup simultaneously.
        action = Action("COLLECT", "S0", "cup")
        self.assertTrue(env.action_mask("A0")[env.codec.encode(action)])
        initial_stock = env.sources["S0"].available["cup"]
        env.step(env.codec.encode(action))
        self.assertEqual(env.sources["S0"].available["cup"], initial_stock - 1)
        env.step(env.codec.encode(Action("WAIT")))
        for _ in range(8):
            if env.robots["A0"].cup:
                break
            env.step(env.codec.encode(Action("WAIT")))
        self.assertTrue(env.robots["A0"].cup)
        self.assertTrue(env.robots["A0"].pin is not None)

    def test_zero_time_two_robot_dispatch(self):
        env = StrategyEnv(self.config, opponent="none", seed=2)
        env.step(env.codec.encode(Action("WAIT")))
        self.assertEqual(env.time, 0.0)
        self.assertEqual(env.active_robot, "A1")
        env.step(env.codec.encode(Action("WAIT")))
        self.assertGreater(env.time, 0.0)


if __name__ == "__main__":
    unittest.main()
