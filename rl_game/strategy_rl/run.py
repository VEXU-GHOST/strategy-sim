"""Windows/VSCode-friendly CLI: run `python run.py --help`."""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from vex_rl.config import load_config
from vex_rl.contract import check_contract
from vex_rl.evaluate import (evaluate_baselines, evaluate_policy, plot_history,
                             print_summary)
from vex_rl.model import load_checkpoint
from vex_rl.policies import choose_action
from vex_rl.simulator import StrategyEnv
from vex_rl.trainer import PPOParams, run_curriculum, train_policy

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="VEX U Override / centralized Masked PPO strategy RL")
    parser.add_argument("--config", default=str(ROOT / "config.json"), help="editable config JSON")
    parser.add_argument("--device", default="cpu", help="torch device, default cpu")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke", help="validate environment + 64-step PPO training")
    sub.add_parser("test", help="run all unit and integration tests")
    demo = sub.add_parser("demo", help="print one match with a heuristic policy")
    demo.add_argument("--policy", choices=["random", "greedy", "coordination"], default="coordination")
    demo.add_argument("--opponent", choices=["none", "greedy", "random", "coordination", "mixed"], default="greedy")
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--limit", type=int, default=30, help="max trace lines to print")
    base = sub.add_parser("baselines", help="compare Random, Greedy, Coordination")
    base.add_argument("--opponent", choices=["none", "greedy", "random", "coordination", "mixed"], default="greedy")
    base.add_argument("--episodes", type=int, default=30)
    base.add_argument("--seed", type=int, default=1000)
    base.add_argument("--out-dir", default="outputs/baselines")
    train = sub.add_parser("train", help="train Masked PPO against one opponent type")
    train.add_argument("--opponent", choices=["none", "greedy", "random", "coordination", "mixed", "selfplay"], default="greedy")
    train.add_argument("--steps", type=int, default=10000)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--save", default="outputs/ppo_custom.pt")
    train.add_argument("--load", default=None, help="initial actor-critic checkpoint")
    train.add_argument("--opponent-checkpoint", default=None, help="frozen opponent checkpoint for selfplay")
    train.add_argument("--bc-episodes", type=int, default=0, help="initial coordination teacher demonstrations")
    curriculum = sub.add_parser("curriculum", help="none -> greedy -> mixed -> snapshot selfplay")
    curriculum.add_argument("--steps-per-stage", type=int, default=20000)
    curriculum.add_argument("--eval-episodes", type=int, default=20)
    curriculum.add_argument("--bc-episodes", type=int, default=12)
    curriculum.add_argument("--seed", type=int, default=42)
    curriculum.add_argument("--out-dir", default="outputs/curriculum")
    evaluate = sub.add_parser("eval", help="evaluate PPO or heuristics with optional CSV")
    evaluate.add_argument("--policy", choices=["random", "greedy", "coordination", "ppo"], default="ppo")
    evaluate.add_argument("--model", help="PPO checkpoint")
    evaluate.add_argument("--opponent", choices=["none", "random", "greedy", "coordination", "mixed", "selfplay"], default="greedy")
    evaluate.add_argument("--opponent-checkpoint", default=None)
    evaluate.add_argument("--episodes", type=int, default=50)
    evaluate.add_argument("--seed", type=int, default=12345)
    evaluate.add_argument("--stochastic", action="store_true", help="sample PPO policy instead of argmax")
    evaluate.add_argument("--csv", default="outputs/evaluation.csv")
    plot = sub.add_parser("plot", help="save training curves (PNG)")
    plot.add_argument("--histories", nargs="+", required=True, help="history_*.csv paths")
    plot.add_argument("--out", default="outputs/training_curve.png")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "test":
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
        outcome = unittest.TextTestRunner(verbosity=2).run(suite)
        if not outcome.wasSuccessful():
            raise SystemExit(1)
    elif args.command == "smoke":
        env = StrategyEnv(cfg, opponent="greedy", seed=44)
        check_contract(env)
        print(f"Environment contract OK: obs={env.encoder.obs_dim}, actions={len(env.codec)}, "
              f"match={env.total_time:.0f}s")
        with tempfile.TemporaryDirectory() as temporary:
            short = replace(PPOParams.from_config(cfg), rollout=32, epochs=1, minibatch=16)
            model = train_policy(cfg, opponent="greedy", steps=64, seed=17,
                                 save=str(Path(temporary) / "smoke.pt"), params=short, device=args.device)
            result = evaluate_policy(cfg, "ppo", "greedy", episodes=2, model=model, seed=81, device=args.device)
            print_summary(result)
        print("SMOKE TEST PASSED")
    elif args.command == "demo":
        env = StrategyEnv(cfg, opponent=args.opponent, seed=args.seed)
        obs, _ = env.reset(seed=args.seed)
        finished = False
        while not finished:
            a = choose_action(env, env.active_robot, args.policy, env.rng)
            obs, reward, finished, _, info = env.step(a)
        for t, robot, action, duration in env.stats["trace"][:args.limit]:
            print(f"{t:6.2f}s {robot}: {action:30s} duration={duration:5.2f}s")
        print("Final match:", {key: info[key] for key in ("our_score", "opponent_score", "win", "awp_ours", "scored_pins", "duplicate_targets")})
    elif args.command == "baselines":
        evaluate_baselines(cfg, args.opponent, args.episodes, args.seed, args.out_dir)
    elif args.command == "train":
        target = Path(args.save)
        train_policy(cfg, opponent=args.opponent, steps=args.steps, seed=args.seed,
                     save=str(target), load=args.load, opponent_checkpoint=args.opponent_checkpoint,
                     bc_episodes=args.bc_episodes, log_csv=str(target.with_suffix(".csv")), device=args.device)
        print(f"Saved model: {target}")
    elif args.command == "curriculum":
        run_curriculum(cfg, steps_per_stage=args.steps_per_stage, seed=args.seed,
                       out_dir=args.out_dir, eval_episodes=args.eval_episodes,
                       bc_episodes=args.bc_episodes, device=args.device)
    elif args.command == "eval":
        probe = StrategyEnv(cfg)
        model = (load_checkpoint(args.model, probe.encoder.obs_dim, len(probe.codec), cfg, args.device)
                 if args.model else None)
        adversary = (load_checkpoint(args.opponent_checkpoint, probe.encoder.obs_dim, len(probe.codec), cfg, args.device)
                     if args.opponent_checkpoint else None)
        s = evaluate_policy(cfg, args.policy, args.opponent, args.episodes, args.seed,
                            model=model, opponent_model=adversary,
                            csv_path=args.csv, deterministic=not args.stochastic,
                            device=args.device)
        print_summary(s)
        print(f"Saved evaluation to {args.csv}")
    elif args.command == "plot":
        print(f"Saved plot to {plot_history(args.histories, args.out)}")


if __name__ == "__main__":
    main()
