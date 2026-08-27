"""
main.py — CLI entry point for the RL game training framework.

Usage examples:
    python main.py train
    python main.py train --episodes 1000 --steps 200
    python main.py train --resume checkpoints/team_a_final.pth checkpoints/team_b_final.pth
    python main.py train --render
"""

from __future__ import annotations

import argparse
import sys
import os

# Ensure the project root is on the path regardless of where main.py is called
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402 — must come after sys.path update
from training.trainer import Trainer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="RL Game — 4-Agent Team Training Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train
  python main.py train --episodes 500 --steps 200
  python main.py train --resume checkpoints/team_a_final.pth checkpoints/team_b_final.pth
  python main.py train --render
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── train subcommand ───────────────────────────────────────────────
    train_parser = subparsers.add_parser("train", help="Start a new training run")

    train_parser.add_argument(
        "--episodes",
        type=int,
        default=config.EPISODES,
        help=f"Number of episodes to train (default: {config.EPISODES})",
    )
    train_parser.add_argument(
        "--steps",
        type=int,
        default=config.STEPS_PER_EPISODE,
        help=f"Max steps per episode (default: {config.STEPS_PER_EPISODE})",
    )
    train_parser.add_argument(
        "--resume",
        nargs=2,
        metavar=("TEAM_A_CKPT", "TEAM_B_CKPT"),
        default=None,
        help="Paths to Team-A and Team-B checkpoint files to resume from",
    )
    train_parser.add_argument(
        "--render",
        action="store_true",
        help="Render the board to the terminal each step (slow, for debugging)",
    )

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.command == "train":
        # Override config values from CLI flags
        config.EPISODES = args.episodes
        config.STEPS_PER_EPISODE = args.steps

        resume_a, resume_b = (args.resume if args.resume else (None, None))

        trainer = Trainer(
            episodes=args.episodes,
            render=args.render,
            resume_team_a=resume_a,
            resume_team_b=resume_b,
        )
        trainer.run()


if __name__ == "__main__":
    main()
