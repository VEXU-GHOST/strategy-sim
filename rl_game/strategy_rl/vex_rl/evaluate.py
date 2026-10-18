"""Paired-seed baseline comparisons, scores/win rates, and CSV reporting."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from .config import load_config
from .model import MaskedActorCritic, load_checkpoint
from .policies import choose_action
from .simulator import StrategyEnv


def policy_predictor(model: MaskedActorCritic, deterministic: bool = True, device="cpu"):
    def predict(obs: np.ndarray, mask: np.ndarray) -> int:
        return model.choose(obs, mask, deterministic=deterministic, device=device)
    return predict


def evaluate_policy(config: dict, policy: str, opponent: str = "greedy", episodes: int = 30,
                    seed: int = 1000, model: MaskedActorCritic | None = None,
                    opponent_model: MaskedActorCritic | None = None,
                    csv_path: str | None = None, deterministic: bool = True,
                    device="cpu") -> dict:
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    if policy == "ppo" and model is None:
        raise ValueError("PPO evaluation requires a trained checkpoint")
    if opponent == "selfplay" and opponent_model is None:
        raise ValueError("selfplay evaluation requires opponent checkpoint")
    predictor = policy_predictor(opponent_model, deterministic, device) if opponent_model is not None else None
    env = StrategyEnv(config, opponent=opponent, seed=seed, opponent_predictor=predictor)
    results = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        finished = False
        while not finished:
            if policy == "ppo":
                action = model.choose(obs, env.action_mask(), deterministic=deterministic, device=device)
            else:
                action = choose_action(env, env.active_robot, policy, env.rng)
            obs, _, finished, _, info = env.step(action)
        results.append({"episode": episode + 1, "seed": seed + episode,
                        "policy": policy, "opponent": opponent, **info})
    keys = ("our_score", "opponent_score", "score_difference", "win", "awp_ours",
            "duplicate_targets", "wait_seconds", "scored_pins", "scored_cups",
            "collected", "toggles_set", "failed_actions")
    summary = {key: float(np.mean([float(row[key]) for row in results])) for key in keys}
    summary.update({"policy": policy, "opponent": opponent, "episodes": episodes, "seed": seed})
    if csv_path:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        summary_path = path.with_name(path.stem + "_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def print_summary(s: dict):
    print(f"{s['policy']:>12s} vs {s['opponent']:<12s} "
          f"score {s['our_score']:6.1f}-{s['opponent_score']:<6.1f} "
          f"win={s['win']:.1%} awp={s['awp_ours']:.1%} "
          f"duplicates={s['duplicate_targets']:.2f} idle={s['wait_seconds']:.1f}s "
          f"pins={s['scored_pins']:.1f}")


def evaluate_baselines(config: dict, opponent: str, episodes: int, seed: int,
                       csv_directory: str | None = None):
    summaries = []
    for policy in ("random", "greedy", "coordination"):
        csv_path = str(Path(csv_directory) / f"{policy}_vs_{opponent}.csv") if csv_directory else None
        summary = evaluate_policy(config, policy, opponent, episodes, seed, csv_path=csv_path)
        print_summary(summary)
        summaries.append(summary)
    return summaries


def plot_history(csv_paths: list[str], output: str):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9, 5))
    for p in csv_paths:
        with Path(p).open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        plt.plot([int(r["steps"]) for r in rows],
                 [float(r["recent_return"]) for r in rows], label=Path(p).stem)
    plt.xlabel("PPO environment decision steps")
    plt.ylabel("Mean recent episodic return")
    plt.title("Masked PPO training")
    plt.legend()
    plt.tight_layout()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    return path
