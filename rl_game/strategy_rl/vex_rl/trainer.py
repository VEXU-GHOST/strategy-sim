"""Semi-Markov PPO: game-time gamma/lambda discounts for variable macro-action durations."""
from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn

from .evaluate import evaluate_policy, policy_predictor, print_summary
from .model import MaskedActorCritic, load_checkpoint, save_checkpoint
from .policies import choose_action
from .simulator import StrategyEnv


@dataclass
class PPOParams:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    clip: float = 0.2
    entropy_coef: float = 0.015
    value_coef: float = 0.5
    grad_clip: float = 0.5
    rollout: int = 512
    epochs: int = 4
    minibatch: int = 128
    hidden: int = 192

    @classmethod
    def from_config(cls, config):
        c = config["training"]
        return cls(c["gamma_per_second"], c["gae_lambda_per_second"], c["learning_rate"],
                   c["clip_epsilon"], c["entropy_coef"], c["value_coef"],
                   c["max_grad_norm"], c["rollout_steps"], c["update_epochs"],
                   c["minibatch_size"], c["hidden_size"])


def time_aware_gae(rewards: np.ndarray, values: np.ndarray,
                   dones: np.ndarray, elapsed_seconds: np.ndarray,
                   last_value: float, gamma_per_second: float, lambda_per_second: float):
    """SMDP advantage: g_t=gamma^dt, trace=(gamma*lambda)^dt.

    dt=0 when assigning the second robot's concurrent action: no artificial
    discount is applied for that zero-time scheduling event.
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for i in range(n - 1, -1, -1):
        cont = 0.0 if dones[i] else 1.0
        next_value = last_value if i == n - 1 else float(values[i + 1])
        discount = gamma_per_second ** float(elapsed_seconds[i])
        trace = lambda_per_second ** float(elapsed_seconds[i])
        td = float(rewards[i]) + cont * discount * next_value - float(values[i])
        gae = td + cont * discount * trace * gae
        advantages[i] = gae
    return advantages, advantages + values


def behavior_clone(model: MaskedActorCritic, config: dict, episodes: int, seed: int,
                   epochs: int = 5, device="cpu") -> int:
    """Optional short heuristic warm start. Subsequent updates optimize PPO reward."""
    if episodes <= 0:
        return 0
    env = StrategyEnv(config, opponent="greedy", seed=seed)
    observations, masks, actions = [], [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        finished = False
        while not finished:
            mask = env.action_mask()
            choice = choose_action(env, env.active_robot, "coordination", env.rng)
            observations.append(obs.copy())
            masks.append(mask.copy())
            actions.append(choice)
            obs, _, finished, _, _ = env.step(choice)
    obs_t = torch.tensor(np.stack(observations), dtype=torch.float32, device=device)
    mask_t = torch.tensor(np.stack(masks), dtype=torch.bool, device=device)
    act_t = torch.tensor(actions, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)
    model.train()
    for _ in range(epochs):
        indices = torch.randperm(len(act_t), device=device)
        for batch in indices.split(256):
            dist, _ = model.policy_value(obs_t[batch], mask_t[batch])
            loss = -dist.log_prob(act_t[batch]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    print(f"warm start: {len(actions)} expert transitions, {epochs} BC epochs", flush=True)
    return len(actions)


def train_policy(config: dict, *, opponent: str, steps: int, seed: int, save: str,
                 load: str | None = None, opponent_checkpoint: str | None = None,
                 bc_episodes: int = 0, log_csv: str | None = None,
                 device="cpu", params: PPOParams | None = None) -> MaskedActorCritic:
    if steps <= 0:
        raise ValueError("training steps must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)
    params = params or PPOParams.from_config(config)
    empty = StrategyEnv(config, seed=seed)
    obs_dim, n_actions = empty.encoder.obs_dim, len(empty.codec)
    opponent_model = (load_checkpoint(opponent_checkpoint, obs_dim, n_actions, config, device=device)
                      if opponent_checkpoint else None)
    if opponent == "selfplay" and opponent_model is None:
        raise ValueError("selfplay needs --opponent-checkpoint")
    opponent_predictor = (policy_predictor(opponent_model, deterministic=False, device=device)
                          if opponent_model else None)
    env = StrategyEnv(config, opponent=opponent, seed=seed, opponent_predictor=opponent_predictor)
    model = (load_checkpoint(load, obs_dim, n_actions, config, device=device) if load else
             MaskedActorCritic(obs_dim, n_actions, params.hidden).to(device))
    if bc_episodes and not load:
        behavior_clone(model, config, bc_episodes, seed + 5000, device=device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    obs, _ = env.reset(seed=seed)
    episodic_return, completed_returns, finished_episodes = 0.0, [], 0
    total_steps, updates = 0, 0
    history: list[dict] = []

    while total_steps < steps:
        n = min(params.rollout, steps - total_steps)
        states, masks, acts, old_logps, rewards, vals, dones, dts = [], [], [], [], [], [], [], []
        for _ in range(n):
            mask = env.action_mask()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
            model.eval()
            with torch.no_grad():
                distribution, value = model.policy_value(obs_t, mask_t)
                a = distribution.sample()
                log_prob = distribution.log_prob(a)
            action = int(a.item())
            states.append(obs.copy())
            masks.append(mask.copy())
            acts.append(action)
            old_logps.append(float(log_prob.item()))
            vals.append(float(value.item()))
            next_obs, reward, done, truncated, info = env.step(action)
            rewards.append(float(reward))
            dones.append(bool(done or truncated))
            dts.append(float(info["delta_time"]))
            episodic_return += reward
            if done or truncated:
                completed_returns.append(episodic_return)
                completed_returns = completed_returns[-50:]
                finished_episodes += 1
                episodic_return = 0.0
                next_obs, _ = env.reset()  # RNG continues instead of replaying identical games
            obs = next_obs
        model.eval()
        with torch.no_grad():
            if dones[-1]:
                last_value = 0.0
            else:
                _, v = model.policy_value(torch.as_tensor(obs, dtype=torch.float32, device=device),
                                          torch.as_tensor(env.action_mask(), dtype=torch.bool, device=device))
                last_value = float(v.item())
        rewards_np = np.asarray(rewards, dtype=np.float32)
        vals_np = np.asarray(vals, dtype=np.float32)
        advantage, targets = time_aware_gae(rewards_np, vals_np, np.asarray(dones),
                                            np.asarray(dts, dtype=np.float32), last_value,
                                            params.gamma, params.gae_lambda)
        observations = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
        action_masks = torch.as_tensor(np.stack(masks), dtype=torch.bool, device=device)
        actions = torch.as_tensor(acts, dtype=torch.long, device=device)
        old_logprobs = torch.as_tensor(old_logps, dtype=torch.float32, device=device)
        advantages = torch.as_tensor(advantage, dtype=torch.float32, device=device)
        returns = torch.as_tensor(targets, dtype=torch.float32, device=device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        model.train()
        last_loss = 0.0
        for _ in range(params.epochs):
            permutation = torch.randperm(n, device=device)
            for inds in permutation.split(params.minibatch):
                distribution, prediction = model.policy_value(observations[inds], action_masks[inds])
                logprobs = distribution.log_prob(actions[inds])
                ratio = torch.exp(logprobs - old_logprobs[inds])
                clipped = torch.clamp(ratio, 1.0 - params.clip, 1.0 + params.clip)
                policy_loss = -torch.min(ratio * advantages[inds], clipped * advantages[inds]).mean()
                value_loss = 0.5 * (prediction - returns[inds]).square().mean()
                entropy = distribution.entropy().mean()
                loss = policy_loss + params.value_coef * value_loss - params.entropy_coef * entropy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), params.grad_clip)
                optimizer.step()
                last_loss = float(loss.item())
        total_steps += n
        updates += 1
        history.append({
            "steps": total_steps,
            "episodes": finished_episodes,
            "recent_return": float(np.mean(completed_returns)) if completed_returns else 0.0,
            "loss": last_loss,
            "stage": opponent,
        })
        if updates == 1 or updates % 10 == 0 or total_steps == steps:
            print(f"[{opponent:>12s}] steps {total_steps:>6}/{steps:<6} "
                  f"episodes {finished_episodes:>4} "
                  f"mean_return {history[-1]['recent_return']:>7.2f} "
                  f"loss {last_loss:>7.3f}", flush=True)
    save_checkpoint(model, save, config, {"stage": opponent, "steps": steps, "seed": seed})
    if log_csv:
        path = Path(log_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    return model


def run_curriculum(config: dict, steps_per_stage: int, seed: int, out_dir: str,
                   eval_episodes: int = 10, bc_episodes: int = 12, device="cpu") -> list[dict]:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    previous = None
    records = []
    for stage_index, opponent in enumerate(("none", "greedy", "mixed", "selfplay")):
        label = ("no_opponent", "greedy", "mixed", "selfplay")[stage_index]
        print(f"\n===== Stage {stage_index + 1}/4: {label} =====", flush=True)
        target = output / f"ppo_{label}.pt"
        frozen = str(previous) if opponent == "selfplay" else None
        model = train_policy(config, opponent=opponent, steps=steps_per_stage,
                             seed=seed + stage_index, save=str(target), load=str(previous) if previous else None,
                             opponent_checkpoint=frozen,
                             bc_episodes=bc_episodes if previous is None else 0,
                             log_csv=str(output / f"history_{label}.csv"), device=device)
        previous = target
        for enemy in ("none", "greedy", "random", "coordination"):
            result = evaluate_policy(config, "ppo", enemy, eval_episodes,
                                     seed=seed + 10000, model=model, device=device)
            print_summary(result)
            records.append({"stage": label, **result})
    import json
    (output / "curriculum_summary.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    with (output / "curriculum_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nFINAL PPO: {previous}\nSUMMARY: {output / 'curriculum_summary.csv'}", flush=True)
    return records
