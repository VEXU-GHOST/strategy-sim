"""Masked, centralized actor-critic network and safe checkpoint metadata."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class MaskedActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int = 192):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, n_actions)
        self.critic = nn.Linear(hidden_size, 1)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def policy_value(self, obs: torch.Tensor, mask: torch.Tensor):
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
            mask = mask.unsqueeze(0)
        if mask.shape != (obs.shape[0], self.n_actions):
            raise ValueError("mismatched action mask shape")
        if not mask.any(dim=-1).all():
            raise ValueError("every observation must permit at least WAIT")
        latent = self.trunk(obs)
        logits = self.actor(latent).masked_fill(~mask.bool(), -1e9)
        return Categorical(logits=logits), self.critic(latent).squeeze(-1)

    @torch.no_grad()
    def choose(self, obs: np.ndarray, mask: np.ndarray, deterministic: bool = False, device="cpu") -> int:
        self.eval()
        x = torch.as_tensor(obs, device=device, dtype=torch.float32)
        m = torch.as_tensor(mask, device=device, dtype=torch.bool)
        d, _ = self.policy_value(x, m)
        action = d.probs.argmax(-1) if deterministic else d.sample()
        return int(action.item())


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def save_checkpoint(model: MaskedActorCritic, filename: str | Path, config: dict, metadata: dict | None = None):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "obs_dim": model.obs_dim,
        "n_actions": model.n_actions,
        "hidden_size": model.hidden_size,
        "config_hash": config_hash(config),
        "metadata": metadata or {},
    }, path)


def load_checkpoint(filename: str | Path, obs_dim: int, n_actions: int,
                    config: dict, device="cpu") -> MaskedActorCritic:
    # Only load your own, trusted local checkpoints.
    payload = torch.load(filename, map_location=device, weights_only=True)
    if (int(payload["obs_dim"]), int(payload["n_actions"])) != (obs_dim, n_actions):
        raise ValueError("checkpoint observation/action dimensions do not match current config")
    if payload.get("config_hash") != config_hash(config):
        print("WARNING: checkpoint was created using different config settings; re-evaluate after changing parameters.")
    model = MaskedActorCritic(obs_dim, n_actions, int(payload["hidden_size"]))
    model.load_state_dict(payload["model_state"])
    return model.to(device)
