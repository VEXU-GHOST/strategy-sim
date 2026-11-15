"""Pick the environment backend named by ``config["backend"]``."""
from __future__ import annotations


def make_env(config: dict, **kwargs):
    backend = config.get("backend", "surrogate")
    if backend == "field":
        from .field_backend import FieldStrategyEnv
        return FieldStrategyEnv(config, **kwargs)
    if backend == "surrogate":
        from .simulator import StrategyEnv
        return StrategyEnv(config, **kwargs)
    raise ValueError(f"unknown backend: {backend}")
