"""Shared helpers for building models in tests."""

from mixres.models import AgeMixAgg, AgeMixAggDataLoader


def build_model(**kwargs):
    """``AgeMixAgg`` from flat keyword arguments, split into loader and model ones."""
    data = {
        key: kwargs.pop(key)
        for key in AgeMixAggDataLoader.PREPARED_INPUT_KEYS
        if key in kwargs
    }
    return AgeMixAgg(AgeMixAggDataLoader(**data), **kwargs)


def hill_effect(trace, rid):
    """Log-scale Hill repeat effect implied by the traced ``hill_*`` parameters."""
    import jax.numpy as jnp

    levels = jnp.asarray(rid, dtype=jnp.float32)
    s = jnp.exp(trace["hill_shape1"]["value"]) * levels ** trace["hill_shape2"]["value"]
    return -trace["hill_scale"]["value"] * s / (1.0 + s)
