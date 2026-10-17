"""Numerical and model checks for age-specific, smoothed allocation."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pytest

from mixres.models import AgeMixAgg, AgeMixAggDataLoader, compute_allocation_probs
from mixres.models._allocation import validate_allocation_inputs

from ._helpers import build_model, hill_effect
from .test_allocation_modes import TinyPrior

MODE = "conditional_intensity"
COMP_MAP = np.array([[1, 0], [1, 1], [0, 1]])
COUNTS = np.array([[9.0, 1.0], [1.0, 9.0], [0.0, 0.0]])
M = np.array([[8.0, 2.0, 1.0], [1.0, 2.0, 8.0], [2.0, 2.0, 2.0]])


def smoothed(counts, alpha=1.0):
    pooled = counts.sum(axis=0)
    return counts + alpha * pooled / pooled.sum()


def conditional(counts=COUNTS, m=M, alpha=1.0):
    validate_allocation_inputs(COMP_MAP, MODE, counts)
    return np.asarray(
        compute_allocation_probs(COMP_MAP, MODE, smoothed(counts, alpha), np.log(m))
    )


def model_inputs():
    return dict(
        y=COUNTS,
        N_c=np.ones(3),
        P=np.ones(3),
        comp_map_cnt=COMP_MAP,
        priors={"rate": TinyPrior()},
        allocation=MODE,
    )


def _trace(model, parameters, observations):
    fn = numpyro.handlers.seed(
        numpyro.handlers.substitute(model.model, data=parameters), 1
    )
    return numpyro.handlers.trace(fn).get_trace(y=observations)


def test_hand_calculation_and_expected_counts():
    q = conditional()
    np.testing.assert_allclose(q[:, 1, 0], [19 / 29, 10 / 29, 0.5], rtol=1e-6)
    np.testing.assert_array_equal(q[..., COMP_MAP == 0], 0)
    np.testing.assert_allclose(q.sum(axis=-1), 1, atol=1e-6)
    np.testing.assert_array_equal(q[:, 0, 0], 1)
    np.testing.assert_array_equal(q[:, 2, 1], 1)
    # Expected counts m[a, d] = sum_b M[a, b] Q[a, b, d] conserve each row total.
    m = np.einsum("ab,abd->ad", M, q)
    expected = [[8 + 38 / 29, 1 + 20 / 29], [1 + 20 / 29, 8 + 38 / 29], [3, 3]]
    np.testing.assert_allclose(m, expected, rtol=1e-6)
    np.testing.assert_allclose(m.sum(axis=1), M.sum(axis=1), rtol=1e-6)


def test_model_mean_uses_respondent_specific_weights():
    model = build_model(**model_inputs(), likelihood="poisson")
    f = jnp.log(jnp.array([[8.0, 2.0, 1.0], [2.0, 2.0, 8.0], [1.0, 8.0, 2.0]]))
    trace = _trace(model, {"baseline": 0.0, "rate/f": f}, model.y)
    gamma = np.exp(np.asarray(trace["log_rate"]["value"]))
    q = np.asarray(trace["alloc_probs"]["value"])
    assert q.shape == (3, 3, 2)
    # With unit population and exposure, mu[a, d] = sum_b gamma[a, b] Q[a, b, d].
    np.testing.assert_allclose(
        model.data.to_grid(trace["mu"]["value"]),
        np.einsum("ab,abd->ad", gamma, q),
        rtol=1e-6,
    )


def test_model_smooths_counts_once_at_construction():
    model = build_model(**model_inputs())
    np.testing.assert_allclose(model.allocation.age_group_counts, smoothed(COUNTS, 1.0))


@pytest.mark.parametrize(
    "counts",
    [
        np.ones((2, 2)),
        np.full((3, 2), -1.0),
        np.full((3, 2), np.nan),
        np.zeros((3, 2)),
    ],
)
def test_invalid_counts(counts):
    with pytest.raises(ValueError, match="counts"):
        conditional(counts=counts)


def test_globally_zero_labels_and_unsupported_ambiguity():
    q = conditional(counts=np.array([[1.0, 0.0], [0.0, 0.0], [2.0, 0.0]]))
    np.testing.assert_array_equal(q[:, 1, 1], 0)
    # The uniquely compatible label still wins even when globally unobserved.
    np.testing.assert_array_equal(q[:, 2, 1], 1)
    with pytest.raises(ValueError, match="ambiguous"):
        validate_allocation_inputs([[1, 1, 0], [0, 0, 1]], MODE, [[0, 0, 1], [0, 0, 1]])


def test_conditional_input_validation():
    with pytest.raises(ValueError, match="support"):
        validate_allocation_inputs([[0, 0], [1, 1]], MODE, np.ones((2, 2)))
    with pytest.raises(ValueError, match="age_group_counts"):
        validate_allocation_inputs(COMP_MAP, MODE)
    with pytest.raises(ValueError, match="age_group_counts"):
        validate_allocation_inputs(COMP_MAP, MODE, COUNTS.sum(axis=0))


def test_extreme_intensities_and_gradients():
    def probability(log_joint):
        return compute_allocation_probs(COMP_MAP, MODE, smoothed(COUNTS), log_joint)[
            0, 1, 0
        ]

    gradient = jax.jit(jax.grad(probability))(jnp.log(M))
    assert np.isfinite(gradient).all() and np.any(np.abs(gradient) > 1e-5)
    extreme = jnp.array([[-1000.0, 0.0, 1000.0]] * 3)
    assert np.isfinite(jax.grad(probability)(extreme)).all()
    q = compute_allocation_probs(COMP_MAP, MODE, smoothed(COUNTS), extreme)
    np.testing.assert_allclose(q.sum(axis=-1), 1, atol=1e-6)


def test_agrees_with_marginal_for_proportional_profiles_and_counts():
    base = np.array([2.0, 3.0, 4.0])
    m = np.array([1.0, 3.0, 7.0])[:, None] * base
    counts = np.array([2.0, 5.0, 9.0])[:, None] * np.array([2.0, 3.0])
    pooled_q = np.asarray(
        compute_allocation_probs(
            COMP_MAP, "marginal_intensity", counts.sum(axis=0), np.log(m)
        )
    )
    np.testing.assert_allclose(
        conditional(counts, m), np.broadcast_to(pooled_q, (3, 3, 2)), rtol=1e-6
    )


def test_smoothing_strength_changes_only_ambiguous_allocations():
    values = [conditional(alpha=alpha) for alpha in (0.1, 1.0, 10.0)]
    assert values[0][0, 1, 0] > values[1][0, 1, 0] > values[2][0, 1, 0]
    for q in values:
        np.testing.assert_array_equal(q[:, 0, 0], 1)
        np.testing.assert_array_equal(q[:, 2, 1], 1)


def test_rejects_grouped_or_non_single_year_respondents():
    args = model_inputs()
    args.update(y=np.ones((1, 2)), N_c=np.ones(1), comp_map_resp=np.ones((3, 1)))
    with pytest.raises(ValueError, match="single-year"):
        build_model(**args)
    args = model_inputs()
    args["age_values"] = [0, 5, 10]
    with pytest.raises(ValueError, match="single-year"):
        build_model(**args)


def test_repeat_means_conservation_frozen_explicit_counts_and_hill_effect_cancellation():
    args = model_inputs()
    y = np.stack([COUNTS, COUNTS], axis=-1)
    n = np.array([[2.0, 1.0], [3.0, 2.0], [0.0, 0.0]])
    args.update(y=y, N_c=None, N_ar=n, rid=[0, 1])
    training_counts = np.array([[2.0, 5.0], [4.0, 1.0], [0.0, 0.0]])
    model = AgeMixAgg(
        AgeMixAggDataLoader.from_prepared_inputs(args),
        priors=args["priors"],
        allocation=args["allocation"],
        age_group_counts=training_counts,
    )
    params = {
        "baseline": 0.0,
        "rate/f": jnp.zeros((3, 3)),
        "inv_disp": 1.0,
        "hill_scale": 1.0,
        "hill_shape1": 0.0,
        "hill_shape2": 1.0,
    }
    train = _trace(model, params, y)
    for observed in (None, model.y * 2):
        np.testing.assert_array_equal(
            _trace(model, params, observed)["alloc_probs"]["value"], train["alloc_probs"]["value"]
        )
    other_fatigue = _trace(model, {**params, "hill_scale": 3.0}, y)
    np.testing.assert_array_equal(other_fatigue["alloc_probs"]["value"], train["alloc_probs"]["value"])
    for t in (train, other_fatigue):
        mu = model.data.to_grid(t["mu"]["value"])
        m = np.exp(np.asarray(t["log_rate"]["value"])) * np.asarray(args["P"])[None, :]
        q = np.asarray(t["alloc_probs"]["value"])
        omega = np.exp(np.asarray(hill_effect(t, [0, 1])))
        expected = np.einsum("ab,abd->ad", m, q)[:, :, None] * n[:, None, :] * omega
        np.testing.assert_allclose(mu, expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(
            np.asarray(mu).sum(axis=1),
            n * omega * m.sum(axis=1)[:, None],
            rtol=1e-6,
        )
