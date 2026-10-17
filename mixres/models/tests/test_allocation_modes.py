"""Formula, exposure, differentiation and predictive checks for the allocation modes."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.util import log_density

from mixres.models import AgeMixAgg, AgeMixAggDataLoader, compute_allocation_probs
from mixres.models._allocation import WEIGHT_MODES, validate_allocation_inputs

from ._helpers import build_model


class TinyPrior:
    prior_type = "global"

    def set_age_bounds(self, lo, hi):
        self.size = hi - lo + 1

    def sample(self):
        return numpyro.sample(
            "f", dist.Normal(0, 0.1).expand((self.size, self.size)).to_event(2)
        )


def inputs():
    return dict(
        y=jnp.array([[25.0, 75.0], [25.0, 75.0]]),
        N_c=jnp.ones(2),
        P=jnp.ones(2),
        comp_map_cnt=jnp.array([[1, 1], [0, 1]]),
        priors={"rate": TinyPrior()},
    )


@pytest.mark.parametrize(
    "mode, expected",
    [("uniform", 0.5), ("empirical", 0.25), ("marginal_intensity", 0.4)],
)
def test_hand_calculated_example(mode, expected):
    # Two respondent ages with 50 contacts each to every contact age, so the
    # contact-age marginal is 100 per age.
    q = compute_allocation_probs(
        inputs()["comp_map_cnt"],
        mode,
        jnp.array([50.0, 150.0]),
        jnp.log(jnp.full((2, 2), 50.0)),
    )
    np.testing.assert_allclose(q[0, 0], expected, rtol=1e-6)
    np.testing.assert_allclose(q.sum(axis=1), 1)
    assert q[1, 0] == 0 and q[1, 1] == 1


@pytest.mark.parametrize("mode", WEIGHT_MODES)
def test_disjoint_allocation_including_zero_count_label(mode):
    counts = [[0, 1, 0]] * 3 if mode == "conditional_intensity" else [0, 3, 0]
    validate_allocation_inputs(jnp.eye(3), mode, counts)
    q = compute_allocation_probs(
        jnp.eye(3),
        mode,
        jnp.asarray(counts, dtype=float),
        jnp.array([[-100.0, 0.0, 100.0]] * 3),
    )
    expected = (
        np.broadcast_to(np.eye(3), (3, 3, 3))
        if mode == "conditional_intensity"
        else np.eye(3)
    )
    np.testing.assert_array_equal(q, expected)


@pytest.mark.parametrize(
    "comp_map, counts, message",
    [
        ([[1, 0], [0, 0]], [1, 2], "support"),
        ([[1, 1]], [0, 0], "ambiguous"),
        ([[1, 1]], [1, -1], "non-negative"),
        ([[1, 1]], [1, float("nan")], "finite"),
    ],
)
def test_invalid_allocation_data(comp_map, counts, message):
    with pytest.raises(ValueError, match=message):
        validate_allocation_inputs(comp_map, "empirical", counts)


def test_model_modes_and_custom_weights_conflict():
    default = build_model(**inputs())
    assert default.allocation.mode == "uniform"
    custom = build_model(**inputs(), allocation_weights=default.allocation.weights)
    assert custom.allocation.mode == "custom"
    with pytest.raises(ValueError, match="either"):
        build_model(
            **inputs(),
            allocation_weights=default.allocation.weights,
            allocation="uniform",
        )
    with pytest.raises(ValueError, match="allocation"):
        build_model(**inputs(), allocation="custom")
    binary = build_model(**inputs(), allocation="binary")
    np.testing.assert_array_equal(binary.allocation.weights, inputs()["comp_map_cnt"])
    dynamic = AgeMixAgg(
        AgeMixAggDataLoader.from_prepared_inputs(inputs()),
        allocation="marginal_intensity",
        priors={"rate": TinyPrior()},
    )
    assert dynamic.allocation.weights is None
    np.testing.assert_array_equal(dynamic.allocation.age_group_counts, [50, 150])


@pytest.mark.parametrize(
    "repeats, exact", [(False, False), (True, False), (True, True)]
)
def test_marginal_matches_direct_exposure_weighted_sum(repeats, exact):
    args = inputs()
    args.update(P=jnp.array([1.0, 3.0]), comp_map_resp=jnp.ones((2, 1)))
    gamma = jnp.array([[2.0, 4.0], [4.0, 3.0]])
    if repeats:
        args.update(
            y=jnp.ones((1, 2, 2)),
            N_c=jnp.array([[8.0, 4.0]]),
            rid=jnp.array([0, 1]),
        )
        exposure = np.array([[2.0, 1.0], [6.0, 3.0]])
        if exact:
            exposure = np.array([[5.0, 0.0], [3.0, 4.0]])
            args.update(N_c=None, N_ar=jnp.array(exposure))
        omega = np.array([1.0, 0.4])
    else:
        args.update(y=jnp.ones((1, 2)), N_c=jnp.array([8.0]))
        exposure = np.array([[2.0], [6.0]])
        omega = np.ones(1)
    model = build_model(**args, allocation="marginal_intensity")
    # Repeat levels 0, 1 coincide with the R-axis positions of age_exposure.
    age_exposure = model.data.age_exposure
    E = age_exposure @ omega if repeats else age_exposure
    np.testing.assert_allclose(E, exposure @ omega, rtol=1e-6)
    log_exposure = jnp.log(E)
    # The marginal the allocation uses: sum_a E[a] gamma[a, b] P[b].
    marginal = jnp.exp(
        jax.scipy.special.logsumexp(
            log_exposure[:, None] + jnp.log(gamma) + model.data.log_P[None, :], axis=0
        )
    )
    expected = ((exposure @ omega)[:, None] * gamma * args["P"][None, :]).sum(axis=0)
    np.testing.assert_allclose(marginal, expected, rtol=1e-6)


def test_dynamic_weights_change_with_surface_and_have_finite_nonzero_gradient():
    comp_map = inputs()["comp_map_cnt"]

    def probability(log_m):
        # Each respondent age contributes log_m, so the marginal is log_m + log 2.
        log_intensity = jnp.broadcast_to(log_m, (2, 2))
        return compute_allocation_probs(
            comp_map, "marginal_intensity", jnp.array([50.0, 150.0]), log_intensity
        )[0, 0]

    a, b = jnp.array([0.0, 0.0]), jnp.array([0.0, 3.0])
    assert not np.isclose(probability(a), probability(b))
    gradient = jax.jit(jax.grad(probability))(a)
    assert np.isfinite(gradient).all() and np.any(np.abs(gradient) > 1e-5)
    assert np.isfinite(jax.grad(probability)(jnp.array([-1000.0, 1000.0]))).all()


@pytest.mark.parametrize("mode", WEIGHT_MODES)
def test_allocation_conserves_total_and_training_counts_in_prediction(mode):
    model = build_model(**inputs(), allocation=mode)
    parameters = {"baseline": 0.0, "rate/f": jnp.zeros((2, 2)), "inv_disp": 1.0}
    fn = numpyro.handlers.seed(
        numpyro.handlers.substitute(model.model, data=parameters), 1
    )
    train = numpyro.handlers.trace(fn).get_trace(y=model.y)
    predicted = numpyro.handlers.trace(fn).get_trace(y=None)
    held_out = numpyro.handlers.trace(fn).get_trace(y=model.y * 2)
    for trace in (predicted, held_out):
        np.testing.assert_allclose(trace["alloc_probs"]["value"], train["alloc_probs"]["value"])
    mu = model.data.to_grid(train["mu"]["value"])
    np.testing.assert_allclose(mu.sum(axis=1), [2.0, 2.0], rtol=1e-6)


@pytest.mark.parametrize("mode", WEIGHT_MODES)
def test_tiny_fatigue_svi_and_predictive_structural_zeros(mode):
    args = inputs()
    args.update(
        y=jnp.array(
            [[[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [2.0, 1.0], [0.0, 0.0]]]
        ),
        N_c=None,
        N_ar=jnp.array([[5.0, 0.0], [3.0, 2.0]]),
        rid=jnp.array([0, 1]),
        comp_map_cnt=jnp.array([[1, 1, 1], [0, 1, 1]]),
    )
    model = build_model(**args, allocation=mode)
    guide = AutoNormal(model.model)
    model.run_inference_svi(jax.random.PRNGKey(1), guide, num_steps=3, peak_lr=0.001)
    assert np.isfinite(model._svi_result.losses).all()
    posterior = model.posterior_predictive_svi(
        jax.random.PRNGKey(2), guide, num_samples=3
    )
    q = np.asarray(posterior["alloc_probs"])
    np.testing.assert_allclose(q.sum(axis=-1), 1, atol=1e-6)
    obs = np.asarray(model.data.to_grid(posterior["obs"]))
    assert obs.shape == (3, *args["y"].shape)
    assert np.all(obs[:, 0, :, 1] == 0)
    if mode != "uniform":
        assert np.all(obs[:, :, 2, :] == 0)
        assert np.all(np.asarray(model.data.to_grid(posterior["mu"]))[:, :, 2, :] == 0)
    # Exercise a differentiated joint density, including zero-weight labels.
    parameters = guide.sample_posterior(jax.random.PRNGKey(3), model._svi_result.params)
    gradients = jax.grad(lambda p: log_density(model.model, (), {"y": model.y}, p)[0])(
        parameters
    )
    assert all(np.isfinite(g).all() for g in jax.tree_util.tree_leaves(gradients))


@pytest.mark.parametrize("mode, n_rows", [("uniform", 4), ("empirical", 2)])
def test_rows_drop_unexposed_strata_and_zero_mass_groups(mode, n_rows):
    args = inputs()
    y = jnp.array([[1.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    args.update(
        comp_map_cnt=jnp.ones((3, 2)),
        P=jnp.ones(3),
        N_c=jnp.array([1.0, 0.0, 2.0]),
        y=y,
    )
    model = build_model(**args, allocation=mode)
    # Respondent group 1 has no respondents; under "empirical" contact group 1
    # has no observed contacts and hence no allocation mass.
    assert model.y.shape == (n_rows,)
    assert not np.any(np.asarray(model.data.cid) == 1)
    if mode == "empirical":
        assert not np.any(np.asarray(model.data.did) == 1)
    np.testing.assert_array_equal(model.data.to_grid(model.y), y)
    parameters = {"baseline": 0.0, "rate/f": jnp.zeros((3, 3)), "inv_disp": 1.0}
    value, _ = log_density(model.model, (), {"y": model.y}, parameters)
    assert np.isfinite(value)
