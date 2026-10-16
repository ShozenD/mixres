import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest

from mixres.models import AgeMixAggReported, make_composition_mapping


class FixedPrior:
    """Small deterministic prior used to make model means testable."""

    def __init__(self, values):
        self.values = jnp.asarray(values, dtype=jnp.float32)

    def set_age_bounds(self, min_age, max_age):
        self.A = max_age - min_age + 1

    def sample(self):
        return self.values[: self.A, : self.A]


def _trace(model, seed=0):
    seeded = numpyro.handlers.seed(model.model, jax.random.PRNGKey(seed))
    return numpyro.handlers.trace(seeded).get_trace(y=model.y)


def _base_inputs():
    return {
        "y": jnp.array([[1, 2], [3, 4]], dtype=jnp.float32),
        "N_c": jnp.array([5, 7], dtype=jnp.float32),
        "P": jnp.array([2, 3], dtype=jnp.float32),
        "comp_map_cnt": jnp.eye(2),
        "comp_map_resp": jnp.eye(2),
        "age_values": jnp.arange(2),
        "priors": {"rate": FixedPrior([[0.0, 1.0], [2.0, 3.0]])},
    }


def test_default_negbin_and_optional_poisson_share_the_mean():
    negbin = AgeMixAggReported(**_base_inputs())
    poisson = AgeMixAggReported(**_base_inputs(), likelihood="poisson")

    negbin_trace = _trace(negbin)
    poisson_trace = _trace(poisson)

    assert isinstance(negbin_trace["obs"]["fn"].base_dist, dist.NegativeBinomial2)
    assert "inv_disp" in negbin_trace
    assert isinstance(poisson_trace["obs"]["fn"].base_dist, dist.Poisson)
    assert "inv_disp" not in poisson_trace
    assert jnp.allclose(negbin_trace["mu"]["value"], poisson_trace["mu"]["value"])


def test_mean_matches_direct_population_volume_calculation():
    model = AgeMixAggReported(**_base_inputs(), likelihood="poisson")
    trace = _trace(model)

    gamma = jnp.exp(trace["log_gamma"]["value"])
    population_volume = model.P[:, None] * model.P[None, :] * gamma
    expected = model.N_c[:, None] / model.P[:, None] * population_volume

    assert jnp.allclose(trace["mu"]["value"], expected)


def test_default_allocation_weights_do_not_duplicate_overlap():
    ages = jnp.arange(3)
    comp_map_cnt = make_composition_mapping(ages, jnp.array([0, 1]), jnp.array([2, 3]))
    model = AgeMixAggReported(
        y=jnp.ones((3, 2)),
        N_c=jnp.ones(3),
        P=jnp.array([1.0, 2.0, 3.0]),
        comp_map_cnt=comp_map_cnt,
        age_values=ages,
        priors={"rate": FixedPrior(jnp.zeros((3, 3)))},
        likelihood="poisson",
    )
    trace = _trace(model)

    assert jnp.allclose(
        model.allocation_weights,
        jnp.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
    )
    gamma = jnp.exp(trace["log_gamma"]["value"])
    expected_total = model.P * (gamma * model.P[None, :]).sum(axis=1)
    reported_total = jnp.exp(trace["log_EZ"]["value"]).sum(axis=1)
    assert jnp.allclose(reported_total, expected_total)


def test_custom_allocation_weights_and_reciprocity():
    inputs = _base_inputs()
    inputs["comp_map_cnt"] = jnp.ones((2, 2))
    model = AgeMixAggReported(
        **inputs,
        allocation_weights=jnp.array([[0.75, 0.25], [0.1, 0.9]]),
        likelihood="poisson",
    )
    trace = _trace(model)

    assert jnp.allclose(model.allocation_weights.sum(axis=1), 1.0)
    assert jnp.array_equal(trace["log_gamma"]["value"], trace["log_gamma"]["value"].T)


def test_grouped_repeat_counts_apply_hill_fatigue():
    inputs = _base_inputs()
    inputs["y"] = jnp.ones((2, 2, 2))
    inputs["N_c"] = jnp.array([[5.0, 4.0], [3.0, 2.0]])
    model = AgeMixAggReported(
        **inputs,
        repeat_levels=jnp.array([0, 2]),
        likelihood="poisson",
    )
    trace = _trace(model)

    completeness = trace["completeness"]["value"]
    reported_volume = jnp.exp(trace["log_EZ"]["value"])
    expected = (
        reported_volume[:, :, None]
        * model.N_c[:, None, :]
        / model.P_respondent[:, None, None]
        * completeness[None, None, :]
    )

    assert completeness[0] == pytest.approx(1.0)
    assert 0 < completeness[1] < 1
    assert jnp.allclose(trace["mu"]["value"], expected)


def test_exact_age_repeat_counts_use_observed_age_composition():
    inputs = _base_inputs()
    inputs["y"] = jnp.ones((2, 2, 2))
    inputs["N_c"] = None
    exact_N = jnp.array([[5.0, 4.0], [3.0, 2.0]])
    model = AgeMixAggReported(
        **inputs,
        repeat_levels=jnp.array([0, 1]),
        N_age_repeat=exact_N,
        likelihood="poisson",
    )
    trace = _trace(model)

    gamma = jnp.exp(trace["log_gamma"]["value"])
    contact_rate = gamma * model.P[None, :]
    expected = (
        contact_rate[:, :, None]
        * exact_N[:, None, :]
        * trace["completeness"]["value"][None, None, :]
    )

    assert jnp.array_equal(model.N_c, exact_N)
    assert jnp.allclose(trace["mu"]["value"], expected)


def test_zero_exposure_repeat_cells_are_masked_with_zero_structural_mean():
    inputs = _base_inputs()
    inputs["y"] = jnp.array([[[1.0, 0.0], [2.0, 0.0]], [[3.0, 1.0], [4.0, 2.0]]])
    inputs["N_c"] = None
    model = AgeMixAggReported(
        **inputs,
        repeat_levels=jnp.array([0, 1]),
        N_age_repeat=jnp.array([[5.0, 0.0], [3.0, 2.0]]),
        likelihood="negbin",
    )
    trace = _trace(model)

    assert jnp.all(trace["mu"]["value"][0, :, 1] == 0)
    observation_mask = trace["obs"]["fn"]._mask
    assert jnp.all(observation_mask.reshape(model.y.shape)[0, :, 1] == 0)


def test_zero_exposure_fine_ages_do_not_leak_through_grouped_respondents():
    model = AgeMixAggReported(
        y=jnp.array([[[1.0, 0.0], [1.0, 0.0]]]),
        N_c=None,
        N_age_repeat=jnp.array([[5.0, 0.0], [0.0, 0.0]]),
        repeat_levels=jnp.array([0, 1]),
        P=jnp.array([1.0, 1.0]),
        comp_map_cnt=jnp.eye(2),
        comp_map_resp=jnp.ones((2, 1)),
        priors={"rate": FixedPrior(jnp.zeros((2, 2)))},
        likelihood="poisson",
    )
    trace = _trace(model)
    mu = trace["mu"]["value"]
    expected = 5.0 * jnp.exp(trace["log_gamma"]["value"])[0, :]
    assert jnp.allclose(mu[0, :, 0], expected)
    assert jnp.all(mu[:, :, 1] == 0)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"likelihood": "gamma"}, "likelihood"),
        ({"allocation_weights": jnp.array([[0.8, 0.0], [0.0, 0.2]])}, "sum to one"),
        (
            {
                "y": jnp.ones((2, 2, 2)),
                "N_c": jnp.ones((2, 2)),
                "repeat_levels": None,
            },
            "repeat_levels",
        ),
    ],
)
def test_invalid_reported_model_inputs(changes, message):
    inputs = _base_inputs()
    inputs.update(changes)
    with pytest.raises(ValueError, match=message):
        AgeMixAggReported(**inputs)


def test_exact_age_counts_reject_overlapping_respondent_groups():
    with pytest.raises(ValueError, match="partition"):
        AgeMixAggReported(
            y=jnp.ones((2, 2, 2)),
            N_c=None,
            P=jnp.ones(2),
            comp_map_cnt=jnp.ones((2, 2)),
            comp_map_resp=jnp.ones((2, 2)),
            priors={"rate": FixedPrior(jnp.zeros((2, 2)))},
            repeat_levels=jnp.array([0, 1]),
            N_age_repeat=jnp.ones((2, 2)),
        )
