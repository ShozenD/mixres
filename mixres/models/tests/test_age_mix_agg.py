import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../cntmosaic"))

from cntmosaic.models.numpyro.priors import Independent2D

from mixres.models import (
    AgeMixAggDataLoader,
    make_composition_mapping,
    prepare_age_mix_agg_inputs,
    sum_EZ,
)

from ._helpers import build_model, hill_effect


def test_aggregation_helpers_on_ones():
    age_values = jnp.arange(80)
    left = jnp.arange(0, 80, 5)
    right = left + 5
    Z = jnp.ones((80, 80))

    comp_map_cnt = make_composition_mapping(age_values, left, right)

    assert comp_map_cnt.shape == (80, 16)
    assert sum_EZ(Z).shape == (80, 80)
    assert sum_EZ(Z, comp_map_cnt).shape == (80, 16)
    assert sum_EZ(Z, comp_map_cnt, comp_map_cnt).shape == (16, 16)
    assert jnp.all(sum_EZ(Z, comp_map_cnt) == 5.0)
    assert jnp.all(sum_EZ(Z, comp_map_cnt, comp_map_cnt) == 25.0)


def test_overlapping_membership_allows_multiple_groups():
    M = make_composition_mapping(jnp.arange(15), jnp.array([0, 5]), jnp.array([10, 15]))

    assert M.shape == (15, 2)
    assert jnp.all(M[:5].sum(axis=1) == 1)
    assert jnp.all(M[5:10].sum(axis=1) == 2)
    assert jnp.all(M[10:15].sum(axis=1) == 1)


def test_prepare_age_mix_agg_inputs_for_scenario_1_like_data():
    labels = ["0-4", "5-9"]
    participants = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "resp_group": pd.Categorical(
                ["0-4", "0-4", "5-9", "5-9"], categories=labels
            ),
        }
    )
    contacts = pd.DataFrame(
        {
            "id": [1, 1, 2, 3, 4],
            "cnt_group": pd.Categorical(
                ["0-4", "5-9", "0-4", "5-9", "0-4"], categories=labels
            ),
            "y": [1, 2, 3, 4, 5],
        }
    )
    scenario = {"participants": participants, "contacts": contacts}

    inputs = prepare_age_mix_agg_inputs(
        scenario,
        P=np.ones(10),
        age_values=np.arange(10),
        left_limits=np.array([0, 5]),
        right_limits=np.array([5, 10]),
    )

    assert inputs["age_agg_dims"] == "both"
    assert inputs["y"].shape == (2, 2)
    assert inputs["N_c"].shape == (2,)
    assert inputs["y"].sum() == contacts["y"].sum()
    assert np.allclose(inputs["N_c"], [2, 2])


def test_prepare_age_mix_agg_inputs_allows_fine_respondent_age():
    contact_labels = ["0-4", "5-9"]
    participants = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "resp_group": [0, 1, 5, 6],
        }
    )
    contacts = pd.DataFrame(
        {
            "id": [1, 1, 2, 3, 4],
            "cnt_group": pd.Categorical(
                ["0-4", "5-9", "0-4", "5-9", "0-4"],
                categories=contact_labels,
            ),
            "y": [1, 2, 3, 4, 5],
        }
    )
    scenario = {"participants": participants, "contacts": contacts}

    inputs = prepare_age_mix_agg_inputs(
        scenario,
        P=np.ones(10),
        age_values=np.arange(10),
        left_limits=np.array([0, 5]),
        right_limits=np.array([5, 10]),
        age_agg_dims="contact",
    )

    assert inputs["comp_map_cnt"].shape == (10, 2)
    assert inputs["age_agg_dims"] == "contact"
    assert inputs["y"].shape == (10, 2)
    assert inputs["N_c"].shape == (10,)
    assert inputs["y"].sum() == contacts["y"].sum()


def test_prepare_inputs_with_coarse_respondent_groups():
    participants = pd.DataFrame({"id": [1, 2, 3], "resp_group": ["0-1", "2-4", "2-4"]})
    contacts = pd.DataFrame(
        {"id": [1, 2, 3], "cnt_group": ["0-2", "0-2", "3-4"], "y": [1, 2, 3]}
    )
    inputs = prepare_age_mix_agg_inputs(
        {"participants": participants, "contacts": contacts},
        P=np.ones(5) / 5,
        age_values=np.arange(5),
        left_limits=np.array([0, 3]),
        right_limits=np.array([3, 5]),
        contact_labels=["0-2", "3-4"],
        respondent_left_limits=np.array([0, 2]),
        respondent_right_limits=np.array([2, 5]),
        respondent_labels=["0-1", "2-4"],
        age_agg_dims="contact",
    )
    assert inputs["comp_map_cnt"].shape == (5, 2)
    assert inputs["comp_map_resp"].shape == (5, 2)
    assert inputs["y"].shape == (2, 2)
    assert np.allclose(inputs["N_c"], [1, 2])


def test_prepare_age_mix_agg_inputs_allows_both_axes_fine():
    participants = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "resp_group": [0, 1, 5, 6],
        }
    )
    contacts = pd.DataFrame(
        {
            "id": [1, 1, 2, 3, 4],
            "cnt_group": [0, 6, 1, 7, 2],
            "y": [1, 2, 3, 4, 5],
        }
    )
    scenario = {"participants": participants, "contacts": contacts}

    inputs = prepare_age_mix_agg_inputs(
        scenario,
        P=np.ones(10),
        age_values=np.arange(10),
        age_agg_dims="none",
    )

    assert inputs["comp_map_cnt"] is None
    assert inputs["age_agg_dims"] == "none"
    assert inputs["y"].shape == (10, 10)
    assert inputs["N_c"].shape == (10,)
    assert inputs["y"].sum() == contacts["y"].sum()


def test_age_mix_agg_model_smoke_trace_shapes():
    age_values = jnp.arange(10)
    left = jnp.array([0, 5])
    right = jnp.array([5, 10])
    P = jnp.ones(10) / 10
    comp_map_cnt = make_composition_mapping(age_values, left, right)
    cases = [  # (comp_map_cnt, comp_map_resp, expected_shape, expected_P_respondent)
        (None, None, (10, 10), jnp.ones(10)),
        (comp_map_cnt, None, (10, 2), jnp.ones(10)),
        (comp_map_cnt, comp_map_cnt, (2, 2), jnp.array([5.0, 5.0])),
    ]

    for (
        case_comp_map_cnt,
        case_comp_map_resp,
        expected_shape,
        expected_P_respondent,
    ) in cases:
        y = jnp.ones(expected_shape)
        model = build_model(
            y=y,
            N_c=jnp.ones(expected_shape[0]),
            P=P,
            comp_map_cnt=case_comp_map_cnt,
            comp_map_resp=case_comp_map_resp,
            age_values=age_values,
            priors={"rate": Independent2D(prior_type="global")},
            # The configuration of the former Poisson-only AgeMixAgg.
            likelihood="poisson",
            allocation="binary",
        )
        seeded = numpyro.handlers.seed(model.model, jax.random.PRNGKey(0))
        trace = numpyro.handlers.trace(seeded).get_trace(y=model.y)

        assert trace["log_rate"]["value"].shape == (10, 10)
        assert trace["mu"]["value"].shape == (y.size,)
        assert model.data.to_grid(trace["mu"]["value"]).shape == expected_shape
        assert jnp.allclose(model.data.P_resp, expected_P_respondent / 10)


def test_age_mix_agg_model_only_conditions_on_explicit_y():
    age_values = jnp.arange(2)
    P = jnp.ones(2) / 2
    model = build_model(
        y=jnp.array([[1.0]]),
        N_c=jnp.array([10.0]),
        P=P,
        comp_map_cnt=jnp.ones((2, 1)),
        comp_map_resp=jnp.ones((2, 1)),
        age_values=age_values,
        priors={"rate": Independent2D(prior_type="global")},
    )

    predictive_trace = numpyro.handlers.trace(
        numpyro.handlers.seed(model.model, jax.random.PRNGKey(0))
    ).get_trace()
    conditioned_trace = numpyro.handlers.trace(
        numpyro.handlers.seed(model.model, jax.random.PRNGKey(0))
    ).get_trace(y=model.y)

    assert predictive_trace["obs"]["is_observed"] is False
    assert conditioned_trace["obs"]["is_observed"] is True


def test_age_mix_agg_requires_rate_prior():
    age_values = jnp.arange(10)
    left = jnp.array([0, 5])
    right = jnp.array([5, 10])
    P = jnp.ones(10) / 10
    comp_map_cnt = make_composition_mapping(age_values, left, right)
    N_c = jnp.array([10, 12], dtype=jnp.float32)
    y = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)

    try:
        build_model(
            y=y,
            N_c=N_c,
            P=P,
            comp_map_cnt=comp_map_cnt,
            comp_map_resp=comp_map_cnt,
            age_values=age_values,
            priors={},
        )
    except ValueError as err:
        assert "priors" in str(err)
        assert "rate" in str(err)
    else:
        raise AssertionError("AgeMixAgg should require priors['rate'].")


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
    negbin = build_model(**_base_inputs())
    poisson = build_model(**_base_inputs(), likelihood="poisson")

    negbin_trace = _trace(negbin)
    poisson_trace = _trace(poisson)

    assert isinstance(negbin_trace["obs"]["fn"], dist.NegativeBinomial2)
    assert "inv_disp" in negbin_trace
    assert isinstance(poisson_trace["obs"]["fn"], dist.Poisson)
    assert "inv_disp" not in poisson_trace
    assert jnp.allclose(negbin_trace["mu"]["value"], poisson_trace["mu"]["value"])


def test_mean_matches_direct_population_volume_calculation():
    model = build_model(**_base_inputs(), likelihood="poisson")
    trace = _trace(model)

    gamma = jnp.exp(trace["log_rate"]["value"])
    population_volume = model.data.P[:, None] * model.data.P[None, :] * gamma
    expected = model.data.N_c[:, None] / model.data.P[:, None] * population_volume

    assert jnp.allclose(model.data.to_grid(trace["mu"]["value"]), expected)


def test_default_allocation_weights_do_not_duplicate_overlap():
    ages = jnp.arange(3)
    comp_map_cnt = make_composition_mapping(ages, jnp.array([0, 1]), jnp.array([2, 3]))
    model = build_model(
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
        model.allocation.weights,
        jnp.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
    )
    gamma = jnp.exp(trace["log_rate"]["value"])
    expected_total = model.data.P * (gamma * model.data.P[None, :]).sum(axis=1)
    # mu = N_c / P_c * EZ with fine respondent ages, so EZ = mu * P / N_c.
    mu = model.data.to_grid(trace["mu"]["value"])
    reported_total = (mu * model.data.P_resp[:, None] / model.data.N_c[:, None]).sum(
        axis=1
    )
    assert jnp.allclose(reported_total, expected_total)


def test_custom_allocation_weights():
    inputs = _base_inputs()
    inputs["comp_map_cnt"] = jnp.ones((2, 2))
    model = build_model(
        **inputs,
        allocation_weights=jnp.array([[0.75, 0.25], [0.1, 0.9]]),
        likelihood="poisson",
    )
    trace = _trace(model)

    assert jnp.allclose(model.allocation.weights.sum(axis=1), 1.0)


def test_grouped_repeat_counts_apply_hill_fatigue():
    inputs = _base_inputs()
    inputs["y"] = jnp.ones((2, 2, 2))
    inputs["N_c"] = jnp.array([[5.0, 4.0], [3.0, 2.0]])
    model = build_model(
        **inputs,
        rid=jnp.array([0, 2]),
        likelihood="poisson",
    )
    trace = _trace(model)

    repeat_multiplier = jnp.exp(hill_effect(trace, [0, 2]))
    # EZ[c, d] = sum_{a in c} P[a] sum_b gamma[a, b] P[b] Q[b, d]
    gamma = jnp.exp(trace["log_rate"]["value"])
    P = model.data.P
    m = (gamma * P[None, :]) @ trace["alloc_probs"]["value"]
    reported_volume = model.data.comp_map_resp.T.astype(jnp.float32) @ (P[:, None] * m)
    expected = (
        reported_volume[:, :, None]
        * model.data.N_c[:, None, :]
        / model.data.P_resp[:, None, None]
        * repeat_multiplier[None, None, :]
    )

    assert repeat_multiplier[0] == pytest.approx(1.0)
    assert 0 < repeat_multiplier[1] < 1
    assert jnp.allclose(model.data.to_grid(trace["mu"]["value"]), expected)


def test_exact_age_repeat_counts_use_observed_age_composition():
    inputs = _base_inputs()
    inputs["y"] = jnp.ones((2, 2, 2))
    inputs["N_c"] = None
    exact_N = jnp.array([[5.0, 4.0], [3.0, 2.0]])
    model = build_model(
        **inputs,
        rid=jnp.array([0, 1]),
        N_ar=exact_N,
        likelihood="poisson",
    )
    trace = _trace(model)

    gamma = jnp.exp(trace["log_rate"]["value"])
    contact_rate = gamma * model.data.P[None, :]
    expected = (
        contact_rate[:, :, None]
        * exact_N[:, None, :]
        * jnp.exp(hill_effect(trace, [0, 1]))[None, None, :]
    )

    assert jnp.array_equal(model.data.N_c, exact_N)
    assert jnp.allclose(model.data.to_grid(trace["mu"]["value"]), expected)



def test_from_prepared_inputs_accepts_exact_age_exposure_without_N_c():
    inputs = {key: value for key, value in _base_inputs().items() if key != "N_c"}
    inputs.update(y=jnp.ones((2, 2, 2)), rid=jnp.array([0, 1]), N_ar=jnp.ones((2, 2)))
    data = AgeMixAggDataLoader.from_prepared_inputs(inputs).load()
    assert jnp.array_equal(data.N_c, jnp.ones((2, 2)))

def test_zero_exposure_repeat_cells_are_dropped_as_structural_zeros():
    inputs = _base_inputs()
    inputs["y"] = jnp.array([[[1.0, 0.0], [2.0, 0.0]], [[3.0, 1.0], [4.0, 2.0]]])
    inputs["N_c"] = None
    model = build_model(
        **inputs,
        rid=jnp.array([0, 1]),
        N_ar=jnp.array([[5.0, 0.0], [3.0, 2.0]]),
        likelihood="negbin",
    )
    trace = _trace(model)

    # Respondent age 0 has no exposure at repeat level 1: those cells have no row.
    assert not jnp.any((model.data.cid == 0) & (model.data.rid == 1))
    assert trace["obs"]["value"].shape == (6,)
    assert jnp.all(model.data.to_grid(trace["mu"]["value"])[0, :, 1] == 0)


def test_zero_exposure_fine_ages_do_not_leak_through_grouped_respondents():
    model = build_model(
        y=jnp.array([[[1.0, 0.0], [1.0, 0.0]]]),
        N_c=None,
        N_ar=jnp.array([[5.0, 0.0], [0.0, 0.0]]),
        rid=jnp.array([0, 1]),
        P=jnp.array([1.0, 1.0]),
        comp_map_cnt=jnp.eye(2),
        comp_map_resp=jnp.ones((2, 1)),
        priors={"rate": FixedPrior(jnp.zeros((2, 2)))},
        likelihood="poisson",
    )
    trace = _trace(model)
    mu = model.data.to_grid(trace["mu"]["value"])
    expected = 5.0 * jnp.exp(trace["log_rate"]["value"])[0, :]
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
                "rid": None,
            },
            "rid",
        ),
    ],
)
def test_invalid_model_inputs(changes, message):
    inputs = _base_inputs()
    inputs.update(changes)
    with pytest.raises(ValueError, match=message):
        build_model(**inputs)


def test_exact_age_counts_reject_overlapping_respondent_groups():
    with pytest.raises(ValueError, match="partition"):
        build_model(
            y=jnp.ones((2, 2, 2)),
            N_c=None,
            P=jnp.ones(2),
            comp_map_cnt=jnp.ones((2, 2)),
            comp_map_resp=jnp.ones((2, 2)),
            priors={"rate": FixedPrior(jnp.zeros((2, 2)))},
            rid=jnp.array([0, 1]),
            N_ar=jnp.ones((2, 2)),
        )
