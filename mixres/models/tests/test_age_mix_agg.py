import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../cntmosaic"))

from mixres.models import (
    AgeMixAgg,
    age_membership,
    aggregate_Z,
    aggregate_log_intensity,
    make_W,
    prepare_age_mix_agg_inputs,
)
from mixres.models._AgeMixAggTotal import AgeMixAggTotal
from cntmosaic.models.numpyro.priors import Independent2D


def test_aggregation_helpers_on_ones():
    age_values = jnp.arange(80)
    left = jnp.arange(0, 80, 5)
    right = left + 5
    Z = jnp.ones((80, 80))

    W = make_W(age_values, left, right)

    assert W.shape == (80, 16)
    assert aggregate_Z(Z).shape == (80, 80)
    assert aggregate_Z(Z, W, aggregation="contact").shape == (80, 16)
    assert aggregate_Z(Z, W, aggregation="both").shape == (16, 16)
    assert jnp.all(aggregate_Z(Z, W, aggregation="contact") == 5.0)
    assert jnp.all(aggregate_Z(Z, W, aggregation="both") == 25.0)


def test_aggregate_log_intensity_matches_direct_sum():
    log_gamma = jnp.log(jnp.arange(1, 17, dtype=jnp.float32).reshape(4, 4))
    P = jnp.array([0.1, 0.2, 0.3, 0.4])
    log_P = jnp.log(P)
    W = make_W(jnp.arange(4), jnp.array([0, 2]), jnp.array([2, 4]))

    actual = aggregate_log_intensity(log_gamma, log_P, W, aggregation="both")
    fine = jnp.exp(log_gamma + log_P[:, None] + log_P[None, :])
    expected = jnp.log(
        jnp.array(
            [
                [fine[:2, :2].sum(), fine[:2, 2:].sum()],
                [fine[2:, :2].sum(), fine[2:, 2:].sum()],
            ]
        )
    )

    assert jnp.allclose(actual, expected)
    assert jnp.allclose(
        aggregate_log_intensity(log_gamma, log_P),
        jnp.log(fine),
    )
    assert jnp.allclose(
        aggregate_log_intensity(log_gamma, log_P, W, aggregation="contact"),
        jnp.log(jnp.stack([fine[:, :2].sum(axis=1), fine[:, 2:].sum(axis=1)], axis=1)),
    )


def test_distinct_respondent_and_contact_aggregation():
    log_gamma = jnp.zeros((4, 4))
    P = jnp.array([0.1, 0.2, 0.3, 0.4])
    contact_W = make_W(jnp.arange(4), jnp.array([0, 2]), jnp.array([2, 4]))
    respondent_W = make_W(jnp.arange(4), jnp.array([0, 1]), jnp.array([1, 4]))
    actual = jnp.exp(
        aggregate_log_intensity(
            log_gamma,
            jnp.log(P),
            contact_W,
            aggregation="contact",
            W_respondent=respondent_W,
        )
    )
    expected = jnp.outer(
        jnp.array([P[:1].sum(), P[1:].sum()]),
        jnp.array([P[:2].sum(), P[2:].sum()]),
    )
    assert actual.shape == (2, 2)
    assert jnp.allclose(actual, expected)


def test_overlapping_membership_allows_multiple_groups():
    M = age_membership(jnp.arange(15), jnp.array([0, 5]), jnp.array([10, 15]))

    assert M.shape == (2, 15)
    assert jnp.all(M[:, :5].sum(axis=0) == 1.0)
    assert jnp.all(M[:, 5:10].sum(axis=0) == 2.0)
    assert jnp.all(M[:, 10:15].sum(axis=0) == 1.0)


def test_prepare_age_mix_agg_inputs_for_scenario_1_like_data():
    labels = ["0-4", "5-9"]
    participants = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "resp_group": pd.Categorical(["0-4", "0-4", "5-9", "5-9"], categories=labels),
        }
    )
    contacts = pd.DataFrame(
        {
            "id": [1, 1, 2, 3, 4],
            "cnt_group": pd.Categorical(["0-4", "5-9", "0-4", "5-9", "0-4"], categories=labels),
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

    assert inputs["aggregation"] == "both"
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
        aggregation="contact",
    )

    assert inputs["W"].shape == (10, 2)
    assert inputs["aggregation"] == "contact"
    assert inputs["y"].shape == (10, 2)
    assert inputs["N_c"].shape == (10,)
    assert inputs["y"].sum() == contacts["y"].sum()


def test_prepare_inputs_with_coarse_respondent_groups():
    participants = pd.DataFrame(
        {"id": [1, 2, 3], "resp_group": ["0-1", "2-4", "2-4"]}
    )
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
        aggregation="contact",
    )
    assert inputs["W"].shape == (5, 2)
    assert inputs["W_respondent"].shape == (5, 2)
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
        aggregation="none",
    )

    assert inputs["W"] is None
    assert inputs["aggregation"] == "none"
    assert inputs["y"].shape == (10, 10)
    assert inputs["N_c"].shape == (10,)
    assert inputs["y"].sum() == contacts["y"].sum()


def test_age_mix_agg_model_smoke_trace_shapes():
    age_values = jnp.arange(10)
    left = jnp.array([0, 5])
    right = jnp.array([5, 10])
    P = jnp.ones(10) / 10
    W = make_W(age_values, left, right)
    cases = [
        ("none", None, (10, 10), jnp.ones(10)),
        ("contact", W, (10, 2), jnp.ones(10)),
        ("both", W, (2, 2), jnp.array([5.0, 5.0])),
    ]

    for aggregation, mode_W, expected_shape, expected_P_respondent in cases:
        y = jnp.ones(expected_shape)
        model = AgeMixAgg(
            y=y,
            N_c=jnp.ones(expected_shape[0]),
            P=P,
            W=mode_W,
            aggregation=aggregation,
            age_values=age_values,
            priors={"rate": Independent2D(prior_type="global")},
        )
        seeded = numpyro.handlers.seed(model.model, jax.random.PRNGKey(0))
        trace = numpyro.handlers.trace(seeded).get_trace(y=y)

        assert trace["log_gamma"]["value"].shape == (10, 10)
        assert trace["aggregated_log_intensity"]["value"].shape == expected_shape
        assert trace["mu"]["value"].shape == expected_shape
        assert jnp.allclose(model.P_respondent, expected_P_respondent / 10)


def test_age_mix_agg_model_only_conditions_on_explicit_y():
    age_values = jnp.arange(2)
    P = jnp.ones(2) / 2
    model = AgeMixAgg(
        y=jnp.array([[1.0]]),
        N_c=jnp.array([10.0]),
        P=P,
        W=jnp.ones((2, 1)),
        aggregation="both",
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
    W = make_W(age_values, left, right)
    N_c = jnp.array([10, 12], dtype=jnp.float32)
    y = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)

    try:
        AgeMixAgg(
            y=y,
            N_c=N_c,
            P=P,
            W=W,
            aggregation="both",
            age_values=age_values,
        )
    except ValueError as err:
        assert "priors" in str(err)
        assert "rate" in str(err)
    else:
        raise AssertionError("AgeMixAgg should require priors['rate'].")


def test_age_mix_agg_total_uses_negative_binomial_total():
    age_values = jnp.arange(4)
    P = jnp.ones(4) / 4
    W = make_W(age_values, jnp.array([0, 2]), jnp.array([2, 4]))
    y = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)
    model = AgeMixAggTotal(
        y=y,
        N_c=jnp.array([10, 12], dtype=jnp.float32),
        P=P,
        W=W,
        aggregation="both",
        age_values=age_values,
        priors={"rate": Independent2D(prior_type="global")},
        inv_disp_prior_rate=2.5,
    )

    seeded = numpyro.handlers.seed(model.model, jax.random.PRNGKey(0))
    trace = numpyro.handlers.trace(seeded).get_trace(y=y)

    assert isinstance(trace["inv_disp"]["fn"], dist.Exponential)
    assert jnp.allclose(trace["inv_disp"]["fn"].rate, 2.5)
    assert isinstance(trace["total_obs"]["fn"], dist.NegativeBinomial2)
    assert jnp.allclose(
        trace["total_obs"]["fn"].concentration,
        1.0 / trace["inv_disp"]["value"],
    )
    assert trace["total_obs"]["is_observed"] is True
    assert trace["obs"]["is_observed"] is True


def test_age_mix_agg_total_rejects_invalid_inv_disp_prior_rate():
    common = {
        "y": jnp.array([[1.0]]),
        "N_c": jnp.array([10.0]),
        "P": jnp.ones(2) / 2,
        "W": jnp.ones((2, 1)),
        "aggregation": "both",
        "age_values": jnp.arange(2),
        "priors": {"rate": Independent2D(prior_type="global")},
    }

    for invalid_rate in (0.0, -1.0, jnp.inf, jnp.nan):
        try:
            AgeMixAggTotal(
                **common,
                inv_disp_prior_rate=invalid_rate,
            )
        except ValueError as err:
            assert "inv_disp_prior_rate" in str(err)
        else:
            raise AssertionError(
                f"Expected inv_disp_prior_rate={invalid_rate} to be rejected."
            )
