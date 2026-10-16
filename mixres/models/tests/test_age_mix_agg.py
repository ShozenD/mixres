import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../cntmosaic"))

from cntmosaic.models.numpyro.priors import Independent2D

from mixres.models import (
    AgeMixAgg,
    log_expected_volume,
    make_composition_mapping,
    prepare_age_mix_agg_inputs,
    sum_EZ,
)


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


def test_log_expected_volume_matches_direct_sum():
    log_gamma = jnp.log(jnp.arange(1, 17, dtype=jnp.float32).reshape(4, 4))
    P = jnp.array([0.1, 0.2, 0.3, 0.4])
    log_P = jnp.log(P)
    comp_map_cnt = make_composition_mapping(jnp.arange(4), jnp.array([0, 2]), jnp.array([2, 4]))

    actual = log_expected_volume(log_gamma, log_P, comp_map_cnt, comp_map_cnt)
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
        log_expected_volume(log_gamma, log_P),
        jnp.log(fine),
    )
    assert jnp.allclose(
        log_expected_volume(log_gamma, log_P, comp_map_cnt),
        jnp.log(jnp.stack([fine[:, :2].sum(axis=1), fine[:, 2:].sum(axis=1)], axis=1)),
    )


def test_distinct_respondent_and_contact_aggregation():
    log_gamma = jnp.zeros((4, 4))
    P = jnp.array([0.1, 0.2, 0.3, 0.4])
    comp_map_cnt = make_composition_mapping(
        jnp.arange(4), jnp.array([0, 2]), jnp.array([2, 4])
    )
    comp_map_resp = make_composition_mapping(
        jnp.arange(4), jnp.array([0, 1]), jnp.array([1, 4])
    )
    actual = jnp.exp(
        log_expected_volume(log_gamma, jnp.log(P), comp_map_cnt, comp_map_resp)
    )
    expected = jnp.outer(
        jnp.array([P[:1].sum(), P[1:].sum()]),
        jnp.array([P[:2].sum(), P[2:].sum()]),
    )
    assert actual.shape == (2, 2)
    assert jnp.allclose(actual, expected)


def test_overlapping_membership_allows_multiple_groups():
    M = make_composition_mapping(
        jnp.arange(15), jnp.array([0, 5]), jnp.array([10, 15])
    )

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

    for case_comp_map_cnt, case_comp_map_resp, expected_shape, expected_P_respondent in cases:
        y = jnp.ones(expected_shape)
        model = AgeMixAgg(
            y=y,
            N_c=jnp.ones(expected_shape[0]),
            P=P,
            comp_map_cnt=case_comp_map_cnt,
            comp_map_resp=case_comp_map_resp,
            age_values=age_values,
            priors={"rate": Independent2D(prior_type="global")},
        )
        seeded = numpyro.handlers.seed(model.model, jax.random.PRNGKey(0))
        trace = numpyro.handlers.trace(seeded).get_trace(y=y)

        assert trace["log_gamma"]["value"].shape == (10, 10)
        assert trace["log_EZ"]["value"].shape == expected_shape
        assert trace["mu"]["value"].shape == expected_shape
        assert jnp.allclose(model.P_respondent, expected_P_respondent / 10)


def test_age_mix_agg_model_only_conditions_on_explicit_y():
    age_values = jnp.arange(2)
    P = jnp.ones(2) / 2
    model = AgeMixAgg(
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
        AgeMixAgg(
            y=y,
            N_c=N_c,
            P=P,
            comp_map_cnt=comp_map_cnt,
            comp_map_resp=comp_map_cnt,
            age_values=age_values,
        )
    except ValueError as err:
        assert "priors" in str(err)
        assert "rate" in str(err)
    else:
        raise AssertionError("AgeMixAgg should require priors['rate'].")
