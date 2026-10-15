"""Tests for the mixres ModelSummariser."""

import numpy as np

from mixres.analysis import ModelSummariser


def _make_summariser() -> ModelSummariser:
    rate_draws = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 3.0], [4.0, 5.0]],
        ]
    )
    return ModelSummariser(
        posterior_samples={"log_gamma": np.log(rate_draws)},
        population=np.array([0.25, 0.75]),
    )


def test_standardised_posterior_quantities():
    summariser = _make_summariser()

    rate = summariser.get_posterior_samples("rate")
    cint = summariser.get_posterior_samples("cint")
    mcint = summariser.get_posterior_samples("mcint")
    column_mcint = summariser.get_posterior_samples("column_mcint")

    expected_cint = rate * np.array([0.25, 0.75])[None, None, :]
    np.testing.assert_allclose(cint, expected_cint)
    np.testing.assert_allclose(mcint, expected_cint.sum(axis=-1))
    np.testing.assert_allclose(column_mcint, expected_cint.sum(axis=-2))


def test_summary_uses_requested_measure_and_interval():
    summariser = _make_summariser()

    summary = summariser.summarise_cint(
        credible_level=0.5,
        measure="mean",
    )
    cint = summariser.get_posterior_samples("cint")

    np.testing.assert_allclose(summary.central, cint.mean(axis=0))
    np.testing.assert_allclose(summary.lower, np.quantile(cint, 0.25, axis=0))
    np.testing.assert_allclose(summary.upper, np.quantile(cint, 0.75, axis=0))
    assert summary.measure == "mean"
    assert summary.credible_level == 0.5


def test_precomputed_cint_does_not_require_population():
    cint = np.ones((3, 2, 2))
    summariser = ModelSummariser(posterior_samples={"cint": cint})

    np.testing.assert_allclose(
        summariser.get_posterior_samples("cint"),
        cint,
    )
