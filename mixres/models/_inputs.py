"""Build ``AgeMixAgg``-family inputs from a simulated survey scenario."""

import jax.numpy as jnp
import pandas as pd

from ._aggregation import make_composition_mapping


def age_bin_labels(left_limits, right_limits):
    """Return labels like 0-4 for half-open integer age bins [0, 5)."""
    return [
        f"{int(left)}-{int(right) - 1}"
        for left, right in zip(left_limits, right_limits)
    ]


def prepare_age_mix_agg_inputs(
    scenario,
    P,
    age_values,
    left_limits=None,
    right_limits=None,
    age_agg_dims="both",
    respondent_labels=None,
    contact_labels=None,
    respondent_left_limits=None,
    respondent_right_limits=None,
):
    """Tabulate contact counts and respondent sizes on the requested age grids.

    Parameters
    ----------
    scenario : dict
        Must contain ``participants`` (columns ``id``, ``resp_group``) and
        ``contacts`` (columns ``id``, ``cnt_group``, ``y``).
    P : array-like, shape (A,)
        Population age distribution on the fine grid ``age_values``.
    left_limits, right_limits : array-like, optional
        Half-open contact age groups; required unless ``age_agg_dims="none"``.
    age_agg_dims : {"none", "contact", "both"}, default="both"
        Which age dimensions are aggregated:

        - ``"none"``: no age aggregation for respondents or contacts.
        - ``"contact"``: age aggregation for the contact age dimension only.
        - ``"both"``: age aggregation for both respondent and contact ages,
          using the contact groups for respondents unless respondent limits
          are given.
    respondent_left_limits, respondent_right_limits : array-like, optional
        Distinct half-open respondent age groups.

    Returns
    -------
    dict
        Keyword arguments for ``<Model>.from_prepared_inputs`` (``y``,
        ``N_c``, ``P``, ``comp_map_cnt``, ``comp_map_resp``, ``age_values``),
        plus the metadata ``age_agg_dims``, ``respondent_labels``,
        ``contact_labels``, ``respondent_left_limits`` and
        ``respondent_right_limits``. An axis that is not aggregated has a
        ``None`` composition matrix.
    """
    if age_agg_dims not in ("none", "contact", "both"):
        raise ValueError(
            f"age_agg_dims must be 'none', 'contact' or 'both', got {age_agg_dims!r}."
        )
    if (respondent_left_limits is None) != (respondent_right_limits is None):
        raise ValueError("Both respondent age-limit arrays must be supplied together.")

    age_values = jnp.asarray(age_values)
    fine_labels = [int(age) for age in age_values]

    if age_agg_dims == "none":
        comp_map_cnt = None
        default_contact_labels = default_respondent_labels = fine_labels
    else:
        if left_limits is None or right_limits is None:
            raise ValueError(
                "Age limits are required when age_agg_dims is 'contact' or 'both'."
            )
        comp_map_cnt = make_composition_mapping(
            age_values, jnp.asarray(left_limits), jnp.asarray(right_limits)
        )
        default_contact_labels = age_bin_labels(left_limits, right_limits)
        default_respondent_labels = (
            default_contact_labels if age_agg_dims == "both" else fine_labels
        )

    comp_map_resp = comp_map_cnt if age_agg_dims == "both" else None
    if respondent_left_limits is not None:
        comp_map_resp = make_composition_mapping(
            age_values, respondent_left_limits, respondent_right_limits
        )
        default_respondent_labels = age_bin_labels(
            respondent_left_limits, respondent_right_limits
        )
        respondent_left_limits = jnp.asarray(respondent_left_limits)
        respondent_right_limits = jnp.asarray(respondent_right_limits)

    if respondent_labels is None:
        respondent_labels = default_respondent_labels
    if contact_labels is None:
        contact_labels = default_contact_labels
    respondent_labels, contact_labels = list(respondent_labels), list(contact_labels)

    participants = scenario["participants"][["id", "resp_group"]].copy()
    contacts = scenario["contacts"][["id", "cnt_group", "y"]].copy()
    participants["resp_group"] = pd.Categorical(
        participants["resp_group"], categories=respondent_labels
    )
    contacts["cnt_group"] = pd.Categorical(
        contacts["cnt_group"], categories=contact_labels
    )

    y = (
        contacts.merge(participants, on="id")
        .groupby(["resp_group", "cnt_group"], observed=False)["y"]
        .sum()
        .unstack("cnt_group")
        .reindex(index=respondent_labels, columns=contact_labels, fill_value=0)
    )
    N_c = (
        participants["resp_group"]
        .value_counts(sort=False)
        .reindex(respondent_labels, fill_value=0)
    )

    return {
        "y": y.to_numpy(dtype=float),
        "N_c": N_c.to_numpy(dtype=float),
        "P": jnp.asarray(P, dtype=jnp.float32),
        "comp_map_cnt": comp_map_cnt,
        "comp_map_resp": comp_map_resp,
        "age_agg_dims": age_agg_dims,
        "age_values": age_values,
        "respondent_labels": respondent_labels,
        "respondent_left_limits": respondent_left_limits,
        "respondent_right_limits": respondent_right_limits,
        "contact_labels": contact_labels,
    }
