import pandas as pd
import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.handlers import scope

from ._aggregation import aggregate_log_intensity, make_W
from ._inference import run_inference_svi, posterior_predictive_svi


def age_bin_labels(left_limits, right_limits):
    """Return labels like 0-4 for half-open integer age bins [0, 5)."""
    return [f"{int(left)}-{int(right) - 1}" for left, right in zip(left_limits, right_limits)]


def prepare_age_mix_agg_inputs(
    scenario,
    P,
    age_values,
    left_limits=None,
    right_limits=None,
    aggregation="both",
    respondent_labels=None,
    contact_labels=None,
    respondent_left_limits=None,
    respondent_right_limits=None,
):
    """Prepare inputs for no, contact-only, or both-axis age aggregation."""
    age_values = jnp.asarray(age_values)
    P = jnp.asarray(P, dtype=jnp.float32)

    if aggregation not in {"none", "contact", "both"}:
        raise ValueError("aggregation must be one of 'none', 'contact', or 'both'.")

    if aggregation == "none":
        W = None
        default_respondent_labels = [int(age) for age in age_values]
        default_contact_labels = default_respondent_labels
    else:
        if left_limits is None or right_limits is None:
            raise ValueError("Age limits are required when aggregation is 'contact' or 'both'.")
        left_limits = jnp.asarray(left_limits)
        right_limits = jnp.asarray(right_limits)
        W = jnp.asarray(make_W(age_values, left_limits, right_limits))
        coarse_labels = age_bin_labels(left_limits, right_limits)
        default_respondent_labels = (
            coarse_labels if aggregation == "both" else [int(age) for age in age_values]
        )
        default_contact_labels = coarse_labels

    if (respondent_left_limits is None) != (respondent_right_limits is None):
        raise ValueError("Both respondent age-limit arrays must be supplied together.")
    W_respondent = None
    if respondent_left_limits is not None:
        W_respondent = jnp.asarray(
            make_W(age_values, respondent_left_limits, respondent_right_limits)
        )
        default_respondent_labels = age_bin_labels(
            respondent_left_limits, respondent_right_limits
        )
    respondent_labels = default_respondent_labels if respondent_labels is None else list(respondent_labels)
    contact_labels = default_contact_labels if contact_labels is None else list(contact_labels)

    participants = scenario["participants"][["id", "resp_group"]].copy()
    contacts = scenario["contacts"][["id", "cnt_group", "y"]].copy()
    participants["resp_group"] = pd.Categorical(
        participants["resp_group"],
        categories=respondent_labels,
    )
    contacts["cnt_group"] = pd.Categorical(
        contacts["cnt_group"],
        categories=contact_labels,
    )

    counts = (
        contacts.merge(participants, on="id")
        .groupby(["resp_group", "cnt_group"], observed=False)["y"]
        .sum()
        .reindex(
            pd.MultiIndex.from_product(
                [respondent_labels, contact_labels],
                names=["resp_group", "cnt_group"],
            ),
            fill_value=0,
        )
        .reset_index(name="y")
    )

    respondent_N = (
        participants["resp_group"]
        .value_counts(sort=False)
        .reindex(respondent_labels, fill_value=0)
        .astype(float)
    )

    return {
        "y": counts["y"].to_numpy(dtype=float).reshape(
            len(respondent_labels), len(contact_labels)
        ),
        "N_c": respondent_N.to_numpy(dtype=float),
        "P": P,
        "W": W,
        "W_respondent": W_respondent,
        "aggregation": aggregation,
        "age_values": age_values,
        "respondent_labels": respondent_labels,
        "respondent_left_limits": (
            None if respondent_left_limits is None else jnp.asarray(respondent_left_limits)
        ),
        "respondent_right_limits": (
            None if respondent_right_limits is None else jnp.asarray(respondent_right_limits)
        ),
        "contact_labels": contact_labels,
    }


class AgeMixAgg:
    """Fine-age NumPyro contact model with optional shared age aggregation.

    ``priors["rate"]`` is expected to follow the Prior2D interface used by
    AgeMixFF: it should provide ``set_age_bounds(min_age, max_age)`` and
    ``sample()`` methods. Concrete examples are Spline2D, HSGP2D, and IGMRF2D.
    """

    def __init__(
        self,
        y,
        N_c,
        P,
        W=None,
        W_respondent=None,
        aggregation="none",
        age_values=None,
        priors=None,
    ):
        if priors is None or "rate" not in priors:
            raise ValueError("AgeMixAgg requires priors with a 'rate' Prior2D instance.")

        self.y = jnp.asarray(y, dtype=jnp.float32)
        self.N_c = jnp.asarray(N_c, dtype=jnp.float32)
        self.P = jnp.asarray(P, dtype=jnp.float32)
        self.W = None if W is None else jnp.asarray(W, dtype=jnp.float32)
        self.W_respondent = (
            None
            if W_respondent is None
            else jnp.asarray(W_respondent, dtype=jnp.float32)
        )
        self.aggregation = aggregation
        self.age_values = (
            jnp.arange(self.P.shape[0], dtype=jnp.float32)
            if age_values is None
            else jnp.asarray(age_values, dtype=jnp.float32)
        )
        self.priors = dict(priors)
        self.validate_inputs()
        self.load_data()
        self.configure_priors()

    @classmethod
    def from_prepared_inputs(cls, inputs, **kwargs):
        return cls(
            y=inputs["y"],
            N_c=inputs["N_c"],
            P=inputs["P"],
            W=inputs["W"],
            W_respondent=inputs.get("W_respondent"),
            aggregation=inputs["aggregation"],
            age_values=inputs.get("age_values"),
            **kwargs,
        )

    def validate_inputs(self):
        if self.aggregation not in {"none", "contact", "both"}:
            raise ValueError("aggregation must be one of 'none', 'contact', or 'both'.")
        if self.P.ndim != 1:
            raise ValueError("P must have shape (A,).")
        if self.age_values.shape != self.P.shape:
            raise ValueError("age_values and P must have the same length.")

        A = self.P.shape[0]
        if self.aggregation == "none":
            if self.W is not None:
                raise ValueError("W must be None when aggregation='none'.")
            expected_y_shape = (A, A)
        else:
            if self.W is None or self.W.ndim != 2 or self.W.shape[0] != A:
                raise ValueError("W must have shape (A, C) when aggregation is enabled.")
            C = self.W.shape[1]
            respondent_count = (
                A if self.W_respondent is None else self.W_respondent.shape[1]
            )
            if self.W_respondent is not None and self.W_respondent.shape[0] != A:
                raise ValueError("W_respondent must have shape (A, R).")
            expected_y_shape = (
                (respondent_count, C)
                if self.aggregation == "contact"
                else ((C if self.W_respondent is None else respondent_count), C)
            )

        if self.y.shape != expected_y_shape:
            raise ValueError(f"y must have shape {expected_y_shape} for aggregation='{self.aggregation}'.")
        if self.N_c.shape != (expected_y_shape[0],):
            raise ValueError(f"N_c must have shape ({expected_y_shape[0]},).")

    def load_data(self):
        if self.W_respondent is not None:
            self.P_respondent = self.W_respondent.T @ self.P
        elif self.aggregation == "both":
            self.P_respondent = self.W.T @ self.P
        else:
            self.P_respondent = self.P
        self.log_P = jnp.log(jnp.clip(self.P, min=1e-12))
        self.log_N_c = jnp.log(jnp.clip(self.N_c, min=1e-12))
        self.log_P_respondent = jnp.log(jnp.clip(self.P_respondent, min=1e-12))

    def configure_priors(self):
        rate_prior = self.priors["rate"]
        min_age = int(jnp.min(self.age_values))
        max_age = int(jnp.max(self.age_values))

        if hasattr(rate_prior, "set_age_bounds"):
            rate_prior.set_age_bounds(min_age, max_age)

        if getattr(rate_prior, "prior_type", None) != "global":
            if hasattr(rate_prior, "set_event_dim") and getattr(rate_prior, "event_dim", None) is None:
                rate_prior.set_event_dim(1)
            if hasattr(rate_prior, "set_loc") and getattr(rate_prior, "loc", None) is None:
                rate_prior.set_loc(0.0)

    def _sample_rate_prior(self):
        with scope(prefix="rate"):
            f = self.priors["rate"].sample()

        if f.ndim == 3 and f.shape[0] == 1:
            f = f[0]

        expected_shape = (self.P.shape[0], self.P.shape[0])
        if f.shape != expected_shape:
            raise ValueError(
                f"priors['rate'].sample() must return shape {expected_shape} "
                f"or (1, {expected_shape[0]}, {expected_shape[1]}), got {f.shape}."
            )

        return f

    def model(self, y=None):
        len_y = self.y.size if y is None else jnp.asarray(y).size

        baseline = numpyro.sample("baseline", dist.Normal(-jnp.mean(self.log_P), 2.5))
        f = self._sample_rate_prior()

        log_gamma = numpyro.deterministic(
            "log_gamma",
            baseline + f,
        )
        aggregated_log_intensity = numpyro.deterministic(
            "aggregated_log_intensity",
            aggregate_log_intensity(
                log_gamma,
                self.log_P,
                self.W,
                aggregation=self.aggregation,
                W_respondent=self.W_respondent,
            ),
        )
        log_mu = numpyro.deterministic(
            "log_mu",
            aggregated_log_intensity
            + self.log_N_c[:, None]
            - self.log_P_respondent[:, None],
        )
        mu = numpyro.deterministic("mu", jnp.exp(log_mu))

        with numpyro.plate("data", len_y):
            numpyro.sample(
                "obs",
                dist.Poisson(rate=mu.reshape(-1)),
                obs=None if y is None else jnp.asarray(y).reshape(-1),
            )

    def run_inference_svi(
        self,
        prng_key: jax.random.PRNGKey,
        guide: callable,
        num_steps: int = 5_000,
        peak_lr: float = 0.01,
        **model_kwargs,
    ):
        self.guide = guide
        model_kwargs.setdefault("y", self.y)
        self.svi = run_inference_svi(
            prng_key=prng_key,
            model=self.model,
            guide=guide,
            num_steps=num_steps,
            peak_lr=peak_lr,
            **model_kwargs,
        )

    def posterior_predictive_svi(
        self,
        prng_key,
        guide: callable,
        num_samples: int = 5_000,
        **model_kwargs,
    ) -> dict[str, jax.Array]:
        if hasattr(self, "svi") is False:
            raise AttributeError("run_inference_svi must be run first.")

        return posterior_predictive_svi(
            prng_key,
            self.model,
            guide,
            self.svi.params,
            num_samples=num_samples,
            **model_kwargs,
        )
