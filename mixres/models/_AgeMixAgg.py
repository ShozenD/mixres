"""Fine-age contact model observed through (overlapping) age-group aggregation."""

from typing import Any, Dict, Literal, Optional

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from cntmosaic.models.numpyro.priors import Hill
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike
from numpyro.handlers import scope

from ._aggregation import safe_log
from ._allocation import (
    DYNAMIC_ALLOCATIONS,
    Allocation,
    AllocationMode,
    compute_allocation_probs,
)
from ._base import ContactModel
from ._data import AgeMixAggData, AgeMixAggDataLoader

LIKELIHOODS = ("negbin", "poisson")


class AgeMixAgg(ContactModel):
    """
    Fine-age contact model for contact counts reported in age groups.

    Mathematical model
    ------------------
    ::

        log gamma[a, b] = baseline + f(a, b),           f ~ priors["rate"]
        m[a, d]         = sum_b gamma[a, b] P[b] Q[b, d]
        EZ[c, d]        = sum_{a in c} P[a] m[a, d]
        mu[c, d]        = N[c] / P[c] * EZ[c, d]
        y[c, d]         ~ NegativeBinomial2(mu, 1 / inv_disp)   or   Poisson(mu)

    Observations are long-format rows ``i = (c, d[, r])`` built by the data
    loader (``data.cid``, ``data.did``, ``data.rid``), and every row mean has
    the form ``mu[i] = sum_a w[i, a] m[a, d]`` with the respondent exposure
    ``w = exp(data.log_row_exposure)``.

    ``EZ`` is the expected contact volume between respondent group ``c`` and
    recorded contact group ``d``. A ``prior_type="global"`` rate prior gives a
    symmetric (reciprocal) surface, ``f = f^T``; ``"partial"`` / ``"full"``
    priors give an asymmetric one. The allocation weights
    ``Q`` are chosen by ``allocation``:

    * ``"uniform"`` (default): an age in ``k`` recorded groups gives ``1/k``
      of its contacts to each.
    * ``"binary"``: ``Q = comp_map_cnt``, so such an age is counted ``k``
      times; only appropriate for non-overlapping groups.
    * ``"empirical"``: shares follow the observed contacts per group.
    * ``"marginal_intensity"``: group counts divided by the current
      exposure-weighted (including the Hill effect) contact-age marginal; recomputed at
      every model evaluation.
    * ``"conditional_intensity"``: ``Q[a, b, d]`` per respondent age, from
      age-specific group counts smoothed towards the pooled frequencies and
      divided by ``gamma[a, b] P[b]``; recomputed at every model evaluation.
      Needs single-year respondents on the latent age grid.

    The formulas are in :mod:`mixres.models._allocation`. Group counts are
    taken from ``y`` unless given, and stay fixed (training counts) when the
    model is evaluated on other data. Cells whose respondent stratum has no
    respondents, or whose contact group receives no allocation mass, are
    structural zeros: they have no row, so they never enter the likelihood.
    ``mu`` and ``obs`` are per row; ``data.to_grid`` scatters them back onto
    the (C, D[, R]) grid with zeros in those cells.

    Repeat-stratified observations
    ------------------------------
    ``y`` may have shape ``(C, D, R)``, indexed by the number of previous
    survey waves ``rid``. As in cntmosaic, repeated participation lowers the
    reported counts through the :class:`Hill` function of each row's repeat
    level ``rid[i]``, added to the log expected counts,

        hill_effect[k] = -scale * s_k / (1 + s_k),   s_k = exp(shape1) * k^shape2

    and exposure is either grouped ``N_c`` with shape ``(C, R)`` or exact-age
    ``N_ar`` with shape ``(A, R)``.

    Parameters
    ----------
    dataloader : AgeMixAggDataLoader
        Validated survey inputs (counts, exposure, population, composition
        matrices, repeat levels); loaded into :attr:`data`.
    priors : dict
        Must contain ``"rate"``, a cntmosaic ``Prior2D`` (Spline2D, HSGP2D, IGMRF2D, ...).
    likelihood : {"negbin", "poisson"}, default="negbin"
        Cell-level count likelihood.
    allocation : AllocationMode, optional
        Rule for building ``Q``; see above. Defaults to ``"uniform"``.
    allocation_weights : array-like, shape (A, D), optional
        Custom allocation weights ``Q``. Rows must sum to one and respect the
        support of ``comp_map_cnt``. Cannot be combined with ``allocation``;
        the model then reports ``allocation.mode == "custom"``.
    age_group_counts : array-like, optional
        Raw contacts per contact group used by the count-based allocations:
        shape (D,), or (A, D) by respondent age for
        ``"conditional_intensity"``. Defaults to ``y`` summed over respondents
        (shape (D,)) or over repeats (shape (A, D)).
    inv_disp_prior_rate : float, default=1.0
        Rate of the exponential prior on the negative-binomial inverse
        concentration ``inv_disp``.

    Attributes
    ----------
    data : AgeMixAggData
        The observation rows (without structural zeros) and grid-level inputs.
    allocation : Allocation
        The allocation configuration (mode, static ``Q``, counts, active groups).

    Examples
    --------
    >>> loader = AgeMixAggDataLoader(y, N_c, P, comp_map_cnt=comp_map_cnt)
    >>> model = AgeMixAgg(loader, priors={"rate": prior}, allocation="empirical")
    >>> model.run_inference_svi(jax.random.PRNGKey(0))
    """

    def __init__(
        self,
        dataloader: AgeMixAggDataLoader,
        priors: Dict[str, Any],
        likelihood: Literal["negbin", "poisson"] = "negbin",
        allocation: Optional[AllocationMode] = None,
        allocation_weights: Optional[ArrayLike] = None,
        age_group_counts: Optional[ArrayLike] = None,
        inv_disp_prior_rate: float = 1.0,
    ) -> None:

        super().__init__()

        if priors is None or "rate" not in priors:
            raise ValueError(
                f"{type(self).__name__} requires priors with a 'rate' Prior2D instance."
            )

        if likelihood not in LIKELIHOODS:
            raise ValueError(f"likelihood must be one of {LIKELIHOODS}.")

        self.inv_disp_prior_rate = float(inv_disp_prior_rate)
        if not jnp.isfinite(self.inv_disp_prior_rate) or self.inv_disp_prior_rate <= 0:
            raise ValueError("inv_disp_prior_rate must be finite and positive.")
        self.priors = dict(priors)
        self.likelihood = likelihood
        data = dataloader.load()
        self.allocation = Allocation.build(
            data, allocation, allocation_weights, age_group_counts
        )
        # Contact groups without allocation mass are structural zeros: drop their rows.
        self.data: AgeMixAggData = data.select(self.allocation.group_active[data.did])
        self.y = self.data.y
        self._configure_rate_prior()
        # Repeat-level (fatigue) effect, evaluated at repeat levels 0, ..., max.
        self.hill = (
            Hill(max_value=self.data.age_exposure.shape[1] - 1)
            if self.data.rid is not None
            else None
        )

    # ------------------------------------------------------------------
    # Model components
    # ------------------------------------------------------------------

    def _configure_rate_prior(self) -> None:
        rate_prior = self.priors["rate"]
        ages = self.data.age_values
        if hasattr(rate_prior, "set_age_bounds"):
            rate_prior.set_age_bounds(int(jnp.min(ages)), int(jnp.max(ages)))
        if getattr(rate_prior, "prior_type", None) == "global":
            return
        if (
            hasattr(rate_prior, "set_event_dim")
            and getattr(rate_prior, "event_dim", None) is None
        ):
            rate_prior.set_event_dim(1)
        if hasattr(rate_prior, "set_loc") and getattr(rate_prior, "loc", None) is None:
            rate_prior.set_loc(0.0)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def model(self, y: Optional[ArrayLike] = None) -> None:
        data, allocation = self.data, self.allocation

        baseline = numpyro.sample("baseline", dist.Normal(-jnp.mean(data.log_P), 2.5))
        with scope(prefix="rate"):
            f = self.priors["rate"].sample()

        log_rate = numpyro.deterministic(
            "log_rate", baseline + (f[0] if f.ndim == 3 else f)
        )
        # Repeat (fatigue) effect
        hill_effect = None if self.hill is None else self.hill.sample()

        # log gamma[a, b] P[b]: population contact intensity from age a to age b.
        log_cint = log_rate + data.log_P[None, :]

        if allocation.mode in DYNAMIC_ALLOCATIONS:

            if allocation.mode == "marginal_intensity":
                # Weight respondent ages by their expected sample exposure,
                # E[a] = sum_r N[a, r] exp(hill_effect[r]).
                exposure = (
                    data.age_exposure
                    if hill_effect is None
                    else data.age_exposure @ jnp.exp(hill_effect)
                )
                log_cint = log_cint + safe_log(exposure)[:, None]

            alloc_probs = compute_allocation_probs(
                allocation.support,
                allocation.mode,
                allocation.age_group_counts,
                log_cint,
            )

        else:
            alloc_probs = allocation.weights

        alloc_probs = numpyro.deterministic("alloc_probs", alloc_probs)  # type: ignore
        log_cint = log_cint[:, :, None] + safe_log(alloc_probs)
        log_cint = logsumexp(jnp.where(allocation.group_active, log_cint, 0.0), axis=1)

        # log mu[i] = log sum_a w[i, a] m[a, did[i]] (+ hill_effect[rid[i]])
        log_mu = logsumexp(log_cint.T[data.did] + data.log_row_exposure, axis=1)
        if hill_effect is not None:
            log_mu = log_mu + hill_effect[data.rid]

        # Expected counts per row; read by downstream analyses (posterior["mu"]).
        mu = jnp.exp(log_mu)
        numpyro.deterministic("mu", mu)

        if self.likelihood == "negbin":
            inv_disp = numpyro.sample(
                "inv_disp", dist.Exponential(self.inv_disp_prior_rate)
            )
            concentration = numpyro.deterministic("concentration", 1.0 / inv_disp)
            obs_dist = dist.NegativeBinomial2(mean=mu, concentration=concentration)
        else:
            obs_dist = dist.Poisson(rate=mu)

        with numpyro.plate("data", mu.size):
            numpyro.sample("obs", obs_dist, obs=y)
