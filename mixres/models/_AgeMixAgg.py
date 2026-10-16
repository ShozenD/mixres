"""Fine-age Poisson contact model observed through age-group aggregation."""

from typing import Optional

import jax.numpy as jnp
import numpyro
from jax.typing import ArrayLike

from ._AgeMixAggBase import AgeMixAggBase
from ._aggregation import log_expected_volume


class AgeMixAgg(AgeMixAggBase):
    """
    Fine-age Poisson contact model observed through age-group aggregation.

    Mathematical model
    ------------------
    ::

        log gamma[a, b] = baseline + f(a, b)
        log mu[c, d]    = log sum_{a in c} sum_{b in d} P[a] gamma[a, b] P[b]
                          - log P[c] + log N[c]
        y[c, d]         ~ Poisson(mu[c, d])

    By default (``allocation="binary"``) the allocation weights are the
    binary ``comp_map_cnt``. An age that falls in ``k`` overlapping recorded
    groups therefore contributes its full contact volume to each of them, i.e.
    it is counted ``k`` times. Use ``allocation="uniform"`` (or
    :class:`AgeMixAggReported`) when recorded groups overlap.

    See :class:`AgeMixAggBase` for the constructor parameters.
    """

    def model(self, y: Optional[ArrayLike] = None) -> None:
        log_gamma = self._sample_log_rate()
        log_EZ = numpyro.deterministic(
            "log_EZ",
            log_expected_volume(
                log_gamma, self.log_P, self.allocation_weights, self.comp_map_resp
            ),
        )
        log_mu = numpyro.deterministic(
            "log_mu",
            log_EZ + self.log_N_c[:, None] - self.log_P_respondent[:, None],
        )
        mu = numpyro.deterministic("mu", jnp.exp(log_mu))
        self._sample_obs(mu, y)
