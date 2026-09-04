"""Standard posterior summary container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ContactSummary:
    """Point estimate and equal-tailed credible interval for one quantity."""

    lower: NDArray[np.float64]
    central: NDArray[np.float64]
    upper: NDArray[np.float64]
    measure: Literal["mean", "median"]
    credible_level: float

    def to_array(self) -> NDArray[np.float64]:
        """Return ``(lower, central, upper)`` stacked along a new first axis."""
        return np.stack((self.lower, self.central, self.upper), axis=0)
