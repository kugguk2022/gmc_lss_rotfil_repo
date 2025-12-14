from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class GMCOperator(Protocol):
    """A minimal interface for your GMC layer.

    The idea: GMC acts as a **connector/mutator** between a noisy point process
    and a coherent filamentary/rotational state.

    This repo does not assume the internal GMC math; it just defines the hook.
    """

    def apply(
        self,
        spins_xyz: np.ndarray,
        filament_hat_xyz: np.ndarray,
        *,
        dt: float = 1.0,
    ) -> np.ndarray:
        """Return updated spins.

        Args:
          spins_xyz: (N,3) unit vectors
          filament_hat_xyz: (3,) unit filament direction
          dt: step size
        """


@dataclass
class SimpleTorqueGMC:
    """Toy GMC: nudges spins toward the filament direction.

    This is not meant to be physically correct. It's a *pluggable placeholder*
    so you can test 'does adding GMC increase alignment metrics?'.
    """

    strength: float = 0.15

    def apply(self, spins_xyz: np.ndarray, filament_hat_xyz: np.ndarray, *, dt: float = 1.0) -> np.ndarray:
        s = np.asarray(spins_xyz, float)
        f = np.asarray(filament_hat_xyz, float)
        f = f / (np.linalg.norm(f) + 1e-12)

        # move along geodesic on the sphere: s <- normalize((1-a)s + a f)
        a = float(np.clip(self.strength * dt, 0.0, 1.0))
        out = (1.0 - a) * s + a * f[None, :]
        n = np.linalg.norm(out, axis=1)
        n = np.where(n == 0.0, 1.0, n)
        out = out / n[:, None]
        return out
