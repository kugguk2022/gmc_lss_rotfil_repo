from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FlatLCDM:
    """Very small cosmology helper.

    This is intentionally lightweight (no astropy). It is *good enough* for
    turning low-z angles into approximate comoving coordinates.

    For serious analysis you should swap this with astropy.cosmology.
    """

    H0_kms_Mpc: float = 70.0
    Omega_m: float = 0.3
    Omega_L: float = 0.7

    def E(self, z: float) -> float:
        return math.sqrt(self.Omega_m * (1 + z) ** 3 + self.Omega_L)

    def comoving_distance_mpc(self, z: float, n: int = 2048) -> float:
        # simple Simpson integration of c/H0 ∫ dz/E(z)
        c_kms = 299792.458
        z = max(0.0, float(z))
        if z == 0.0:
            return 0.0
        xs = np.linspace(0.0, z, n)
        ys = 1.0 / np.vectorize(self.E)(xs)
        h = z / (n - 1)
        # Simpson's rule requires odd number of samples; ensure n is odd
        if (n % 2) == 0:
            xs = np.linspace(0.0, z, n + 1)
            ys = 1.0 / np.vectorize(self.E)(xs)
            h = z / n
        s = ys[0] + ys[-1] + 4 * ys[1:-1:2].sum() + 2 * ys[2:-2:2].sum()
        integral = (h / 3.0) * s
        return (c_kms / self.H0_kms_Mpc) * integral


def project_ra_dec_to_tangent_plane_mpc(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    z: np.ndarray,
    ra0_deg: Optional[float] = None,
    dec0_deg: Optional[float] = None,
    cosmo: Optional[FlatLCDM] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project (ra,dec,z) to a local tangent plane in Mpc.

    Returns (x_mpc, y_mpc, d_mpc) where d_mpc is the comoving distance.

    Assumptions:
    - small field (gnomonic approximation)
    - low z (uses simple LCDM distance)
    """

    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    zz = np.asarray(z, dtype=float)

    if ra0_deg is None:
        ra0_deg = float(np.nanmedian(ra))
    if dec0_deg is None:
        dec0_deg = float(np.nanmedian(dec))

    ra0 = math.radians(ra0_deg)
    dec0 = math.radians(dec0_deg)

    ra_r = np.radians(ra)
    dec_r = np.radians(dec)

    # angular offsets
    dra = (ra_r - ra0) * np.cos(dec0)
    ddec = (dec_r - dec0)

    if cosmo is None:
        cosmo = FlatLCDM()

    d_mpc = np.array([cosmo.comoving_distance_mpc(float(zv)) for zv in zz], dtype=float)

    x = d_mpc * dra
    y = d_mpc * ddec
    return x, y, d_mpc
