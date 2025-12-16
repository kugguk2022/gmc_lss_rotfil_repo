from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .rotation import v_pseudo_isothermal_cylinder_kms


@dataclass
class SynthParams:
    n: int = 60
    spine_length_mpc: float = 10.0
    radius_mpc: float = 1.0
    z0: float = 0.03
    # rotation model params (roughly in the paper's ballpark)
    Rc_kpc: float = 50.0
    rho0_msun_kpc3: float = 2.0e5
    v_noise_frac: float = 0.20
    spin_alignment_strength: float = 0.75  # closer to 1 => more aligned


def generate_synthetic_catalog(
    params: SynthParams,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Generate a toy filament catalog.

    - straight filament in x
    - galaxies with radial offset y
    - v_los follows a pseudo-isothermal cylinder curve with sign set by side
    - spins biased to align with filament direction
    """

    if rng is None:
        rng = np.random.default_rng(0)

    n = int(params.n)
    # along-spine coordinate
    s = rng.uniform(-0.5, 0.5, size=n) * params.spine_length_mpc

    # radial distances (more near spine)
    r = np.abs(rng.normal(0.0, params.radius_mpc / 2.0, size=n))
    r = np.clip(r, 0.0, params.radius_mpc * 2.5)

    side = rng.choice([-1.0, 1.0], size=n)
    x = s
    y = side * r

    # pretend comoving distance (line of sight) ~ constant at low z
    d_mpc = np.full(n, 120.0)  # arbitrary, just to define a 3D endpoint vector

    # rotation curve gives speed vs radius
    R_kpc = 1000.0 * np.abs(y)
    v_mag = v_pseudo_isothermal_cylinder_kms(R_kpc, params.rho0_msun_kpc3, params.Rc_kpc)
    v = side * v_mag
    v = v + rng.normal(0.0, np.abs(v_mag) * params.v_noise_frac)

    # small sky patch: map x,y to ra/dec (degrees) for compatibility
    ra0, dec0 = 150.0, 2.0
    ra = ra0 + (x / 120.0) * (180.0 / np.pi)  # small-angle
    dec = dec0 + (y / 120.0) * (180.0 / np.pi)

    # build spins: aligned => PA such that spin points along x
    # This is not exact inverse mapping; it's just consistent enough for demos.
    # We'll set inclination moderate and PA near 0 for alignment, with noise.
    inc = rng.uniform(30.0, 75.0, size=n)
    pa = rng.normal(0.0, 15.0, size=n)  # degrees
    # weaken alignment by mixing with uniform PA
    mix = rng.random(n) > params.spin_alignment_strength
    pa[mix] = rng.uniform(0.0, 180.0, size=mix.sum())

    # make redshift consistent with v_los around z0 (non-relativistic)
    c_kms = 299792.458
    z = params.z0 + (v / c_kms)

    df = pd.DataFrame(
        {
            "ra_deg": ra,
            "dec_deg": dec,
            "z": z,
            "x_mpc": x,
            "y_mpc": y,
            "d_mpc": d_mpc,
            "pa_deg": pa,
            "inc_deg": inc,
            "v_los_kms": v,
            "sample": rng.choice(["optical", "HI"], size=n, p=[0.75, 0.25]),
        }
    )

    return df
