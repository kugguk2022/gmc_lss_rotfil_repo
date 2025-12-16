from __future__ import annotations

import math

# Gravitational constant in units convenient for v(R) fitting.
# If rho0 is in Msun / kpc^3 and R, Rc in kpc, then v is km/s with this G.
# Source: standard astrophysical unit conversion.
G_KPC_KMS2_PER_MSUN = 4.30091e-6  # kpc * (km/s)^2 / Msun

# Unit conversions
CM_PER_KPC = 3.0856775814913673e21
G_PER_MSUN = 1.98847e33


def rho_cgs_to_msun_per_kpc3(rho_g_cm3: float) -> float:
    """Convert density from g/cm^3 to Msun/kpc^3."""
    # (g/cm^3) * (kpc^3 / cm^3) * (Msun / g)
    return rho_g_cm3 * (CM_PER_KPC**3) / G_PER_MSUN


def rho_msun_per_kpc3_to_cgs(rho_msun_kpc3: float) -> float:
    """Convert density from Msun/kpc^3 to g/cm^3."""
    return rho_msun_kpc3 * G_PER_MSUN / (CM_PER_KPC**3)


def safe_arctan(x: float) -> float:
    # helper to keep mypy happy and avoid importing numpy here
    return math.atan(x)
