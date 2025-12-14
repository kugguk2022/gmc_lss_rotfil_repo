from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit

from .constants import G_KPC_KMS2_PER_MSUN


@dataclass
class RotationFitResult:
    Rc_kpc: float
    rho0_msun_kpc3: float
    Rc_kpc_ci: Tuple[float, float]
    rho0_msun_kpc3_ci: Tuple[float, float]
    n_success: int


def v_pseudo_isothermal_cylinder_kms(
    R_kpc: np.ndarray,
    rho0_msun_kpc3: float,
    Rc_kpc: float,
) -> np.ndarray:
    """Paper Eq. (13) pseudo-isothermal cylinder rotation curve.

    v(R) = sqrt(4*pi*G*rho0*Rc^2 * (1 - (Rc/R) * arctan(R/Rc)))

    Inputs:
      - R_kpc: radius from spine (kpc)
      - rho0_msun_kpc3: central density (Msun/kpc^3)
      - Rc_kpc: core radius (kpc)

    Returns:
      - v_kms: km/s
    """

    R = np.asarray(R_kpc, dtype=float)
    Rc = float(max(1e-6, Rc_kpc))
    rho0 = float(max(1e-30, rho0_msun_kpc3))

    # avoid division by zero at R=0
    R_safe = np.where(R == 0.0, 1e-6, R)
    term = 1.0 - (Rc / R_safe) * np.arctan(R_safe / Rc)
    term = np.clip(term, 0.0, None)
    v2 = 4.0 * np.pi * G_KPC_KMS2_PER_MSUN * rho0 * (Rc**2) * term
    return np.sqrt(np.clip(v2, 0.0, None))


def fit_rotation_curve_mc(
    R_mpc: np.ndarray,
    v_kms: np.ndarray,
    *,
    sigma_R_mpc: Optional[np.ndarray] = None,
    sigma_v_frac: float = 0.20,
    n_mc: int = 500,
    rng: Optional[np.random.Generator] = None,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((1.0, 1e-4), (2000.0, 1e12)),
) -> RotationFitResult:
    """Fit Eq. (13) with simple Monte Carlo error propagation.

    The paper uses ROXY with X/Y errors and adopts ~20% velocity uncertainty.
    Here we emulate that with MC sampling + curve_fit.

    bounds are (Rc_kpc, rho0_msun_kpc3) lower/upper.
    """

    if rng is None:
        rng = np.random.default_rng(0)

    R_mpc = np.asarray(R_mpc, float)
    v_kms = np.asarray(v_kms, float)

    mask = np.isfinite(R_mpc) & np.isfinite(v_kms)
    R_mpc = R_mpc[mask]
    v_kms = v_kms[mask]

    if sigma_R_mpc is None:
        sigma_R_mpc = np.full_like(R_mpc, 0.05)  # a small default
    else:
        sigma_R_mpc = np.asarray(sigma_R_mpc, float)[mask]

    R_kpc = 1000.0 * R_mpc
    sig_R_kpc = 1000.0 * sigma_R_mpc

    def model(R_kpc_arr: np.ndarray, Rc_kpc: float, rho0_msun_kpc3: float) -> np.ndarray:
        return v_pseudo_isothermal_cylinder_kms(R_kpc_arr, rho0_msun_kpc3=rho0_msun_kpc3, Rc_kpc=Rc_kpc)

    # initial guess (rough)
    p0 = (50.0, 1e6)

    Rc_samps = []
    rho_samps = []

    n_success = 0
    for _ in range(int(n_mc)):
        R_draw = R_kpc + rng.normal(0.0, sig_R_kpc)
        v_draw = v_kms + rng.normal(0.0, np.abs(v_kms) * sigma_v_frac)
        # enforce positive radii
        R_draw = np.clip(R_draw, 1e-3, None)
        v_draw = np.clip(v_draw, 0.0, None)
        try:
            popt, _pcov = curve_fit(
                model,
                R_draw,
                v_draw,
                p0=p0,
                bounds=bounds,
                maxfev=20000,
            )
            Rc_fit, rho_fit = float(popt[0]), float(popt[1])
            Rc_samps.append(Rc_fit)
            rho_samps.append(rho_fit)
            n_success += 1
        except Exception:
            continue

    if n_success < max(10, n_mc // 10):
        # too few successful fits; return NaNs with count
        return RotationFitResult(
            Rc_kpc=float("nan"),
            rho0_msun_kpc3=float("nan"),
            Rc_kpc_ci=(float("nan"), float("nan")),
            rho0_msun_kpc3_ci=(float("nan"), float("nan")),
            n_success=n_success,
        )

    Rc_arr = np.asarray(Rc_samps)
    rho_arr = np.asarray(rho_samps)

    Rc_med = float(np.median(Rc_arr))
    rho_med = float(np.median(rho_arr))
    Rc_ci = (float(np.percentile(Rc_arr, 16)), float(np.percentile(Rc_arr, 84)))
    rho_ci = (float(np.percentile(rho_arr, 16)), float(np.percentile(rho_arr, 84)))

    return RotationFitResult(
        Rc_kpc=Rc_med,
        rho0_msun_kpc3=rho_med,
        Rc_kpc_ci=Rc_ci,
        rho0_msun_kpc3_ci=rho_ci,
        n_success=n_success,
    )


def signed_perp_distance(
    xy_mpc: np.ndarray,
    origin_mpc: np.ndarray,
    tangent_hat: np.ndarray,
) -> np.ndarray:
    """Signed perpendicular distance to a 2D spine."""

    pts = np.asarray(xy_mpc, float)
    origin = np.asarray(origin_mpc, float)
    t = np.asarray(tangent_hat, float)
    t = t / np.linalg.norm(t)

    # perpendicular unit vector (rotate 90 degrees)
    n = np.array([-t[1], t[0]], dtype=float)
    centered = pts - origin
    return centered @ n


def dynamical_temperature_from_sides(
    z: np.ndarray,
    signed_d_perp: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    """Compute Td = sigma_z / Delta z_AB.

    zA is mean z for receding side; zB is mean z for approaching side.
    This implementation uses signed perpendicular distance as the side label.
    """

    z = np.asarray(z, float)
    sd = np.asarray(signed_d_perp, float)

    mask = np.isfinite(z) & np.isfinite(sd)
    z = z[mask]
    sd = sd[mask]

    if len(z) < 4:
        return float("nan"), {"sigma_z": float("nan"), "delta_z_ab": float("nan")}

    z0 = float(np.mean(z))
    sigma_z = float(np.sqrt(np.mean((z - z0) ** 2)))

    zA = float(np.mean(z[sd >= 0])) if np.any(sd >= 0) else float("nan")
    zB = float(np.mean(z[sd < 0])) if np.any(sd < 0) else float("nan")
    delta = zA - zB

    Td = sigma_z / delta if (delta != 0 and np.isfinite(delta)) else float("nan")

    return float(Td), {"sigma_z": sigma_z, "delta_z_ab": float(delta), "zA": zA, "zB": zB}
