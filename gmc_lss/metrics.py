from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .alignment import summarize_alignment
from .gmc import SimpleTorqueGMC
from .io import GalaxyCatalog
from .rotation import (
    dynamical_temperature_from_sides,
    fit_rotation_curve_mc,
    signed_perp_distance,
)
from .spine import distance_uncertainty_from_spines, fit_spine_pca, jackknife_spines
from .spin import filament_vector_from_endpoints, spin_unit_vector_xyz, cos_psi


def compute_all_metrics(
    cat: GalaxyCatalog,
    *,
    n_spine_iter: int = 100,
    omit_frac: float = 0.05,
    n_tilt_iter: int = 2000,
    use_gmc: bool = False,
    gmc_strength: float = 0.15,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """Compute the main observables used in the rotating-filament analysis.

    Returns a JSON-serializable dict.
    """

    if rng is None:
        rng = np.random.default_rng(0)

    df = cat.df
    x = cat.x_mpc
    y = cat.y_mpc
    d = cat.d_mpc

    # 1) spine + jackknife uncertainty
    base_spine = fit_spine_pca(x, y)
    spines = [base_spine] + jackknife_spines(x, y, n_iter=n_spine_iter, omit_frac=omit_frac, rng=rng)
    d16, d84 = distance_uncertainty_from_spines(spines)
    sigma_R = 0.5 * (d84 - d16)

    # 2) rotation curve fit: use |perp distance| as radius, v_los as relative velocity
    R_mpc = base_spine.d_perp_mpc
    v = df["v_los_kms"].to_numpy(float)
    rot_fit = fit_rotation_curve_mc(R_mpc, np.abs(v), sigma_R_mpc=sigma_R, sigma_v_frac=0.20, n_mc=500, rng=rng)

    # 3) dynamical temperature Td = sigma_z / delta z_AB
    sd = signed_perp_distance(np.column_stack([x, y]), base_spine.origin_mpc, base_spine.tangent_hat)
    Td, Td_parts = dynamical_temperature_from_sides(df["z"].to_numpy(float), sd)

    # 4) alignment products
    align = summarize_alignment(
        df.get("ra_deg", np.zeros_like(x)).to_numpy(float),
        df.get("dec_deg", np.zeros_like(y)).to_numpy(float),
        df["pa_deg"].to_numpy(float),
        df["inc_deg"].to_numpy(float),
        x,
        y,
        d,
        spines,
        n_tilt_iter=n_tilt_iter,
        rng=rng,
    )

    # 5) optional GMC pass: does alignment improve?
    gmc_report = None
    if use_gmc:
        f_hat = filament_vector_from_endpoints(x, y, d, base_spine.s_mpc)
        L0 = spin_unit_vector_xyz(
            df.get("ra_deg", np.zeros_like(x)).to_numpy(float),
            df.get("dec_deg", np.zeros_like(y)).to_numpy(float),
            df["pa_deg"].to_numpy(float),
            df["inc_deg"].to_numpy(float),
        ).L_hat_xyz
        c0 = np.abs(cos_psi(L0, f_hat))
        op = SimpleTorqueGMC(strength=gmc_strength)
        L1 = op.apply(L0, f_hat, dt=1.0)
        c1 = np.abs(cos_psi(L1, f_hat))
        gmc_report = {
            "gmc_strength": float(gmc_strength),
            "median_abs_cospsi_before": float(np.nanmedian(c0)),
            "median_abs_cospsi_after": float(np.nanmedian(c1)),
        }

    return {
        "spine": {
            "origin_mpc": base_spine.origin_mpc.tolist(),
            "tangent_hat": base_spine.tangent_hat.tolist(),
        },
        "distance_uncertainty": {
            "d16_mpc": d16.tolist(),
            "d84_mpc": d84.tolist(),
        },
        "rotation_fit": {
            "Rc_kpc": rot_fit.Rc_kpc,
            "Rc_kpc_ci": list(rot_fit.Rc_kpc_ci),
            "rho0_msun_kpc3": rot_fit.rho0_msun_kpc3,
            "rho0_msun_kpc3_ci": list(rot_fit.rho0_msun_kpc3_ci),
            "n_success": rot_fit.n_success,
        },
        "dynamical_temperature": {
            "Td": Td,
            **Td_parts,
        },
        "alignment": align,
        "gmc": gmc_report,
    }


def save_metrics(metrics: Dict[str, object], outpath: str | Path) -> None:
    outpath = Path(outpath)
    outpath.write_text(json.dumps(metrics, indent=2))
