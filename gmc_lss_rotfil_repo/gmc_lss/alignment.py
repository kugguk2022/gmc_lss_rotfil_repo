from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .spine import SpineResult
from .spin import cos_psi, filament_vector_from_endpoints, spin_unit_vector_xyz


@dataclass
class AlignmentResult:
    abs_cospsi_per_galaxy: np.ndarray  # (N,)
    abs_cospsi_std_per_galaxy: np.ndarray  # (N,)
    median_abs_cospsi: float


def compute_abs_cospsi_with_filament_uncertainty(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    pa_deg: np.ndarray,
    inc_deg: np.ndarray,
    x_mpc: np.ndarray,
    y_mpc: np.ndarray,
    d_mpc: np.ndarray,
    spines: List[SpineResult],
    *,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-galaxy |cosψ| with spine uncertainty.

    For each spine realization:
      - build filament direction from endpoints
      - compute spin vector (no tilt randomization here)
      - compute |cosψ|

    Returns per-galaxy (median, std) across spine realizations.
    """

    if rng is None:
        rng = np.random.default_rng(0)

    L = spin_unit_vector_xyz(ra_deg, dec_deg, pa_deg, inc_deg).L_hat_xyz

    vals = []
    for sp in spines:
        f = filament_vector_from_endpoints(x_mpc, y_mpc, d_mpc, sp.s_mpc)
        c = cos_psi(L, f)
        vals.append(np.abs(c))

    A = np.stack(vals, axis=0)  # (K,N)
    med = np.nanmedian(A, axis=0)
    std = np.nanstd(A, axis=0)
    return med, std


def tilt_degeneracy_median_distribution(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    pa_deg: np.ndarray,
    inc_deg: np.ndarray,
    x_mpc: np.ndarray,
    y_mpc: np.ndarray,
    d_mpc: np.ndarray,
    spine: SpineResult,
    *,
    n_iter: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Monte Carlo over the ±π PA degeneracy.

    Each iteration:
      - independently flip each galaxy's PA by +180° with p=0.5
      - compute |cosψ|
      - record the median |cosψ| across the sample

    Returns array of length n_iter.
    """

    if rng is None:
        rng = np.random.default_rng(0)

    pa = np.asarray(pa_deg, float)
    f = filament_vector_from_endpoints(x_mpc, y_mpc, d_mpc, spine.s_mpc)

    meds = np.empty(int(n_iter), dtype=float)
    for i in range(int(n_iter)):
        flip = rng.random(pa.shape[0]) < 0.5
        pa_i = pa.copy()
        pa_i[flip] = pa_i[flip] + 180.0
        L = spin_unit_vector_xyz(ra_deg, dec_deg, pa_i, inc_deg).L_hat_xyz
        c = np.abs(cos_psi(L, f))
        meds[i] = float(np.nanmedian(c))

    return meds


def summarize_alignment(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    pa_deg: np.ndarray,
    inc_deg: np.ndarray,
    x_mpc: np.ndarray,
    y_mpc: np.ndarray,
    d_mpc: np.ndarray,
    spines: List[SpineResult],
    *,
    n_tilt_iter: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """Compute alignment products used in the paper.

    Returns a JSON-serializable dict.
    """

    if rng is None:
        rng = np.random.default_rng(0)

    # spine uncertainty -> per-galaxy med/std
    med_gal, std_gal = compute_abs_cospsi_with_filament_uncertainty(
        ra_deg,
        dec_deg,
        pa_deg,
        inc_deg,
        x_mpc,
        y_mpc,
        d_mpc,
        spines,
        rng=rng,
    )

    # tilt degeneracy -> distribution of sample medians (using the first spine)
    tilt_meds = tilt_degeneracy_median_distribution(
        ra_deg,
        dec_deg,
        pa_deg,
        inc_deg,
        x_mpc,
        y_mpc,
        d_mpc,
        spines[0],
        n_iter=n_tilt_iter,
        rng=rng,
    )

    # a crude "peak" estimate: mode of histogram
    hist, edges = np.histogram(tilt_meds[np.isfinite(tilt_meds)], bins=30, range=(0, 1))
    peak_bin = int(np.argmax(hist))
    peak = float(0.5 * (edges[peak_bin] + edges[peak_bin + 1]))

    return {
        "per_galaxy_abs_cospsi_median": med_gal.tolist(),
        "per_galaxy_abs_cospsi_std": std_gal.tolist(),
        "median_abs_cospsi": float(np.nanmedian(med_gal)),
        "tilt_median_distribution": tilt_meds.tolist(),
        "tilt_peak_estimate": peak,
        "tilt_peak_ci_approx": [
            float(np.nanpercentile(tilt_meds, 16)),
            float(np.nanpercentile(tilt_meds, 84)),
        ],
    }
