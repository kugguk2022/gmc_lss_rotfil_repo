from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class SpineResult:
    """A filament spine representation.

    This scaffold uses a **linear PCA spine** by default. Swap this
    for a DISPERSE-based skeleton once you wire it in.
    """

    origin_mpc: np.ndarray  # (2,)
    tangent_hat: np.ndarray  # (2,)
    s_mpc: np.ndarray  # (N,) projected coordinate along spine
    d_perp_mpc: np.ndarray  # (N,) perpendicular distance to spine


def fit_spine_pca(x_mpc: np.ndarray, y_mpc: np.ndarray) -> SpineResult:
    """Fit a straight spine via 2D PCA.

    Returns a best-fit line through the point cloud.
    """

    pts = np.column_stack([np.asarray(x_mpc, float), np.asarray(y_mpc, float)])
    origin = np.nanmean(pts, axis=0)
    centered = pts - origin
    # covariance
    c = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(c)
    direction = eigvecs[:, np.argmax(eigvals)]
    direction = direction / np.linalg.norm(direction)

    s = centered @ direction
    # perpendicular distance using projection residual
    proj = np.outer(s, direction)
    resid = centered - proj
    d_perp = np.linalg.norm(resid, axis=1)

    return SpineResult(origin_mpc=origin, tangent_hat=direction, s_mpc=s, d_perp_mpc=d_perp)


def jackknife_spines(
    x_mpc: np.ndarray,
    y_mpc: np.ndarray,
    n_iter: int = 100,
    omit_frac: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> List[SpineResult]:
    """Jackknife resampling of the spine.

    The paper regenerates the filament many times by omitting a small fraction
    of galaxies; we mimic that here.
    """

    if rng is None:
        rng = np.random.default_rng(0)

    n = len(x_mpc)
    k = max(1, int(round(omit_frac * n)))
    idx_all = np.arange(n)
    results: List[SpineResult] = []
    for _ in range(int(n_iter)):
        omit = rng.choice(idx_all, size=k, replace=False)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[omit] = False
        res = fit_spine_pca(np.asarray(x_mpc)[keep_mask], np.asarray(y_mpc)[keep_mask])
        # re-evaluate s,d for full sample using res origin/tangent
        pts = np.column_stack([np.asarray(x_mpc, float), np.asarray(y_mpc, float)])
        centered = pts - res.origin_mpc
        s = centered @ res.tangent_hat
        proj = np.outer(s, res.tangent_hat)
        resid = centered - proj
        d_perp = np.linalg.norm(resid, axis=1)
        results.append(
            SpineResult(
                origin_mpc=res.origin_mpc,
                tangent_hat=res.tangent_hat,
                s_mpc=s,
                d_perp_mpc=d_perp,
            )
        )
    return results


def distance_uncertainty_from_spines(spines: Iterable[SpineResult]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (d16, d84) per galaxy from multiple spine realizations."""

    ds = np.stack([s.d_perp_mpc for s in spines], axis=0)  # (K,N)
    d16 = np.percentile(ds, 16, axis=0)
    d84 = np.percentile(ds, 84, axis=0)
    return d16, d84
