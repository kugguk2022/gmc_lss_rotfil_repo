from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class SpinVectors:
    L_hat_xyz: np.ndarray  # (N,3)


def spin_unit_vector_xyz(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    pa_deg: np.ndarray,
    inc_deg: np.ndarray,
    *,
    assume_positive_Lr: bool = True,
) -> SpinVectors:
    """Compute galaxy spin unit vectors in Cartesian coords.

    Follows the paper's construction:
    - L_r = cos(i), L_theta = sin(i) * sin(PA), L_phi = sin(i) * cos(PA)
    - Convert spherical (r,theta,phi) to Cartesian using their matrix form.

    Notes:
    - There is a sign ambiguity in L_r; the paper takes the positive sign.
    """

    ra = np.radians(np.asarray(ra_deg, float))
    dec = np.radians(np.asarray(dec_deg, float))
    pa = np.radians(np.asarray(pa_deg, float))
    inc = np.radians(np.asarray(inc_deg, float))

    # paper notation
    alpha = np.pi / 2.0 - dec
    beta = ra

    Lr = np.cos(inc)
    if assume_positive_Lr:
        Lr = np.abs(Lr)

    Ltheta = np.sin(inc) * np.sin(pa)
    Lphi = np.sin(inc) * np.cos(pa)

    # Transformation matrix from the paper (Eq. 6)
    # [Lx]   [ sinα cosβ   cosα cosβ   -sinβ ] [Lr]
    # [Ly] = [ sinα sinβ   cosα sinβ    cosβ ] [Lθ]
    # [Lz]   [ cosα        -sinα        0    ] [Lφ]
    sin_a = np.sin(alpha)
    cos_a = np.cos(alpha)
    sin_b = np.sin(beta)
    cos_b = np.cos(beta)

    Lx = sin_a * cos_b * Lr + cos_a * cos_b * Ltheta - sin_b * Lphi
    Ly = sin_a * sin_b * Lr + cos_a * sin_b * Ltheta + cos_b * Lphi
    Lz = cos_a * Lr - sin_a * Ltheta

    L = np.column_stack([Lx, Ly, Lz]).astype(float)
    # normalize (should be unit-ish already)
    n = np.linalg.norm(L, axis=1)
    n = np.where(n == 0.0, 1.0, n)
    L /= n[:, None]

    return SpinVectors(L_hat_xyz=L)


def filament_vector_from_endpoints(
    x_mpc: np.ndarray,
    y_mpc: np.ndarray,
    d_mpc: np.ndarray,
    s_mpc: np.ndarray,
) -> np.ndarray:
    """Construct a 3D filament direction from endpoints.

    Uses (x,y,d) as a simple Cartesian representation of positions.
    The direction is (endpoint_max_s - endpoint_min_s) normalized.
    """

    x = np.asarray(x_mpc, float)
    y = np.asarray(y_mpc, float)
    d = np.asarray(d_mpc, float)
    s = np.asarray(s_mpc, float)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(d) & np.isfinite(s)
    if mask.sum() < 2:
        return np.array([np.nan, np.nan, np.nan])

    idx_min = int(np.nanargmin(s[mask]))
    idx_max = int(np.nanargmax(s[mask]))
    # careful: indices are within masked array; map back
    global_idx = np.flatnonzero(mask)
    i0 = global_idx[idx_min]
    i1 = global_idx[idx_max]

    p0 = np.array([x[i0], y[i0], d[i0]], dtype=float)
    p1 = np.array([x[i1], y[i1], d[i1]], dtype=float)
    v = p1 - p0
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return np.array([np.nan, np.nan, np.nan])
    return v / norm


def cos_psi(
    L_hat_xyz: np.ndarray,
    f_hat_xyz: np.ndarray,
) -> np.ndarray:
    """Cosine of angle between spin vectors and filament direction."""

    L = np.asarray(L_hat_xyz, float)
    f = np.asarray(f_hat_xyz, float)
    if f.shape != (3,):
        raise ValueError("f_hat_xyz must be shape (3,)")

    mask = np.isfinite(L).all(axis=1) & np.isfinite(f).all()
    out = np.full(L.shape[0], np.nan, dtype=float)
    if not np.isfinite(f).all():
        return out

    out[mask] = L[mask] @ f
    return out
