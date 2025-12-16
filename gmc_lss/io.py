from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .coords import FlatLCDM, project_ra_dec_to_tangent_plane_mpc


@dataclass
class GalaxyCatalog:
    df: pd.DataFrame
    x_mpc: np.ndarray
    y_mpc: np.ndarray
    d_mpc: np.ndarray


def load_galaxy_csv(
    path: str,
    *,
    cosmo: Optional[FlatLCDM] = None,
) -> GalaxyCatalog:
    """Load a galaxy catalog from CSV.

    Required columns:
      - ra_deg, dec_deg, z
      - pa_deg, inc_deg

    Optional:
      - x_mpc, y_mpc (if provided, we skip projection)
      - v_los_kms (if provided, used directly)
      - sample, m_star, m_hi
    """

    df = pd.read_csv(path)

    required = {"pa_deg", "inc_deg"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise ValueError(f"Missing required columns: {missing}")

    if {"x_mpc", "y_mpc", "z"}.issubset(df.columns):
        x = df["x_mpc"].to_numpy(float)
        y = df["y_mpc"].to_numpy(float)
        if "d_mpc" in df.columns:
            d = df["d_mpc"].to_numpy(float)
        elif "z" in df.columns:
            if cosmo is None:
                cosmo = FlatLCDM()
            d = np.array([cosmo.comoving_distance_mpc(float(zv)) for zv in df["z"].to_numpy(float)])
        else:
            d = np.full_like(x, np.nan)
    else:
        for col in ("ra_deg", "dec_deg", "z"):
            if col not in df.columns:
                raise ValueError(
                    "Provide either (x_mpc,y_mpc,z) or (ra_deg,dec_deg,z) columns"
                )
        x, y, d = project_ra_dec_to_tangent_plane_mpc(
            df["ra_deg"].to_numpy(float),
            df["dec_deg"].to_numpy(float),
            df["z"].to_numpy(float),
            cosmo=cosmo,
        )
        df = df.copy()
        df["x_mpc"] = x
        df["y_mpc"] = y
        df["d_mpc"] = d

    # derive a crude LOS velocity if absent (peculiar velocities ignored)
    if "v_los_kms" not in df.columns and "z" in df.columns:
        c_kms = 299792.458
        z0 = float(np.nanmedian(df["z"].to_numpy(float)))
        df = df.copy()
        df["v_los_kms"] = c_kms * (df["z"].to_numpy(float) - z0)

    if "sample" not in df.columns:
        df = df.copy()
        df["sample"] = "unknown"

    return GalaxyCatalog(df=df, x_mpc=np.asarray(x), y_mpc=np.asarray(y), d_mpc=np.asarray(d))


def save_catalog_csv(cat: GalaxyCatalog, path: str) -> None:
    cat.df.to_csv(path, index=False)
