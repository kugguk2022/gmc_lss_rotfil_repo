import sys
from pathlib import Path

import numpy as np

# allow running tests without installation
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from gmc_lss.synth import SynthParams, generate_synthetic_catalog
from gmc_lss.io import GalaxyCatalog
from gmc_lss.metrics import compute_all_metrics


def test_end_to_end_synthetic():
    rng = np.random.default_rng(123)
    df = generate_synthetic_catalog(SynthParams(n=80, spin_alignment_strength=0.85), rng=rng)
    cat = GalaxyCatalog(df=df, x_mpc=df["x_mpc"].to_numpy(float), y_mpc=df["y_mpc"].to_numpy(float), d_mpc=df["d_mpc"].to_numpy(float))

    metrics = compute_all_metrics(cat, n_spine_iter=20, n_tilt_iter=200, rng=rng)

    assert 0.0 <= metrics["alignment"]["median_abs_cospsi"] <= 1.0
    # on this synthetic setup the tilt distribution should show strong alignment
    assert metrics["alignment"]["tilt_peak_estimate"] > 0.55


def test_rotation_curve_positive():
    rng = np.random.default_rng(42)
    df = generate_synthetic_catalog(SynthParams(n=100), rng=rng)
    cat = GalaxyCatalog(df=df, x_mpc=df["x_mpc"].to_numpy(float), y_mpc=df["y_mpc"].to_numpy(float), d_mpc=df["d_mpc"].to_numpy(float))
    metrics = compute_all_metrics(cat, n_spine_iter=10, n_tilt_iter=50, rng=rng)

    Rc = metrics["rotation_fit"]["Rc_kpc"]
    rho0 = metrics["rotation_fit"]["rho0_msun_kpc3"]
    assert np.isfinite(Rc) and Rc > 0
    assert np.isfinite(rho0) and rho0 > 0
