from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .io import load_galaxy_csv
from .metrics import compute_all_metrics, save_metrics
from .synth import SynthParams, generate_synthetic_catalog
from .rotation import v_pseudo_isothermal_cylinder_kms


def synth_main() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic filament+galaxy catalog.")
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    df = generate_synthetic_catalog(SynthParams(n=args.n), rng=rng)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


def run_main() -> None:
    p = argparse.ArgumentParser(description="Run the rotating-filament analysis pipeline.")
    p.add_argument("--galaxies", required=True, help="Input galaxy CSV")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--n-spine", type=int, default=100)
    p.add_argument("--omit-frac", type=float, default=0.05)
    p.add_argument("--n-tilt", type=int, default=2000)
    p.add_argument("--use-gmc", action="store_true")
    p.add_argument("--gmc-strength", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    cat = load_galaxy_csv(args.galaxies)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metrics = compute_all_metrics(
        cat,
        n_spine_iter=args.n_spine,
        omit_frac=args.omit_frac,
        n_tilt_iter=args.n_tilt,
        use_gmc=args.use_gmc,
        gmc_strength=args.gmc_strength,
        rng=rng,
    )

    save_metrics(metrics, outdir / "metrics.json")

    # Plot: rotation curve
    df = cat.df
    R_mpc = np.asarray(metrics["distance_uncertainty"]["d16_mpc"]) * 0  # placeholder for array size
    # recompute R using x,y from catalog (distance to PCA spine is in metrics? not stored)
    # For plotting, use the stored spine and direct recompute
    origin = np.array(metrics["spine"]["origin_mpc"], float)
    t = np.array(metrics["spine"]["tangent_hat"], float)
    pts = np.column_stack([cat.x_mpc, cat.y_mpc])
    centered = pts - origin
    s = centered @ t
    proj = np.outer(s, t)
    resid = centered - proj
    R_mpc = np.linalg.norm(resid, axis=1)

    v = np.abs(df["v_los_kms"].to_numpy(float))

    plt.figure()
    plt.scatter(R_mpc * 1000.0, v, s=20)
    # overlay fitted curve if available
    Rc = metrics["rotation_fit"]["Rc_kpc"]
    rho0 = metrics["rotation_fit"]["rho0_msun_kpc3"]
    if np.isfinite(Rc) and np.isfinite(rho0):
        xs = np.linspace(0, max(10.0, float(np.nanmax(R_mpc * 1000.0))), 200)
        ys = v_pseudo_isothermal_cylinder_kms(xs, rho0, Rc)
        plt.plot(xs, ys)
    plt.xlabel("R (kpc)")
    plt.ylabel("|v| (km/s)")
    plt.title("Rotation curve")
    plt.tight_layout()
    plt.savefig(outdir / "rotation_curve.png", dpi=150)
    plt.close()

    # Plot: tilt distribution
    tilt = np.array(metrics["alignment"]["tilt_median_distribution"], float)
    plt.figure()
    plt.hist(tilt[np.isfinite(tilt)], bins=30, range=(0, 1))
    plt.xlabel("median |cosψ| per iteration")
    plt.ylabel("count")
    plt.title("Tilt-degeneracy Monte Carlo")
    plt.tight_layout()
    plt.savefig(outdir / "tilt_distribution.png", dpi=150)
    plt.close()

    print(json.dumps({
        "median_abs_cospsi": metrics["alignment"]["median_abs_cospsi"],
        "tilt_peak_estimate": metrics["alignment"]["tilt_peak_estimate"],
        "Td": metrics["dynamical_temperature"]["Td"],
        "Rc_kpc": metrics["rotation_fit"]["Rc_kpc"],
        "rho0_msun_kpc3": metrics["rotation_fit"]["rho0_msun_kpc3"],
        "gmc": metrics.get("gmc"),
    }, indent=2))
