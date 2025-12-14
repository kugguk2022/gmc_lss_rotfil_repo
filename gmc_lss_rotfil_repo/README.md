# GMC-LSS (Rotating Filament) — repo scaffold

This repo is a **practical scaffold** to test whether a model (SPT / Synaionikon / Driftedcz / etc.) can reproduce the key observables of a **rotating cosmic filament** + **spin–filament alignment** analysis.

It implements:
- Filament spine estimation (starter: PCA spine; swap-in DISPERSE later)
- Distance-to-spine + along-spine coordinate
- **Pseudo-isothermal cylinder** rotation curve fit (paper Eq. 13)
- **Dynamical temperature** \(T_d = \sigma_z / \Delta z_{AB}\)
- Spin vector construction from \(PA\), \(i\), \(RA\), \(Dec\)
- Tilt degeneracy handling (\(\pm \pi\) PA) with Monte Carlo re-assignments
- A simple, pluggable **GMC operator** interface (for torque / alignment updates)

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Quickstart (synthetic demo)

```bash
# 1) generate a synthetic filament+galaxy catalog
gmc-lss-synth --out data/synth_galaxies.csv --n 60

# 2) run the pipeline and print summary metrics
gmc-lss-run --galaxies data/synth_galaxies.csv --outdir examples/out
```

Outputs:
- `examples/out/metrics.json`
- `examples/out/rotation_fit.json`
- `examples/out/alignment.json`
- diagnostic plots

## Input schema (CSV)

Required columns:
- `ra_deg`, `dec_deg`, `z`  (or you can pass pre-projected `x_mpc`,`y_mpc`)
- `pa_deg` (position angle)
- `inc_deg` (inclination)

Optional:
- `v_los_kms` (line-of-sight velocity residual). If missing, we estimate from `z`.
- `sample` (e.g., `HI` or `optical`)
- `m_star`, `m_hi`

See `data/example_galaxies.csv`.

## What “paper tests” we target

The code is structured so you can drop in real data and test:
- alignment peaks of \(|\cos\psi|\) after tilt randomization
- rotation dominance via \(T_d\)
- rotation curve parameters \(R_c\), \(\rho_0\)
- density-dependent misalignment

These correspond to definitions and reported values in the rotating-filament paper:
- \(T_d\) definition and Eq. 13 rotation curve
- 20% velocity uncertainty + spine jackknife distance errors
- alignment peak values after 2000 tilt iterations

## Repo layout

```
.
├── gmc_lss/
│   ├── cli.py
│   ├── io.py
│   ├── coords.py
│   ├── spine.py
│   ├── rotation.py
│   ├── spin.py
│   ├── alignment.py
│   ├── gmc.py
│   └── metrics.py
├── tests/
├── data/
└── examples/
```

## Extending to “real” DISPERSE

The PCA spine is a placeholder. To match the paper more closely:
- Replace `gmc_lss.spine.fit_spine_pca()` with a DISPERSE wrapper
- Use jackknife resampling of galaxies to regenerate spine realizations
- Use the 16th/84th percentiles of distance-to-spine across realizations as \(\sigma_{d}\)

## License

MIT.
