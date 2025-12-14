"""GMC-LSS rotating-filament toolkit.

This package is designed as a **scaffold**: it makes the paper's observables
computable, so future physics/symbolic models can be judged by pass/fail tests.

Main entrypoints:
- CLI: `gmc-lss-synth`, `gmc-lss-run`
- Python: see `gmc_lss.metrics.compute_all_metrics`
"""

from .metrics import compute_all_metrics

__all__ = ["compute_all_metrics"]
