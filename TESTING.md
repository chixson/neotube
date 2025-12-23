# Testing Log

This repository now tracks the `neotube` pipeline, including the updated fitting and replica sampling workflow built around data-only seeds and Student-t inference.

- **Dense MOST span**: see `docs/ceres_most_ztf.md` for the list of ZTF exposures; the densest coverage window is documented there as **2018-04-03 11:45 UTC → 2018-04-04 06:00 UTC**. That span is currently the reference for all repeatable tests.
- **Test plan**: fetch the ten MPC observations starting at 2018-04-06 06:00 UTC (done via `python -m neotube.mpc_obs_cli --target 1 --start "2018-04-06T06:00:00" --n 10 --sigma-arcsec 0.5 --out runs/ceres/obs.csv`). Those rows will seed the attributable/Student-t pipeline (optionally with `--estimate-site-scales`), and every subsequent run in `runs/ceres/` should note this span and reference `docs/ceres_most_ztf.md`.
- **Fit characterization**: use `PYTHONPATH=neotube/src python scripts/fit_diagnostics.py --obs runs/ceres/obs.csv --posterior runs/ceres/fit_studentt/posterior.npz` to capture RMS/chi2 and per-observation normalized residuals.
- **JPL comparison**: use `PYTHONPATH=neotube/src python scripts/compare_jpl_posterior.py` to fetch a Horizons state (ecliptic -> ICRS) and compare posterior vs JPL, and `PYTHONPATH=neotube/src python scripts/diagnose_jpl_diff.py` to write per-observation residual plots/CSV under `runs/ceres/diagnostics_jpl_vs_post/`.

Continue capturing diagnostics (fit summaries, replica diagrams, inference traces) under `runs/ceres/` once the necessary MPC observations/cutouts exist.
