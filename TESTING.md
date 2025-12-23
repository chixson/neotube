# Testing Log

This repository now tracks the `neotube` pipeline, including the updated fitting and replica sampling workflow built around data-only seeds and Student-t inference.

- **Dense MOST span**: see `docs/ceres_most_ztf.md` for the list of ZTF exposures; the densest coverage window is documented there as **2018-04-03 11:45 UTC → 2018-04-04 06:00 UTC**. That span is currently the reference for all repeatable tests.
- **Test plan**: fetch the ten MPC observations starting at 2018-04-06 06:00 UTC (done via `python -m neotube.mpc_obs_cli --target 1 --start "2018-04-06T06:00:00" --n 10 --sigma-arcsec 0.5 --out runs/ceres/obs.csv`). Those rows will seed the attributable/Student-t pipeline, and every subsequent run in `runs/ceres/` should note this span and reference `docs/ceres_most_ztf.md`.

Continue capturing diagnostics (fit summaries, replica diagrams, inference traces) under `runs/ceres/` once the necessary MPC observations/cutouts exist.
