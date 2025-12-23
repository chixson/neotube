# Testing Log

This repository now tracks the `neotube` pipeline, including the updated fitting and replica sampling workflow built around data-only seeds and Student-t inference.

- **Dense MOST span**: see `docs/ceres_most_ztf.md` for the list of ZTF exposures; the densest coverage window is documented there as **2018-04-03 11:45 UTC → 2018-04-04 06:00 UTC**. That span is currently the reference for all repeatable tests.
- **Test plan**: target the ten MPC observations taken starting two days after that window (2018-04-06 06:00 UTC) and progressively move forward in time (the table will be used for data-only seed comparisons). Each run in `runs/ceres/` should cite this span and reference `docs/ceres_most_ztf.md`.

Continue capturing diagnostics (fit summaries, replica diagrams, inference traces) under `runs/ceres/` once the necessary MPC observations/cutouts exist.
