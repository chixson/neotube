# NEOTube CLI Bundle

This package implements the complete NEOTube / SSO precovery pipeline: ingest observations, solve for an orbit posterior (with Horizons/observation/Gauss/attributable seeds), sample replicas, propagate them across exposures, compress them into tube nodes, query ZTF IRSA (with optional ZOGY subtraction), and run matched-filter inference to flag strong candidates.

## Quick orientation

- **Observation ingestion**: input CSV must include `t_utc`, `ra_deg`, `dec_deg`, `sigma_arcsec`, `site` (MPC code). Example: `runs/ceres/obs.csv`.
- **Orbit posterior**: `neotube-fit` produces `posterior.npz/json`, logs residuals, supports deterministic seeds (`horizons`, `observations`, `gauss`, `attributable`), and can estimate per-site sigma scales (`--estimate-site-scales`).
- **Replica/tube**: `neotube-replicas`, `neotube-propcloud`, and `neotube-tube` convert the posterior into per-exposure tube nodes that drive planning.
- **Archive planning**: `neotube-plan` plus `neotube-ztf` fetch metadata + science/reference/ZOGY cutouts while respecting IRSA rate limits and caching.
- **Inference**: `neotube-infer` reweights replicas using matched-filter SNR maps (ZOGY when available), and the diagnostic CLIs expose coverage plots and overlays.
- **Fit diagnostics**: `scripts/fit_diagnostics.py` prints RMS/chi2 and per-observation residuals from an observation CSV + posterior.
- **Replica diagnostics**: `scripts/replica_spread.py` now emits three plots by default: RA/Dec, PCA1 vs PCA2, and PCA1 vs heliocentric distance (AU). Use `--jpl-ra/--jpl-dec/--jpl-r-au` to crosshair JPL.

## Run example (Ceres)

```bash
neotube-fit --obs runs/ceres/obs.csv --target 00001 --seed-method attributable --out-dir runs/ceres/fit_seed_attributable --estimate-site-scales
neotube-replicas --posterior runs/ceres/fit_seed_attributable/posterior.npz --n 2000 --output runs/ceres/replicas_attributable.csv
neotube-propcloud --replicas runs/ceres/replicas_attributable.csv --meta runs/ceres/replicas_attributable_meta.json --times runs/ceres/times_first_cutout.csv --output runs/ceres/cloud_attributable.csv
neotube-tube --cloud runs/ceres/cloud_attributable.csv --cred 0.99 --margin-arcsec 20 --output runs/ceres/tube_attributable.csv
neotube-plan --tube runs/ceres/tube_attributable.csv --out runs/ceres/plan_attributable.csv --filter zr --slot-s 1800
neotube-ztf --plan runs/ceres/plan_attributable.csv --out runs/ceres/cutouts --size-arcsec 120 --user-agent 'neotube/0.1 (contact: chixson@fourshadows.org)'
neotube-infer --posterior-json runs/ceres/fit_seed_attributable/posterior.json --replicas runs/ceres/replicas_attributable.csv --cutouts-index runs/ceres/cutouts/cutouts_index.csv --out-dir runs/ceres/infer_attributable
```

Quick fit check:

```bash
PYTHONPATH=neotube/src python scripts/fit_diagnostics.py \
  --obs runs/ceres/obs.csv \
  --posterior runs/ceres/fit_seed_attributable/posterior.npz
```

## Philosophy

- **Seed flexibility**: use the attributable grid by default, or switch to Horizons/observations/Gauss seeds via `--seed-method`.
- **Replica/tube clarity**: propagate once per replica, compute tube nodes (time + radius), and use them for targeted metadata queries.
- **Polite IRSA usage**: `neotube-plan` caches requests, respects rate limits, and logs request IDs/`Server-Timing`; `neotube-ztf` exposes the downloaded cutouts and ZOGY outputs.
- **Inference diagnostics**: overlay replica clouds via `neotube-diag-overlay`, inspect spreads with the provided plots, and monitor evidence logs in `runs/ceres/`.

## Development notes

- `pyproject.toml` declares the CLI entry points.
- Install dev dependencies with `pip install -e .[dev]`. Tests rely on `pytest`, formatting on `black`, linting on `ruff`.
*** End Patch***  
