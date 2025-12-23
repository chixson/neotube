# FITSAlongFit / NEOTube

FITSAlongFit is the structured driver for the **NEOTube / SSO precovery** ecosystem: start with observations, build a probabilistic orbit/tube, query archives, download cutouts (science/reference/difference), and then stack/detect along candidate paths. The polite downloader is one piece of this stack, but the **primary mission** is to coordinate uncertainty-aware search plans + inference across archives such as ZTF, NOIRLab, and future services.

## Project overview

- **Observation ingestion** (`runs/ceres/obs.csv` is an example): capture `t_utc`, `ra_deg`, `dec_deg`, `sigma_arcsec`, and `site` (MPC code). All downstream steps rely on this canonical input.
- **Orbit posterior**: `neotube-fit` solves the orbit via LM, supports seeds from Horizons/observations/Gauss/attributable, and writes `posterior.{npz,json}` plus residuals so replicas can be sampled reproducibly.
- **Replica/tube workflow**: `neotube-replicas`, `neotube-propcloud`, and `neotube-tube` turn the posterior into spatial tubes (per-exposure center + radius quantiles) that safely describe where your object could be.
- **Archive planning**: `neotube-plan` and `neotube-ztf` (with ZOGY subtraction support) query ZTF/IRSA, download cutouts, and cache metadata for later inference.
- **Inference & diagnostics**: `neotube-infer` reweights replicas per exposure using matched-filter SNR statistics, and `neotube-diag`/`neotube-diag-overlay` visualize coverage and the replica-to-image alignment.

## Getting started

1. Create a venv and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

2. Prepare your observation CSV (a sample is `runs/ceres/obs.csv`). `sigma_arcsec` defaults to 0.5″ if absent, but the fitter logs problems when all measurements share the same error.
3. Run the CLI stack. The README below documents the full command chain, but the most critical commands are `neotube-fit`, `neotube-plan`, `neotube-ztf`, and `neotube-infer`.

## CLI workflow

Each command writes artifacts that feed the next stage. A high-level run looks like:

```bash
neotube-fit --obs runs/ceres/obs.csv --target 00001 --seed-method attributable --out-dir runs/ceres/fit_seed_attributable
neotube-replicas --posterior runs/ceres/fit_seed_attributable/posterior.npz --n 2000 --output runs/ceres/replicas_attributable.csv
neotube-propcloud --replicas runs/ceres/replicas_attributable.csv --meta runs/ceres/replicas_attributable_meta.json --times runs/ceres/times_first_cutout.csv --output runs/ceres/cloud_attributable.csv
neotube-tube --cloud runs/ceres/cloud_attributable.csv --cred 0.99 --margin-arcsec 20 --output runs/ceres/tube_attributable.csv
neotube-plan --tube runs/ceres/tube_attributable.csv --out runs/ceres/plan_attributable.csv --filter zr --slot-s 1800
neotube-ztf --plan runs/ceres/plan_attributable.csv --out runs/ceres/cutouts --size-arcsec 120 --user-agent 'neotube/0.1 (contact: chixson@fourshadows.org)'
neotube-infer --posterior-json runs/ceres/fit_seed_attributable/posterior.json --replicas runs/ceres/replicas_attributable.csv --cutouts-index runs/ceres/cutouts/cutouts_index.csv --out-dir runs/ceres/infer_attributable
```

## Tip highlights

- **Seed strategy**: `neotube-fit` now defaults to the attributable grid initializer; `--seed-method` lets you switch between Horizons/observation-based/Gauss seeds when appropriate.
- **Polite archive usage**: `neotube-plan` caches metadata, respects rate limits/Retry-After, and logs `Server-Timing` when available. `neotube-ztf` downloads science/reference/ZOGY outputs and records request IDs for observability.
- **ZOGY awareness**: When ZOGY difference cutouts exist, inference uses the matched-filter SNR maps to evaluate each replica’s predicted position.
- **Diagnostics**: `neotube-diag` compares planned centers vs. metadata, `neotube-diag-overlay` paints replica clouds on cutouts, and `runs/ceres/replica_spread_vs_jpl.png` and `attributable_spread_first_image.png` are concrete outputs you can inspect.

## Repo organization

- `neotube/`: Python package implementing clients, fit/infer logic, propagation, tube planning, and CLI anchors.
- `runs/`: example pipelines (Ceres, fit artifacts, cached cutouts, diagnostic plots, etc.). Big cutout directories are ignored via `.gitignore`.
- `docs/`: architectural notes plus CLI reference documentation for NEOTube steps.
- `deprecated/`: legacy SSOIS/Scout prototypes for historical reference.

## See also

- `docs/ARCHITECTURE.md` for a deep dive into the NEOTube pipeline shape.
- `runs/ceres/` for concrete experiments (fit outputs, propagation logs, cutouts, inference traces).
