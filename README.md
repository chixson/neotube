# NEOTube / SSO Precovery

NEOTube is the NEA-focused precovery toolchain sitting on top of IRSA ZTF and the broader SSO archive ecosystem. It implements polite metadata queries, cutout downloads, replica propagation, and pixel-level inference to help you find archival detections of Solar System objects.

## Getting started

1. Create an environment and install dependencies.

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

2. Prep your observations (CSV with `t_utc, ra_deg, dec_deg, sigma_arcsec, site`), or use the provided `runs/ceres/obs.csv`.
3. Run the CLI suite (fit → replicas → propagation → tube → plan → ztf → infer) in order; each stage writes JSON/CSV artifacts describing its inputs/outputs.

## CLI summary

| Command | Purpose |
| --- | --- |
| `neotube-fit` | Walks MPC-style astrometry to an orbit posterior, optionally seeding from Horizons, observations, Gauss, or the new attributable grid. |
| `neotube-replicas` | Samples replicas from the posterior and optionally annotates them with RA/Dec. |
| `neotube-propcloud` | Propagates replicas through requested times (few exposures) with multiprocessing. |
| `neotube-tube` | Compresses clouds into 2D “tube” nodes (center + radius) that drive metadata queries. |
| `neotube-plan` | Queries the archive (ZTF IRSA) for exposures whose foot-prints intersect the tube nodes. |
| `neotube-ztf` | Downloads the science, reference, and ZOGY difference cutouts for the planned rows. |
| `neotube-infer` | Updates replica weights exposure-by-exposure via matched-filter SNR statistics, optionally running ZOGY when reference cutouts exist. |
| `neotube-diag` | Produces diagnostics for coverage, plus overlays and metadata reports. |

## Example workflow

```
neotube-fit --obs runs/ceres/obs.csv --target 00001 --seed-method attributable --out-dir runs/ceres/fit_seed_attributable
neotube-replicas --posterior runs/ceres/fit_seed_attributable/posterior.npz --n 2000 --output runs/ceres/replicas_attributable.csv
neotube-propcloud --replicas runs/ceres/replicas_attributable.csv --meta runs/ceres/replicas_attributable_meta.json --times runs/ceres/times_first_cutout.csv --output runs/ceres/cloud_attributable.csv
neotube-tube --cloud runs/ceres/cloud_attributable.csv --cred 0.99 --margin-arcsec 20 --output runs/ceres/tube_attributable.csv
neotube-plan --tube runs/ceres/tube_attributable.csv --out runs/ceres/plan_attributable.csv --filter zr --slot-s 1800
neotube-ztf --plan runs/ceres/plan_attributable.csv --out runs/ceres/cutouts --size-arcsec 120 --user-agent "neotube/0.1 (contact: chixson@fourshadows.org)"
neotube-infer --posterior-json runs/ceres/fit_seed_attributable/posterior.json --replicas runs/ceres/replicas_attributable.csv --cutouts-index runs/ceres/cutouts/cutouts_index.csv --out-dir runs/ceres/infer_attributable
```

## Philosophy

- **Data-first seeds:** `neotube-fit` now defaults to the attributable initializer, scans a small range + range-rate grid, and falls back to observation heuristics or Horizons only when requested.
- **Replica propagation:** `neotube-propcloud` + `propagate_replicas` batch the expensive propagation once per replica, then `neotube-tube` summarizes the clouds as centers + radii.
- **Polite IRSA usage:** `neotube-plan` and `neotube-ztf` add rate limiting, caching, and logging of server timing/IDs to avoid bot detection.
- **Inference readiness:** `neotube-infer` works with matched-filter SNR maps (ZOGY when available) and enforces a miss model so it can reject exposures that lacked data.

## Contributions

Contributions are welcome. Please file issues for new archives, improved inference, or CLI refinements. The pipeline assumes ZTF/IRSA for now but the architecture can host additional sources (SSOIS, NOIRLab) via the `neotube.plan` + `neotube-ztf` adapters.
