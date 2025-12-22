# NEOTube (ZTF Data Harvester)

This repository now hosts the polite ZTF metadata + cutout downloader that feeds the next stage of the NEOTube workflow.
It is **not** the old SSOIS/Scout prototype (that work lives under `deprecated/neotube_legacy` and is ignored by git).

## Key behaviour

1. Query IRSA’s ZTF IBE science table with a single normalized metadata request (filtered by JD range, RA/Dec, optional filter code).
2. Construct the documented science product URLs from the resulting rows.
3. Download cutouts via the same URL with `center=RA,Dec`, `size=...`, `gzip=false`, and polite rate-limiting + backoff.
4. Cache cutouts and write an index CSV so repeated runs resume without re-downloading.
5. Log the CADC-style headers you need (request IDs, payload size, `Server-Timing`), and the code gracefully backs off on `429`/`5xx` responses.

## Requirements

Primarily this tool depends on `requests`. Install it via your preferred workflow (a virtualenv, `pip install -e .`, etc.).

## Workflow

NEOTube now exposes a small CLI suite that mirrors how the pipeline should be run:

1. `neotube-fit` — ingests MPC-style astrometry and produces an orbit posterior + covariance.
2. `neotube-replicas` — samples an ensemble of `x_0` replicas from the posterior.
3. `neotube-propcloud` — propagates those replicas to requested epochs and emits RA/Dec clouds.
4. `neotube-tube` — compresses the clouds into tube nodes (center + radius at a quantile), which are the inputs to `neotube-ztf`.
5. `neotube-ztf`/`neotube-plan` — fetch metadata + cutouts for the exposures the tube intersects.

Here's a high-level example run directory layout you can reproduce:

```
runs/ceres/
  00_fit/
    posterior.npz
    posterior.json
    fit_summary.json
    residuals.csv
  01_replicas/
    replicas.csv
    replicas_meta.json
  02_cloud/
    cloud.csv
  03_tube/
    tube_nodes.csv
  cutouts/
    cutouts_index.csv
```

## CLI reference

### `neotube-fit`

Inputs: CSV with `t_utc`, `ra_deg`, `dec_deg`, `sigma_arcsec`.

```
neotube-fit \
  --obs runs/ceres/obs.csv \
  --target 00001 \
  --perturbers earth mars jupiter saturn \
  --out-dir runs/ceres/00_fit/
```

Outputs: `posterior.npz`, `posterior.json`, `fit_summary.json`, `residuals.csv`, `fit_params.json`.

### `neotube-replicas`

```
neotube-replicas \
  --posterior runs/ceres/00_fit/posterior.npz \
  --n 2000 \
  --seed 42 \
  --output runs/ceres/01_replicas/replicas.csv
```

Outputs: CSV of state vectors + RA/Dec, plus `replicas_meta.json` describing the epoch and seed.

### `neotube-propcloud`

```
neotube-propcloud \
  --replicas runs/ceres/01_replicas/replicas.csv \
  --meta runs/ceres/01_replicas/replicas_meta.json \
  --times runs/ceres/times.csv \
  --output runs/ceres/02_cloud/cloud.csv
```

`times.csv` only needs a `time_utc` column describing the exposures/midpoints you care about.

### `neotube-tube`

```
neotube-tube \
  --cloud runs/ceres/02_cloud/cloud.csv \
  --cred 0.99 \
  --margin-arcsec 10 \
  --output runs/ceres/03_tube/tube_nodes.csv
```

`tube_nodes.csv` becomes the canonical input for `neotube-ztf`.

### `neotube-plan`

```
neotube-plan \
  --tube runs/ceres/03_tube/tube_nodes.csv \
  --out runs/ceres/03_plan/plan.csv \
  --slot-s 1800 \
  --min-size-arcsec 30 \
  --filter zr
```

It queries IRSA for each tube node, records per-exposure planned centers + radii, and deduplicates repeated hits. The resulting `plan.csv` becomes the direct input to `neotube-ztf` via `--plan`.

### `neotube-diag`

```
neotube-diag \
  --plan runs/ceres/03_plan/plan.csv \
  --clean-index results/ceres_precise_cutouts_clean_index.csv \
  --output runs/ceres/diag/coverage.json
```

This command compares each planned exposure center against the observed metadata (and optionally the cleaned index) to verify the tube actually covers the expected frames.

### `neotube-ztf`

```
neotube-ztf \
  --ra 255.57691 --dec 12.28378 \
  --jd-start 2460000.0 --jd-end 2460005.0 \
  --filter zr \
  --size-arcsec 50 \
  --out cutouts \
  --user-agent "neotube/0.1 (contact: chixson@fourshadows.org)"
```

The command writes `cutouts_index.csv` and downloads FITS cutouts to the `--out` directory. Logs include request identifiers and server timing when available.

## How to extend

- Add ephemeris-driven centers so each exposure is retrieved at the predicted location.
- Use the resulting index to seed the planner’s tube quantiles and motion-model logic.
- Cache the metadata query results if you plan to iterate over the same time span.
