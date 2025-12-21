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

## Usage

```
neotube-ztf \
  --ra 255.57691 --dec 12.28378 \
  --jd-start 2460000.0 --jd-end 2460005.0 \
  --filter zr \
  --size-arcsec 50 \
  --out cutouts \
  --user-agent "neotube/0.1 (contact: chixson@fourshadows.org)"
```

The command will write `cutouts_index.csv` and download FITS cutouts to the `--out` directory. Logs include request identifiers and server timing when available.

## How to extend

- Add ephemeris-driven centers so each exposure is retrieved at the predicted location.
- Use the resulting index to seed the planner’s tube quantiles and motion-model logic.
- Cache the metadata query results if you plan to iterate over the same time span.
