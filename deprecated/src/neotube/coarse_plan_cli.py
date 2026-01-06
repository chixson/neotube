from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from astropy.time import Time, TimeDelta
from astropy import units as u
from astropy.coordinates import get_body_barycentric_posvel

from .cli import DEFAULT_COLUMNS, GlobalRateLimiter, exposure_unique_id, query_exposures
from .propagate import propagate_state_sun


@dataclass(frozen=True)
class OrbitPosterior:
    epoch: Time
    state_km: np.ndarray  # shape (6,)
    perturbers: tuple[str, ...]


def load_posterior(path: Path) -> OrbitPosterior:
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        epoch = Time(data["epoch_utc"], scale="utc")
        state = np.array(data["state_km"], dtype=float)
        perturbers = tuple(data.get("fit", {}).get("perturbers", ["earth", "mars", "jupiter"]))
        return OrbitPosterior(epoch=epoch, state_km=state, perturbers=perturbers)
    if path.suffix == ".npz":
        arr = np.load(path)
        epoch = Time(str(arr["epoch_utc"]), scale="utc")
        state = np.array(arr["state_km"], dtype=float)
        perturbers = tuple(arr.get("perturbers", ["earth", "mars", "jupiter"]))
        return OrbitPosterior(epoch=epoch, state_km=state, perturbers=perturbers)
    raise ValueError(f"Unsupported posterior format: {path}")


def iter_chunks(start: Time, end: Time, chunk_days: float) -> list[tuple[Time, Time, Time]]:
    if chunk_days <= 0:
        raise ValueError("chunk_days must be > 0")
    out: list[tuple[Time, Time, Time]] = []
    cur = start
    while cur < end:
        nxt = cur + TimeDelta(chunk_days, format="jd")
        if nxt > end:
            nxt = end
        mid = cur + (nxt - cur) / 2.0
        out.append((cur, nxt, mid))
        cur = nxt
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Conservative day-by-day (or chunk-by-chunk) planner: propagate nominal orbit to each chunk mid-time, then query IRSA for exposures within a large search radius."
    )
    parser.add_argument("--posterior", type=Path, required=True, help="Posterior JSON/NPZ containing epoch + state_km.")
    parser.add_argument("--start", type=str, required=True, help="Start time (UTC, ISO; e.g. 2020-12-01T00:00:00)")
    parser.add_argument("--end", type=str, required=True, help="End time (UTC, ISO; e.g. 2021-01-01T00:00:00)")
    parser.add_argument("--chunk-days", type=float, default=1.0, help="Chunk size in days (default: 1 day).")
    parser.add_argument("--radius-deg", type=float, default=1.0, help="Conservative search radius in degrees (default: 1).")
    parser.add_argument("--filter", type=str, default=None, help="Optional filter code to restrict exposures (zg/zr/zi).")
    parser.add_argument("--max-step", type=float, default=200.0, help="Max step size (seconds) for orbit propagation.")
    parser.add_argument("--perturbers", nargs="+", default=None, help="Override perturbers (default: from posterior).")
    parser.add_argument("--max-rps", type=float, default=0.2, help="Max requests/sec for IRSA metadata queries.")
    parser.add_argument(
        "--user-agent",
        type=str,
        default="neotube/0.1 (contact:chixson@fourshadows.org)",
        help="User-Agent header.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output plan CSV path.")
    args = parser.parse_args()

    posterior = load_posterior(args.posterior)
    if args.perturbers:
        raise SystemExit("--perturbers is not used in coarse Kepler planning; use fine propagation for N-body.")

    start = Time(args.start, scale="utc")
    end = Time(args.end, scale="utc")
    chunks = iter_chunks(start, end, args.chunk_days)
    if not chunks:
        raise SystemExit("No chunks to query.")

    seen: set[str] = set()
    rows: list[dict] = []

    headers = {"User-Agent": args.user_agent}
    limiter = GlobalRateLimiter(args.max_rps)

    mids = [mid for _, _, mid in chunks]
    # Sun-only Keplerian dynamics (but integrated via solve_ivp for robustness).
    states_mid = propagate_state_sun(posterior.state_km, posterior.epoch, mids, max_step=args.max_step)

    # Vectorized Earth heliocentric positions for topocentric-ish RA/Dec conversion.
    earth_pos, _ = get_body_barycentric_posvel("earth", Time(mids))
    sun_pos, _ = get_body_barycentric_posvel("sun", Time(mids))
    earth_helio = (earth_pos.xyz - sun_pos.xyz).to(u.km).value.T  # (n,3)

    def radec_from_vectors(obj_helio: np.ndarray, earth_helio_vec: np.ndarray) -> tuple[float, float]:
        vec = obj_helio - earth_helio_vec
        x, y, z = vec
        lon = np.degrees(np.arctan2(y, x)) % 360.0
        lat = np.degrees(np.arctan2(z, np.hypot(x, y)))
        return float(lon), float(lat)

    mid_radec = [radec_from_vectors(st[:3], e) for st, e in zip(states_mid, earth_helio)]

    with requests.Session() as session:
        for (chunk_start, chunk_end, mid), (ra_mid, dec_mid) in zip(chunks, mid_radec):

            exposures = query_exposures(
                session,
                ra=ra_mid,
                dec=dec_mid,
                jd_start=chunk_start.jd,
                jd_end=chunk_end.jd,
                columns=DEFAULT_COLUMNS,
                headers=headers,
                limiter=limiter,
                size_deg=args.radius_deg * 2.0,
                filtercode=args.filter,
            )
            for exp in exposures:
                eid = exposure_unique_id(exp)
                if eid in seen:
                    continue
                seen.add(eid)
                rows.append(
                    {
                        "exposure_id": eid,
                        "obsjd": exp.obsjd,
                        "obsdate": exp.obsdate,
                        "filefracday": exp.filefracday,
                        "field": exp.field,
                        "ccdid": exp.ccdid,
                        "qid": exp.qid,
                        "filtercode": exp.filtercode,
                        "imgtypecode": exp.imgtypecode,
                        "ra": exp.ra,
                        "dec": exp.dec,
                        "coarse_center_ra": ra_mid,
                        "coarse_center_dec": dec_mid,
                        "coarse_radius_deg": args.radius_deg,
                        "chunk_start_utc": chunk_start.isot,
                        "chunk_end_utc": chunk_end.isot,
                        "chunk_mid_utc": mid.isot,
                    }
                )

    if not rows:
        raise SystemExit("No exposures found for the requested time window and radius.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote coarse plan with {len(rows)} exposures to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
