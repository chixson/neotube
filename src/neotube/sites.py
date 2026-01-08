from __future__ import annotations

import csv
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import requests
from astropy.coordinates import EarthLocation

from .observatory_entry import ObservatoryEntry
from .site_kind import SiteKind

OBS_CODES_URL = "https://minorplanetcenter.net/iau/lists/ObsCodes.html"
CACHE_PATH = Path.home() / ".cache" / "neotube" / "observatories.csv"
OVERRIDES_PATH = Path(__file__).resolve().parents[2] / "data" / "site_overrides.json"


def _parse_line(
    line: str,
) -> tuple[str, float | None, float | None, float | None, str | None] | None:
    line = line.strip()
    if not line or line.startswith("</pre>") or line.startswith("<"):
        return None
    parts = re.match(r"^(?P<code>\S+)\s+(?P<rest>.*)$", line)
    if not parts:
        return None
    code = parts.group("code").upper()
    rest = parts.group("rest")
    match = re.match(
        r"^\s*(?P<lon>[-+]?\d*\.\d+|[-+]?\d+)\s+"
        r"(?P<rho_cos>[-+]?\d*\.\d+|[-+]?\d+)\s+"
        r"(?P<rho_sin>[-+]?\d*\.\d+|[-+]?\d+)\s*(?P<name>.*)$",
        rest,
    )
    if match:
        lon = float(match.group("lon"))
        rho_cos_phi = float(match.group("rho_cos"))
        rho_sin_phi = float(match.group("rho_sin"))
        name = match.group("name").strip() or None
        return code, lon, rho_cos_phi, rho_sin_phi, name
    name = rest.strip() or None
    return code, None, None, None, name


def _fetch_catalog() -> Mapping[str, ObservatoryEntry]:
    resp = requests.get(OBS_CODES_URL, timeout=30)
    resp.raise_for_status()
    lines = resp.text.splitlines()
    entries: dict[str, ObservatoryEntry] = {}
    start_idx = 0
    for idx, line in enumerate(lines):
        if line.strip().startswith("Code"):
            start_idx = idx + 1
            break
    for line in lines[start_idx:]:
        parsed = _parse_line(line)
        if parsed is None:
            continue
        code, lon, rho_cos_phi, rho_sin_phi, description = parsed
        entries[code] = ObservatoryEntry(
            code=code,
            lon_deg=lon,
            rho_cos_phi=rho_cos_phi,
            rho_sin_phi=rho_sin_phi,
            description=description,
        )
    return entries


@lru_cache(maxsize=1)
def _load_site_overrides() -> Mapping[str, dict]:
    if not OVERRIDES_PATH.exists():
        return {}
    try:
        data = json.loads(OVERRIDES_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    normalized: dict[str, dict] = {}
    for key, payload in data.items():
        if not isinstance(payload, dict):
            continue
        code = str(key).strip().upper()
        if not code:
            continue
        normalized[code] = payload
    return normalized


def _apply_site_overrides(
    entries: Mapping[str, ObservatoryEntry],
) -> Mapping[str, ObservatoryEntry]:
    overrides = _load_site_overrides()
    if not overrides:
        return entries
    updated = dict(entries)
    for code, payload in overrides.items():
        entry = updated.get(code)
        description = payload.get("description")
        site_kind = payload.get("site_kind")
        ephemeris_id = payload.get("ephemeris_id")
        ephemeris_id_type = payload.get("ephemeris_id_type")
        ephemeris_location = payload.get("ephemeris_location")
        ephemeris_frame = payload.get("ephemeris_frame")
        if entry is None:
            updated[code] = ObservatoryEntry(
                code=code,
                lon_deg=None,
                rho_cos_phi=None,
                rho_sin_phi=None,
                description=description,
                site_kind=site_kind,
                ephemeris_id=ephemeris_id,
                ephemeris_id_type=ephemeris_id_type,
                ephemeris_location=ephemeris_location,
                ephemeris_frame=ephemeris_frame,
            )
        else:
            updated[code] = ObservatoryEntry(
                code=entry.code,
                lon_deg=entry.lon_deg,
                rho_cos_phi=entry.rho_cos_phi,
                rho_sin_phi=entry.rho_sin_phi,
                description=description or entry.description,
                site_kind=site_kind or entry.site_kind,
                ephemeris_id=ephemeris_id or entry.ephemeris_id,
                ephemeris_id_type=ephemeris_id_type or entry.ephemeris_id_type,
                ephemeris_location=ephemeris_location or entry.ephemeris_location,
                ephemeris_frame=ephemeris_frame or entry.ephemeris_frame,
            )
    return updated


def _read_cache() -> Mapping[str, ObservatoryEntry] | None:
    if not CACHE_PATH.exists():
        return None
    try:
        with CACHE_PATH.open() as fh:
            reader = csv.DictReader(fh)
            entries: dict[str, ObservatoryEntry] = {}
            for row in reader:
                code = str(row.get("code", "")).strip().upper()
                if not code:
                    continue
                def _as_float(val: str | None) -> float | None:
                    if val in (None, ""):
                        return None
                    try:
                        return float(val)
                    except ValueError:
                        return None
                entries[code] = ObservatoryEntry(
                    code=code,
                    lon_deg=_as_float(row.get("lon_deg")),
                    rho_cos_phi=_as_float(row.get("rho_cos_phi")),
                    rho_sin_phi=_as_float(row.get("rho_sin_phi")),
                    description=row.get("description") or None,
                    site_kind=row.get("site_kind") or None,
                    ephemeris_id=row.get("ephemeris_id") or None,
                    ephemeris_id_type=row.get("ephemeris_id_type") or None,
                    ephemeris_location=row.get("ephemeris_location") or None,
                    ephemeris_frame=row.get("ephemeris_frame") or None,
                )
            return entries
    except OSError:
        return None


def _write_cache(entries: Mapping[str, ObservatoryEntry]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "code",
        "lon_deg",
        "rho_cos_phi",
        "rho_sin_phi",
        "description",
        "site_kind",
        "ephemeris_id",
        "ephemeris_id_type",
        "ephemeris_location",
        "ephemeris_frame",
    ]
    with CACHE_PATH.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries.values():
            writer.writerow(
                {
                    "code": entry.code,
                    "lon_deg": entry.lon_deg,
                    "rho_cos_phi": entry.rho_cos_phi,
                    "rho_sin_phi": entry.rho_sin_phi,
                    "description": entry.description,
                    "site_kind": entry.site_kind,
                    "ephemeris_id": entry.ephemeris_id,
                    "ephemeris_id_type": entry.ephemeris_id_type,
                    "ephemeris_location": entry.ephemeris_location,
                    "ephemeris_frame": entry.ephemeris_frame,
                }
            )


@lru_cache(maxsize=1)
def get_observatory_catalog() -> Mapping[str, ObservatoryEntry]:
    cached = _read_cache()
    if cached is None:
        try:
            cached = _fetch_catalog()
            _write_cache(cached)
        except Exception:
            cached = {}
    return _apply_site_overrides(cached)


def classify_site(code: str, entry: ObservatoryEntry | None) -> SiteKind:
    key = code.strip().upper()
    if entry and entry.site_kind:
        try:
            return SiteKind(entry.site_kind)
        except ValueError:
            pass
    if entry and entry.ephemeris_id:
        return SiteKind.SPACECRAFT
    if key in {"500", "244"}:
        return SiteKind.GEOCENTER
    if key in {"247", "270"}:
        return SiteKind.ROVING
    if entry and entry.description and "geocentric" in entry.description.lower():
        return SiteKind.GEOCENTER
    if (
        entry is None
        or entry.lon_deg is None
        or entry.rho_cos_phi is None
        or entry.rho_sin_phi is None
        or not math.isfinite(entry.lon_deg)
        or not math.isfinite(entry.rho_cos_phi)
        or not math.isfinite(entry.rho_sin_phi)
    ):
        return SiteKind.SPACECRAFT if entry is not None else SiteKind.UNKNOWN
    return SiteKind.FIXED


def get_site_entry(code: str | None) -> ObservatoryEntry | None:
    if code is None:
        return None
    return get_observatory_catalog().get(code.strip().upper())


def get_site_location(code: str | None) -> EarthLocation | None:
    entry = get_site_entry(code)
    if entry is None:
        return None
    return entry.to_location()


def get_site_ephemeris(code: str | None) -> dict | None:
    entry = get_site_entry(code)
    if entry is None:
        return None
    if not entry.ephemeris_id:
        return None
    return {
        "ephemeris_id": entry.ephemeris_id,
        "ephemeris_id_type": entry.ephemeris_id_type,
        "ephemeris_location": entry.ephemeris_location,
        "ephemeris_frame": entry.ephemeris_frame,
    }
