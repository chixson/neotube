from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping
import json

import requests
from astropy import units as u
from astropy.constants import R_earth
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astroquery.jplhorizons import Horizons

OBS_CODES_URL = "https://minorplanetcenter.net/iau/lists/ObsCodes.html"
CACHE_PATH = Path.home() / ".cache" / "neotube" / "observatories.csv"
OVERRIDES_PATH = Path(__file__).resolve().parents[2] / "data" / "site_overrides.json"
_OBS_CACHE: Mapping[str, "ObservatoryEntry"] | None = None
EARTH_RADIUS_KM = float(R_earth.to(u.km).value)
_SPACECRAFT_ALIAS = {
    "WISE": "-163",
    "NEOWISE": "-163",
    "NEO-WISE": "-163",
    "WISE/NEOWISE": "-163",
}


@dataclass(frozen=True)
class ObservatoryEntry:
    code: str
    lon_deg: float | None
    rho_cos_phi: float | None
    rho_sin_phi: float | None
    description: str | None
    site_kind: str | None = None
    ephemeris_id: str | None = None
    ephemeris_id_type: str | None = None
    ephemeris_location: str | None = None
    ephemeris_frame: str | None = None

    @property
    def rho(self) -> float:
        if self.rho_cos_phi is None or self.rho_sin_phi is None:
            return 0.0
        return math.hypot(self.rho_cos_phi, self.rho_sin_phi)

    def to_location(self) -> EarthLocation | None:
        if (
            self.lon_deg is None
            or self.rho_cos_phi is None
            or self.rho_sin_phi is None
            or not math.isfinite(self.lon_deg)
            or not math.isfinite(self.rho_cos_phi)
            or not math.isfinite(self.rho_sin_phi)
        ):
            return None
        # compute geocentric cartesian offset using MPC parallax constants
        lam = math.radians(self.lon_deg)
        rho = self.rho
        if rho == 0.0:
            # fallback to mean Earth radius at ellipsoid (lat derived from sin)
            lat = math.degrees(math.atan2(self.rho_sin_phi, max(self.rho_cos_phi, 1e-12)))
            return EarthLocation.from_geodetic(
                lon=self.lon_deg * u.deg,
                lat=lat * u.deg,
                height=0.0 * u.m,
            )
        rho_cos_phi = self.rho_cos_phi
        rho_sin_phi = self.rho_sin_phi
        x = rho_cos_phi * math.cos(lam) * EARTH_RADIUS_KM
        y = rho_cos_phi * math.sin(lam) * EARTH_RADIUS_KM
        z = rho_sin_phi * EARTH_RADIUS_KM
        return EarthLocation.from_geocentric(x * u.km, y * u.km, z * u.km)


class SiteKind(str, Enum):
    FIXED = "fixed"
    GEOCENTER = "geocenter"
    ROVING = "roving"
    SPACECRAFT = "spacecraft"
    UNKNOWN = "unknown"


def _parse_line(line: str) -> tuple[str, float | None, float | None, float | None, str | None] | None:
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


def _apply_site_overrides(entries: Mapping[str, ObservatoryEntry]) -> Mapping[str, ObservatoryEntry]:
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


def _normalize_spacecraft_name(description: str | None) -> list[str]:
    if not description:
        return []
    text = description.strip()
    # Keep full description and a few simplified tokens.
    tokens = [text]
    for part in re.split(r"[(),;/]", text):
        part = part.strip()
        if part:
            tokens.append(part)
    # Uppercased aliases for lookup.
    tokens.extend([t.upper() for t in tokens])
    return list(dict.fromkeys(tokens))


def _resolve_horizons_id(candidate: str) -> str | None:
    """Return candidate if Horizons accepts it as an id at @ssb, else None."""
    try:
        t = Time.now()
        obj = Horizons(id=candidate, location="@ssb", epochs=t.jd)
        _ = obj.vectors()
    except Exception:
        return None
    return candidate


def _write_cache(entries: Iterable[ObservatoryEntry]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["code", "lon_deg", "rho_cos_phi", "rho_sin_phi", "description"])
        for entry in entries:
            writer.writerow(
                [entry.code, entry.lon_deg, entry.rho_cos_phi, entry.rho_sin_phi, entry.description]
            )


def _read_cache() -> Mapping[str, ObservatoryEntry]:
    if not CACHE_PATH.exists():
        return {}
    entries: dict[str, ObservatoryEntry] = {}
    with CACHE_PATH.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                code = row["code"].strip().upper()
                lon = float(row["lon_deg"]) if row["lon_deg"] else None
                rho_cos_phi = float(row["rho_cos_phi"]) if row["rho_cos_phi"] else None
                rho_sin_phi = float(row["rho_sin_phi"]) if row["rho_sin_phi"] else None
            except (KeyError, ValueError):
                continue
            entries[code] = ObservatoryEntry(
                code=code,
                lon_deg=lon,
                rho_cos_phi=rho_cos_phi,
                rho_sin_phi=rho_sin_phi,
                description=row.get("description") or None,
            )
    return entries


def load_observatories(refresh: bool = False) -> Mapping[str, ObservatoryEntry]:
    global _OBS_CACHE
    if _OBS_CACHE is not None and not refresh:
        return _OBS_CACHE
    if not refresh:
        cached = _read_cache()
        if cached:
            _OBS_CACHE = _apply_site_overrides(cached)
            return _OBS_CACHE
    fetched = _fetch_catalog()
    if fetched:
        _write_cache(fetched.values())
        _OBS_CACHE = _apply_site_overrides(fetched)
        return _OBS_CACHE
    if _OBS_CACHE is not None:
        return _OBS_CACHE
    cached = _read_cache()
    _OBS_CACHE = _apply_site_overrides(cached)
    return _OBS_CACHE


@lru_cache(maxsize=1024)
def get_site_location(code: str | None) -> EarthLocation | None:
    if code is None:
        return None
    key = code.strip().upper()
    if not key:
        return None
    catalog = load_observatories()
    entry = catalog.get(key)
    if entry is None:
        return None
    if entry.site_kind and entry.site_kind.lower() == SiteKind.SPACECRAFT.value:
        return None
    return entry.to_location()


@lru_cache(maxsize=1024)
def get_site_kind(code: str | None) -> SiteKind:
    if code is None:
        return SiteKind.UNKNOWN
    key = code.strip().upper()
    if not key:
        return SiteKind.UNKNOWN
    catalog = load_observatories()
    entry = catalog.get(key)
    return classify_site(key, entry)


@lru_cache(maxsize=1024)
def get_site_ephemeris(code: str | None) -> dict | None:
    if code is None:
        return None
    key = code.strip().upper()
    if not key:
        return None
    catalog = load_observatories()
    entry = catalog.get(key)
    if entry is None:
        return None
    if not entry.ephemeris_id:
        # Best-effort automatic lookup for spacecraft sites based on description.
        if classify_site(key, entry) == SiteKind.SPACECRAFT:
            candidates = _normalize_spacecraft_name(entry.description)
            for cand in candidates:
                alias = _SPACECRAFT_ALIAS.get(cand.upper())
                if alias:
                    return {
                        "ephemeris_id": alias,
                        "ephemeris_id_type": "id",
                        "ephemeris_location": "@ssb",
                        "ephemeris_frame": "icrs",
                    }
                resolved = _resolve_horizons_id(cand)
                if resolved:
                    return {
                        "ephemeris_id": resolved,
                        "ephemeris_id_type": "id",
                        "ephemeris_location": "@ssb",
                        "ephemeris_frame": "icrs",
                    }
        return None
    return {
        "ephemeris_id": entry.ephemeris_id,
        "ephemeris_id_type": entry.ephemeris_id_type or "spacecraft",
        "ephemeris_location": entry.ephemeris_location or "@ssb",
        "ephemeris_frame": entry.ephemeris_frame or "icrs",
    }
