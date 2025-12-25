from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

import requests
from astropy import units as u
from astropy.constants import R_earth
from astropy.coordinates import EarthLocation

OBS_CODES_URL = "https://minorplanetcenter.net/iau/lists/ObsCodes.html"
CACHE_PATH = Path.home() / ".cache" / "neotube" / "observatories.csv"
_OBS_CACHE: Mapping[str, "ObservatoryEntry"] | None = None
EARTH_RADIUS_KM = float(R_earth.to(u.km).value)


@dataclass(frozen=True)
class ObservatoryEntry:
    code: str
    lon_deg: float | None
    rho_cos_phi: float | None
    rho_sin_phi: float | None
    description: str | None

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
            _OBS_CACHE = cached
            return cached
    fetched = _fetch_catalog()
    if fetched:
        _write_cache(fetched.values())
        _OBS_CACHE = fetched
        return fetched
    if _OBS_CACHE is not None:
        return _OBS_CACHE
    cached = _read_cache()
    _OBS_CACHE = cached
    return cached


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
