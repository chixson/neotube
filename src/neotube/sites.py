from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

import requests
from astropy import units as u
from astropy.constants import R_earth
from astropy.coordinates import EarthLocation

OBS_CODES_URL = "https://minorplanetcenter.net/iau/lists/ObsCodes.html"
CACHE_PATH = Path.home() / ".cache" / "neotube" / "observatories.csv"
EARTH_RADIUS_KM = float(R_earth.to(u.km).value)


@dataclass(frozen=True)
class ObservatoryEntry:
    code: str
    lon_deg: float
    rho_cos_phi: float
    rho_sin_phi: float

    @property
    def rho(self) -> float:
        return math.hypot(self.rho_cos_phi, self.rho_sin_phi)

    def to_location(self) -> EarthLocation:
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


def _parse_line(line: str) -> tuple[str, float, float, float] | None:
    line = line.strip()
    if not line or line.startswith("</pre>") or line.startswith("<"):
        return None
    parts = re.match(r"^(?P<code>\S+)\s+(?P<rest>.*)$", line)
    if not parts:
        return None
    code = parts.group("code").upper()
    rest = parts.group("rest")
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", rest)
    if len(nums) < 3:
        return None
    lon = float(nums[0])
    rho_cos_phi = float(nums[1])
    rho_sin_phi = float(nums[2])
    return code, lon, rho_cos_phi, rho_sin_phi


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
        code, lon, rho_cos_phi, rho_sin_phi = parsed
        entries[code] = ObservatoryEntry(
            code=code,
            lon_deg=lon,
            rho_cos_phi=rho_cos_phi,
            rho_sin_phi=rho_sin_phi,
        )
    return entries


def _write_cache(entries: Iterable[ObservatoryEntry]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["code", "lon_deg", "rho_cos_phi", "rho_sin_phi"])
        for entry in entries:
            writer.writerow([entry.code, entry.lon_deg, entry.rho_cos_phi, entry.rho_sin_phi])


def _read_cache() -> Mapping[str, ObservatoryEntry]:
    if not CACHE_PATH.exists():
        return {}
    entries: dict[str, ObservatoryEntry] = {}
    with CACHE_PATH.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                code = row["code"].strip().upper()
                lon = float(row["lon_deg"])
                rho_cos_phi = float(row["rho_cos_phi"])
                rho_sin_phi = float(row["rho_sin_phi"])
            except (KeyError, ValueError):
                continue
            entries[code] = ObservatoryEntry(
                code=code,
                lon_deg=lon,
                rho_cos_phi=rho_cos_phi,
                rho_sin_phi=rho_sin_phi,
            )
    return entries


def load_observatories() -> Mapping[str, ObservatoryEntry]:
    cached = _read_cache()
    if cached:
        return cached
    fetched = _fetch_catalog()
    if fetched:
        _write_cache(fetched.values())
    return fetched


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
