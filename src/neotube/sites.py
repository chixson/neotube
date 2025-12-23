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
from astropy.coordinates import EarthLocation

OBS_CODES_URL = "https://minorplanetcenter.net/iau/lists/ObsCodes.html"
CACHE_PATH = Path.home() / ".cache" / "neotube" / "observatories.csv"


@dataclass(frozen=True)
class ObservatoryEntry:
    code: str
    lon_deg: float
    lat_deg: float


def _parse_line(line: str) -> tuple[str, float, float] | None:
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
    sin_lat = float(nums[2])
    # Ensure sin is within [-1,1]
    sin_lat = max(-1.0, min(1.0, sin_lat))
    lat_rad = math.asin(sin_lat)
    lat = math.degrees(lat_rad)
    return code, lon, lat


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
        code, lon, lat = parsed
        entries[code] = ObservatoryEntry(code=code, lon_deg=lon, lat_deg=lat)
    return entries


def _write_cache(entries: Iterable[ObservatoryEntry]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["code", "lon_deg", "lat_deg"])
        for entry in entries:
            writer.writerow([entry.code, entry.lon_deg, entry.lat_deg])


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
                lat = float(row["lat_deg"])
            except (KeyError, ValueError):
                continue
            entries[code] = ObservatoryEntry(code=code, lon_deg=lon, lat_deg=lat)
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
    return EarthLocation.from_geodetic(
        lon=entry.lon_deg * u.deg,
        lat=entry.lat_deg * u.deg,
        height=0.0 * u.m,
    )
