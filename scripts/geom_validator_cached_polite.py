#!/usr/bin/env python3
"""
geom_validator_cached_polite.py

Self-contained geometry validator (updated):
 - All geometry math performed locally (no astropy for the math)
 - Persistent caching for Horizons queries and MPC ObsCodes
 - Uses NEOTube polite query scheme (GlobalRateLimiter + request_with_backoff) if available,
   otherwise falls back to a local polite requester with backoff.
 - Explicit comments about frames & approximations and TODOs for Gaia/LSST-grade accuracy.

Notes about precision / frames (CRITICAL)
----------------------------------------
This validator currently uses a Meeus-style GMST rotation (UTC ~ UT1) to convert
ECEF -> ECI and a first-order aberration model. That is *sufficient for coarse
validation* and debugging geometry flows, but is NOT sufficient for Gaia/LSST-level
astrometry (sub-0.01 arcsec). For production-grade (Gaia / LSST) accuracy we must:

  * handle Earth orientation parameters (EOP) and UT1-UTC offsets,
  * use IAU-2000A / IAU-2006 precession-nutation (nutation + frame bias),
  * correct for polar motion (x,y), and apply true-of-date transformations,
  * transform time scales correctly (UTC -> TAI -> TT -> TDB) for ephemerides and light-time,
  * include full relativistic modelling: light deflection by Sun/planets, higher-order aberration terms,
  * use high-fidelity ephemeris frames and ensure we match JPL/Horizons frame choices (ICRS / FK5 / TETE),
  * handle catalog reference frames and proper motions (e.g., Gaia DR2/DR3 reference epoch),
  * for LSST-style reductions account for site-specific calibrations (thermal flexure, mount offsets) if needed.

Where I marked TODOs in the code you can slot in those pieces. The code is intentionally modular
so we can replace the GMST/rotation/time-conversion pieces with IERS/EOP and IAU routines later.

Caching & polite queries
------------------------
Horizons queries are cached persistently in:
   ~/.cache/neotube/horizons/<sha256>.json

MPC obs-codes catalog cached in:
   ~/.cache/neotube/mpc_sites.json

We *use* NEOTube's polite query scheme where available (GlobalRateLimiter + request_with_backoff).
See `src/neotube/cli.py` for NEOTube's implementation (GlobalRateLimiter, request_with_backoff).
If NEOTube isn't importable the script falls back to an embedded request_with_backoff + simple
GlobalRateLimiter (exponential backoff + jitter + retry on 429/5xx).

Run:
  python scripts/geom_validator_cached_polite.py --obs-csv runs/ceres/obs.csv --target 1 --output runs/ceres/jpl_dump.csv

Dependencies: Python 3.8+, numpy, requests (astropy/astroquery optional for TDB-aware paths)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
import urllib.parse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Try to import NEOTube polite requester; if unavailable, fall back to local implementation.
try:
    from neotube.cli import request_with_backoff, GlobalRateLimiter  # type: ignore
    _HAVE_NEOTUBE_POLITE = True
except Exception:
    _HAVE_NEOTUBE_POLITE = False
    import requests

    class GlobalRateLimiter:
        def __init__(self, max_rps: float):
            self.min_interval = 1.0 / max_rps if max_rps > 0 else 0.0
            self._lock = threading.Lock()
            self._next_time = 0.0

        def wait(self) -> None:
            with self._lock:
                now = time.time()
                if now < self._next_time:
                    time.sleep(self._next_time - now)
                jitter = random.uniform(0.02, 0.1)
                self._next_time = time.time() + self.min_interval + jitter

    def request_with_backoff(
        session: "requests.Session",
        url: str,
        *,
        headers: Dict[str, str],
        stream: bool = False,
        timeout: Tuple[int, int] = (10, 60),
        max_tries: int = 6,
        limiter: Optional[GlobalRateLimiter] = None,
    ):
        backoff = 1.0
        for attempt in range(1, max_tries + 1):
            if limiter:
                limiter.wait()
            try:
                resp = session.get(url, headers=headers, stream=stream, timeout=timeout)
            except requests.RequestException:
                if attempt == max_tries:
                    raise
                sleep_s = backoff + random.uniform(0, 0.5)
                time.sleep(sleep_s)
                backoff = min(backoff * 2.0, 60.0)
                continue
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = resp.headers.get("Retry-After")
                try:
                    sleep_s = float(retry_after) if retry_after else backoff
                except Exception:
                    sleep_s = backoff
                resp.close()
                if attempt == max_tries:
                    raise RuntimeError(
                        f"Request failed after {max_tries} attempts; last status={resp.status_code}"
                    )
                time.sleep(sleep_s + random.uniform(0, 0.5))
                backoff = min(backoff * 2.0, 120.0)
                continue
            body_snip = ""
            try:
                body_snip = resp.text[:300]
            except Exception:
                pass
            resp.close()
            raise RuntimeError(
                f"Non-retryable HTTP {resp.status_code} for {url}. Body: {body_snip}"
            )
        raise RuntimeError("request_with_backoff reached unreachable code")

# Try to import Astropy / Astroquery for TDB-aware epoch handling and Horizons access.
try:
    from astropy.time import Time, TimeDelta  # type: ignore
    from astropy.coordinates import (  # type: ignore
        AltAz,
        EarthLocation,
        GCRS,
        ICRS,
        SkyCoord,
        CartesianDifferential,
        CartesianRepresentation,
    )
    from astropy import units as u  # type: ignore
    from astroquery.jplhorizons import Horizons as AQ_Horizons  # type: ignore

    _HAVE_ASTROPY = True
except Exception:
    _HAVE_ASTROPY = False

# Constants
AU_KM = 149597870.7
DAY_S = 86400.0
C_KM_S = 299792.458
WGS84_A_KM = 6378.137
WGS84_F = 1.0 / 298.257223563
OMEGA_EARTH = 7.2921150e-5

HORIZONS_API = "https://ssd.jpl.nasa.gov/api/horizons.api"
MPC_OBS_CODES_URL = "https://minorplanetcenter.net/iau/lists/ObsCodes.html"

# Cache locations
CACHE_BASE = os.path.expanduser("~/.cache/neotube")
HORIZONS_CACHE_DIR = os.path.join(CACHE_BASE, "horizons")
MPC_SITES_CACHE = os.path.join(CACHE_BASE, "mpc_sites.json")
os.makedirs(HORIZONS_CACHE_DIR, exist_ok=True)
os.makedirs(CACHE_BASE, exist_ok=True)

# Politeness defaults (we use NEOTube polite defaults where possible)
DEFAULT_MAX_RPS = 1.0  # requests per second to remote services (tunable)


# ----------------------
# Time and frame helpers
# ----------------------

def parse_iso_utc(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def datetime_to_iso(dt: datetime) -> str:
    dt = dt.astimezone(timezone.utc)
    if dt.microsecond:
        return dt.isoformat().replace("+00:00", "Z")
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def iso_minus_seconds(iso: str, seconds: float) -> str:
    dt = parse_iso_utc(iso)
    dt2 = dt - timedelta(seconds=seconds)
    return datetime_to_iso(dt2)


def iso_plus_seconds(iso: str, seconds: float) -> str:
    dt = parse_iso_utc(iso)
    dt2 = dt + timedelta(seconds=seconds)
    return datetime_to_iso(dt2)

def utc_to_jd(dt: datetime) -> float:
    dt = dt.astimezone(timezone.utc)
    year = dt.year
    month = dt.month
    day_frac = dt.day + (
        dt.hour + (dt.minute + (dt.second + dt.microsecond / 1e6) / 60.0) / 60.0
    ) / 24.0
    if month <= 2:
        year -= 1
        month += 12
    A = int(year / 100)
    B = 2 - A + int(A / 4)
    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day_frac + B - 1524.5
    return jd


def gmst_from_jd_ut1(jd_ut1: float) -> float:
    T = (jd_ut1 - 2451545.0) / 36525.0
    gmst_sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * T
        + 0.093104 * (T**2)
        - (6.2e-6) * (T**3)
    )
    gmst_sec = gmst_sec % 86400.0
    gmst = (gmst_sec / 86400.0) * 2.0 * math.pi
    return gmst


# ----------------------
# Vector math helpers
# ----------------------

def _normalize_horizons_id(raw: str) -> str:
    """Normalize common MPC-style identifiers into something Horizons accepts."""
    s = raw.strip()
    if s.isdigit():
        n = int(s)
        if 1 <= n < 2000000:
            return str(2000000 + n)
    return s


def radec_to_unit(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    x = math.cos(dec) * math.cos(ra)
    y = math.cos(dec) * math.sin(ra)
    z = math.sin(dec)
    return np.array([x, y, z], dtype=float)


def unit_to_radec(u: np.ndarray) -> Tuple[float, float]:
    x, y, z = float(u[0]), float(u[1]), float(u[2])
    xy = math.hypot(x, y)
    ra = math.degrees(math.atan2(y, x))
    if ra < 0.0:
        ra += 360.0
    dec = math.degrees(math.atan2(z, xy))
    return ra, dec


def signed_deltas(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    ra1 = np.deg2rad(np.asarray(ra1_deg, dtype=float))
    ra2 = np.deg2rad(np.asarray(ra2_deg, dtype=float))
    dec1 = np.deg2rad(np.asarray(dec1_deg, dtype=float))
    dec2 = np.deg2rad(np.asarray(dec2_deg, dtype=float))
    dra = ra1 - ra2
    dra = (dra + math.pi) % (2.0 * math.pi) - math.pi
    dx_arcsec = dra * np.cos(dec1) * 206265.0
    dy_arcsec = (dec1 - dec2) * 206265.0
    return dx_arcsec, dy_arcsec


# ----------------------
# Persistent cache helpers
# ----------------------

def _cache_key_hex(prefix: str, payload: str) -> str:
    h = hashlib.sha256()
    h.update(prefix.encode("utf-8"))
    h.update(b":")
    h.update(payload.encode("utf-8"))
    return h.hexdigest()


def horizons_cache_get(key_hex: str) -> Optional[List[float]]:
    path = os.path.join(HORIZONS_CACHE_DIR, f"{key_hex}.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as fh:
                data = json.load(fh)
            return data
        except Exception:
            try:
                os.remove(path)
            except Exception:
                pass
    return None


def horizons_cache_put(key_hex: str, vec: List[float]) -> None:
    path = os.path.join(HORIZONS_CACHE_DIR, f"{key_hex}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(vec, fh)
    os.replace(tmp, path)


def mpc_sites_cache_get() -> Optional[Dict[str, Any]]:
    if os.path.exists(MPC_SITES_CACHE):
        try:
            with open(MPC_SITES_CACHE, "r") as fh:
                return json.load(fh)
        except Exception:
            try:
                os.remove(MPC_SITES_CACHE)
            except Exception:
                pass
    return None


def mpc_sites_cache_put(dct: Dict[str, Any]) -> None:
    tmp = MPC_SITES_CACHE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dct, fh)
    os.replace(tmp, MPC_SITES_CACHE)


# ----------------------
# Horizons query (polite + cached)
# ----------------------

def horizons_vectors_cached(
    command: str,
    epoch_iso: str,
    center: str = "@ssb",
    refplane: str = "earth",
    limiter: Optional[GlobalRateLimiter] = None,
    max_tries: int = 6,
) -> np.ndarray:
    """
    Cached, polite Horizons VECTORS request returning 6-vector (units km, km/s)
    Keyed by (command, epoch_iso, center, refplane).
    """
    payload = f"{command}|{epoch_iso}|{center}|{refplane}"
    key = _cache_key_hex("horizons", payload)
    cached = horizons_cache_get(key)
    if cached is not None:
        return np.array(cached, dtype=float)

    refplane_norm = refplane.strip().upper()
    if refplane_norm == "EARTH":
        refplane_norm = "FRAME"
    if refplane_norm not in {"ECLIPTIC", "FRAME", "BODY EQUATOR"}:
        refplane_norm = "FRAME"

    if _HAVE_ASTROPY:
        try:
            t = Time(epoch_iso, scale="utc")
            t_tdb = t.tdb
            obj_id = str(command).strip()
            id_type = None
            if obj_id.isdigit():
                n = int(obj_id)
                if n >= 2000000:
                    id_type = "smallbody"
                    obj_id = f"DES={obj_id}"
            else:
                id_type = "smallbody"
            kwargs = {"id": obj_id, "location": center, "epochs": t_tdb.jd}
            if id_type is not None:
                kwargs["id_type"] = id_type
            h = AQ_Horizons(**kwargs)
            vec = h.vectors(refplane=refplane_norm.lower())
            row = vec[0]
            x = float(row["x"]) * AU_KM
            y = float(row["y"]) * AU_KM
            z = float(row["z"]) * AU_KM
            vx = float(row["vx"]) * AU_KM / DAY_S
            vy = float(row["vy"]) * AU_KM / DAY_S
            vz = float(row["vz"]) * AU_KM / DAY_S
            vals = [x, y, z, vx, vy, vz]
            horizons_cache_put(key, vals)
            return np.array(vals, dtype=float)
        except Exception:
            pass

    stop_iso = iso_plus_seconds(epoch_iso, 60.0)
    params = {
        "format": "json",
        "COMMAND": f"'{command}'",
        "EPHEM_TYPE": "VECTORS",
        "CENTER": f"'{center}'",
        "REF_PLANE": f"'{refplane_norm}'",
        "REF_SYSTEM": "'ICRF'",
        "VEC_TABLE": "2",
        "OUT_UNITS": "'KM-S'",
        "START_TIME": f"'{epoch_iso}'",
        "STOP_TIME": f"'{stop_iso}'",
        "STEP_SIZE": "'1 m'",
    }
    url = HORIZONS_API + "?" + urllib.parse.urlencode(params, safe="'@:,")
    import requests

    session = requests.Session()
    headers = {"User-Agent": "NEOTube-GeomValidator/1.0 (+your-email@domain)"}
    if limiter is None:
        limiter = GlobalRateLimiter(DEFAULT_MAX_RPS)
    resp = request_with_backoff(session, url, headers=headers, limiter=limiter, max_tries=max_tries)
    txt = resp.text
    resp.close()
    j = json.loads(txt)
    if "result" not in j:
        raise RuntimeError("Horizons returned no 'result' for " + url)
    result_text = j["result"]
    m_so = result_text.find("$$SOE")
    m_eo = result_text.find("$$EOE")
    if m_so == -1 or m_eo == -1 or m_eo <= m_so:
        raise RuntimeError("Horizons missing SOE/EOE")
    block = result_text[m_so:m_eo]
    pattern = re.compile(
        r"X\s*=\s*([-\d.+eE]+)\s*Y\s*=\s*([-\d.+eE]+)\s*Z\s*=\s*([-\d.+eE]+)\s*VX\s*=\s*([-\d.+eE]+)\s*VY\s*=\s*([-\d.+eE]+)\s*VZ\s*=\s*([-\d.+eE]+)",
        re.IGNORECASE | re.DOTALL,
    )
    m = pattern.search(block)
    if not m:
        joined = " ".join(block.splitlines())
        m = pattern.search(joined)
        if not m:
            raise RuntimeError("Could not parse Horizons vector block for " + url)
    vals = [float(m.group(i)) for i in range(1, 7)]
    horizons_cache_put(key, vals)
    return np.array(vals, dtype=float)


# ----------------------
# MPC ObsCodes parsing (cached)
# ----------------------

def fetch_mpc_obs_catalog_cached(timeout: int = 30, use_cache: bool = True) -> Dict[str, Dict[str, Any]]:
    if use_cache:
        c = mpc_sites_cache_get()
        if c is not None:
            return c
    import requests

    resp = requests.get(MPC_OBS_CODES_URL, timeout=timeout)
    resp.raise_for_status()
    html = resp.text
    lines = html.splitlines()
    start_idx = 0
    for i, L in enumerate(lines):
        if L.strip().startswith("Code"):
            start_idx = i + 1
            break
    entries: Dict[str, Dict[str, Any]] = {}
    for line in lines[start_idx:]:
        parsed = _parse_mpc_line(line)
        if parsed is None:
            continue
        code, lon, rho_cos, rho_sin, name = parsed
        entries[code] = {
            "lon_deg": lon,
            "rho_cos_phi": rho_cos,
            "rho_sin_phi": rho_sin,
            "description": name,
        }
    mpc_sites_cache_put(entries)
    return entries


def _parse_mpc_line(line: str):
    line = line.strip()
    if not line or line.startswith("</pre>") or line.startswith("<"):
        return None
    m = re.match(r"^(?P<code>\S+)\s+(?P<rest>.*)$", line)
    if not m:
        return None
    code = m.group("code").upper()
    rest = m.group("rest")
    match = re.match(
        r"^\s*(?P<lon>[-+]?\d*\.\d+|[-+]?\d+)\s+(?P<rho_cos>[-+]?\d*\.\d+|[-+]?\d+)\s+(?P<rho_sin>[-+]?\d*\.\d+|[-+]?\d+)\s*(?P<name>.*)$",
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


def mpc_site_to_ecef_km(entry: Dict[str, Any]) -> Optional[np.ndarray]:
    if entry is None:
        return None
    lon = entry.get("lon_deg")
    rho_cos = entry.get("rho_cos_phi")
    rho_sin = entry.get("rho_sin_phi")
    if lon is None or rho_cos is None or rho_sin is None:
        return None
    lam = math.radians(lon)
    x = rho_cos * math.cos(lam) * WGS84_A_KM
    y = rho_cos * math.sin(lam) * WGS84_A_KM
    z = rho_sin * WGS84_A_KM
    return np.array([x, y, z], dtype=float)


# ----------------------
# ECEF -> ECI via GMST
# ----------------------

def ecef_to_eci(ecef: np.ndarray, gmst_rad: float) -> np.ndarray:
    c = math.cos(gmst_rad)
    s = math.sin(gmst_rad)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R @ ecef


# ----------------------
# Aberration & Shapiro (first-order)
# ----------------------

def aberrate_direction_first_order(topovec: np.ndarray, obs_vel_km_s: np.ndarray) -> np.ndarray:
    n = topovec / (np.linalg.norm(topovec) + 1e-30)
    beta = obs_vel_km_s / C_KM_S
    nb = float(np.dot(n, beta))
    s = n + beta - nb * n
    s = s / (np.linalg.norm(s) + 1e-30)
    return s


def shapiro_delay_sun(
    obj_pos_bary_km: np.ndarray, obs_pos_bary_km: np.ndarray, sun_pos_bary_km: np.ndarray
) -> float:
    GM_SUN = 1.32712440018e11
    r_e = float(np.linalg.norm(obj_pos_bary_km - sun_pos_bary_km))
    r_o = float(np.linalg.norm(obs_pos_bary_km - sun_pos_bary_km))
    R = float(np.linalg.norm(obj_pos_bary_km - obs_pos_bary_km))
    denom = max(r_e + r_o - R, 1e-12)
    arg = (r_e + r_o + R) / denom
    return 2.0 * GM_SUN / (C_KM_S**3) * math.log(arg)


# ----------------------
# Light-time iterate: predict apparent RA/Dec
# ----------------------

@dataclass
class PredictDebug:
    iterations: int
    tau_s: float
    obj_bary_km: List[float]
    r_topo_km: List[float]
    unit_topovec: List[float]


def predict_apparent_radec_for_obs(
    jpl_cmd: str,
    obs_time_iso: str,
    site_bary_km: np.ndarray,
    site_vel_bary_km_s: np.ndarray,
    earth_bary_km: Optional[np.ndarray] = None,
    earth_vel_km_s: Optional[np.ndarray] = None,
    site_ecef_km: Optional[np.ndarray] = None,
    limiter: Optional[GlobalRateLimiter] = None,
    max_iters: int = 6,
    tol_s: float = 1e-4,
) -> Tuple[float, float, PredictDebug]:
    """
    TDB-aware light-time iteration to get object barycentric vector at emission and compute
    predicted apparent RA/Dec (unit vector aberrated by first-order SR).

    If astropy is available, iterate in TDB and request vectors at t_guess.tdb.
    Otherwise, fall back to the older UTC-based iteration.
    """
    if _HAVE_ASTROPY:
        t_obs = Time(obs_time_iso, scale="utc")
        t_obs_tdb = t_obs.tdb
        t_guess = t_obs_tdb
        last_tau = None
        obj_bary = None
        obj_vel = None
        for it in range(max_iters):
            obj_vec = horizons_vectors_cached(jpl_cmd, t_guess.iso, center="@ssb", refplane="frame", limiter=limiter)
            obj_bary = obj_vec.copy()
            obj_vel = obj_vec[3:].copy()
            r_topo = obj_bary[:3] - site_bary_km
            R = float(np.linalg.norm(r_topo))
            tau = R / C_KM_S
            t_new = t_obs_tdb - TimeDelta(tau, format="sec")
            if last_tau is not None and abs(tau - last_tau) < tol_s:
                break
            last_tau = tau
            t_guess = t_new
        r_topo = obj_bary[:3] - site_bary_km
        if site_ecef_km is not None and obj_vel is not None:
            rep = CartesianRepresentation(obj_bary[:3] * u.km)
            rep = rep.with_differentials(CartesianDifferential(obj_vel * u.km / u.s))
            sc_obj_icrs = SkyCoord(rep, frame=ICRS(), obstime=t_guess)
            site_loc = EarthLocation.from_geocentric(
                site_ecef_km[0] * u.km,
                site_ecef_km[1] * u.km,
                site_ecef_km[2] * u.km,
            )
            altaz = AltAz(obstime=t_obs, location=site_loc, pressure=0.0 * u.bar)
            sc_obj_altaz = sc_obj_icrs.transform_to(altaz)
            sc_apparent_icrs = sc_obj_altaz.transform_to(ICRS())
            ra_deg = float(sc_apparent_icrs.ra.deg)
            dec_deg = float(sc_apparent_icrs.dec.deg)
            s_ab = radec_to_unit(ra_deg, dec_deg)
        else:
            s_unit = r_topo / (np.linalg.norm(r_topo) + 1e-30)
            s_ab = aberrate_direction_first_order(s_unit, site_vel_bary_km_s)
            ra_deg, dec_deg = unit_to_radec(s_ab)
        debug = PredictDebug(
            iterations=it + 1,
            tau_s=float(last_tau if last_tau is not None else 0.0),
            obj_bary_km=list(obj_bary[:3]),
            r_topo_km=list(r_topo),
            unit_topovec=list(s_ab),
        )
        return ra_deg, dec_deg, debug

    t_guess = obs_time_iso
    last_tau = None
    obj_bary = None
    for it in range(max_iters):
        obj_vec = horizons_vectors_cached(jpl_cmd, t_guess, center="@ssb", refplane="earth", limiter=limiter)
        obj_bary = obj_vec.copy()
        r_topo = obj_bary[:3] - site_bary_km
        R = float(np.linalg.norm(r_topo))
        tau = R / C_KM_S
        t_new = iso_minus_seconds(obs_time_iso, tau)
        if last_tau is not None and abs(tau - last_tau) < tol_s:
            break
        last_tau = tau
        t_guess = t_new
    r_topo = obj_bary[:3] - site_bary_km
    s_unit = r_topo / (np.linalg.norm(r_topo) + 1e-30)
    s_ab = aberrate_direction_first_order(s_unit, site_vel_bary_km_s)
    ra_deg, dec_deg = unit_to_radec(s_ab)
    debug = PredictDebug(
        iterations=it + 1,
        tau_s=float(last_tau if last_tau is not None else 0.0),
        obj_bary_km=list(obj_bary[:3]),
        r_topo_km=list(r_topo),
        unit_topovec=list(s_ab),
    )
    return ra_deg, dec_deg, debug


# ----------------------
# Inversion: apparent -> obj bary
# ----------------------

def invert_apparent_to_obj_bary(
    jpl_cmd: str,
    obs_time_iso: str,
    site_bary_km: np.ndarray,
    site_vel_bary_km_s: np.ndarray,
    s_unit: np.ndarray,
    limiter: Optional[GlobalRateLimiter] = None,
    max_iters: int = 8,
    tol_r_km: float = 1e-6,
    tol_t_s: float = 1e-4,
) -> Tuple[np.ndarray, str, int]:
    """
    TDB-aware inversion: Given an apparent direction unit vector s_unit (aberrated) and site bary pos,
    iteratively recover the object barycentric state and emission time (TDB).
    """
    if _HAVE_ASTROPY:
        t_obs = Time(obs_time_iso, scale="utc")
        t_obs_tdb = t_obs.tdb
        t_em_guess = t_obs_tdb - TimeDelta(0.5 * DAY_S, format="sec")
        last_rho = None
        for it in range(max_iters):
            obj_vec = horizons_vectors_cached(jpl_cmd, t_em_guess.iso, center="@ssb", refplane="frame", limiter=limiter)
            obj_bary = obj_vec.copy()
            diff = obj_bary[:3] - site_bary_km
            rho = float(np.dot(diff, s_unit))
            if rho <= 0:
                rho = float(np.linalg.norm(diff))
            tau = float(np.linalg.norm(diff)) / C_KM_S
            t_em_new = t_obs_tdb - TimeDelta(tau, format="sec")
            dr = 0.0 if last_rho is None else abs(rho - last_rho)
            dt_sec = abs((t_em_new - t_em_guess).to("s").value)
            last_rho = rho
            t_em_guess = t_em_new
            if dr < tol_r_km and dt_sec < tol_t_s:
                return obj_bary, t_em_guess.iso, it + 1
        return obj_bary, t_em_guess.iso, max_iters

    t_em_guess = iso_minus_seconds(obs_time_iso, 0.5 * DAY_S)
    last_rho = None
    for it in range(max_iters):
        obj_vec = horizons_vectors_cached(jpl_cmd, t_em_guess, center="@ssb", refplane="earth", limiter=limiter)
        obj_bary = obj_vec.copy()
        diff = obj_bary[:3] - site_bary_km
        rho = float(np.dot(diff, s_unit))
        if rho <= 0:
            rho = float(np.linalg.norm(diff))
        tau = float(np.linalg.norm(diff)) / C_KM_S
        t_em_new = iso_minus_seconds(obs_time_iso, tau)
        dr = 0.0 if last_rho is None else abs(rho - last_rho)
        dt_sec = abs((parse_iso_utc(t_em_new) - parse_iso_utc(t_em_guess)).total_seconds())
        last_rho = rho
        t_em_guess = t_em_new
        if dr < tol_r_km and dt_sec < tol_t_s:
            return obj_bary, t_em_guess, it + 1
    return obj_bary, t_em_guess, max_iters


# ----------------------
# Site barycentric pos/vel builder (from MPC obs-codes)
# ----------------------

def site_bary_and_vel_from_mpc_site(
    site_code: str,
    obs_time_iso: str,
    mpc_catalog: Dict[str, Any],
    earth_bary_km: np.ndarray,
    earth_vel_km_s: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    entry = mpc_catalog.get(site_code)
    if entry is None:
        return earth_bary_km, earth_vel_km_s
    ecef = mpc_site_to_ecef_km(entry)
    if ecef is None:
        return earth_bary_km, earth_vel_km_s
    if _HAVE_ASTROPY:
        try:
            t_obs = Time(obs_time_iso, scale="utc")
            t_obs_tdb = t_obs.tdb
            loc = EarthLocation.from_geocentric(ecef[0] * u.km, ecef[1] * u.km, ecef[2] * u.km)
            gcrs = loc.get_gcrs(obstime=t_obs_tdb)
            site_eci = gcrs.cartesian.xyz.to(u.km).value
            try:
                v_site = gcrs.velocity.d_xyz.to(u.km / u.s).value
            except Exception:
                omega = np.array([0.0, 0.0, OMEGA_EARTH])
                v_site = np.cross(omega, site_eci)
            site_bary = earth_bary_km + site_eci
            site_vel_bary = earth_vel_km_s + v_site
            return site_bary, site_vel_bary
        except Exception:
            pass
    dt = parse_iso_utc(obs_time_iso)
    jd_ut1 = utc_to_jd(dt)
    gmst = gmst_from_jd_ut1(jd_ut1)
    site_eci = ecef_to_eci(ecef, gmst)
    omega = np.array([0.0, 0.0, OMEGA_EARTH])
    v_site = np.cross(omega, site_eci)
    site_bary = earth_bary_km + site_eci
    site_vel_bary = earth_vel_km_s + v_site
    return site_bary, site_vel_bary


# ----------------------
# Per-site stats & main orchestration
# ----------------------

def per_site_summary(obs_list: List[Dict[str, Any]], pred_ra: List[float], pred_dec: List[float]) -> Dict[str, Any]:
    obs_ra = np.array([o["ra_deg"] for o in obs_list], dtype=float)
    obs_dec = np.array([o["dec_deg"] for o in obs_list], dtype=float)
    dx, dy = signed_deltas(np.array(pred_ra), np.array(pred_dec), obs_ra, obs_dec)
    sep = np.sqrt(dx * dx + dy * dy)
    groups = defaultdict(list)
    for i, o in enumerate(obs_list):
        groups[(o.get("site") or "UNK").strip().upper()].append(i)
    summary = {}
    for site, idxs in groups.items():
        arr_dx = dx[idxs]
        arr_dy = dy[idxs]
        arr_sep = sep[idxs]
        median_dx = float(np.median(arr_dx))
        median_dy = float(np.median(arr_dy))
        mad_dx = float(np.median(np.abs(arr_dx - median_dx))) * 1.4826
        mad_dy = float(np.median(np.abs(arr_dy - median_dy))) * 1.4826
        rms = float(np.sqrt(np.mean(arr_sep**2)))
        cov = (
            np.cov(np.vstack([arr_dx, arr_dy]), bias=True)
            if len(idxs) >= 2
            else np.array([[mad_dx * mad_dx, 0.0], [0.0, mad_dy * mad_dy]])
        )
        summary[site] = {
            "n": len(idxs),
            "median_dx": median_dx,
            "median_dy": median_dy,
            "mad_dx": mad_dx,
            "mad_dy": mad_dy,
            "rms": rms,
            "cov": cov.tolist(),
        }
    return summary


def load_obs_csv(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, newline="") as fh:
        r = csv.DictReader(fh)
        for row in r:
            out.append(
                {
                    "t_utc": row["t_utc"],
                    "ra_deg": float(row["ra_deg"]),
                    "dec_deg": float(row["dec_deg"]),
                    "sigma_arcsec": float(row.get("sigma_arcsec", "0.5")),
                    "site": (row.get("site") or "").strip().upper(),
                }
            )
    return out


def run_validator(
    obs_list: List[Dict[str, Any]],
    jpl_cmd: str,
    site_catalog: Dict[str, Any],
    fixed_only: bool,
    limiter: Optional[GlobalRateLimiter],
    out_csv: str = "geom_validator_out.csv",
):
    if fixed_only:
        obs_list = [
            o
            for o in obs_list
            if (o.get("site") or "").strip().upper() in site_catalog
            and site_catalog[(o.get("site") or "").strip().upper()].get("lon_deg") is not None
        ]
    if len(obs_list) == 0:
        raise RuntimeError("No observations to process")

    unique_epochs = sorted(set([o["t_utc"] for o in obs_list]))
    print(f"[INFO] unique epochs: {len(unique_epochs)}")
    epoch_to_earth = {}
    for epoch in unique_epochs:
        v = horizons_vectors_cached("399", epoch, center="@ssb", refplane="earth", limiter=limiter)
        epoch_to_earth[epoch] = v

    results = []
    pred_ra = []
    pred_dec = []
    for i, ob in enumerate(obs_list):
        site = (ob.get("site") or "").strip().upper()
        epoch = ob["t_utc"]
        earth_vec = epoch_to_earth[epoch]
        earth_bary = earth_vec[:3].copy()
        earth_vel = earth_vec[3:].copy()
        site_bary, site_vel = site_bary_and_vel_from_mpc_site(site, epoch, site_catalog, earth_bary, earth_vel)
        site_entry = site_catalog.get(site)
        site_ecef = mpc_site_to_ecef_km(site_entry) if site_entry is not None else None
        ra_pred, dec_pred, dbg = predict_apparent_radec_for_obs(
            jpl_cmd,
            epoch,
            site_bary,
            site_vel,
            earth_bary_km=earth_bary,
            earth_vel_km_s=earth_vel,
            site_ecef_km=site_ecef,
            limiter=limiter,
        )
        pred_ra.append(ra_pred)
        pred_dec.append(dec_pred)
        s_unit = np.array(dbg.unit_topovec, dtype=float)
        obj_bary_inv, t_em_iso, inv_iters = invert_apparent_to_obj_bary(
            jpl_cmd, epoch, site_bary, site_vel, s_unit, limiter=limiter
        )
        obj_bary_forward = np.array(dbg.obj_bary_km, dtype=float)
        km_diff = np.linalg.norm(obj_bary_inv[:3] - obj_bary_forward[:3])
        r_topo_geoc = obj_bary_forward[:3] - earth_bary
        s_g = r_topo_geoc / (np.linalg.norm(r_topo_geoc) + 1e-30)
        s_g_ab = aberrate_direction_first_order(s_g, earth_vel)
        ra_500, dec_500 = unit_to_radec(s_g_ab)
        s_500_unit = radec_to_unit(ra_500, dec_500)
        obj_bary_from_500, t_em_from_500, iters_from_500 = invert_apparent_to_obj_bary(
            jpl_cmd, epoch, earth_bary, earth_vel, s_500_unit, limiter=limiter
        )
        r_topo_back = obj_bary_from_500[:3] - site_bary
        s_back = r_topo_back / (np.linalg.norm(r_topo_back) + 1e-30)
        s_back_ab = aberrate_direction_first_order(s_back, site_vel)
        ra_back_site, dec_back_site = unit_to_radec(s_back_ab)
        dx_back, dy_back = signed_deltas(ra_back_site, dec_back_site, ra_pred, dec_pred)
        sep_back = math.hypot(dx_back, dy_back)
        results.append(
            {
                "idx": i,
                "time_utc": epoch,
                "site": site,
                "obs_ra": ob["ra_deg"],
                "obs_dec": ob["dec_deg"],
                "pred_ra": ra_pred,
                "pred_dec": dec_pred,
                "km_diff_forward_inverse": float(km_diff),
                "ra_500": ra_500,
                "dec_500": dec_500,
                "ra_back_site": ra_back_site,
                "dec_back_site": dec_back_site,
                "sep_back_arcsec": float(sep_back),
                "inv_iters": int(inv_iters),
                "inv_from_500_iters": int(iters_from_500),
                "predict_debug": dbg,
            }
        )
    summary = per_site_summary(obs_list, pred_ra, pred_dec)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        hdr = [
            "idx",
            "time_utc",
            "site",
            "obs_ra",
            "obs_dec",
            "pred_ra",
            "pred_dec",
            "km_diff_forward_inverse",
            "ra_500",
            "dec_500",
            "ra_back_site",
            "dec_back_site",
            "sep_back_arcsec",
            "inv_iters",
            "inv_from_500_iters",
            "predict_debug",
        ]
        w.writerow(hdr)
        for r in results:
            w.writerow(
                [
                    r["idx"],
                    r["time_utc"],
                    r["site"],
                    r["obs_ra"],
                    r["obs_dec"],
                    r["pred_ra"],
                    r["pred_dec"],
                    r["km_diff_forward_inverse"],
                    r["ra_500"],
                    r["dec_500"],
                    r["ra_back_site"],
                    r["dec_back_site"],
                    r["sep_back_arcsec"],
                    r["inv_iters"],
                    r["inv_from_500_iters"],
                    json.dumps(getattr(r["predict_debug"], "__dict__", r["predict_debug"])),
                ]
            )
    return results, summary


# ----------------------
# CLI
# ----------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obs-csv", help="CSV with columns t_utc,ra_deg,dec_deg,sigma_arcsec,site")
    p.add_argument("--target", required=True, help="MPC target id or name")
    p.add_argument("--fixed-only", action="store_true")
    p.add_argument("--output", default="geom_validator_out.csv")
    p.add_argument(
        "--max-rps", type=float, default=DEFAULT_MAX_RPS, help="Max requests per second for polite querying"
    )
    args = p.parse_args()

    if not args.obs_csv:
        print("Please generate an MPC-style obs CSV (or change this script to fetch MPC directly).")
        return 1

    target_id = _normalize_horizons_id(args.target)
    if target_id != args.target:
        print(f"[INFO] normalized target '{args.target}' -> '{target_id}' for Horizons")

    obs_list = load_obs_csv(args.obs_csv)
    print("[INFO] loaded", len(obs_list), "observations")
    print("[INFO] reading MPC site catalog (cached)")
    site_catalog = fetch_mpc_obs_catalog_cached()
    print("[INFO] site catalog entries:", len(site_catalog))
    limiter = GlobalRateLimiter(args.max_rps)
    print("[INFO] running validator (polite: max_rps=", args.max_rps, ")")
    results, summary = run_validator(
        obs_list, target_id, site_catalog, args.fixed_only, limiter, out_csv=args.output
    )
    print("\nPer-site summary:")
    for site, s in sorted(summary.items(), key=lambda kv: -kv[1]["n"]):
        print(site, s)
    print("\nWrote per-observation CSV:", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
