"""Canonical geometry utilities for NEOTube.

This module centralizes geometry transforms, light-time iteration, site helpers,
aberration, refraction, and the Astropy Path-B wrapper (ICRS@t_em -> AltAz@t_obs -> ICRS).

Semantics / conventions:
- Times passed to light-time routines should be Astropy Time objects. Light-time
  iteration must be done in TDB (see GEOMETRY.md).
- All position vectors are kilometers (km) unless otherwise documented.
- The module provides a convenience compute_q1_q45(...) to produce canonical Q1/Q45
  RA/Dec outputs used by the repo.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    from astropy import units as u
    from astropy.coordinates import (
        EarthLocation,
        SkyCoord,
        ICRS,
        AltAz,
        CartesianRepresentation,
        CartesianDifferential,
        TETE,
    )
    from astropy.time import Time

    _HAVE_ASTROPY = True
except Exception:
    _HAVE_ASTROPY = False
    Time = object

from .sites import get_site_location

C_KM_S = 299792.458


def light_time_iterate(
    obj_bary_km: np.ndarray,
    obj_vel_km_s: Optional[np.ndarray],
    obs_bary_km: np.ndarray,
    iters: int = 2,
) -> np.ndarray:
    """Perform simple light-time iteration returning obj_bary at emission epoch."""
    obj_bary_km = np.asarray(obj_bary_km, dtype=float)
    obs_bary_km = np.asarray(obs_bary_km, dtype=float)
    out = obj_bary_km.copy()
    if obj_vel_km_s is None:
        for _ in range(iters):
            rho = np.linalg.norm(out - obs_bary_km, axis=1)
            _ = rho / C_KM_S
            break
        return out
    obj_vel_km_s = np.asarray(obj_vel_km_s, dtype=float)
    for _ in range(iters):
        rho = np.linalg.norm(out - obs_bary_km, axis=1)
        dt = rho / C_KM_S
        out = obj_bary_km - obj_vel_km_s * dt[:, None]
    return out


def mpc_site_to_ecef_km(site: Optional[str] | np.ndarray) -> Optional[np.ndarray]:
    """Return ECEF (x,y,z) km for MPC site or already-expressed ECEF vector."""
    if site is None:
        return None
    if isinstance(site, (list, tuple, np.ndarray)):
        arr = np.asarray(site, dtype=float)
        if arr.shape == (3,):
            return arr.copy()
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr.copy()
        raise ValueError("site array must be shape (3,) or (N,3)")
    loc = get_site_location(str(site))
    if loc is None:
        return None
    return np.array(
        [float(loc.x.to(u.km).value), float(loc.y.to(u.km).value), float(loc.z.to(u.km).value)],
        dtype=float,
    )


def site_ecef_to_barycentric(
    site_ecef_km: np.ndarray, time_obs: Time, get_earth_bary_callable
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute site barycentric position and velocity."""
    if not _HAVE_ASTROPY:
        raise RuntimeError("Astropy required for site_ecef_to_barycentric")
    tdb = time_obs.tdb if hasattr(time_obs, "tdb") else time_obs
    earth_bary_km, earth_bary_vel_km_s = get_earth_bary_callable(tdb)
    earth_bary_km = np.asarray(earth_bary_km, dtype=float)
    earth_bary_vel_km_s = np.asarray(earth_bary_vel_km_s, dtype=float)
    single = earth_bary_km.ndim == 1
    if single:
        earth_bary_km = earth_bary_km.reshape((1, 3))
        earth_bary_vel_km_s = earth_bary_vel_km_s.reshape((1, 3))
    site_ecef_km = np.asarray(site_ecef_km, dtype=float)
    if site_ecef_km.ndim == 1:
        site_ecef_km = site_ecef_km.reshape((1, 3))
    out_pos = []
    out_vel = []
    for i in range(site_ecef_km.shape[0]):
        loc = EarthLocation.from_geocentric(
            site_ecef_km[i, 0] * u.km,
            site_ecef_km[i, 1] * u.km,
            site_ecef_km[i, 2] * u.km,
        )
        gcrs = loc.get_gcrs(obstime=tdb)
        site_gcrs_xyz = gcrs.cartesian.xyz.to(u.km).value
        idx = 0 if single else i
        out_pos.append(earth_bary_km[idx] + np.asarray(site_gcrs_xyz).reshape(3,))
        out_vel.append(earth_bary_vel_km_s[idx])
    pos = np.vstack(out_pos)
    vel = np.vstack(out_vel)
    if single and pos.shape[0] == 1:
        return pos[0], vel[0]
    return pos, vel


def topocentric_vector(obj_bary_km: np.ndarray, site_bary_km: np.ndarray) -> np.ndarray:
    """Return r_topo = obj_bary - site_bary (supports vectorized inputs)."""
    return np.asarray(obj_bary_km, dtype=float) - np.asarray(site_bary_km, dtype=float)


def unit_to_radec(unit_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized unit-vector -> (ra_deg, dec_deg)."""
    uvec = np.asarray(unit_vec, dtype=float)
    if uvec.ndim == 1:
        uvec = uvec.reshape((1, 3))
    x = uvec[:, 0]
    y = uvec[:, 1]
    z = uvec[:, 2]
    ra = np.degrees(np.arctan2(y, x)) % 360.0
    dec = np.degrees(np.arctan2(z, np.hypot(x, y)))
    if ra.size == 1:
        return float(ra[0]), float(dec[0])
    return ra, dec


def radec_to_unit(ra_deg: float, dec_deg: float) -> np.ndarray:
    """(Possibly vectorized) RA/Dec to unit vector."""
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    return np.stack((x, y, z), axis=-1)


def aberrate_direction_first_order(topovec: np.ndarray, obs_vel_km_s: np.ndarray) -> np.ndarray:
    """Apply first-order special-relativistic aberration to a topocentric direction."""
    topovec = np.asarray(topovec, dtype=float)
    vel = np.asarray(obs_vel_km_s, dtype=float)
    if topovec.ndim == 1:
        topovec = topovec.reshape((1, 3))
    if vel.ndim == 1:
        vel = vel.reshape((1, 3))
    n = topovec / (np.linalg.norm(topovec, axis=1)[:, None] + 1e-30)
    beta = vel / C_KM_S
    nb_dot = (n * beta).sum(axis=1)
    s = n + beta - (nb_dot[:, None] * n)
    s = s / (np.linalg.norm(s, axis=1)[:, None] + 1e-30)
    if s.shape[0] == 1:
        return s[0]
    return s


def bennett_refraction(zenith_rad: float) -> float:
    """Bennett refraction approximation (radians)."""
    z_deg = np.degrees(zenith_rad)
    if z_deg >= 89.999:
        return 0.0
    r_arcmin = 1.02 / np.tan(np.radians(z_deg + 10.3 / (z_deg + 5.11)))
    return float(np.radians(r_arcmin / 60.0))


def apply_bennett_refraction(topounit: np.ndarray, site_pos_km: np.ndarray) -> np.ndarray:
    """Apply Bennett refraction to a unit topocentric vector (vectorized)."""
    topounit = np.asarray(topounit, dtype=float)
    site_pos_km = np.asarray(site_pos_km, dtype=float)
    single = False
    if topounit.ndim == 1:
        topounit = topounit.reshape((1, 3))
        single = True
    out = []
    for i in range(topounit.shape[0]):
        site = site_pos_km[i] if site_pos_km.ndim > 1 else site_pos_km
        site_norm = float(np.linalg.norm(site))
        if site_norm <= 0.0 or site_norm < 5000.0 or site_norm > 7000.0:
            out.append(topounit[i])
            continue
        z_hat = site / site_norm
        cos_z = float(np.clip(np.dot(topounit[i], z_hat), -1.0, 1.0))
        z = float(np.arccos(cos_z))
        if not np.isfinite(z) or z <= 0.0:
            out.append(topounit[i])
            continue
        delta = bennett_refraction(z)
        if delta <= 0.0 or delta >= z:
            out.append(topounit[i])
            continue
        perp = topounit[i] - cos_z * z_hat
        perp_norm = float(np.linalg.norm(perp))
        if perp_norm <= 0.0:
            out.append(topounit[i])
            continue
        e_perp = perp / perp_norm
        z_prime = z - delta
        out.append(e_perp * np.sin(z_prime) + z_hat * np.cos(z_prime))
    out = np.vstack(out)
    if single:
        return out[0]
    return out


def astropy_apparent_radec(
    obj_geoc_km: np.ndarray,
    obj_vel_km_s: Optional[np.ndarray],
    t_em: Time,
    t_obs_tdb: Time,
    site_ecef_km: np.ndarray,
    product: str = "Q45",
) -> Tuple[np.ndarray, np.ndarray]:
    """Astropy-native Path B: ICRS@t_em -> AltAz@t_obs.tdb -> ICRS."""
    if not _HAVE_ASTROPY:
        raise RuntimeError("Astropy required for astropy_apparent_radec")
    rep = CartesianRepresentation(obj_geoc_km * u.km)
    if obj_vel_km_s is not None:
        rep = rep.with_differentials(CartesianDifferential(obj_vel_km_s * u.km / u.s))
    sc_obj_icrs = SkyCoord(rep, frame=ICRS(), obstime=t_em)
    site_loc = EarthLocation.from_geocentric(
        site_ecef_km[0] * u.km, site_ecef_km[1] * u.km, site_ecef_km[2] * u.km
    )
    altaz = AltAz(obstime=t_obs_tdb, location=site_loc, pressure=0.0 * u.bar)
    sc_obj_altaz = sc_obj_icrs.transform_to(altaz)
    sc_app_icrs = sc_obj_altaz.transform_to(ICRS())
    if product == "Q45":
        return sc_app_icrs.ra.deg, sc_app_icrs.dec.deg
    if product == "Q2":
        sc_tete = sc_app_icrs.transform_to(TETE(obstime=t_obs_tdb))
        return sc_tete.ra.deg, sc_tete.dec.deg
    raise ValueError("Unsupported product; use 'Q45' or 'Q2'")


def compute_q1_q45(
    obj_bary_km: np.ndarray,
    obj_vel_km_s: Optional[np.ndarray],
    site_entry: Optional[str] | np.ndarray,
    t_em: Time,
    t_obs: Time,
    get_earth_bary_callable,
) -> dict:
    """Compute Q1 and Q45 RA/Dec for the supplied geometry."""
    site_ecef = mpc_site_to_ecef_km(site_entry)
    if site_ecef is None:
        raise ValueError("site_entry could not be resolved to ECEF")
    t_obs_tdb = t_obs.tdb if hasattr(t_obs, "tdb") else t_obs
    _ = get_earth_bary_callable(t_obs_tdb)
    site_bary_km, site_bary_vel_km_s = site_ecef_to_barycentric(
        site_ecef, t_obs, get_earth_bary_callable
    )
    r_topo = topocentric_vector(obj_bary_km, site_bary_km)
    if r_topo.ndim == 1:
        s_unit = r_topo / (np.linalg.norm(r_topo) + 1e-30)
    else:
        s_unit = r_topo / (np.linalg.norm(r_topo, axis=1)[:, None] + 1e-30)
    ra_q1, dec_q1 = unit_to_radec(s_unit)
    s_ab = aberrate_direction_first_order(
        s_unit if s_unit.ndim == 2 else s_unit.reshape((1, 3)), site_bary_vel_km_s
    )
    ra_q45, dec_q45 = unit_to_radec(s_ab)
    return {"q1": (ra_q1, dec_q1), "q45": (ra_q45, dec_q45)}
