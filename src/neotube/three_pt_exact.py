from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from astropy.time import Time, TimeDelta

from .constants import AU_KM, C_KM_S, GM_SUN
from .models import Observation
from .propagate import _body_posvel_km_single, _site_states
from .site_checks import filter_special_sites


PLANET_GM = {
    "mercury": 22032.080,
    "venus": 324858.592,
    "earth": 398600.435436,
    "mars": 42828.375214,
    "jupiter": 126686534.0,
    "saturn": 37940626.0,
    "uranus": 5794548.0,
    "neptune": 6836527.0,
}


@dataclass
class ThreePointRoot:
    rho_km: float
    rhodot_km_s: float
    state: np.ndarray
    s_em: np.ndarray
    sdot_em: np.ndarray
    sddot_em: np.ndarray
    ok: bool
    error: str | None = None
    mc: dict[str, float] | None = None


def _radec_to_unit_vector(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    return np.array(
        [math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)],
        dtype=float,
    )


def _wrap_ra_delta(ra_ref_deg: float, ra_deg: float) -> float:
    return (ra_deg - ra_ref_deg + 180.0) % 360.0 - 180.0


def _fit_tangent_plane_quadratic_coeffs(
    observations: Sequence[Observation], epoch: Time
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(observations) != 3:
        raise ValueError("Exactly 3 observations required for quadratic fit.")
    ref = observations[1]
    ra0_deg = ref.ra_deg
    dec0_deg = ref.dec_deg
    ra0 = math.radians(ra0_deg)
    dec0 = math.radians(dec0_deg)
    cos_dec0 = math.cos(dec0)
    s0 = np.array(
        [math.cos(dec0) * math.cos(ra0), math.cos(dec0) * math.sin(ra0), math.sin(dec0)],
        dtype=float,
    )
    e_ra = np.array([-math.sin(ra0), math.cos(ra0), 0.0], dtype=float)
    e_dec = np.array(
        [-math.cos(ra0) * math.sin(dec0), -math.sin(ra0) * math.sin(dec0), math.cos(dec0)],
        dtype=float,
    )
    times = np.array([(ob.time.tdb - epoch.tdb).to_value("s") for ob in observations], dtype=float)
    A = np.vstack([np.ones_like(times), times, times**2]).T
    x = np.empty(3, dtype=float)
    y = np.empty(3, dtype=float)
    for i, ob in enumerate(observations):
        dra_deg = _wrap_ra_delta(ra0_deg, ob.ra_deg)
        dra_rad = math.radians(dra_deg)
        ddec_rad = math.radians(ob.dec_deg - dec0_deg)
        x[i] = cos_dec0 * dra_rad
        y[i] = ddec_rad
    coef_x = np.linalg.solve(A, x)
    coef_y = np.linalg.solve(A, y)
    return coef_x, coef_y, s0, e_ra, e_dec


def _fit_tangent_plane_quadratic_coeffs_with_times(
    observations: Sequence[Observation],
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(observations) != 3:
        raise ValueError("Exactly 3 observations required for quadratic fit.")
    if times.shape[0] != 3:
        raise ValueError("Times array must have length 3.")
    ref = observations[1]
    ra0_deg = ref.ra_deg
    dec0_deg = ref.dec_deg
    ra0 = math.radians(ra0_deg)
    dec0 = math.radians(dec0_deg)
    cos_dec0 = math.cos(dec0)
    s0 = np.array(
        [math.cos(dec0) * math.cos(ra0), math.cos(dec0) * math.sin(ra0), math.sin(dec0)],
        dtype=float,
    )
    e_ra = np.array([-math.sin(ra0), math.cos(ra0), 0.0], dtype=float)
    e_dec = np.array(
        [-math.cos(ra0) * math.sin(dec0), -math.sin(ra0) * math.sin(dec0), math.cos(dec0)],
        dtype=float,
    )
    A = np.vstack([np.ones_like(times), times, times**2]).T
    x = np.empty(3, dtype=float)
    y = np.empty(3, dtype=float)
    for i, ob in enumerate(observations):
        dra_deg = _wrap_ra_delta(ra0_deg, ob.ra_deg)
        dra_rad = math.radians(dra_deg)
        ddec_rad = math.radians(ob.dec_deg - dec0_deg)
        x[i] = cos_dec0 * dra_rad
        y[i] = ddec_rad
    coef_x = np.linalg.solve(A, x)
    coef_y = np.linalg.solve(A, y)
    return coef_x, coef_y, s0, e_ra, e_dec


def _eval_tangent_plane_derivs(
    coef_x: np.ndarray,
    coef_y: np.ndarray,
    s0: np.ndarray,
    e_ra: np.ndarray,
    e_dec: np.ndarray,
    epoch: Time,
    t: Time,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_rel = (t.tdb - epoch.tdb).to_value("s")
    a_x, b_x, c_x = coef_x
    a_y, b_y, c_y = coef_y
    x = a_x + b_x * t_rel + c_x * (t_rel**2)
    xdot = b_x + 2.0 * c_x * t_rel
    xddot = 2.0 * c_x
    y = a_y + b_y * t_rel + c_y * (t_rel**2)
    ydot = b_y + 2.0 * c_y * t_rel
    yddot = 2.0 * c_y
    s = s0 + x * e_ra + y * e_dec
    s = s / np.linalg.norm(s)
    sdot = xdot * e_ra + ydot * e_dec
    sddot = xddot * e_ra + yddot * e_dec
    sdot = sdot - np.dot(s, sdot) * s
    sddot = sddot - np.dot(s, sddot) * s
    return s, sdot, sddot


def _tangent_basis(
    s: np.ndarray, sdot: np.ndarray, tol: float = 1e-16
) -> tuple[np.ndarray, np.ndarray]:
    s = s / np.linalg.norm(s)
    norm_sdot = np.linalg.norm(sdot)
    if norm_sdot > tol:
        t1 = sdot / norm_sdot
        t1 = t1 - np.dot(s, t1) * s
        t1 = t1 / np.linalg.norm(t1)
    else:
        z = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(z, s)) > 0.9:
            z = np.array([0.0, 1.0, 0.0])
        t1 = np.cross(s, z)
        t1 = t1 / np.linalg.norm(t1)
    t2 = np.cross(s, t1)
    t2 = t2 / np.linalg.norm(t2)
    return t1, t2


def _default_nbody_accel(r_helio: np.ndarray, t: Time) -> np.ndarray:
    rnorm = float(np.linalg.norm(r_helio))
    if rnorm <= 0:
        raise RuntimeError("Zero heliocentric radius for accel.")
    acc = -float(GM_SUN) * r_helio / (rnorm**3)
    sun_pos, _ = _body_posvel_km_single("sun", t)
    for body, gm in PLANET_GM.items():
        pos_bary, _ = _body_posvel_km_single(body, t)
        body_helio = pos_bary - sun_pos
        rel = body_helio - r_helio
        d = float(np.linalg.norm(rel))
        if d <= 0:
            continue
        acc += gm * rel / (d**3)
    return acc


def _emission_epoch_for_rho(rho_km: float, obs_time: Time) -> Time:
    tau = rho_km / C_KM_S
    return obs_time - TimeDelta(tau, format="sec")


def _compute_accels_for_rho(
    rho_km: float,
    fit: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    epoch: Time,
    obs_ref: Observation,
    accel_func: Callable[[np.ndarray, Time], np.ndarray],
    dt_site_fd: float = 60.0,
    *,
    light_time_iter: int = 2,
    debug: bool = False,
) -> dict[str, object]:
    coef_x, coef_y, s0, e_ra, e_dec = fit
    t_obs = obs_ref.time
    t_em = _emission_epoch_for_rho(rho_km, t_obs)
    site_pos_obs, _ = _site_states(
        [t_obs],
        [obs_ref.site],
        observer_positions_km=[obs_ref.observer_pos_km],
        observer_velocities_km_s=None,
        allow_unknown_site=True,
    )
    sun_pos_obs, sun_vel_obs = _body_posvel_km_single("sun", t_obs)
    earth_bary_obs, earth_bary_vel_obs = _body_posvel_km_single("earth", t_obs)
    earth_helio_obs = earth_bary_obs - sun_pos_obs
    obs_pos_helio = earth_helio_obs + site_pos_obs[0]
    for _ in range(max(0, light_time_iter)):
        s_em, sdot_em, sddot_em = _eval_tangent_plane_derivs(
            coef_x, coef_y, s0, e_ra, e_dec, epoch, t_em
        )
        sun_pos_em, sun_vel_em = _body_posvel_km_single("sun", t_em)
        earth_bary_em, earth_bary_vel_em = _body_posvel_km_single("earth", t_em)
        earth_helio = earth_bary_em - sun_pos_em
        site_pos_arr, site_vel_arr = _site_states(
            [t_em],
            [obs_ref.site],
            observer_positions_km=[obs_ref.observer_pos_km],
            observer_velocities_km_s=None,
            allow_unknown_site=True,
        )
        site_pos = site_pos_arr[0]
        site_vel = site_vel_arr[0]
        r_helio = earth_helio + site_pos + rho_km * s_em
        dist = float(np.linalg.norm(r_helio - obs_pos_helio))
        t_new = t_obs - TimeDelta(dist / C_KM_S, format="sec")
        if abs((t_new - t_em).to_value("s")) < 1e-6:
            break
        t_em = t_new

    s_em, sdot_em, sddot_em = _eval_tangent_plane_derivs(
        coef_x, coef_y, s0, e_ra, e_dec, epoch, t_em
    )
    site_pos_arr, site_vel_arr = _site_states(
        [t_em],
        [obs_ref.site],
        observer_positions_km=[obs_ref.observer_pos_km],
        observer_velocities_km_s=None,
        allow_unknown_site=True,
    )
    site_pos = site_pos_arr[0]
    site_vel = site_vel_arr[0]

    sun_pos, sun_vel = _body_posvel_km_single("sun", t_em)
    earth_bary, earth_bary_vel = _body_posvel_km_single("earth", t_em)
    earth_helio = earth_bary - sun_pos
    earth_helio_vel = earth_bary_vel - sun_vel

    r_helio = earth_helio + site_pos + rho_km * s_em
    a_total = accel_func(r_helio, t_em)

    dt = float(dt_site_fd)
    times = [t_em - TimeDelta(dt, format="sec"), t_em + TimeDelta(dt, format="sec")]
    site_positions, site_velocities = _site_states(
        times,
        [obs_ref.site] * 2,
        observer_positions_km=[obs_ref.observer_pos_km] * 2,
        observer_velocities_km_s=None,
        allow_unknown_site=True,
    )
    ddot_site = (site_velocities[1] - site_velocities[0]) / (2.0 * dt)

    _, vel_minus_e = _body_posvel_km_single("earth", times[0])
    _, vel_plus_e = _body_posvel_km_single("earth", times[1])
    _, sun_vel_minus = _body_posvel_km_single("sun", times[0])
    _, sun_vel_plus = _body_posvel_km_single("sun", times[1])
    ddot_earth = ((vel_plus_e - sun_vel_plus) - (vel_minus_e - sun_vel_minus)) / (2.0 * dt)

    if debug:
        print("DEBUG: t_em:", t_em.iso)
        print(
            "DEBUG: |s_em|, |sdot_em|, |sddot_em|:",
            np.linalg.norm(s_em),
            np.linalg.norm(sdot_em),
            np.linalg.norm(sddot_em),
        )
        print("DEBUG: site_pos (km) norm, type:", np.linalg.norm(site_pos), type(site_pos))
        print("DEBUG: site_vel (km/s) norm:", np.linalg.norm(site_vel))
        print("DEBUG: earth_helio norm (km):", np.linalg.norm(earth_helio))
        print("DEBUG: r_helio norm (km):", np.linalg.norm(r_helio))
        print("DEBUG: a_total norm (km/s^2):", np.linalg.norm(a_total))
        print("DEBUG: ddot_site norm (km/s^2):", np.linalg.norm(ddot_site))
        print("DEBUG: ddot_earth norm (km/s^2):", np.linalg.norm(ddot_earth))
        print("DEBUG: expected solar accel at 1 AU: ~", float(GM_SUN) / (AU_KM**2))

    return {
        "t_em": t_em,
        "s_em": s_em,
        "sdot_em": sdot_em,
        "sddot_em": sddot_em,
        "site_pos": site_pos,
        "site_vel": site_vel,
        "earth_helio": earth_helio,
        "earth_helio_vel": earth_helio_vel,
        "r_helio": r_helio,
        "a_total": a_total,
        "ddot_site": ddot_site,
        "ddot_earth": ddot_earth,
    }


def _F_of_rho(
    rho_km: float,
    fit: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    epoch: Time,
    obs_ref: Observation,
    accel_func: Callable[[np.ndarray, Time], np.ndarray],
) -> float:
    data = _compute_accels_for_rho(rho_km, fit, epoch, obs_ref, accel_func)
    s_em = data["s_em"]
    sdot_em = data["sdot_em"]
    sddot_em = data["sddot_em"]
    ddot_site = data["ddot_site"]
    ddot_earth = data["ddot_earth"]
    a_total = data["a_total"]
    _, t2 = _tangent_basis(s_em, sdot_em)
    lhs = rho_km * float(np.dot(t2, sddot_em)) + float(np.dot(t2, ddot_site))
    rhs = float(np.dot(t2, (-a_total - ddot_earth)))
    return lhs - rhs


def _find_rho_roots(
    fit: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    epoch: Time,
    obs_ref: Observation,
    accel_func: Callable[[np.ndarray, Time], np.ndarray],
    rho_min_km: float,
    rho_max_km: float,
    n_grid: int = 300,
) -> list[float]:
    logmin = math.log10(max(1e-12, rho_min_km))
    logmax = math.log10(max(rho_min_km, rho_max_km))
    grid = np.logspace(logmin, logmax, num=n_grid, base=10.0)
    fvals = np.zeros(len(grid), dtype=float)
    for i, rho in enumerate(grid):
        try:
            fvals[i] = _F_of_rho(rho, fit, epoch, obs_ref, accel_func)
        except Exception:
            fvals[i] = np.nan

    roots: list[float] = []
    for i in range(len(grid) - 1):
        f1 = fvals[i]
        f2 = fvals[i + 1]
        if not np.isfinite(f1) or not np.isfinite(f2):
            continue
        if f1 == 0.0:
            roots.append(grid[i])
        if f1 * f2 < 0.0:
            ra = grid[i]
            rb = grid[i + 1]
            fa = f1
            fb = f2
            for _ in range(80):
                rm = math.sqrt(ra * rb)
                try:
                    fm = _F_of_rho(rm, fit, epoch, obs_ref, accel_func)
                except Exception:
                    rm = 0.5 * (ra + rb)
                    fm = _F_of_rho(rm, fit, epoch, obs_ref, accel_func)
                if not np.isfinite(fm):
                    ra = rm
                    continue
                if fa * fm <= 0:
                    rb = rm
                    fb = fm
                else:
                    ra = rm
                    fa = fm
                if abs(rb - ra) / max(1.0, ra) < 1e-12:
                    break
            roots.append(0.5 * (ra + rb))
    roots_sorted = sorted(roots)
    uniq: list[float] = []
    for r in roots_sorted:
        if not uniq:
            uniq.append(r)
            continue
        if abs(math.log10(r) - math.log10(uniq[-1])) > 1e-6:
            uniq.append(r)
    return uniq


def _rhodot_and_state_for_rho(
    rho_km: float,
    fit: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    epoch: Time,
    obs_ref: Observation,
    accel_func: Callable[[np.ndarray, Time], np.ndarray],
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = _compute_accels_for_rho(rho_km, fit, epoch, obs_ref, accel_func)
    s_em = data["s_em"]
    sdot_em = data["sdot_em"]
    sddot_em = data["sddot_em"]
    site_vel = data["site_vel"]
    a_total = data["a_total"]
    ddot_site = data["ddot_site"]
    ddot_earth = data["ddot_earth"]
    earth_helio = data["earth_helio"]
    earth_helio_vel = data["earth_helio_vel"]
    r_helio = data["r_helio"]

    t1, _ = _tangent_basis(s_em, sdot_em)
    numer = float(np.dot(t1, (-a_total - ddot_earth))) - rho_km * float(
        np.dot(t1, sddot_em)
    ) - float(np.dot(t1, ddot_site))
    denom = 2.0 * float(np.linalg.norm(sdot_em))
    if denom <= 0:
        raise RuntimeError("sdot norm zero; cannot compute rhodot.")
    rhodot_km_s = numer / denom

    v_geo = site_vel + rhodot_km_s * s_em + rho_km * sdot_em
    v_helio = earth_helio_vel + v_geo
    state = np.hstack([r_helio, v_helio])
    return rhodot_km_s, state, s_em, sdot_em, sddot_em


def solve_three_point_exact(
    observations: Sequence[Observation],
    *,
    rho_min_au: float = 1e-4,
    rho_max_au: float = 100.0,
    n_grid: int = 300,
    nmc: int = 0,
    rng_seed: int | None = None,
    accel_func: Callable[[np.ndarray, Time], np.ndarray] | None = None,
    debug: bool = False,
) -> dict[str, object]:
    observations = filter_special_sites(
        observations, skip_special_sites=False, fail_unknown_site=True
    )
    if len(observations) != 3:
        raise ValueError("Need exactly 3 observations.")
    obs_ref = observations[1]
    epoch = Time(
        np.mean([ob.time.tdb.jd for ob in observations]),
        format="jd",
        scale="tdb",
    )

    if accel_func is None:
        accel_func = _default_nbody_accel

    fit = _fit_tangent_plane_quadratic_coeffs(observations, epoch)
    s0, sdot0, sddot0 = _eval_tangent_plane_derivs(*fit, epoch, epoch)

    rho_min_km = float(rho_min_au) * AU_KM
    rho_max_km = float(rho_max_au) * AU_KM
    if debug:
        rho_probe = math.sqrt(rho_min_km * rho_max_km)
        _compute_accels_for_rho(rho_probe, fit, epoch, obs_ref, accel_func, debug=True)

    roots = _find_rho_roots(fit, epoch, obs_ref, accel_func, rho_min_km, rho_max_km, n_grid)

    root_objs: list[ThreePointRoot] = []
    for rho in roots:
        try:
            rhodot, state, s_em, sdot_em, sddot_em = _rhodot_and_state_for_rho(
                rho, fit, epoch, obs_ref, accel_func
            )
            root_objs.append(
                ThreePointRoot(
                    rho_km=float(rho),
                    rhodot_km_s=float(rhodot),
                    state=state,
                    s_em=s_em,
                    sdot_em=sdot_em,
                    sddot_em=sddot_em,
                    ok=True,
                )
            )
        except Exception as exc:
            root_objs.append(
                ThreePointRoot(
                    rho_km=float(rho),
                    rhodot_km_s=float("nan"),
                    state=np.full(6, float("nan")),
                    s_em=np.full(3, float("nan")),
                    sdot_em=np.full(3, float("nan")),
                    sddot_em=np.full(3, float("nan")),
                    ok=False,
                    error=str(exc),
                )
            )

    mc = None
    if nmc > 0 and root_objs:
        rng = np.random.default_rng(rng_seed)
        rho_samples = [[] for _ in root_objs]
        rhodot_samples = [[] for _ in root_objs]
        s_samples = [[] for _ in root_objs]
        sdot_samples = [[] for _ in root_objs]
        sddot_samples = [[] for _ in root_objs]

        for _ in range(int(nmc)):
            perturbed = []
            for ob in observations:
                sigma_arc = max(1e-6, float(ob.sigma_arcsec))
                sigma_dec_deg = sigma_arc / 3600.0
                sigma_ra_deg = sigma_arc / 3600.0 / max(1e-8, math.cos(math.radians(ob.dec_deg)))
                ra_i = ob.ra_deg + rng.normal(0.0, sigma_ra_deg)
                dec_i = ob.dec_deg + rng.normal(0.0, sigma_dec_deg)
                perturbed.append(
                    Observation(
                        time=ob.time,
                        ra_deg=float(ra_i),
                        dec_deg=float(dec_i),
                        sigma_arcsec=ob.sigma_arcsec,
                        site=ob.site,
                        observer_pos_km=ob.observer_pos_km,
                        mag=ob.mag,
                        sigma_mag=ob.sigma_mag,
                    )
                )
            try:
                fit_p = _fit_tangent_plane_quadratic_coeffs(perturbed, epoch)
            except Exception:
                continue
            roots_p = _find_rho_roots(
                fit_p, epoch, perturbed[1], accel_func, rho_min_km, rho_max_km, n_grid
            )
            if not roots_p:
                continue
            for idx, root in enumerate(root_objs):
                nominal = root.rho_km
                best_idx = int(
                    np.argmin([abs(math.log10(rp) - math.log10(nominal)) for rp in roots_p])
                )
                rho_p = roots_p[best_idx]
                try:
                    rhodot_p, _, s_em_p, sdot_em_p, sddot_em_p = _rhodot_and_state_for_rho(
                        rho_p, fit_p, epoch, perturbed[1], accel_func
                    )
                except Exception:
                    continue
                rho_samples[idx].append(float(rho_p))
                rhodot_samples[idx].append(float(rhodot_p))
                s_samples[idx].append(s_em_p)
                sdot_samples[idx].append(sdot_em_p)
                sddot_samples[idx].append(sddot_em_p)

        for idx, root in enumerate(root_objs):
            if not rho_samples[idx]:
                root.mc = {"n_samples": 0}
                continue
            rho_arr = np.array(rho_samples[idx], dtype=float)
            rhodot_arr = np.array(rhodot_samples[idx], dtype=float)
            s_arr = np.vstack(s_samples[idx])
            sdot_arr = np.vstack(sdot_samples[idx])
            sddot_arr = np.vstack(sddot_samples[idx])
            root.mc = {
                "n_samples": int(len(rho_arr)),
                "rho_mean": float(np.mean(rho_arr)),
                "rho_std": float(np.std(rho_arr)),
                "rhodot_mean": float(np.mean(rhodot_arr)),
                "rhodot_std": float(np.std(rhodot_arr)),
                "s_mean": np.mean(s_arr, axis=0).tolist(),
                "s_std": np.std(s_arr, axis=0).tolist(),
                "sdot_mean": np.mean(sdot_arr, axis=0).tolist(),
                "sdot_std": np.std(sdot_arr, axis=0).tolist(),
                "sddot_mean": np.mean(sddot_arr, axis=0).tolist(),
                "sddot_std": np.std(sddot_arr, axis=0).tolist(),
            }
        mc = {"nmc": int(nmc)}

    return {
        "epoch": epoch,
        "s": s0,
        "sdot": sdot0,
        "sddot": sddot0,
        "roots": root_objs,
        "mc": mc,
    }


def solve_three_point_coupled(
    observations: Sequence[Observation],
    *,
    rho_min_au: float = 1e-4,
    rho_max_au: float = 100.0,
    n_grid: int = 200,
    n_iter: int = 3,
    accel_func: Callable[[np.ndarray, Time], np.ndarray] | None = None,
) -> dict[str, object]:
    observations = filter_special_sites(
        observations, skip_special_sites=False, fail_unknown_site=True
    )
    if len(observations) != 3:
        raise ValueError("Need exactly 3 observations.")
    obs_ref = observations[1]
    epoch = Time(
        np.mean([ob.time.tdb.jd for ob in observations]),
        format="jd",
        scale="tdb",
    )
    if accel_func is None:
        accel_func = _default_nbody_accel

    rho_min_km = float(rho_min_au) * AU_KM
    rho_max_km = float(rho_max_au) * AU_KM

    times = np.array([(ob.time.tdb - epoch.tdb).to_value("s") for ob in observations], dtype=float)
    fit = _fit_tangent_plane_quadratic_coeffs_with_times(observations, times)
    roots = _find_rho_roots(fit, epoch, obs_ref, accel_func, rho_min_km, rho_max_km, n_grid)
    if not roots:
        return {"epoch": epoch, "s": np.nan, "sdot": np.nan, "sddot": np.nan, "roots": []}

    root_objs: list[ThreePointRoot] = []
    for rho_guess in roots:
        rho_ref = float(rho_guess)
        t_em = [ob.time for ob in observations]
        fit_it = fit
        for _ in range(max(1, int(n_iter))):
            rho_list: list[float] = []
            for ob in observations:
                roots_i = _find_rho_roots(
                    fit_it, epoch, ob, accel_func, rho_min_km, rho_max_km, n_grid
                )
                if roots_i:
                    rho_pick = min(roots_i, key=lambda r: abs(math.log10(r) - math.log10(rho_ref)))
                else:
                    rho_pick = rho_ref
                rho_list.append(float(rho_pick))
            t_em = [ob.time - TimeDelta(r / C_KM_S, format="sec") for ob, r in zip(observations, rho_list)]
            times = np.array([(t.tdb - epoch.tdb).to_value("s") for t in t_em], dtype=float)
            fit_it = _fit_tangent_plane_quadratic_coeffs_with_times(observations, times)
            roots_ref = _find_rho_roots(
                fit_it, epoch, obs_ref, accel_func, rho_min_km, rho_max_km, n_grid
            )
            if roots_ref:
                rho_ref = min(roots_ref, key=lambda r: abs(math.log10(r) - math.log10(rho_ref)))

        try:
            rhodot, state, s_em, sdot_em, sddot_em = _rhodot_and_state_for_rho(
                rho_ref, fit_it, epoch, obs_ref, accel_func
            )
            root_objs.append(
                ThreePointRoot(
                    rho_km=float(rho_ref),
                    rhodot_km_s=float(rhodot),
                    state=state,
                    s_em=s_em,
                    sdot_em=sdot_em,
                    sddot_em=sddot_em,
                    ok=True,
                )
            )
        except Exception as exc:
            root_objs.append(
                ThreePointRoot(
                    rho_km=float(rho_ref),
                    rhodot_km_s=float("nan"),
                    state=np.full(6, float("nan")),
                    s_em=np.full(3, float("nan")),
                    sdot_em=np.full(3, float("nan")),
                    sddot_em=np.full(3, float("nan")),
                    ok=False,
                    error=str(exc),
                )
            )

    s0, sdot0, sddot0 = _eval_tangent_plane_derivs(*fit_it, epoch, obs_ref.time)
    return {"epoch": epoch, "s": s0, "sdot": sdot0, "sddot": sddot0, "roots": root_objs}


def solve_three_point_gauss(
    observations: Sequence[Observation],
    *,
    accel_mu: float = GM_SUN,
) -> dict[str, object]:
    observations = filter_special_sites(
        observations, skip_special_sites=False, fail_unknown_site=True
    )
    if len(observations) != 3:
        raise ValueError("Need exactly 3 observations.")
    obs1, obs2, obs3 = observations
    t1 = obs1.time.tdb
    t2 = obs2.time.tdb
    t3 = obs3.time.tdb
    tau1 = float((t1 - t2).to_value("s"))
    tau3 = float((t3 - t2).to_value("s"))
    tau = tau3 - tau1

    rhohat1 = _radec_to_unit_vector(obs1.ra_deg, obs1.dec_deg)
    rhohat2 = _radec_to_unit_vector(obs2.ra_deg, obs2.dec_deg)
    rhohat3 = _radec_to_unit_vector(obs3.ra_deg, obs3.dec_deg)

    site_pos, _ = _site_states(
        [obs1.time, obs2.time, obs3.time],
        [obs1.site, obs2.site, obs3.site],
        observer_positions_km=[obs1.observer_pos_km, obs2.observer_pos_km, obs3.observer_pos_km],
        observer_velocities_km_s=None,
        allow_unknown_site=True,
    )
    sun1, _ = _body_posvel_km_single("sun", obs1.time)
    sun2, _ = _body_posvel_km_single("sun", obs2.time)
    sun3, _ = _body_posvel_km_single("sun", obs3.time)
    earth1, _ = _body_posvel_km_single("earth", obs1.time)
    earth2, _ = _body_posvel_km_single("earth", obs2.time)
    earth3, _ = _body_posvel_km_single("earth", obs3.time)
    R1 = (earth1 - sun1) + site_pos[0]
    R2 = (earth2 - sun2) + site_pos[1]
    R3 = (earth3 - sun3) + site_pos[2]

    D0 = float(np.dot(rhohat1, np.cross(rhohat2, rhohat3)))
    D11 = float(np.dot(np.cross(R1, rhohat2), rhohat3))
    D12 = float(np.dot(np.cross(R2, rhohat2), rhohat3))
    D13 = float(np.dot(np.cross(R3, rhohat2), rhohat3))
    D21 = float(np.dot(np.cross(rhohat1, R1), rhohat3))
    D22 = float(np.dot(np.cross(rhohat1, R2), rhohat3))
    D23 = float(np.dot(np.cross(rhohat1, R3), rhohat3))
    D31 = float(np.dot(rhohat1, np.cross(rhohat2, R1)))
    D32 = float(np.dot(rhohat1, np.cross(rhohat2, R2)))
    D33 = float(np.dot(rhohat1, np.cross(rhohat2, R3)))

    if abs(D0) < 1e-12:
        return {"roots": []}

    A = (-D12 * (tau3 / tau) + D22 + D32 * (tau1 / tau)) / D0
    B = (
        D12 * (tau3**2 - tau**2) * (tau3 / tau)
        + D32 * (tau**2 - tau1**2) * (tau1 / tau)
    ) / (6.0 * D0)

    R2_norm = float(np.linalg.norm(R2))
    R2_dot = float(np.dot(R2, rhohat2))
    a = -(A**2 + 2.0 * A * R2_dot + R2_norm**2)
    b = -2.0 * accel_mu * B * (A + R2_dot)
    c = -(accel_mu * B) ** 2

    coeffs = [1.0, 0.0, a, 0.0, 0.0, b, 0.0, 0.0, c]
    roots_r2 = np.roots(coeffs)
    roots_r2 = [r.real for r in roots_r2 if abs(r.imag) < 1e-6 and r.real > 0]

    root_objs: list[ThreePointRoot] = []
    for r2 in roots_r2:
        rho2 = A + accel_mu * B / (r2**3)
        rho1 = (
            (
                D11 * (tau3 / tau)
                - D21
                - D31 * (tau1 / tau)
                + (accel_mu * B / (r2**3))
                * (
                    6.0 * (D11 * (tau3 / tau) + D21 + D31 * (tau1 / tau))
                )
            )
            / D0
        )
        rho3 = (
            (
                D13 * (tau3 / tau)
                - D23
                - D33 * (tau1 / tau)
                + (accel_mu * B / (r2**3))
                * (
                    6.0 * (D13 * (tau3 / tau) + D23 + D33 * (tau1 / tau))
                )
            )
            / D0
        )

        r1 = R1 + rho1 * rhohat1
        r2_vec = R2 + rho2 * rhohat2
        r3 = R3 + rho3 * rhohat3
        r2_norm = float(np.linalg.norm(r2_vec))
        mu = float(accel_mu)
        f1 = 1.0 - (mu / (2.0 * r2_norm**3)) * (tau1**2)
        f3 = 1.0 - (mu / (2.0 * r2_norm**3)) * (tau3**2)
        g1 = tau1 - (mu / (6.0 * r2_norm**3)) * (tau1**3)
        g3 = tau3 - (mu / (6.0 * r2_norm**3)) * (tau3**3)
        denom = f1 * g3 - f3 * g1
        if abs(denom) < 1e-12:
            continue
        v2 = (-f3 * r1 + f1 * r3) / denom
        state = np.hstack([r2_vec, v2])

        s = r2_vec / (np.linalg.norm(r2_vec) + 1e-12)
        sdot = np.zeros(3, dtype=float)
        sddot = np.zeros(3, dtype=float)
        root_objs.append(
            ThreePointRoot(
                rho_km=float(rho2),
                rhodot_km_s=float(np.dot(v2 - (earth2 - sun2), s)),
                state=state,
                s_em=s,
                sdot_em=sdot,
                sddot_em=sddot,
                ok=True,
            )
        )

    return {"roots": root_objs}
