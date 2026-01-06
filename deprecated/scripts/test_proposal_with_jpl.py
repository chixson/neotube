#!/usr/bin/env python3
"""
scripts/test_proposal_with_jpl.py

Load runs/ceres/obs.csv, use the first two observations.
Fetch JPL/Horizons full state for Ceres at each obs epoch.
Propagate the JPL state at epoch1 to epoch2 using the repository propagator.
Print:
 - Observation RA/Dec (from CSV)
 - JPL RA/Dec at each obs time
 - RA/Dec computed from the propagated state (at epoch2)
 - rho (range) for JPL and propagated states at the observation times
 - position/velocity difference between JPL(epoch2) and the propagated state

Designed as a quick check that the repo propagator reproduces JPL state at epoch2.
"""
from pathlib import Path
import sys
import numpy as np
from astropy.time import Time, TimeDelta
import astropy.units as u

# Repo helpers
from neotube.fit_cli import load_observations
from neotube.fit import _initial_state_from_horizons
from neotube.sites import get_site_location, load_observatories
from neotube import geometry
from neotube.propagate import (
    # The repo exposes a couple of propagate helpers; we'll try variants if needed.
    # Common ones used by repo scripts:
    propagate_state,
    propagate_state_kepler,
    predict_radec_from_epoch,
)
from neotube.propagate import _prepare_obs_cache, _body_posvel_km_single, shapiro_delay_sun

# astroquery / astropy helpers for observer position
from astropy.coordinates import get_body_barycentric_posvel


N_PROPOSALS = 50
JITTER_POS_KM = 10.0
JITTER_VEL_KM_S = 1e-5


def site_description(code: str | None) -> str | None:
    if not code:
        return None
    entry = load_observatories().get(code.strip().upper())
    if entry and entry.description:
        return entry.description.strip()
    return None


def format_site_label(code: str | None) -> str:
    if not code:
        return "UNKNOWN"
    desc = site_description(code)
    if desc:
        return f"{code.strip().upper()} ({desc})"
    return code.strip().upper()


def site_geocentric_gcrs_km(site_code: str | None, t: Time) -> np.ndarray:
    """
    Return the geocentric site vector (GCRS) in km for a fixed observatory,
    or zeros if site is None or not found.
    """
    if not site_code:
        return np.zeros(3, dtype=float)
    loc = get_site_location(site_code)
    if loc is None:
        # fallback to geocenter
        return np.zeros(3, dtype=float)
    # EarthLocation.get_gcrs returns a GCRS frame object with cartesian attribute
    gcrs = loc.get_gcrs(obstime=t)
    return gcrs.cartesian.xyz.to(u.km).value.flatten()


def earth_heliocentric_km(t: Time) -> np.ndarray:
    """
    Return Earth's heliocentric vector (km) at time t (Time object).
    Implemented similarly to other repo helpers: get_body_barycentric_posvel('earth') - sun.
    Use t.tdb for solar system ephemerides.
    """
    t_tdb = t.tdb
    pb_earth = get_body_barycentric_posvel("earth", t_tdb)
    pb_sun = get_body_barycentric_posvel("sun", t_tdb)
    # pb_xxx[0].xyz is a CartesianRepresentation
    r_earth = (pb_earth[0].xyz - pb_sun[0].xyz).to(u.km).value.flatten()
    return r_earth


def observer_heliocentric_km(site: str | None, t: Time) -> np.ndarray:
    """
    Return observer position in heliocentric frame (km): Earth_heliocentric + site_geocentric (GCRS -> km).
    For missing site, returns Earth's center heliocentric.
    """
    r_earth = earth_heliocentric_km(t)
    site_geo = site_geocentric_gcrs_km(site, t)
    return r_earth + site_geo


def radec_from_state_at_epoch(state6: np.ndarray, epoch: Time, obs_list, **predict_kwargs):
    """
    Wrapper to call predict_radec_from_epoch and return ra/dec for the first state
    as flat arrays (one value per observation).
    predict_radec_from_epoch expects states shaped (Nstates,6) and returns arrays.
    """
    state = np.asarray(state6, dtype=float).reshape(-1)
    if state.size != 6:
        raise ValueError("state6 must be a 6-element state vector")
    # predict_radec_from_epoch returns (ra_pred, dec_pred)
    ra_pred, dec_pred = predict_radec_from_epoch(
        state,
        epoch,
        obs_list,
        **predict_kwargs,
    )
    return np.asarray(ra_pred, dtype=float), np.asarray(dec_pred, dtype=float)


def radec_from_states_at_epoch(states6: np.ndarray, epoch: Time, obs_list, **predict_kwargs):
    """
    Loop over multiple states and return RA/Dec arrays (one per state per observation).
    """
    states = np.asarray(states6, dtype=float)
    if states.ndim != 2 or states.shape[1] != 6:
        raise ValueError("states6 must be shape (N, 6)")
    ra_all = np.empty((states.shape[0], len(obs_list)), dtype=float)
    dec_all = np.empty((states.shape[0], len(obs_list)), dtype=float)
    for i, state in enumerate(states):
        ra_i, dec_i = radec_from_state_at_epoch(state, epoch, obs_list, **predict_kwargs)
        ra_all[i] = ra_i
        dec_all[i] = dec_i
    return ra_all, dec_all


def try_propagate_to_time(state6: np.ndarray, epoch_from: Time, epoch_to: Time):
    """
    Try to propagate state6 from epoch_from to epoch_to using the repo propagators.
    We attempt a few call signatures used elsewhere in the repo:
     - propagate_state(state6, epoch_from, [epoch_to])
     - propagate_state((r,v), epoch_from, epoch_to)
     - propagate_state_kepler(...)
    Return a 6-vector state at epoch_to or raise an exception.
    """
    state6 = np.asarray(state6, dtype=float)
    r1 = state6[:3]
    v1 = state6[3:]
    # Try 1: propagate_state(state, epoch_from, [epoch_to]) -> array-like
    try:
        out = propagate_state(state6, epoch_from, [epoch_to])
        out = np.asarray(out)
        # out might be shape (1,6) or (len(states), n_times, 6) etc.
        # repo helper patterns usually return shape (n_times, 6) for single-state input.
        if out.ndim == 2 and out.shape[1] == 6:
            # if output is (1,6), take that
            return out[0].astype(float)
        if out.ndim == 3:
            # (n_states, n_times, 6) -> extract first state, first time
            return out[0, -1, :].astype(float)
        # fallback to interpreting as r_t,v_t tuple
    except Exception:
        pass

    # Try 2: propagate_state((r,v), epoch_from, epoch_to) -> (r_t, v_t)
    try:
        rval = propagate_state((r1, v1), epoch_from, epoch_to)
        # If returns tuple (r_t, v_t)
        if isinstance(rval, tuple) and len(rval) == 2:
            r_t = np.asarray(rval[0], dtype=float)
            v_t = np.asarray(rval[1], dtype=float)
            return np.concatenate([r_t, v_t])
        # If returns single array (r_t, v_t) in some other shape
        rval = np.asarray(rval, dtype=float)
        if rval.size == 6:
            return rval.flatten()
    except Exception:
        pass

    # Try propagate_state_kepler signature variants if available
    try:
        out = propagate_state_kepler(state6, epoch_from, [epoch_to])
        out = np.asarray(out)
        if out.ndim == 2 and out.shape[1] == 6:
            return out[0].astype(float)
        if out.ndim == 3:
            return out[0, -1, :].astype(float)
    except Exception:
        pass

    # Try second signature for propagate_state_kepler
    try:
        rval = propagate_state_kepler((r1, v1), epoch_from, epoch_to)
        if isinstance(rval, tuple) and len(rval) == 2:
            r_t = np.asarray(rval[0], dtype=float)
            v_t = np.asarray(rval[1], dtype=float)
            return np.concatenate([r_t, v_t])
    except Exception:
        pass

    raise RuntimeError("All propagation attempts failed (tried propagate_state and propagate_state_kepler).")


def light_time_emit_info(state6: np.ndarray, obs_cache, idx: int, iters: int = 2) -> dict:
    """
    Compute emission time using geometry.light_time_iterate (constant-velocity iteration).
    Returns a dict with t_emit (Time), rho_km, dt_s, and c_est_km_s.
    """
    obs_bary = obs_cache.earth_bary_km[idx] + obs_cache.site_pos_km[idx]
    sun_bary, sun_vel = _body_posvel_km_single("sun", obs_cache.times_tdb[idx])
    obj_bary = state6[:3] + sun_bary
    obj_bary_vel = state6[3:] + sun_vel
    obj_emit = geometry.light_time_iterate(
        np.asarray(obj_bary, dtype=float).reshape((1, 3)),
        np.asarray(obj_bary_vel, dtype=float).reshape((1, 3)),
        np.asarray(obs_bary, dtype=float).reshape((1, 3)),
        iters=iters,
    )[0]
    rho_km = float(np.linalg.norm(obj_emit - obs_bary))
    dt_geom_s = rho_km / geometry.C_KM_S
    t_emit = obs_cache.times_tdb[idx] - TimeDelta(dt_geom_s, format="sec")
    dt_obs_s = float((obs_cache.times_tdb[idx] - t_emit).to_value("sec"))
    c_est_km_s = rho_km / dt_obs_s if dt_obs_s > 0 else float("nan")
    return dict(
        t_emit=t_emit,
        rho_km=rho_km,
        dt_geom_s=dt_geom_s,
        dt_obs_s=dt_obs_s,
        c_est_km_s=c_est_km_s,
    )


def light_time_emit_info_full(
    state6: np.ndarray,
    epoch: Time,
    obs_cache,
    idx: int,
    *,
    perturbers: tuple[str, ...],
    max_step: float,
    use_kepler: bool,
    iters: int = 2,
    full_physics: bool = True,
) -> dict:
    """
    Compute emission time using the same light-time loop as predict_radec_from_epoch,
    including Shapiro delay when full_physics=True.
    """
    t_obs = obs_cache.times_tdb[idx]
    obs_bary = obs_cache.earth_bary_km[idx] + obs_cache.site_pos_km[idx]
    t_emit = t_obs
    obj_bary = None
    dt_geom = 0.0
    dt_sh = 0.0
    for _ in range(max(1, iters)):
        if use_kepler:
            try:
                emit_state = propagate_state_kepler(state6, epoch, (t_emit,))[0]
            except Exception:
                emit_state = propagate_state(state6, epoch, (t_emit,), perturbers=perturbers, max_step=max_step)[0]
        else:
            emit_state = propagate_state(state6, epoch, (t_emit,), perturbers=perturbers, max_step=max_step)[0]
        sun_bary, _ = _body_posvel_km_single("sun", t_emit)
        obj_bary = emit_state[:3] + sun_bary
        rho = float(np.linalg.norm(obj_bary - obs_bary))
        dt_geom = rho / geometry.C_KM_S
        dt_sh = shapiro_delay_sun(obj_bary, obs_bary, sun_bary) if full_physics else 0.0
        t_emit = t_obs - TimeDelta(dt_geom + dt_sh, format="sec")
    if obj_bary is None:
        obj_bary = state6[:3]
    rho_km = float(np.linalg.norm(obj_bary - obs_bary))
    dt_obs_s = float((t_obs - t_emit).to_value("sec"))
    c_est_km_s = rho_km / dt_obs_s if dt_obs_s > 0 else float("nan")
    c_geom_km_s = rho_km / dt_geom if dt_geom > 0 else float("nan")
    c_geom_no_sh_km_s = rho_km / (dt_obs_s - dt_sh) if (dt_obs_s - dt_sh) > 0 else float("nan")
    return dict(
        t_emit=t_emit,
        rho_km=rho_km,
        dt_geom_s=dt_geom,
        dt_shapiro_s=dt_sh,
        dt_obs_s=dt_obs_s,
        c_est_km_s=c_est_km_s,
        c_geom_km_s=c_geom_km_s,
        c_geom_no_sh_km_s=c_geom_no_sh_km_s,
    )


def _resolve_obs_csv() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    return Path("runs/ceres-ground-test/obs.csv")


def main():
    obs_csv = _resolve_obs_csv()
    if not obs_csv.exists():
        raise SystemExit(f"Expected obs CSV at {obs_csv}.")

    # Load observations -> neotube.models.Observation objects
    obs_all = load_observations(obs_csv, None)
    if len(obs_all) < 2:
        raise SystemExit("Need at least two observations in CSV.")

    obs1 = obs_all[0]
    obs2 = obs_all[1]

    # Target (Ceres)
    target = "1"

    # Fetch full JPL states at each epoch (heliocentric equatorial/ICRS km, km/s)
    jpl_state1 = _initial_state_from_horizons(target, obs1.time)  # shape (6,)
    jpl_state2 = _initial_state_from_horizons(target, obs2.time)

    # Observer heliocentric positions (for rho)
    obs1_obs_helio = observer_heliocentric_km(obs1.site, obs1.time)
    obs2_obs_helio = observer_heliocentric_km(obs2.site, obs2.time)

    # Compute rho for JPL states at obs times:
    rho1_jpl = float(np.linalg.norm(jpl_state1[:3] - obs1_obs_helio))
    rho2_jpl = float(np.linalg.norm(jpl_state2[:3] - obs2_obs_helio))

    # Compute RA/Dec from JPL states at their own epochs for the matching observation:
    # (use high-fidelity prediction options similar to repo defaults)
    predict_kwargs = dict(
        perturbers=("earth", "mars", "jupiter"),
        max_step=3600.0,
        use_kepler=False,
        allow_unknown_site=True,
        light_time_iters=2,
        full_physics=True,
        include_refraction=False,
    )

    jpl_ra1, jpl_dec1 = radec_from_state_at_epoch(jpl_state1, obs1.time, [obs1], **predict_kwargs)
    jpl_ra2, jpl_dec2 = radec_from_state_at_epoch(jpl_state2, obs2.time, [obs2], **predict_kwargs)
    jpl_ra1 = float(jpl_ra1[0])
    jpl_dec1 = float(jpl_dec1[0])
    jpl_ra2 = float(jpl_ra2[0])
    jpl_dec2 = float(jpl_dec2[0])

    # Propagate jpl_state1 -> epoch2 using repo propagator (with several fallback signatures)
    try:
        prop_state_at_2 = try_propagate_to_time(jpl_state1, obs1.time, obs2.time)
    except Exception as exc:
        raise RuntimeError(f"Propagation failed: {exc}")

    # Compute rho for propagated state at obs2
    rho2_prop = float(np.linalg.norm(prop_state_at_2[:3] - obs2_obs_helio))
    # Propagated at obs1 is just jpl_state1 (no change): rho1_prop = rho1_jpl
    rho1_prop = rho1_jpl

    # RA/Dec from propagated state (treat propagated state as state at epoch2)
    prop_ra2, prop_dec2 = radec_from_state_at_epoch(prop_state_at_2, obs2.time, [obs2], **predict_kwargs)
    prop_ra2 = float(prop_ra2[0])
    prop_dec2 = float(prop_dec2[0])

    # Light-time emission estimates for obs1/obs2 using repo geometry helper
    obs_cache = _prepare_obs_cache([obs1, obs2], allow_unknown_site=True)
    emit1 = light_time_emit_info(jpl_state1, obs_cache, 0, iters=2)
    emit2 = light_time_emit_info(jpl_state2, obs_cache, 1, iters=2)
    emit1_full = light_time_emit_info_full(
        jpl_state1,
        obs1.time,
        obs_cache,
        0,
        perturbers=tuple(predict_kwargs["perturbers"]),
        max_step=float(predict_kwargs["max_step"]),
        use_kepler=bool(predict_kwargs["use_kepler"]),
        iters=int(predict_kwargs["light_time_iters"]),
        full_physics=bool(predict_kwargs["full_physics"]),
    )
    emit2_full = light_time_emit_info_full(
        jpl_state2,
        obs2.time,
        obs_cache,
        1,
        perturbers=tuple(predict_kwargs["perturbers"]),
        max_step=float(predict_kwargs["max_step"]),
        use_kepler=bool(predict_kwargs["use_kepler"]),
        iters=int(predict_kwargs["light_time_iters"]),
        full_physics=bool(predict_kwargs["full_physics"]),
    )

    # JPL states at emission times (for proposal comparisons)
    jpl_state_emit1 = _initial_state_from_horizons(target, emit1_full["t_emit"])
    jpl_state_emit2 = _initial_state_from_horizons(target, emit2_full["t_emit"])

    # Generate jittered proposals at t_emit1 and propagate to t_emit2
    rng = np.random.default_rng(42)
    proposals_emit1 = np.tile(jpl_state_emit1, (N_PROPOSALS, 1))
    proposals_emit1[:, :3] += rng.normal(scale=JITTER_POS_KM, size=(N_PROPOSALS, 3))
    proposals_emit1[:, 3:] += rng.normal(scale=JITTER_VEL_KM_S, size=(N_PROPOSALS, 3))
    proposals_emit2 = []
    for state0 in proposals_emit1:
        proposals_emit2.append(try_propagate_to_time(state0, emit1_full["t_emit"], emit2_full["t_emit"]))
    proposals_emit2 = np.asarray(proposals_emit2, dtype=float)

    # For completeness, RA/Dec of propagated state evaluated at epoch1 equals jpl_ra1/jpl_dec1
    # (we'll display that directly).

    # Compare propagated state to JPL state at epoch2 (pos, vel differences)
    dpos = prop_state_at_2[:3] - jpl_state2[:3]
    dvel = prop_state_at_2[3:] - jpl_state2[3:]
    pos_sep_km = float(np.linalg.norm(dpos))
    vel_sep_kms = float(np.linalg.norm(dvel))

    # Print results (obs2-focused summary)
    from astropy.coordinates import SkyCoord

    c_jpl2 = SkyCoord(jpl_ra2 * u.deg, jpl_dec2 * u.deg, frame="icrs")
    c_prop2 = SkyCoord(prop_ra2 * u.deg, prop_dec2 * u.deg, frame="icrs")
    c_obs2 = SkyCoord(float(obs2.ra_deg) * u.deg, float(obs2.dec_deg) * u.deg, frame="icrs")
    sep_prop_jpl_arcsec = float(c_jpl2.separation(c_prop2).arcsecond)
    sep_obs_jpl_arcsec = float(c_obs2.separation(c_jpl2).arcsecond)
    sep_obs_prop_arcsec = float(c_obs2.separation(c_prop2).arcsecond)

    prop_ra2_all, prop_dec2_all = radec_from_states_at_epoch(
        proposals_emit2,
        emit2_full["t_emit"],
        [obs2],
        **predict_kwargs,
    )
    prop_ra2_all = prop_ra2_all[:, 0]
    prop_dec2_all = prop_dec2_all[:, 0]
    c_prop_all = SkyCoord(prop_ra2_all * u.deg, prop_dec2_all * u.deg, frame="icrs")
    sep_prop_jpl_arcsec_all = c_prop_all.separation(c_jpl2).arcsecond
    sep_obs_prop_arcsec_all = c_prop_all.separation(c_obs2).arcsecond

    dpos = proposals_emit2[:, :3] - jpl_state_emit2[:3]
    dvel = proposals_emit2[:, 3:] - jpl_state_emit2[3:]
    dpos_norm = np.linalg.norm(dpos, axis=1)
    dvel_norm = np.linalg.norm(dvel, axis=1)
    rho_prop_all = np.linalg.norm(proposals_emit2[:, :3] - obs2_obs_helio, axis=1)

    def _stats(arr: np.ndarray) -> tuple[float, float, float, float]:
        arr = np.asarray(arr, dtype=float)
        return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())

    print("\n=== Obs2 Summary ===")
    print(f"Time: {obs2.time.isot}")
    print(f"Site: {format_site_label(obs2.site)}")
    print("")
    print("RA/Dec (deg)")
    print(f"  OBS : {float(obs2.ra_deg):.9f} , {float(obs2.dec_deg):.9f}")
    print(f"  JPL : {jpl_ra2:.9f} , {jpl_dec2:.9f}")
    print(f"  PROP: {prop_ra2:.9f} , {prop_dec2:.9f}")
    print("")
    print("Range rho (km)")
    print(f"  JPL : {rho2_jpl:.3f}")
    print(f"  PROP: {rho2_prop:.3f}")
    print("")
    print("Angular separation (arcsec)")
    print(f"  OBS vs JPL : {sep_obs_jpl_arcsec:.6f}")
    print(f"  OBS vs PROP: {sep_obs_prop_arcsec:.6f}")
    print(f"  PROP vs JPL: {sep_prop_jpl_arcsec:.6f}")
    print("")
    print("State diff at obs2 (propagated - JPL)")
    print(f"  |dpos| = {pos_sep_km:.6f} km")
    print(f"  |dvel| = {vel_sep_kms:.6e} km/s")
    print("")
    print("Light-time emission (geometry.light_time_iterate)")
    print(
        f"  obs1 t_emit: {emit1['t_emit'].isot}  "
        f"dt_obs={emit1['dt_obs_s']:.6f} s  dt_geom={emit1['dt_geom_s']:.6f} s"
    )
    print(
        f"  obs1 c_est:  {emit1['c_est_km_s']:.6f} km/s (delta {emit1['c_est_km_s'] - geometry.C_KM_S:+.6f})"
    )
    print(
        f"  obs2 t_emit: {emit2['t_emit'].isot}  "
        f"dt_obs={emit2['dt_obs_s']:.6f} s  dt_geom={emit2['dt_geom_s']:.6f} s"
    )
    print(
        f"  obs2 c_est:  {emit2['c_est_km_s']:.6f} km/s (delta {emit2['c_est_km_s'] - geometry.C_KM_S:+.6f})"
    )
    print("")
    print("Light-time emission (full physics loop)")
    print(
        f"  obs1 t_emit: {emit1_full['t_emit'].isot}  "
        f"dt_obs={emit1_full['dt_obs_s']:.6f} s  dt_geom={emit1_full['dt_geom_s']:.6f} s  "
        f"dt_sh={emit1_full['dt_shapiro_s']:.6e} s"
    )
    print(
        f"  obs1 c_est:  {emit1_full['c_est_km_s']:.6f} km/s "
        f"(geom {emit1_full['c_geom_km_s']:.6f}, no_sh {emit1_full['c_geom_no_sh_km_s']:.6f})"
    )
    print(
        f"  obs2 t_emit: {emit2_full['t_emit'].isot}  "
        f"dt_obs={emit2_full['dt_obs_s']:.6f} s  dt_geom={emit2_full['dt_geom_s']:.6f} s  "
        f"dt_sh={emit2_full['dt_shapiro_s']:.6e} s"
    )
    print(
        f"  obs2 c_est:  {emit2_full['c_est_km_s']:.6f} km/s "
        f"(geom {emit2_full['c_geom_km_s']:.6f}, no_sh {emit2_full['c_geom_no_sh_km_s']:.6f})"
    )
    print("")
    print("Proposals: t_emit1 -> t_emit2 (jittered JPL state)")
    print(f"  n={N_PROPOSALS}  pos_sigma={JITTER_POS_KM:.3f} km  vel_sigma={JITTER_VEL_KM_S:.3e} km/s")
    dpos_stats = _stats(dpos_norm)
    dvel_stats = _stats(dvel_norm)
    sep_jpl_stats = _stats(sep_prop_jpl_arcsec_all)
    sep_obs_stats = _stats(sep_obs_prop_arcsec_all)
    rho_stats = _stats(rho_prop_all)
    print("  |dpos| km  : mean {:.6f}  std {:.6f}  min {:.6f}  max {:.6f}".format(*dpos_stats))
    print("  |dvel| km/s: mean {:.6e}  std {:.6e}  min {:.6e}  max {:.6e}".format(*dvel_stats))
    print("  rho km     : mean {:.3f}  std {:.3f}  min {:.3f}  max {:.3f}".format(*rho_stats))
    print(
        "  sep vs JPL arcsec: mean {:.6f}  std {:.6f}  min {:.6f}  max {:.6f}".format(
            *sep_jpl_stats
        )
    )
    print(
        "  sep vs OBS arcsec: mean {:.6f}  std {:.6f}  min {:.6f}  max {:.6f}".format(
            *sep_obs_stats
        )
    )
    print("\nDone.\n")


if __name__ == "__main__":
    main()
