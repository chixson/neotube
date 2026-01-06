#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from math import sqrt
import numpy as np
from astropy.time import Time

from neotube.fit_cli import load_observations
from neotube.propagate import predict_radec_from_epoch, GM_SUN

OBS_FILE = Path("runs/ceres-ground-test/obs.csv")
CKPT_FILE = Path("runs/ceres-ground-test/fit_smc_ckpt.npz")


def energy_and_orbital_stats(state_vec: np.ndarray) -> dict[str, float]:
    r = np.asarray(state_vec[:3], dtype=float)
    v = np.asarray(state_vec[3:6], dtype=float)
    rnorm = float(np.linalg.norm(r))
    vnorm = float(np.linalg.norm(v))
    if rnorm <= 0.0 or not np.isfinite(rnorm) or not np.isfinite(vnorm):
        return dict(eps=np.nan, a=np.nan, e=np.nan, rnorm=rnorm, vnorm=vnorm)
    eps = 0.5 * vnorm * vnorm - GM_SUN / rnorm
    if eps >= 0.0:
        a = np.inf
    else:
        a = -GM_SUN / (2.0 * eps)
    h = np.cross(r, v)
    hnorm = float(np.linalg.norm(h))
    if hnorm <= 0.0:
        e = np.nan
    else:
        evec = (np.cross(v, h) / GM_SUN) - (r / rnorm)
        e = float(np.linalg.norm(evec))
    return dict(eps=float(eps), a=float(a), e=e, rnorm=rnorm, vnorm=vnorm)


def ang_res_arcsec_for_state_and_obs(
    state_vec: np.ndarray, epoch, ob
) -> float:
    ra_pred, dec_pred = predict_radec_from_epoch(
        state_vec,
        epoch,
        [ob],
        perturbers=("earth", "mars", "jupiter"),
        max_step=3600.0,
        use_kepler=True,
        allow_unknown_site=True,
        light_time_iters=2,
        full_physics=True,
        include_refraction=False,
    )
    dra = (float(ra_pred[0]) - float(ob.ra_deg))
    ddec = (float(dec_pred[0]) - float(ob.dec_deg))
    dra = (dra + 180.0) % 360.0 - 180.0
    return sqrt((dra * 3600.0) ** 2 + (ddec * 3600.0) ** 2)


def summarize_resids(mat: np.ndarray, idxs: list[int]) -> None:
    print("--- per-particle residuals (arcsec) for obs1/2/3 ---")
    for j, col in enumerate(mat.T):
        print(
            "obs{}: nfin={}/{} min/med/mean/max: {:.3f} {:.3f} {:.3f} {:.3f}".format(
                idxs[j],
                int(np.sum(np.isfinite(col))),
                len(col),
                float(np.nanmin(col)),
                float(np.nanmedian(col)),
                float(np.nanmean(col)),
                float(np.nanmax(col)),
            )
        )


def main() -> None:
    obs = list(load_observations(OBS_FILE, None))
    if not CKPT_FILE.exists():
        raise FileNotFoundError(f"checkpoint not found: {CKPT_FILE}")
    ckpt = np.load(CKPT_FILE, allow_pickle=True)
    states = np.array(ckpt["states"], dtype=float)
    epoch_isot = str(ckpt["epoch_isot"])
    epoch = Time(epoch_isot) if epoch_isot else obs[0].time
    print("loaded states shape:", states.shape)
    print("checkpoint epoch:", epoch.isot)

    idxs = [0, 1, 2]
    n = len(states)
    resid_matrix = np.full((n, len(idxs)), np.nan, dtype=float)
    energy_stats: list[dict[str, float]] = []

    for i in range(n):
        state = states[i]
        energy_stats.append(energy_and_orbital_stats(state))
        for j, oi in enumerate(idxs):
            ob = obs[oi]
            try:
                resid = ang_res_arcsec_for_state_and_obs(state, epoch, ob)
            except Exception:
                resid = np.nan
            resid_matrix[i, j] = resid

    summarize_resids(resid_matrix, idxs)

    order = np.argsort(resid_matrix[:, 2])
    topk = order[:10]
    print("Top 10 particles by obs3 residual (smallest):", topk.tolist())
    for k in topk:
        print(
            k,
            "resids obs1/2/3:",
            resid_matrix[k, :],
            "energy:",
            energy_stats[k],
        )

    best_idx = int(topk[0])
    print("Best particle index at obs3:", best_idx)
    print("State vector for best particle:", states[best_idx])


if __name__ == "__main__":
    main()
