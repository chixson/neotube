from __future__ import annotations

import numpy as np
from astropy.time import Time
from dataclasses import dataclass

from .propagate import ReplicaCloud, propagate_replicas, predict_radec


@dataclass
class TubeNode:
    time: Time
    center_ra: float
    center_dec: float
    radius_arcsec: float


def build_tube(
    cloud: ReplicaCloud,
    times: list[Time],
    perturbers: tuple[str, ...] = ("earth", "mars", "jupiter"),
    coverage: float = 0.99,
    margin_arcsec: float = 5.0,
) -> list[TubeNode]:
    nodes: list[TubeNode] = []
    propagated = propagate_replicas(cloud, times, perturbers)
    for epoch, states in zip(times, propagated):
        ras = []
        decs = []
        for idx in range(states.shape[1]):
            r, d = predict_radec(states[:, idx], epoch)
            ras.append(r)
            decs.append(d)
        ras = np.array(ras)
        decs = np.array(decs)
        center_ra = np.rad2deg(
            np.arctan2(np.sum(np.sin(np.deg2rad(ras))), np.sum(np.cos(np.deg2rad(ras))))
        ) % 360.0
        center_dec = float(np.mean(decs))
        delta_ra = ((ras - center_ra + 180.0) % 360.0) - 180.0
        delta_tile = np.sqrt(
            (delta_ra * np.cos(np.deg2rad(center_dec))) ** 2 + (decs - center_dec) ** 2
        )
        quantile = np.quantile(delta_tile * 3600.0, coverage)
        nodes.append(TubeNode(epoch, center_ra, center_dec, quantile + margin_arcsec))
    return nodes
