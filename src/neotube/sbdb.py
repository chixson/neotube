from __future__ import annotations

import requests
import numpy as np

SBDB_URL = "https://ssd-api.jpl.nasa.gov/sbdb.api"


def fetch_sbdb_covariance(target: str) -> np.ndarray:
    params = {
        "sstr": target,
        "cov": "mat",
        "full-prec": "true",
    }
    resp = requests.get(SBDB_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    cov = payload.get("orbit", {}).get("covariance")
    if not isinstance(cov, dict):
        raise ValueError("SBDB response missing covariance data.")
    labels = cov.get("labels") or cov.get("label")
    if labels is not None:
        normalized = [str(lbl).strip().lower() for lbl in labels]
        needed = {"x", "y", "z", "vx", "vy", "vz"}
        if set(normalized) != needed:
            raise ValueError("SBDB covariance is not Cartesian; provide jpl_cov explicitly.")

    data = cov.get("data") or cov.get("matrix")
    if data is None:
        raise ValueError("SBDB covariance matrix missing data.")
    if isinstance(data, list) and data and isinstance(data[0], list):
        mat = np.array(data, dtype=float)
    else:
        flat = np.array(data, dtype=float).ravel()
        if flat.size != 36:
            raise ValueError("SBDB covariance matrix size unexpected.")
        mat = flat.reshape(6, 6)
    if mat.shape != (6, 6):
        raise ValueError("SBDB covariance matrix shape unexpected.")
    return mat
