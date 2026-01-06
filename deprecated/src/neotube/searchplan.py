from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional

import yaml

from astropy.time import Time


@dataclass
class ObjectInfo:
    id: str
    id_type: str = "designation"
    sbdb_uri: Optional[str] = None


@dataclass
class TimeWindow:
    start: str
    end: str


@dataclass
class Observer:
    site_code: str
    lat_deg: Optional[float] = None
    lon_deg: Optional[float] = None
    elev_m: Optional[float] = None


@dataclass
class Archive:
    name: str
    dataset: str
    filters: list[str] = field(default_factory=lambda: ["zg", "zr"])
    product_level: str = "sci"


@dataclass
class Uncertainty:
    method: str = "sbdb_covariance"
    clones: int = 1000
    coverage_quantile: float = 0.99
    seed: int = 0
    covariance_epoch: Optional[str] = None
    clones_file: Optional[str] = None


@dataclass
class Propagation:
    backend: str = "horizons"
    step: str = "1m"
    aberrations: str = "airless"
    pyoorb_enabled: bool = False


@dataclass
class MotionModel:
    psf_fwhm_arcsec: float = 2.0
    pixel_scale_arcsec: float = 1.0
    stack_window_s: int = 1800
    tolerance_arcsec: float = 1.0
    allow_acceleration: bool = True


@dataclass
class TrackClustering:
    method: str = "kmeans"
    max_tracks: int = 200
    min_tracks: int = 10
    target_coverage: float = 0.99
    features: list[str] = field(default_factory=lambda: ["dra", "ddec", "ddra", "dddec"])


@dataclass
class PlanningConfig:
    tube_quantiles: list[float] = field(default_factory=lambda: [0.5, 0.9, 0.99])
    track_clustering: TrackClustering = field(default_factory=TrackClustering)


@dataclass
class Thresholds:
    max_total_pixel_ops: float = 5e10
    max_tracks_per_group: int = 250
    max_cutout_size_px: int = 1024
    max_total_io_gb: float = 50.0


@dataclass
class Preflight:
    thresholds: Thresholds = field(default_factory=Thresholds)
    estimates: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    go: bool = False


@dataclass
class Exposure:
    exposure_id: str
    time_mid_utc: str
    exptime_s: float
    filter: str
    pointing_ra_deg: float
    pointing_dec_deg: float
    product_url: str
    tube_r_q99: float
    motion_model_required: str = "v"
    tracks: list[dict] = field(default_factory=list)


@dataclass
class SearchPlan:
    version: int = 1
    object: ObjectInfo = field(default_factory=ObjectInfo)
    time_window: TimeWindow = field(default_factory=TimeWindow)
    observer: Observer = field(default_factory=Observer)
    archive: Archive = field(default_factory=Archive)
    uncertainty: Uncertainty = field(default_factory=Uncertainty)
    propagation: Propagation = field(default_factory=Propagation)
    motion_model: MotionModel = field(default_factory=MotionModel)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    preflight: Preflight = field(default_factory=Preflight)
    exposures: list[Exposure] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        if self.provenance:
            data["provenance"] = self.provenance
        else:
            data["provenance"] = {}
        return data

    def write_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fout:
            yaml.safe_dump(self.to_dict(), fout, sort_keys=False)

    @classmethod
    def sample_plan(cls, object_id: str, start: str, end: str, site_code: str) -> "SearchPlan":
        plan = cls(
            object=ObjectInfo(id=object_id),
            time_window=TimeWindow(start=start, end=end),
            observer=Observer(site_code=site_code),
            provenance={
                "created_utc": Time.now().utc.iso,
                "code_version": "git:latest",
                "python": ".".join(map(str, (Time.now().utc.datetime.year,))),
                "dependencies": {"typer": ">=0.9"},
            },
        )
        plan.preflight.estimates = {
            "total_ops": 0,
            "total_io_gb": 0.0,
            "group_count": 0,
        }
        return plan
