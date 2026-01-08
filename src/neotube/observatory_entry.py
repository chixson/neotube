from __future__ import annotations

from dataclasses import dataclass
import math

from astropy import units as u
from astropy.constants import R_earth
from astropy.coordinates import EarthLocation

EARTH_RADIUS_KM = float(R_earth.to(u.km).value)


@dataclass(frozen=True)
class ObservatoryEntry:
    code: str
    lon_deg: float | None
    rho_cos_phi: float | None
    rho_sin_phi: float | None
    description: str | None
    site_kind: str | None = None
    ephemeris_id: str | None = None
    ephemeris_id_type: str | None = None
    ephemeris_location: str | None = None
    ephemeris_frame: str | None = None

    @property
    def rho(self) -> float:
        if self.rho_cos_phi is None or self.rho_sin_phi is None:
            return 0.0
        return math.hypot(self.rho_cos_phi, self.rho_sin_phi)

    def to_location(self) -> EarthLocation | None:
        if (
            self.lon_deg is None
            or self.rho_cos_phi is None
            or self.rho_sin_phi is None
            or not math.isfinite(self.lon_deg)
            or not math.isfinite(self.rho_cos_phi)
            or not math.isfinite(self.rho_sin_phi)
        ):
            return None
        lam = math.radians(self.lon_deg)
        rho = self.rho
        if rho == 0.0:
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
