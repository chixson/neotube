# GEOMETRY.md — Notes, conventions and practical recipes

This doc is written PI -> grad-student style: precise, pragmatic, and verifiable.

## Goal (short)

This document explains the different "places" (RA/Dec products) we can compute or
request from JPL/Horizons, how they differ physically, when to use each for NEOTube
workflows (orbit fitting, site diagnostics, telescope pointing), and exactly how those
map to the two validator code paths in this repo:

* Path A — the currently canonical, hand-rolled geometry pipeline (light-time solved,
  topocentric geometry; no manual stellar aberration by default). This matches
  Horizons Quantity 1 (astrometric) very closely.
* Path B — the optional Astropy-native pipeline (ICRS@t_em -> AltAz@t_obs.tdb -> ICRS),
  which we use as an independent double-check and to build a standards-correct
  "apparent" inertial product (Q45) or to transform into the true equator/equinox of
  date (Q2/TETE) when needed.

Read this doc before you add any further transformations — the single biggest source of
confusion is mixing models (what physics is included) with frames (what coordinate
basis RA/Dec are expressed in).

## 1 — What Horizons offers (quantities you must know)

When you call Horizons/ephemerides you can request different quantities. Three are the
most important for our work:

* Quantity 1 — "Astrometric RA/Dec" (Q1).

  * What it is: geometric apparent direction corrected only for light-time (i.e., the
    classical "astrometric" place used for catalog/fit comparisons).
  * What it does NOT include: stellar (observer) aberration, gravitational light
    deflection, or atmospheric refraction.
  * When to use: orbit fitting / catalog comparisons where you want the geometric,
    light-time corrected direction that is directly comparable to an astrometric model
    or catalog reduction.
  * NEOTube mapping: Path A (no manual aberration) reproduces this to <= 0.3".

* Quantity 2 — "Apparent RA/Dec" (Q2) for Earth observers.

  * What it is: the "apparent" place in the true equator and true equinox of date for
    an Earth observer; it includes light-time, stellar aberration, gravitational light
    deflection, precession and nutation (true equator), and optionally refraction if
    requested. In short: the direction an observer on Earth would report after full IAU
    reductions.
  * When to use: true local apparent coordinates / telescope pointing that match an
    of-date equator/equinox standard.
  * Caveat: Q2 is an of-date, non-inertial frame; a direct comparison to an ICRS RA/Dec
    will produce a large offset (hundreds to thousands of arcseconds) unless you rotate
    and transform correctly.

* Quantity 45 — "Apparent (inertial) RA/Dec" (Q45).

  * What it is: the full apparent corrections (light-time, deflection, stellar
    aberration, etc.) but expressed in an inertial frame (ICRF/ICRS). This is Horizons’
    inertial apparent product — useful when you want physical apparent direction but
    still in an inertial basis.
  * When to use: if you want an apparent direction but expressed in ICRS so it can be
    compared directly to inertial sky coordinates (e.g., for validating an inertial
    aberration implementation).
  * NEOTube mapping: Path A with stellar aberration enabled matches Q45 to ~0.02".

Bottom line: Q1 ~= light-time only (astrometric), Q2 ~= apparent-of-date (true
equator/equinox, topocentric), Q45 ~= apparent-inertial (apparent physics but inertial
basis).

## 2 — Physics checklist (what each step does)

When you convert a state -> observed RA/Dec you may include any subset of:

* Light-time (photon emission epoch calculation) — must be done to get the emission
  state. Use TDB for iteration and ephemeris queries.
* Topocentric parallax (object minus site barycentric) — converts barycentric ->
  topocentric geometric vector.
* Stellar aberration (SR effect of observer velocity) — shifts direction by O(v/c) ~
  20" for Earth orbital velocity; must be applied exactly once in the correct
  frame/sign.
* Gravitational light deflection (Sun and planets) — included in high-accuracy apparent
  transforms.
* Precession/nutation / true equator vs mean equator — historical "equinox-of-date"
  bookkeeping.
* Atmospheric refraction — local; only if you intend to compare to observed Az/El.

Rule: pick the physics stack you need, and implement it exactly once. Mixing an Astropy
transform that already applies aberration with a manual aberration will double-count
and ruin the result.

## 3 — Time and frames: the most important practical rules

* Always do light-time iteration in an ephemeris time scale (TDB/TT).
  Example: t_obs = Time(obs_iso, scale="utc"); t_obs_tdb = t_obs.tdb. Run your t_em
  iteration in tdb. Query Horizons with TDB JD for ephemeris vectors.
* When building SkyCoord / transforms use explicit obstime and units.

  * SkyCoord(CartesianRepresentation(... * u.km), frame=ICRS(), obstime=t_em) for the
    object at emission time.
  * AltAz(obstime=t_obs_tdb, location=site_loc, pressure=0.0*u.bar) for the
    observer/local sky.
* EOP/IERS matters for GCRS <-> ITRS conversions (polar motion, UT1). Enable
  astropy.utils.iers.conf.auto_download = True in networked environments for best
  fidelity. In this repo, enable via NEOTUBE_ENABLE_IERS_AUTO=1.
* EarthLocation.from_geocentric expects ECEF / ITRS coordinates. Do not pass an
  already-rotated ECI vector into it. Use mpc_site_to_ecef_km(...) to compute ECEF from
  MPC site values and pass that to Astropy with units.

## 4 — How this maps to our codebase (quick)

* Path A (canonical by default)

  * Light-time iterate in TDB -> obj_bary(t_em) (existing solver).
  * Compute site_bary from Earth barycentric at t_obs + site GCRS/ECI (we validated
    obj_geoc = obj_bary(t_em) − earth_bary(t_obs) worked best empirically).
  * Default: no manual stellar aberration (best match to Horizons Q1).
  * If you enable aberration manually (or enable Astropy aberration), Path A with
    aberration matches Horizons Q45 extremely well (~0.02").
  * Environment flags: NEOTUBE_DEBUG_NO_ABERRATION, NEOTUBE_DEBUG_VEL, etc.

* Path B (Astropy-native; opt-in)

  * astropy_apparent_radec() implements:
    ICRS@t_em -> AltAz@t_obs.tdb -> ICRS (AltAz does aberration/deflection).
  * Needs: iers.conf.auto_download = True for best fidelity, EarthLocation.from_geocentric
    (site_ecef*kms), t_obs.tdb for obstime.
  * Path B as implemented gives a good inertial apparent (Q45-ish) result when fed the
    validated geocentric emission vector, but in our tests Path A with aberration matched
    Q45 better (Path A + correct aberration).
  * Path B -> to produce Q2 (true equator/equinox) you must transform the inertial
    result to TETE (true equator true equinox of date).

Important code rules in the repo:

* Never double-apply aberration. If you call transform_to(AltAz(...)), do not call
  aberrate_direction_first_order() or SR aberration afterwards.
* Choose one canonical product to validate against (we recommend Q45 for
  apparent-inertial, Q1 for astrometric). Q2 is a special "of-date" product — treat
  comparisons to it explicitly and transform into TETE.

## 5 — Recipes (exact code snippets)

### A — Canonical Path A (astrometric/Q1 by default; optional aberration => Q45)

```py
# 1. times
t_obs = Time(obs_iso, scale="utc")
t_obs_tdb = t_obs.tdb
# light-time iteration -> t_em (Time, scale="tdb") and obj_bary_km (np.array, km)

# 2. build site barycentric (earth_bary at t_obs + site_ECI_at_t_obs)
earth_bary_tobs = horizons_vectors_cached("399", t_obs_tdb.iso, center="@ssb", refplane="frame")[:3]
site_ecef_km = mpc_site_to_ecef_km(site_entry)
site_gcrs = EarthLocation.from_geocentric(...).get_gcrs(obstime=t_obs_tdb)
site_bary = earth_bary_tobs + site_gcrs.cartesian.xyz.to(u.km).value

# 3. topocentric geometry (no stellar aberration)
r_topo = obj_bary_km - site_bary
s_unit = r_topo / np.linalg.norm(r_topo)

# 4a. For Q1 (ASTROMETRIC) -> convert s_unit -> RA/Dec:
ra, dec = unit_to_radec(s_unit)

# 4b. To produce Q45 (apparent inertial), include aberration (via SR or Astropy)
s_ab = aberrate_relativistic(s_unit, site_vel_bary_km_s)
ra_q45, dec_q45 = unit_to_radec(s_ab)
```

### B — Astropy-native Path B (ICRS@t_em -> AltAz@t_obs.tdb -> ICRS)

```py
from astropy.utils import iers
iers.conf.auto_download = True   # optional / opt-in in repo

rep = CartesianRepresentation(obj_geoc_km * u.km)
if obj_vel_km_s is not None:
    rep = rep.with_differentials(CartesianDifferential(obj_vel_km_s * u.km/u.s))
sc_obj_icrs = SkyCoord(rep, frame=ICRS(), obstime=t_em)

site_loc = EarthLocation.from_geocentric(site_ecef_km[0]*u.km, site_ecef_km[1]*u.km, site_ecef_km[2]*u.km)
altaz = AltAz(obstime=t_obs_tdb, location=site_loc, pressure=0.0*u.bar)

sc_obj_altaz = sc_obj_icrs.transform_to(altaz)               # Astropy applies aberration/deflection
sc_app_icrs = sc_obj_altaz.transform_to(ICRS())              # inertial apparent (Q45-like)

# For Q2 (true equator/equinox of date)
sc_tete = sc_app_icrs.transform_to(TETE(obstime=t_obs_tdb))
ra_q2, dec_q2 = sc_tete.ra.deg, sc_tete.dec.deg
```

Notes:

* obj_geoc_km = obj_bary(t_em) - earth_bary(t_obs) is the validated variant that matched
  best empirically; test earth_bary(t_em) only if needed.
* TETE is Astropy’s True Equator/True Equinox-of-date frame; use it for Q2 comparisons.

## 6 — When to use which product (practical guidance)

* Orbit fitting / residual analysis against catalogs: use Q1 (astrometric). This is the
  canonical target for fits because it is light-time corrected but not
  stellar-aberrated (it matches catalog reductions). Path A (no manual stellar
  aberration) is the right validator for Q1.
* Pointing / what the telescope "sees" in RA/Dec (apparent inertial): use Q45
  (apparent but in inertial frame) — convenient if you want an inertial coordinate that
  includes the apparent physics. Path A with aberration or Path B (Astropy AltAz->ICRS)
  produce Q45 when done right.
* Telescope control or historical equinox-of-date formats: use Q2 (true equator/equinox
  of date) or TETE. This is the format Horizons returns for "apparent" RA/Dec for Earth
  observers. Use it only when you specifically need the of-date equator/equinox
  representation.
* Raw telescope pixels -> cataloged astrometry: that reduction pipeline usually outputs
  positions reduced to a chosen catalog (often ICRS/Gaia) and epoch; be aware the
  personal observatory may have used older catalogs or applied refraction assumptions —
  check observatory metadata.

## 7 — Practical validation checklist (to avoid regressions)

1. Unit tests

   * Assert Path A vs Horizons Q1: median_sep < 0.3" on canonical dataset (Z22 + another).
   * Assert Path A with aberration vs Horizons Q45: median_sep < 0.1".
   * Assert Path B Q45 ~= Path A Q45 to small tolerance.

2. Runtime flags

   * NEOTUBE_COMPARE_PATHS=1 — optional A/B compare, writes pred_ra_b/pred_dec_b.
   * NEOTUBE_ENABLE_IERS_AUTO=1 — use EOP for best fidelity (network required).
   * NEOTUBE_DEBUG_NO_ABERRATION, NEOTUBE_DEBUG_VEL — diagnostics.

3. CSV fields

   * When writing outputs, include which Horizons quantity you compared to (hz_ra_q1,
     hz_ra_q2, hz_ra_q45), and which internal product you wrote (pred_ra, pred_ra_b,
     pred_ra_b_tete, etc.). Make comparisons explicit.

4. "Do not double-apply" policy

   * If your code calls transform_to(AltAz(...)) or transform_to(TETE(...)), do not call
     manual aberration afterwards. Put a comment guard in code to explain why.

## 8 — Glossary (ultra short)

* ICRS/ICRF — inertial reference frame used by JPL/Astropy.
* Q1 (Astrometric) — Horizons: light-time only; the astrometric place.
* Q2 (Apparent-of-date) — Horizons: true equator/equinox of date; full apparent stack
  for Earth observers.
* Q45 (Apparent-inertial) — Horizons: apparent stack but expressed in ICRF/ICRS.
* TDB/TT/UTC — time scales; use TDB/ephemeris time for light-time iteration and Horizons
  ephemeris queries.
* TETE — True Equator, True Equinox of date (Astropy frame; useful for Q2).
* EOP/IERS — Earth Orientation Parameters (UT1/polar motion); necessary for precise
  GCRS <-> ITRS conversion.

## Final words (pragmatic)

* If you want a single canonical validator for orbit fitting: keep Path A (no manual
  aberration) as the default and compare to Horizons Q1. That is stable and matches
  catalog/fit conventions.
* If you want the "apparent inertial" product: compare Path A (with aberration) or
  Path B against Q45 and require IERS/TDB consistency.
* If you must produce "apparent-of-date" RA/Dec for pointing: transform your inertial
  apparent vector to TETE and compare to Q2; you will need to match Horizons’ exact
  precession/nutation model to close the remaining few arcseconds.

