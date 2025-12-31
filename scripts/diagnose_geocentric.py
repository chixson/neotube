#!/usr/bin/env python3
import numpy as np, math, json
from astropy.time import Time
from astropy import units as u
from astroquery.jplhorizons import Horizons
from astropy.coordinates import CartesianRepresentation

# Path to validator output CSV (adjust if needed)
CSV = "runs/ceres/geom_validator_out.csv"

# Pick an index or a site+index; here we search for the first Z22 row:
import pandas as pd
df = pd.read_csv(CSV)
row = df[df['site']=='Z22'].iloc[0]   # adjust if needed
print("Using row index:", row.name)

# Extract values from CSV (adapt keys if different)
t_obs_iso = row['time_utc']                 # observation time
pred_ra = row['pred_ra']; pred_dec = row['pred_dec']
obs_ra = row['obs_ra']; obs_dec = row['obs_dec']
# parse debug JSON if present
dbg = row.get('predict_debug')
try:
    dbg = json.loads(dbg) if isinstance(dbg, str) else dbg
except Exception:
    dbg = row.get('predict_debug')
print("predict_debug keys:", list(dbg.keys()) if isinstance(dbg, dict) else dbg)

# Emission time and object bary from predict_debug (t_em should be stored as string or we reconstruct)
# We expect t_em diags in dbg; if not present, use dbg['tau_s'] to compute t_em = t_obs - tau
t_obs = Time(t_obs_iso, scale='utc')
if dbg and isinstance(dbg, dict) and 'tau_s' in dbg:
    tau = float(dbg['tau_s'])
    t_em = (t_obs.tdb - (tau * u.s)).tdb
else:
    # fallback: if dbg contains obj emission time string
    if dbg and isinstance(dbg, dict) and 't_em_iso' in dbg:
        t_em = Time(dbg['t_em_iso'], scale='tdb')
    else:
        raise RuntimeError("No tau/t_em in predict_debug; please include emission time.")

# Get our obj_bary from predict_debug (forward)
obj_bary = np.array(dbg['obj_bary_km'], dtype=float)   # km

# Need Earth bary at observation epoch (t_obs). Query Horizons for body 399 at t_obs (TDB)
def hz_vec_km(idstr, epoch_time, center='@ssb', refplane='frame'):
    # returns 6-vector in km, km/s using astroquery Horizons
    h = Horizons(id=str(idstr), location=center, epochs=epoch_time.tdb.jd, id_type='majorbody' if str(idstr).isdigit() and int(str(idstr))<2000000 else 'smallbody')
    vec = h.vectors(refplane=refplane)[0]
    AU = 149597870.7
    DAY_S = 86400.0
    x = float(vec['x'])*AU
    y = float(vec['y'])*AU
    z = float(vec['z'])*AU
    vx = float(vec['vx'])*AU/DAY_S
    vy = float(vec['vy'])*AU/DAY_S
    vz = float(vec['vz'])*AU/DAY_S
    return np.array([x,y,z,vx,vy,vz], dtype=float)

# Fetch horizons object bary at t_em and earth bary at t_obs
obj_hz = hz_vec_km("1", t_em, center='@ssb', refplane='frame')
earth_hz = hz_vec_km("399", t_obs, center='@ssb', refplane='frame')

obj_hz_pos = obj_hz[:3]; earth_hz_pos = earth_hz[:3]

# Compute our geocentric vector (obj_bary(t_em) - earth_bary(t_obs))
our_geoc = obj_bary - earth_hz_pos
norm_our_geoc = np.linalg.norm(our_geoc)

# Compute horizons geocentric (obj_hz(t_em) - earth_hz(t_obs))
hz_geoc = obj_hz_pos - earth_hz_pos
norm_hz_geoc = np.linalg.norm(hz_geoc)

# compute unit vectors
u_our_geoc = our_geoc / norm_our_geoc
u_hz_geoc = hz_geoc / norm_hz_geoc

# compute dot/angle
dot_geoc = float(np.dot(u_our_geoc, u_hz_geoc))
angle_geoc_deg = math.degrees(math.acos(max(-1.0,min(1.0,dot_geoc))))

print("our_geoc_norm_km:", norm_our_geoc, "hz_geoc_norm_km:", norm_hz_geoc)
print("dot(our_geoc,hz_geoc)=", dot_geoc, "angle deg=", angle_geoc_deg)

# Also compare sc_obj_gcrs and site_gcrs we used earlier from your script (extract from CSV debug)
# We expect predict_debug to contain sc_obj_gcrs and site_gcrs; adapt naming to your debug
if dbg and isinstance(dbg, dict):
    # attempt to print sc_obj_gcrs, site_gcrs if available
    print("debug keys:", list(dbg.keys()))
    if 'r_topo_km' in dbg:
        print("r_topo_km (our) from debug:", dbg['r_topo_km'])
    # If you included sc_obj_gcrs or site_gcrs names, print them here.

# Also fetch Horizons per-observer vector at t_obs (location=site code) to compare topocentric
site_code = row['site']
hz_obs = Horizons(id="1", location=site_code, epochs=t_obs.tdb.jd, id_type='smallbody')
try:
    hz_vecs = hz_obs.vectors(refplane='frame')
    # If Horizons can return per-observer vectors, parse the x,y,z that will be topocentric or
    # wavefront? We'll try to get 'x', 'y', 'z' as the object's vector in AU relative to observer
    # Many Horizons per-observer calls return RA/DEC in ephemerides; we can instead call ephemerides()
    eph = hz_obs.ephemerides()
    hz_ra = float(eph['RA'][0]); hz_dec = float(eph['DEC'][0])
    print("Horizons apparent (observer) RA/Dec:", hz_ra, hz_dec)
except Exception as e:
    print("Could not fetch per-observer vectors from Horizons vectors(); trying ephemerides():", e)
    eph = hz_obs.ephemerides()
    hz_ra = float(eph['RA'][0]); hz_dec = float(eph['DEC'][0])
    print("Horizons apparent (observer) RA/Dec:", hz_ra, hz_dec)

# Finally compute our topo (obj_bary - site_bary(=earth_bary+site_eci)) unit and compare to horizon topo if available
# If you have site_bary stored in your debug, print comparison here.
