from neotube.sites import load_observatories, get_site_location, get_site_kind, get_site_ephemeris

for code in ("Z22", "B50", "D29"):
    catalog = load_observatories(refresh=False)
    entry = catalog.get(code)
    print("===", code, "===")
    print("catalog entry:", entry)
    print("site_kind:", get_site_kind(code))
    print("site_location (EarthLocation or None):", get_site_location(code))
    print("site_ephemeris:", get_site_ephemeris(code))
    print()
