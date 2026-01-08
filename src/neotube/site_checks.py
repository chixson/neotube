from __future__ import annotations

from typing import Sequence

from .models import Observation
from .site_kind import SiteKind
from .sites import classify_site, get_site_entry


def filter_special_sites(
    observations: Sequence[Observation],
    *,
    skip_special_sites: bool,
    fail_unknown_site: bool = True,
) -> list[Observation]:
    if not observations:
        return []
    filtered: list[Observation] = []
    for ob in observations:
        if not ob.site:
            if fail_unknown_site and not skip_special_sites:
                raise ValueError("Missing site code. Provide a flag to skip these observations.")
            filtered.append(ob)
            continue
        entry = get_site_entry(ob.site)
        if entry is None:
            if skip_special_sites:
                continue
            raise ValueError(
                f"Unknown site '{ob.site}'. Provide a flag to skip these observations."
            )
        kind = classify_site(ob.site, entry)
        if kind in {SiteKind.ROVING, SiteKind.SPACECRAFT}:
            if skip_special_sites:
                continue
            raise ValueError(
                f"Unsupported site kind '{kind.value}' for site '{ob.site}'. "
                "Provide a flag to skip these observations."
            )
        filtered.append(ob)
    return filtered
