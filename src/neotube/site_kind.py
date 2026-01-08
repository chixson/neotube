from __future__ import annotations

from enum import Enum


class SiteKind(str, Enum):
    FIXED = "fixed"
    GEOCENTER = "geocenter"
    ROVING = "roving"
    SPACECRAFT = "spacecraft"
    UNKNOWN = "unknown"
