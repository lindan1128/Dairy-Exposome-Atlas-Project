#!/usr/bin/env python3
"""Five-period sensitivity for humid-vs-dry heat lag-recovery profiles."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().with_name("10_build_humid_dry_era_lag_profile.py")
spec = importlib.util.spec_from_file_location("humid_dry_era_lag_profile", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

module.ERAS = {
    "2000-2004": (2000, 2004),
    "2005-2009": (2005, 2009),
    "2010-2014": (2010, 2014),
    "2015-2019": (2015, 2019),
    "2020-2025": (2020, 2025),
}
module.OUT_PROFILE = module.TAB / "point2_humid_dry_era_lag_profile_5era.csv"

if __name__ == "__main__":
    raise SystemExit(module.main())
