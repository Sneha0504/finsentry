"""Copy the key figures from a real pipeline run into the committed ``assets/``.

``reports/*`` is gitignored (regenerated on every run), but the README needs
images that render on GitHub, so the four headline figures are copied into a
tracked ``assets/`` directory. Run by ``make figures`` / ``make reproduce`` after
the pipeline has produced ``reports/*.png``.

Every copied figure therefore comes from the real run that just happened - this
script never draws anything, it only promotes the real PNGs to a committed path.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from finsentry.config import load_config

# The four headline figures the README references (all produced by Unit 3's
# deterministic figure step; see finsentry/figures.py).
FIGURES = (
    "revenue_timeseries_anomalies.png",
    "consumption_timeseries_anomalies.png",
    "reconciliation.png",
    "precision_recall.png",
)


def main() -> int:
    cfg = load_config()
    reports_dir = cfg.reports_dir
    assets_dir = cfg.project_root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    copied = []
    for name in FIGURES:
        src = reports_dir / name
        if not src.is_file() or src.stat().st_size == 0:
            missing.append(name)
            continue
        dst = assets_dir / name
        shutil.copyfile(src, dst)
        copied.append(name)

    for name in copied:
        print(f"  copied reports/{name} -> assets/{name}")
    if missing:
        print(
            "  ! missing figures (run the pipeline first: "
            "`python -m finsentry.pipeline_runner`): " + ", ".join(missing),
            file=sys.stderr,
        )
        return 1
    print(f"  {len(copied)} figure(s) committed to assets/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
