"""Drop structurally constant columns from the extracted view families.

A family must declare the width it actually carries.  A constant column is
harmless to the classifier — `StandardScaler` leaves zero-variance features at
scale 1 and centring turns them into zeros — but it inflates the declared
width, and the placebo control of section 6.3 matches a family's
dimensionality with random features.  A placebo built against an inflated
width is not the control it claims to be.

Run after `extract-views`.  Idempotent: a second run finds nothing to drop.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Every visibility-graph generator emits the edge (i, i+1), so the graph is
#: connected by construction and both columns are exactly 1.0 for every
#: record.  See views/series_to_graph/visibility.py.
STRUCTURAL = {
    "graph_hvg": ("n_components", "largest_cc_frac"),
    "graph_lphvg1": ("n_components", "largest_cc_frac"),
}

#: The quantile transition graph is not connected by construction — an
#: unvisited bin would split it — but with 1000 samples over 10 bins it is
#: connected in every record extracted so far.  Dropped on the empirical
#: ground, which `--keep-empirical` overrides.
EMPIRICAL = {
    "graph_transition": ("n_components", "largest_cc_frac"),
}


def _suffix(column: str) -> str:
    return column.split("__", 2)[2]


def normalize(cache_dir: Path, *, keep_empirical: bool, dry_run: bool) -> None:
    rules = dict(STRUCTURAL)
    if not keep_empirical:
        rules.update(EMPIRICAL)

    for family, measures in sorted(rules.items()):
        path = cache_dir / f"{family}.parquet"
        if not path.exists():
            print(f"{family:20s} skipped: no cache")
            continue
        frame = pd.read_parquet(path)
        doomed = [
            column
            for column in frame.columns
            if column != "ecg_id" and _suffix(column) in measures
        ]
        if not doomed:
            print(f"{family:20s} already normalized")
            continue

        varying = [column for column in doomed if frame[column].nunique(dropna=False) > 1]
        if varying:
            raise SystemExit(
                f"{family}: refusing to drop {len(varying)} column(s) that are NOT "
                f"constant, e.g. {varying[0]}. Re-check the assumption before "
                "dropping anything."
            )

        kept = [column for column in frame.columns if column not in doomed]
        before = len(frame.columns) - 1
        after = len(kept) - 1
        print(
            f"{family:20s} {before} -> {after} features "
            f"({len(doomed)} constant columns: {', '.join(sorted(measures))})"
            + (" [dry run]" if dry_run else "")
        )
        if dry_run:
            continue

        temporary = path.with_suffix(".parquet.tmp")
        frame[kept].to_parquet(temporary, index=False)
        os.replace(temporary, path)

        manifest_path = path.with_suffix(".parquet.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        n_leads = len(manifest["leads"])
        manifest["feature_count"] = after
        manifest["features_per_lead"] = after // n_leads
        manifest["dropped_constant_columns"] = sorted(measures)
        manifest["dropped_constant_reason"] = (
            "connected by construction: every visibility-graph generator emits (i, i+1)"
            if family in STRUCTURAL
            else "constant in every extracted record, though not guaranteed by construction"
        )
        manifest_tmp = manifest_path.with_suffix(".json.tmp")
        manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(manifest_tmp, manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path, default=PROJECT_ROOT / "data" / "features" / "views"
    )
    parser.add_argument(
        "--keep-empirical",
        action="store_true",
        help="Keep the columns that are constant in practice but not by construction.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report without writing.")
    args = parser.parse_args()
    normalize(args.cache_dir, keep_empirical=args.keep_empirical, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
