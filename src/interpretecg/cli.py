"""Command-line entry point for the Catch22 baseline."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .data import (
    LEADS,
    SEALED_FOLD,
    SPLIT_REGIMES,
    SUPERCLASSES,
    DatasetIncompleteError,
    PTBXLDataLoader,
)
from .evaluate import Evaluate
from .features import Catch22Extractor
from .view_features import DEFAULT_FAMILIES, FAMILY_OPERATORS, ViewExtractor


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Extract Catch22 features and evaluate diagnostic superclasses on PTB-XL."
    )
    parser.add_argument("--data-dir", type=Path, default=project_root / "data" / "ptb-xl")
    parser.add_argument(
        "--cache",
        type=Path,
        default=project_root / "data" / "features" / "catch22_12leads.parquet",
    )
    parser.add_argument(
        "--regime",
        choices=sorted(SPLIT_REGIMES),
        default="exploratory",
        help=(
            f"Evaluation regime. 'exploratory' never reads fold {SEALED_FOLD}: "
            "use it for every comparison between views, families, parameters "
            "and controls. 'confirmatory' is the official PTB-XL protocol and "
            f"evaluates on fold {SEALED_FOLD}; it requires --confirm-test-fold."
        ),
    )
    parser.add_argument(
        "--confirm-test-fold",
        metavar="REASON",
        help=(
            f"Unseal fold {SEALED_FOLD} for one confirmatory run. The reason is "
            "appended to results/test_fold_usage.jsonl so the number of "
            "evaluations on the test fold stays auditable."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--leads", nargs="+", default=list(LEADS), choices=LEADS)
    parser.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow a partial extraction/evaluation while signals are still downloading.",
    )
    parser.add_argument(
        "--extract-only", action="store_true", help="Populate the feature cache and stop."
    )
    parser.add_argument(
        "--max-records-per-fold",
        type=int,
        help="Development-only cap sampled deterministically from each official fold.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]

    if args.regime == "confirmatory" and not args.confirm_test_fold:
        raise SystemExit(
            f"The confirmatory regime evaluates on fold {SEALED_FOLD}, which is "
            "sealed.\nEvery exploratory comparison must use --regime "
            "exploratory, so that the number of test-fold evaluations stays "
            "small and declared.\nTo proceed, state why this run needs the "
            'test fold:\n  catch22-baseline --regime confirmatory '
            '--confirm-test-fold "<reason>"'
        )
    if args.confirm_test_fold and args.regime != "confirmatory":
        raise SystemExit(
            "--confirm-test-fold only applies to --regime confirmatory."
        )
    if args.output_dir is None:
        suffix = "" if args.regime == "confirmatory" else f"_{args.regime}"
        args.output_dir = project_root / "results" / f"catch22_baseline{suffix}"

    loader = PTBXLDataLoader(args.data_dir)
    print(json.dumps(loader.summary(), indent=2))
    if not args.allow_incomplete:
        try:
            loader.assert_complete()
        except DatasetIncompleteError as error:
            raise SystemExit(f"Dataset incomplete: {error}\nRerun later or use --allow-incomplete.")

    records = loader.records
    if args.allow_incomplete:
        records = records.loc[records["signal_available"]].copy()
        if not loader.is_complete:
            print("WARNING: using an incomplete, non-final subset of PTB-XL.")
    if args.max_records_per_fold is not None:
        if args.max_records_per_fold < 1:
            raise SystemExit("--max-records-per-fold must be positive.")
        records = (
            records.sort_values("ecg_id")
            .groupby("strat_fold", group_keys=False)
            .head(args.max_records_per_fold)
            .copy()
        )

    extractor = Catch22Extractor(
        cache_path=args.cache,
        leads=tuple(args.leads),
        n_jobs=args.n_jobs,
        checkpoint_every=args.checkpoint_every,
    )
    features = extractor.extract(records)
    print(f"Feature matrix: {features.shape[0]:,} × {features.shape[1] - 1:,}")
    if args.extract_only:
        return

    evaluator = Evaluate(labels=SUPERCLASSES, n_jobs=args.n_jobs, regime=args.regime)
    results = evaluator.run(records, features)
    output_dir = args.output_dir
    if not loader.is_complete:
        output_dir = output_dir.with_name(output_dir.name + "_PARTIAL")
        results["warning"] = "NON-FINAL result from an incomplete download"
    evaluator.results_ = results
    evaluator.save(output_dir)

    if args.regime == "confirmatory":
        usage = project_root / "results" / "test_fold_usage.jsonl"
        usage.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reason": args.confirm_test_fold,
            "test_fold": SEALED_FOLD,
            "leads": list(args.leads),
            "feature_count": results["feature_count"],
            "test_macro_auroc": results["test"]["macro_auroc"],
            "complete_dataset": loader.is_complete,
            "output_dir": str(output_dir),
        }
        with usage.open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        print(f"Recorded test-fold evaluation #{sum(1 for _ in usage.open())} in {usage}")

    print(json.dumps(results, indent=2))
    print(f"Saved metrics and model under {output_dir}")


if __name__ == "__main__":
    main()


def build_views_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Extract the derived image and graph views for PTB-XL."
    )
    parser.add_argument("--data-dir", type=Path, default=project_root / "data" / "ptb-xl")
    parser.add_argument(
        "--cache-dir", type=Path, default=project_root / "data" / "features" / "views"
    )
    parser.add_argument(
        "--embedding",
        type=Path,
        default=project_root / "data" / "features" / "takens_embedding.json",
        help=(
            "Frozen Takens embedding per lead. Estimated once on training folds "
            "and reused for every record, so that the family is comparable "
            "across records."
        ),
    )
    parser.add_argument("--leads", nargs="+", default=list(LEADS), choices=LEADS)
    parser.add_argument(
        "--families",
        nargs="+",
        default=list(DEFAULT_FAMILIES),
        choices=sorted(FAMILY_OPERATORS),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=project_root / "data" / "features" / "absolute_calibration.json",
        help=(
            "Training-fold constants for the amplitude-preserving `_abs` families "
            "(scripts/calibrate_absolute_views.py)."
        ),
    )
    parser.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow a partial extraction while signals are still downloading.",
    )
    parser.add_argument(
        "--max-records-per-fold",
        type=int,
        help="Development-only cap sampled deterministically from each official fold.",
    )
    return parser


def extract_views() -> None:
    args = build_views_parser().parse_args()
    loader = PTBXLDataLoader(args.data_dir)
    if not args.allow_incomplete:
        try:
            loader.assert_complete()
        except DatasetIncompleteError as error:
            raise SystemExit(f"Dataset incomplete: {error}\nRerun later or use --allow-incomplete.")

    records = loader.records
    records = records.loc[records["signal_available"]].copy()
    if args.max_records_per_fold is not None:
        if args.max_records_per_fold < 1:
            raise SystemExit("--max-records-per-fold must be positive.")
        records = (
            records.sort_values("ecg_id")
            .groupby("strat_fold", group_keys=False)
            .head(args.max_records_per_fold)
            .copy()
        )

    extractor = ViewExtractor(
        cache_dir=args.cache_dir,
        embedding_path=args.embedding,
        leads=tuple(args.leads),
        families=tuple(args.families),
        n_jobs=args.n_jobs,
        checkpoint_every=args.checkpoint_every,
        calibration_path=args.calibration,
    )
    print(
        f"Extracting {len(args.families)} families on {len(args.leads)} leads "
        f"for {len(records):,} records with {args.n_jobs} workers."
    )
    frames = extractor.extract(records)
    total = 0
    for family, frame in frames.items():
        columns = [column for column in frame.columns if column != "ecg_id"]
        total += len(columns)
        print(f"  {family:22s} {frame.shape[0]:,} × {len(columns):,}")
    print(f"Total: {total:,} features across {len(frames)} families in {args.cache_dir}")
