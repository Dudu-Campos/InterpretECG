"""PTB-XL metadata, labels, official splits, and signal loading."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb


SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")


#: Fold assignments per evaluation regime.  ``confirmatory`` is the official
#: PTB-XL protocol (Wagner et al. 2020) and the one Strodthoff et al. report
#: on, so it is the regime whose numbers are comparable to the literature.
#: It is also the only regime that reads fold 10, which is sealed: see
#: ``PTBXLDataLoader.split_masks`` and the ``--confirm-test-fold`` flag.
SPLIT_REGIMES: dict[str, dict[str, object]] = {
    "exploratory": {"train": (1, 7), "validation": 8, "test": 9},
    "confirmatory": {"train": (1, 8), "validation": 9, "test": 10},
}

#: The fold kept out of every exploratory analysis.
SEALED_FOLD = 10


class DatasetIncompleteError(RuntimeError):
    """Raised when a full experiment is requested before all signals exist."""


class PTBXLDataLoader:
    """Load PTB-XL 100 Hz records without changing the official patient folds.

    The current project stores the WFDB pairs in a flat directory, whereas
    ``filename_lr`` contains the original PhysioNet subdirectories.  This
    loader deliberately resolves records by basename to support that layout.
    """

    def __init__(self, data_dir: str | Path, *, drop_unlabeled: bool = True) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.drop_unlabeled = drop_unlabeled
        self.database_path = self.data_dir / "ptbxl_database.csv"
        self.statements_path = self.data_dir / "scp_statements.csv"
        if not self.database_path.is_file() or not self.statements_path.is_file():
            raise FileNotFoundError(
                f"PTB-XL metadata not found under {self.data_dir}. "
                "Expected ptbxl_database.csv and scp_statements.csv."
            )
        self._all_records = self._build_index()

    @staticmethod
    def _parse_codes(raw: str) -> dict[str, float]:
        parsed = ast.literal_eval(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected an SCP-code dictionary, got {type(parsed)!r}")
        return parsed

    def _diagnostic_class_map(self) -> dict[str, str]:
        statements = pd.read_csv(self.statements_path, index_col=0)
        diagnostic = statements.loc[statements["diagnostic"].eq(1.0), "diagnostic_class"]
        return diagnostic.dropna().astype(str).to_dict()

    @staticmethod
    def _is_complete_record(record_path: Path) -> bool:
        header_path = record_path.with_suffix(".hea")
        signal_path = record_path.with_suffix(".dat")
        if not header_path.is_file() or not signal_path.is_file():
            return False
        try:
            first_line = header_path.read_text(encoding="utf-8").splitlines()[0].split()
            signal_count = int(first_line[1])
            sample_count = int(first_line[3])
            expected_bytes = signal_count * sample_count * 2  # WFDB format 16.
            return signal_path.stat().st_size == expected_bytes
        except (IndexError, OSError, ValueError):
            return False

    def _build_index(self) -> pd.DataFrame:
        records = pd.read_csv(self.database_path)
        class_by_code = self._diagnostic_class_map()
        label_sets: list[set[str]] = []
        for raw in records["scp_codes"]:
            codes = self._parse_codes(raw)
            label_sets.append({class_by_code[code] for code in codes if code in class_by_code})

        basenames = records["filename_lr"].map(lambda value: Path(value).name)
        records = records.assign(
            record_name=basenames,
            record_path=basenames.map(lambda name: str(self.data_dir / name)),
            diagnostic_superclasses=label_sets,
        )
        for label in SUPERCLASSES:
            records[label] = np.fromiter(
                (label in labels for labels in label_sets), dtype=np.int8, count=len(records)
            )
        records["has_diagnostic_label"] = records.loc[:, SUPERCLASSES].any(axis=1)
        records["signal_available"] = [
            self._is_complete_record(Path(path)) for path in records["record_path"]
        ]
        return records

    @property
    def all_records(self) -> pd.DataFrame:
        """All metadata rows, including 411 records without diagnostic labels."""
        return self._all_records.copy()

    @property
    def records(self) -> pd.DataFrame:
        """Experiment index; by default excludes records with no target label."""
        records = self._all_records
        if self.drop_unlabeled:
            records = records.loc[records["has_diagnostic_label"]]
        return records.copy()

    @property
    def is_complete(self) -> bool:
        return bool(self._all_records["signal_available"].all())

    def assert_complete(self) -> None:
        missing = self._all_records.loc[~self._all_records["signal_available"], "ecg_id"]
        if not missing.empty:
            preview = ", ".join(map(str, missing.head(8)))
            raise DatasetIncompleteError(
                f"Only {len(self._all_records) - len(missing):,}/{len(self._all_records):,} "
                f"PTB-XL signals are complete. Missing/incomplete ECG ids include: {preview}."
            )

    @staticmethod
    def split_masks(
        records: pd.DataFrame, regime: str = "exploratory"
    ) -> dict[str, pd.Series]:
        """Return train/validation/test masks for one evaluation regime.

        The confirmatory regime is the official PTB-XL protocol and is the
        only one that touches fold 10.  Everything comparative — views,
        feature families, parameter choices, controls — must run in the
        exploratory regime, so that the number of fold-10 evaluations stays
        small, declared, and auditable.
        """
        try:
            folds = SPLIT_REGIMES[regime]
        except KeyError as error:
            available = ", ".join(sorted(SPLIT_REGIMES))
            raise KeyError(
                f"unknown regime {regime!r}; available: {available}"
            ) from error
        column = records["strat_fold"]
        low, high = folds["train"]
        return {
            "train": column.between(low, high),
            "validation": column.eq(folds["validation"]),
            "test": column.eq(folds["test"]),
        }

    def load_signal(self, record: int | pd.Series) -> tuple[np.ndarray, tuple[str, ...], float]:
        """Return ``(samples × leads, lead_names, sampling_rate)`` in mV."""
        if isinstance(record, (int, np.integer)):
            matches = self._all_records.loc[self._all_records["ecg_id"].eq(int(record))]
            if matches.empty:
                raise KeyError(f"Unknown ecg_id: {record}")
            row = matches.iloc[0]
        else:
            row = record
        if not bool(row["signal_available"]):
            raise DatasetIncompleteError(f"ECG {row['ecg_id']} is missing or incomplete.")
        signal, fields = wfdb.rdsamp(str(row["record_path"]))
        return signal, tuple(fields["sig_name"]), float(fields["fs"])

    def summary(self) -> dict[str, object]:
        records = self.records
        return {
            "metadata_records": len(self._all_records),
            "target_records": len(records),
            "available_signals": int(self._all_records["signal_available"].sum()),
            "dropped_without_target": int((~self._all_records["has_diagnostic_label"]).sum()),
            "label_counts": {label: int(records[label].sum()) for label in SUPERCLASSES},
            "fold_counts": records.groupby("strat_fold").size().astype(int).to_dict(),
        }
