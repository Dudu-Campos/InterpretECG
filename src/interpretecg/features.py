"""Cacheable Catch22 feature extraction for PTB-XL."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pycatch22
import wfdb
from tqdm import tqdm

from .data import LEADS


def _extract_record(job: tuple[int, str, tuple[str, ...]]) -> dict[str, float | int]:
    ecg_id, record_path, requested_leads = job
    try:
        signal, fields = wfdb.rdsamp(record_path)
        lead_positions = {name: index for index, name in enumerate(fields["sig_name"])}
        missing = set(requested_leads) - set(lead_positions)
        if missing:
            raise ValueError(f"missing leads: {sorted(missing)}")
        result: dict[str, float | int] = {"ecg_id": ecg_id}
        for lead in requested_leads:
            values = signal[:, lead_positions[lead]].astype(float).tolist()
            features = pycatch22.catch22_all(values, catch24=False)
            for name, value in zip(features["names"], features["values"]):
                result[f"{lead}__{name}"] = float(value)
        return result
    except Exception as error:
        raise RuntimeError(f"Catch22 extraction failed for ECG {ecg_id}: {error}") from error


@dataclass
class Catch22Extractor:
    """Extract 22 features per lead and incrementally persist them to Parquet."""

    cache_path: Path
    leads: tuple[str, ...] = LEADS
    n_jobs: int = 8
    checkpoint_every: int = 500

    def __post_init__(self) -> None:
        self.cache_path = Path(self.cache_path).resolve()
        self.leads = tuple(self.leads)
        unknown = set(self.leads) - set(LEADS)
        if unknown:
            raise ValueError(f"Unknown PTB-XL leads: {sorted(unknown)}")
        if not self.leads:
            raise ValueError("At least one lead is required.")
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be at least 1.")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be at least 1.")

    @property
    def manifest_path(self) -> Path:
        return self.cache_path.with_suffix(self.cache_path.suffix + ".json")

    def _read_cache(self) -> pd.DataFrame:
        if not self.cache_path.exists():
            return pd.DataFrame(columns=["ecg_id"])
        if not self.manifest_path.exists():
            raise RuntimeError(f"Cache manifest is missing: {self.manifest_path}")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if tuple(manifest["leads"]) != self.leads:
            raise RuntimeError(
                f"Cache uses leads {manifest['leads']}, requested {list(self.leads)}. "
                "Use a different --cache path."
            )
        return pd.read_parquet(self.cache_path)

    def _write_cache(self, frame: pd.DataFrame) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame = frame.sort_values("ecg_id").drop_duplicates("ecg_id", keep="last")
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, self.cache_path)
        manifest = {
            "extractor": "catch22",
            "catch24": False,
            "leads": list(self.leads),
            "features_per_lead": 22,
            "feature_count": 22 * len(self.leads),
        }
        manifest_tmp = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(manifest_tmp, self.manifest_path)

    def extract(self, records: pd.DataFrame) -> pd.DataFrame:
        required = {"ecg_id", "record_path", "signal_available"}
        missing_columns = required - set(records.columns)
        if missing_columns:
            raise ValueError(f"Record index is missing columns: {sorted(missing_columns)}")
        unavailable = records.loc[~records["signal_available"].astype(bool), "ecg_id"]
        if not unavailable.empty:
            raise ValueError(
                f"Feature extraction received {len(unavailable)} unavailable records. "
                "Filter signal_available first or complete the download."
            )

        cached = self._read_cache()
        requested_ids = records["ecg_id"].astype(int)
        completed_ids = set(cached["ecg_id"].astype(int)) if not cached.empty else set()
        pending = records.loc[~requested_ids.isin(completed_ids)]
        jobs = [
            (int(row.ecg_id), str(row.record_path), self.leads)
            for row in pending.itertuples(index=False)
        ]

        additions: list[dict[str, float | int]] = []
        if jobs:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                iterator: Iterable[dict[str, float | int]] = executor.map(
                    _extract_record, jobs, chunksize=max(1, min(20, len(jobs) // self.n_jobs))
                )
                for result in tqdm(iterator, total=len(jobs), unit="ECG", desc="Catch22"):
                    additions.append(result)
                    if len(additions) >= self.checkpoint_every:
                        cached = pd.concat([cached, pd.DataFrame(additions)], ignore_index=True)
                        self._write_cache(cached)
                        additions.clear()
        if additions:
            cached = pd.concat([cached, pd.DataFrame(additions)], ignore_index=True)
            self._write_cache(cached)

        if jobs and not self.cache_path.exists():
            self._write_cache(cached)
        selected = cached.loc[cached["ecg_id"].astype(int).isin(set(requested_ids))].copy()
        if len(selected) != len(records):
            raise RuntimeError(f"Expected {len(records)} feature rows, found {len(selected)}.")
        return records[["ecg_id"]].merge(selected, on="ecg_id", how="left", validate="one_to_one")
