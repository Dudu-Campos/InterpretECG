"""Cacheable extraction of the derived views (image and graph) for PTB-XL.

One Parquet per family, so that a family can be added, recomputed or dropped
without touching the others, and so that the redundancy analysis of section
6.4 can carry a family label per feature.

The recurrence families are parameter-matched on purpose: ``image_recurrence``
reads the very same recurrence matrix that ``graph_takens`` turns into a
network, using the frozen embedding and the same recurrence rate and Theiler
window.  That is the controlled experiment of section 4.4 — identical data,
only the reading differs.

The recurrence family uses recurrence quantification (section 4.2), not
Haralick texture: the thresholded matrix is binary, and per-image
equal-probability quantization collapses it to one gray level, which makes
every texture measure constant across records.  Texture is kept for the
continuous-valued images.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from .data import LEADS
from .views.image_to_tabular import (
    haralick_feature_names,
    haralick_features,
    rqa_feature_names,
    rqa_features,
)
from .views.series_to_graph import (
    takens_recurrence_graph,
    transition_graph,
    visibility_graph,
)
from .views.series_to_graph.features import extract_features, features_to_array
from .views.series_to_image import (
    gramian_angular_field,
    markov_transition_field,
    recurrence_plot,
    scalogram,
    spectrogram,
)

#: Recurrence rate and Theiler window shared by `graph_takens` and
#: `image_recurrence`, so that both read one matrix.
RECURRENCE_RATE = 0.05
THEILER_WINDOW = 1

#: Columns of the 27-value graph descriptor that are only defined for weighted
#: graphs.  Dropping them for unweighted families is deliberate: left in, they
#: are all-NaN columns that `SimpleImputer` discards silently, which would make
#: the effective dimensionality differ from the declared one and break the
#: dimensionality matching of the placebo control (section 6.3).
WEIGHT_ONLY_COLUMNS = ("w_mean", "w_std")


def _graph_descriptor(graph, kind: str, *, weighted: bool) -> dict[str, float]:
    values, names = features_to_array(extract_features(graph, kind=kind))
    pairs = dict(zip(names, (float(value) for value in values)))
    if not weighted:
        for column in WEIGHT_ONLY_COLUMNS:
            pairs.pop(column, None)
    return pairs


#: `recurrence_rate` is fixed at RECURRENCE_RATE by construction, so as a
#: feature it is a constant column.  Dropping it keeps the declared family
#: width equal to the effective one, as for the graph weight columns.
CONSTANT_RQA_COLUMNS = ("rqa__recurrence_rate",)


def _rqa_descriptor(image: np.ndarray, *, fixed_rate: bool = True) -> dict[str, float]:
    values = rqa_features(image, theiler_window=THEILER_WINDOW)
    pairs = dict(zip(rqa_feature_names(), (float(value) for value in values)))
    if fixed_rate:
        for column in CONSTANT_RQA_COLUMNS:
            pairs.pop(column, None)
    return pairs


def _haralick_descriptor(image: np.ndarray, thresholds=None) -> dict[str, float]:
    values = haralick_features(image, thresholds=thresholds)
    return dict(zip(haralick_feature_names(), (float(value) for value in values)))


# Each family maps a lead's samples plus its frozen embedding to a flat dict.
FAMILY_OPERATORS: dict[str, Callable[[np.ndarray, dict[str, int]], dict[str, float]]] = {
    "image_spectrogram": lambda x, p: _haralick_descriptor(spectrogram(x)),
    "image_scalogram": lambda x, p: _haralick_descriptor(scalogram(x)),
    "image_gasf": lambda x, p: _haralick_descriptor(
        gramian_angular_field(x, method="summation")
    ),
    "image_gadf": lambda x, p: _haralick_descriptor(
        gramian_angular_field(x, method="difference")
    ),
    "image_mtf": lambda x, p: _haralick_descriptor(markov_transition_field(x)),
    "image_recurrence": lambda x, p: _rqa_descriptor(
        recurrence_plot(
            x,
            dimension=p["m"],
            delay=p["tau"],
            recurrence_rate=RECURRENCE_RATE,
            theiler_window=THEILER_WINDOW,
        )
    ),
    "graph_hvg": lambda x, p: _graph_descriptor(
        visibility_graph(x, kind="hvg", rho=0), "hvg", weighted=False
    ),
    "graph_lphvg1": lambda x, p: _graph_descriptor(
        visibility_graph(x, kind="hvg", rho=1), "hvg", weighted=False
    ),
    "graph_takens": lambda x, p: _graph_descriptor(
        takens_recurrence_graph(
            x,
            m=p["m"],
            tau=p["tau"],
            recurrence_rate=RECURRENCE_RATE,
            theiler_window=THEILER_WINDOW,
        ),
        "takens",
        weighted=False,
    ),
    "graph_transition": lambda x, p: _graph_descriptor(
        transition_graph(x), "transition", weighted=True
    ),
}

# Amplitude-preserving variants (suffix `_abs`).  Every family above discards
# the lead's scale somewhere: min-max in the GAF, per-series quantiles in the
# MTF and the transition graph, a fixed recurrence rate in the recurrence
# plot and network, and per-image equal-probability quantization in Haralick.
# The `_abs` variants replace each of those with a constant estimated on the
# exploratory training folds (1-7) by `scripts/calibrate_absolute_views.py`,
# so the same transform is applied to every record and nothing is fitted on
# validation or test.  The visibility graphs have no variant: they depend only
# on the order of the samples, so no fixed constant can give them amplitude.
ABSOLUTE_IMAGES: dict[str, Callable[[np.ndarray, dict], np.ndarray]] = {
    "image_spectrogram_abs": lambda x, p: spectrogram(x),
    "image_scalogram_abs": lambda x, p: scalogram(x),
    "image_gasf_abs": lambda x, p: gramian_angular_field(
        x, method="summation", scale=p["scale"]
    ),
    "image_gadf_abs": lambda x, p: gramian_angular_field(
        x, method="difference", scale=p["scale"]
    ),
    "image_mtf_abs": lambda x, p: markov_transition_field(x, bin_edges=p["mtf_edges"]),
}


def _absolute_haralick(family: str):
    image = ABSOLUTE_IMAGES[family]
    return lambda x, p: _haralick_descriptor(image(x, p), thresholds=p["haralick"][family])


FAMILY_OPERATORS.update({family: _absolute_haralick(family) for family in ABSOLUTE_IMAGES})
FAMILY_OPERATORS.update({
    "image_recurrence_abs": lambda x, p: _rqa_descriptor(
        recurrence_plot(
            x,
            dimension=p["m"],
            delay=p["tau"],
            epsilon=p["epsilon"],
            theiler_window=THEILER_WINDOW,
        ),
        fixed_rate=False,
    ),
    "graph_takens_abs": lambda x, p: _graph_descriptor(
        takens_recurrence_graph(
            x,
            m=p["m"],
            tau=p["tau"],
            recurrence_rate=None,
            epsilon=p["epsilon"],
            theiler_window=THEILER_WINDOW,
        ),
        "takens",
        weighted=False,
    ),
    "graph_transition_abs": lambda x, p: _graph_descriptor(
        transition_graph(x, bin_edges=p["transition_edges"]), "transition", weighted=True
    ),
})
ABSOLUTE_FAMILIES: tuple[str, ...] = tuple(f for f in FAMILY_OPERATORS if f.endswith("_abs"))

FAMILY_VIEW = {
    name: ("image" if name.startswith("image_") else "graph") for name in FAMILY_OPERATORS
}

DEFAULT_FAMILIES: tuple[str, ...] = tuple(
    f for f in FAMILY_OPERATORS if f not in ABSOLUTE_FAMILIES
)


def _extract_record(job: tuple) -> tuple[int, dict[str, dict[str, float]]]:
    ecg_id, record_path, leads, families, embedding = job
    try:
        signal, fields = wfdb.rdsamp(record_path)
        positions = {name: index for index, name in enumerate(fields["sig_name"])}
        missing = set(leads) - set(positions)
        if missing:
            raise ValueError(f"missing leads: {sorted(missing)}")
        rows: dict[str, dict[str, float]] = {family: {} for family in families}
        for lead in leads:
            samples = signal[:, positions[lead]].astype(float)
            params = embedding[lead]
            for family in families:
                descriptor = FAMILY_OPERATORS[family](samples, params)
                for name, value in descriptor.items():
                    rows[family][f"{lead}__{family}__{name}"] = value
        return ecg_id, rows
    except Exception as error:
        raise RuntimeError(f"View extraction failed for ECG {ecg_id}: {error}") from error


@dataclass
class ViewExtractor:
    """Extract the derived-view families and persist one Parquet per family."""

    cache_dir: Path
    embedding_path: Path
    leads: tuple[str, ...] = LEADS
    families: tuple[str, ...] = DEFAULT_FAMILIES
    n_jobs: int = 8
    checkpoint_every: int = 250
    calibration_path: Path | None = None
    embedding: dict[str, dict[str, int]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir).resolve()
        self.leads = tuple(self.leads)
        self.families = tuple(self.families)
        unknown_leads = set(self.leads) - set(LEADS)
        if unknown_leads:
            raise ValueError(f"Unknown PTB-XL leads: {sorted(unknown_leads)}")
        unknown_families = set(self.families) - set(FAMILY_OPERATORS)
        if unknown_families:
            raise ValueError(
                f"Unknown families: {sorted(unknown_families)}; "
                f"available: {sorted(FAMILY_OPERATORS)}"
            )
        if not self.leads or not self.families:
            raise ValueError("At least one lead and one family are required.")
        if self.n_jobs < 1 or self.checkpoint_every < 1:
            raise ValueError("n_jobs and checkpoint_every must be at least 1.")

        frozen = json.loads(Path(self.embedding_path).read_text(encoding="utf-8"))
        self.embedding = {
            lead: {"m": int(values["m"]), "tau": int(values["tau"])}
            for lead, values in frozen["per_lead"].items()
        }
        absent = set(self.leads) - set(self.embedding)
        if absent:
            raise ValueError(f"Frozen embedding is missing leads: {sorted(absent)}")
        # The job parameters per lead are the embedding plus, for the `_abs`
        # families, the training-fold constants.  `embedding` stays m and tau
        # only, because that is what the manifest records.
        self.parameters = {lead: dict(values) for lead, values in self.embedding.items()}
        if set(self.families) & set(ABSOLUTE_FAMILIES):
            if self.calibration_path is None:
                raise ValueError("the _abs families need --calibration")
            calibration = json.loads(Path(self.calibration_path).read_text(encoding="utf-8"))
            for lead in self.leads:
                if calibration["per_lead"][lead]["m"] != self.embedding[lead]["m"] or (
                    calibration["per_lead"][lead]["tau"] != self.embedding[lead]["tau"]
                ):
                    raise ValueError(f"calibration was built for another embedding ({lead})")
                self.parameters[lead].update(calibration["per_lead"][lead])

    def family_path(self, family: str) -> Path:
        return self.cache_dir / f"{family}.parquet"

    def _manifest_path(self, family: str) -> Path:
        return self.family_path(family).with_suffix(".parquet.json")

    def _read_family(self, family: str) -> pd.DataFrame:
        path = self.family_path(family)
        if not path.exists():
            return pd.DataFrame(columns=["ecg_id"])
        manifest = json.loads(self._manifest_path(family).read_text(encoding="utf-8"))
        if tuple(manifest["leads"]) != self.leads:
            raise RuntimeError(
                f"{path.name} was built for leads {manifest['leads']}, "
                f"requested {list(self.leads)}. Use a different --cache-dir."
            )
        return pd.read_parquet(path)

    def _write_family(self, family: str, frame: pd.DataFrame) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        frame = frame.sort_values("ecg_id").drop_duplicates("ecg_id", keep="last")
        path = self.family_path(family)
        temporary = path.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)

        columns = [column for column in frame.columns if column != "ecg_id"]
        manifest = {
            "family": family,
            "view": FAMILY_VIEW[family],
            "leads": list(self.leads),
            "features_per_lead": len(columns) // len(self.leads) if columns else 0,
            "feature_count": len(columns),
            "recurrence_rate": RECURRENCE_RATE,
            "theiler_window": THEILER_WINDOW,
            "embedding": {lead: self.embedding[lead] for lead in self.leads},
            "dropped_weight_columns": (
                [] if family.startswith("graph_transition") else list(WEIGHT_ONLY_COLUMNS)
            ),
        }
        if family in ABSOLUTE_FAMILIES:
            manifest["recurrence_rate"] = None
            manifest["calibration"] = str(self.calibration_path)
        manifest_path = self._manifest_path(family)
        manifest_tmp = manifest_path.with_suffix(".json.tmp")
        manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(manifest_tmp, manifest_path)

    def extract(self, records: pd.DataFrame) -> dict[str, pd.DataFrame]:
        required = {"ecg_id", "record_path", "signal_available"}
        missing = required - set(records.columns)
        if missing:
            raise ValueError(f"Record index is missing columns: {sorted(missing)}")
        unavailable = records.loc[~records["signal_available"].astype(bool), "ecg_id"]
        if not unavailable.empty:
            raise ValueError(
                f"View extraction received {len(unavailable)} unavailable records. "
                "Filter signal_available first or complete the download."
            )

        cached = {family: self._read_family(family) for family in self.families}
        requested = set(records["ecg_id"].astype(int))
        # A record is pending unless every requested family already has it.
        done = set.intersection(
            *(
                set(frame["ecg_id"].astype(int)) if not frame.empty else set()
                for frame in cached.values()
            )
        ) if cached else set()
        pending = records.loc[~records["ecg_id"].astype(int).isin(done)]
        jobs = [
            (int(row.ecg_id), str(row.record_path), self.leads, self.families, self.parameters)
            for row in pending.itertuples(index=False)
        ]

        buffer: dict[str, list[dict[str, float]]] = {family: [] for family in self.families}
        pending_count = 0

        def flush() -> None:
            for family, rows in buffer.items():
                if rows:
                    fresh = pd.DataFrame(rows)
                    # A cache written by an earlier readout has different column
                    # names.  `pd.concat` would union them and leave both halves
                    # half-NaN, which is silent and survives into the features,
                    # so refuse instead of merging two incompatible schemas.
                    if not cached[family].empty:
                        stale = set(cached[family].columns) ^ set(fresh.columns)
                        if stale:
                            raise RuntimeError(
                                f"{family}: cached columns do not match the current "
                                f"readout ({len(stale)} differ, e.g. "
                                f"{sorted(stale)[:3]}). Delete "
                                f"{self.family_path(family).name} and recompute."
                            )
                    cached[family] = pd.concat(
                        [cached[family], fresh], ignore_index=True
                    )
                    self._write_family(family, cached[family])
                    rows.clear()

        if jobs:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                stream = executor.map(
                    _extract_record, jobs, chunksize=max(1, min(4, len(jobs) // self.n_jobs))
                )
                for ecg_id, rows in tqdm(stream, total=len(jobs), unit="ECG", desc="Views"):
                    for family, values in rows.items():
                        buffer[family].append({"ecg_id": ecg_id, **values})
                    pending_count += 1
                    if pending_count >= self.checkpoint_every:
                        flush()
                        pending_count = 0
            flush()

        out: dict[str, pd.DataFrame] = {}
        for family in self.families:
            frame = cached[family]
            selected = frame.loc[frame["ecg_id"].astype(int).isin(requested)].copy()
            if len(selected) != len(records):
                raise RuntimeError(
                    f"{family}: expected {len(records)} rows, found {len(selected)}."
                )
            out[family] = records[["ecg_id"]].merge(
                selected, on="ecg_id", how="left", validate="one_to_one"
            )
        return out
