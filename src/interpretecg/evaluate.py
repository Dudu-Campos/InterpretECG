"""Leakage-safe evaluation on the official PTB-XL folds."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import SPLIT_REGIMES, SUPERCLASSES, PTBXLDataLoader


@dataclass
class Evaluate:
    """Tune on the validation fold and evaluate once on the test fold.

    Which folds those are comes from ``regime``; see ``data.SPLIT_REGIMES``.
    The default is exploratory, which never reads fold 10.
    """

    labels: tuple[str, ...] = SUPERCLASSES
    regime: str = "exploratory"
    c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    random_state: int = 42
    n_jobs: int = 8
    max_iter: int = 3000
    model_: Pipeline | None = field(default=None, init=False, repr=False)
    thresholds_: np.ndarray | None = field(default=None, init=False, repr=False)
    results_: dict[str, object] | None = field(default=None, init=False, repr=False)
    test_predictions_: pd.DataFrame | None = field(default=None, init=False, repr=False)
    feature_columns_: list[str] | None = field(default=None, init=False, repr=False)

    def _pipeline(self, c_value: float) -> Pipeline:
        classifier = OneVsRestClassifier(
            LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=self.max_iter,
                random_state=self.random_state,
                solver="liblinear",
            ),
            n_jobs=self.n_jobs,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ]
        )

    @staticmethod
    def _validate_targets(targets: np.ndarray, split_name: str, labels: tuple[str, ...]) -> None:
        invalid = [
            label for index, label in enumerate(labels) if np.unique(targets[:, index]).size < 2
        ]
        if invalid:
            raise ValueError(f"{split_name} lacks positive or negative examples for: {invalid}")

    @staticmethod
    def _optimal_f1_thresholds(targets: np.ndarray, scores: np.ndarray) -> np.ndarray:
        thresholds: list[float] = []
        for index in range(targets.shape[1]):
            precision, recall, candidates = precision_recall_curve(
                targets[:, index], scores[:, index]
            )
            if not len(candidates):
                thresholds.append(0.5)
                continue
            f1_values = 2 * precision[:-1] * recall[:-1] / (
                precision[:-1] + recall[:-1] + 1e-12
            )
            thresholds.append(float(candidates[int(np.nanargmax(f1_values))]))
        return np.asarray(thresholds)

    def run(self, records: pd.DataFrame, features: pd.DataFrame) -> dict[str, object]:
        merged = records.merge(features, on="ecg_id", how="inner", validate="one_to_one")
        if len(merged) != len(records):
            raise ValueError(f"Features cover {len(merged)}/{len(records)} requested records.")
        feature_columns = [column for column in features.columns if column != "ecg_id"]
        if not feature_columns:
            raise ValueError("No feature columns were supplied.")

        masks = PTBXLDataLoader.split_masks(merged, self.regime)
        arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for split_name, mask in masks.items():
            x_values = merged.loc[mask, feature_columns].to_numpy(dtype=float)
            y_values = merged.loc[mask, self.labels].to_numpy(dtype=np.int8)
            if not len(x_values):
                raise ValueError(f"Official {split_name} split is empty.")
            self._validate_targets(y_values, split_name, self.labels)
            arrays[split_name] = (x_values, y_values)

        x_train, y_train = arrays["train"]
        x_validation, y_validation = arrays["validation"]
        x_test, y_test = arrays["test"]
        tuning: dict[str, float] = {}
        best_score = -np.inf
        best_model: Pipeline | None = None
        best_c: float | None = None
        for c_value in self.c_grid:
            model = self._pipeline(c_value)
            model.fit(x_train, y_train)
            validation_scores = model.predict_proba(x_validation)
            macro_auc = float(roc_auc_score(y_validation, validation_scores, average="macro"))
            tuning[str(c_value)] = macro_auc
            if macro_auc > best_score:
                best_score, best_model, best_c = macro_auc, model, c_value

        assert best_model is not None and best_c is not None
        validation_scores = best_model.predict_proba(x_validation)
        thresholds = self._optimal_f1_thresholds(y_validation, validation_scores)
        test_scores = best_model.predict_proba(x_test)
        test_predictions = (test_scores >= thresholds).astype(np.int8)

        per_label: dict[str, dict[str, float | int]] = {}
        for index, label in enumerate(self.labels):
            tn, fp, fn, tp = confusion_matrix(
                y_test[:, index], test_predictions[:, index], labels=[0, 1]
            ).ravel()
            per_label[label] = {
                "support": int(y_test[:, index].sum()),
                "prevalence": float(y_test[:, index].mean()),
                "auroc": float(roc_auc_score(y_test[:, index], test_scores[:, index])),
                "average_precision": float(
                    average_precision_score(y_test[:, index], test_scores[:, index])
                ),
                "f1": float(f1_score(y_test[:, index], test_predictions[:, index])),
                "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
                "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
                "threshold": float(thresholds[index]),
            }

        folds = SPLIT_REGIMES[self.regime]
        results: dict[str, object] = {
            "task": "PTB-XL diagnostic superclasses",
            "labels": list(self.labels),
            "feature_count": len(feature_columns),
            "protocol": {
                "regime": self.regime,
                "train_folds": list(range(folds["train"][0], folds["train"][1] + 1)),
                "validation_fold": folds["validation"],
                "test_fold": folds["test"],
                "classifier": "one-vs-rest logistic regression",
                "class_weight": "balanced",
                "preprocessing": "median imputation followed by standard scaling",
                "threshold_selection": "maximum F1 on validation, independently per label",
                "random_state": self.random_state,
            },
            "split_sizes": {name: len(values[0]) for name, values in arrays.items()},
            "selection_metric": "validation macro-AUROC",
            "c_grid_validation_macro_auroc": tuning,
            "selected_c": best_c,
            "validation_macro_auroc": best_score,
            "test": {
                "macro_auroc": float(roc_auc_score(y_test, test_scores, average="macro")),
                "micro_auroc": float(roc_auc_score(y_test, test_scores, average="micro")),
                "macro_average_precision": float(
                    average_precision_score(y_test, test_scores, average="macro")
                ),
                "macro_f1": float(f1_score(y_test, test_predictions, average="macro")),
                "micro_f1": float(f1_score(y_test, test_predictions, average="micro")),
                "exact_match_accuracy": float(np.all(y_test == test_predictions, axis=1).mean()),
                "hamming_loss": float(hamming_loss(y_test, test_predictions)),
                "per_label": per_label,
            },
        }
        self.model_ = best_model
        self.thresholds_ = thresholds
        self.results_ = results
        test_rows = merged.loc[masks["test"], ["ecg_id"]].reset_index(drop=True)
        for index, label in enumerate(self.labels):
            test_rows[f"y_true_{label}"] = y_test[:, index]
            test_rows[f"score_{label}"] = test_scores[:, index]
            test_rows[f"y_pred_{label}"] = test_predictions[:, index]
        self.test_predictions_ = test_rows
        self.feature_columns_ = feature_columns
        return results

    def save(self, output_dir: str | Path) -> None:
        if (
            self.model_ is None
            or self.results_ is None
            or self.thresholds_ is None
            or self.test_predictions_ is None
            or self.feature_columns_ is None
        ):
            raise RuntimeError("Call run() before save().")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(
            json.dumps(self.results_, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        joblib.dump(
            {
                "model": self.model_,
                "labels": self.labels,
                "thresholds": self.thresholds_,
                "feature_columns": self.feature_columns_,
            },
            output_dir / "model.joblib",
        )
        self.test_predictions_.to_parquet(output_dir / "test_predictions.parquet", index=False)
