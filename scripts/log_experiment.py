"""Append one experiment to the project log and re-render the results table.

The log is append-only and lives in `results/experiments.jsonl`. The human
readable view, `results/RESULTS.md`, is always regenerated from it, so the
JSONL is the single source of truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "results" / "experiments.jsonl"
TABLE_PATH = PROJECT_ROOT / "results" / "RESULTS.md"


def _source_fingerprint() -> dict[str, str]:
    """Hash the source tree so a logged number can be traced to the code."""
    files = sorted((PROJECT_ROOT / "src").rglob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    fingerprint = {"src_sha256": digest.hexdigest()[:16], "n_source_files": str(len(files))}
    try:
        head = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if head.returncode == 0:
            fingerprint["git"] = head.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return fingerprint


def build_entry(args: argparse.Namespace) -> dict:
    metrics = json.loads(Path(args.metrics).read_text())
    protocol = metrics["protocol"]
    test = metrics["test"]
    return {
        "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": args.label,
        "question": args.question,
        "view": args.view,
        "families": args.family or [],
        "regime": protocol.get("regime", "unknown"),
        "folds": {
            "train": protocol.get("train_folds"),
            "validation": protocol.get("validation_fold"),
            "test": protocol.get("test_fold"),
        },
        "leads": args.leads or [],
        "params": json.loads(args.params) if args.params else {},
        "classifier": protocol.get("classifier"),
        "n_features": metrics.get("feature_count"),
        "split_sizes": metrics.get("split_sizes"),
        "selected_c": metrics.get("selected_c"),
        "metrics": {
            key: test[key]
            for key in (
                "macro_auroc",
                "micro_auroc",
                "macro_average_precision",
                "macro_f1",
            )
            if key in test
        },
        "per_label_auroc": {
            label: values["auroc"] for label, values in test.get("per_label", {}).items()
        },
        "finding": args.finding,
        "artifacts": {"metrics": str(Path(args.metrics).resolve())},
        "code": _source_fingerprint(),
    }


def render_table(entries: list[dict]) -> str:
    lines = [
        "# Registro de experimentos",
        "",
        "Gerado por `scripts/log_experiment.py` a partir de",
        "`results/experiments.jsonl`. **Não editar à mão**: as alterações são",
        "sobrescritas na próxima execução de `/log`.",
        "",
        f"{len(entries)} experimento(s) registrado(s).",
        "",
        "| # | data | rótulo | visão | features | regime | teste | AUROC macro | macro-AP |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for index, entry in enumerate(entries, start=1):
        metrics = entry.get("metrics", {})
        auroc = metrics.get("macro_auroc")
        ap = metrics.get("macro_average_precision")
        lines.append(
            f"| {index} "
            f"| {entry['logged_at'][:10]} "
            f"| {entry['label']} "
            f"| {entry.get('view') or '—'} "
            f"| {entry.get('n_features') or '—'} "
            f"| {entry['regime']} "
            f"| fold {entry['folds']['test']} "
            f"| {'—' if auroc is None else format(auroc, '.4f')} "
            f"| {'—' if ap is None else format(ap, '.4f')} |"
            + (" ⚠️" if entry.get("correction") else "")
        )

    for index, entry in enumerate(entries, start=1):
        lines += ["", f"## {index}. {entry['label']}", ""]
        if entry.get("question"):
            lines.append(f"**Pergunta:** {entry['question']}")
            lines.append("")
        if entry.get("finding"):
            lines.append(f"**Achado:** {entry['finding']}")
            lines.append("")
        correction = entry.get("correction")
        if correction:
            lines.append(
                f"> **Corrigido por `{correction['superseded_by']}`.** "
                f"{correction['what_was_wrong']}"
            )
            lines.append("")
            if correction.get("what_still_holds"):
                lines.append(f"> O que permanece: {correction['what_still_holds']}")
                lines.append("")
        folds = entry["folds"]
        lines.append(
            f"- Regime `{entry['regime']}`: treino {folds['train']}, "
            f"validação fold {folds['validation']}, teste fold {folds['test']}"
        )
        if entry.get("families"):
            lines.append(f"- Famílias: {', '.join(entry['families'])}")
        if entry.get("leads"):
            lines.append(f"- Derivações: {', '.join(entry['leads'])}")
        if entry.get("params"):
            lines.append(f"- Parâmetros: `{json.dumps(entry['params'], sort_keys=True)}`")
        if entry.get("selected_c") is not None:
            lines.append(f"- `C` selecionado na validação: {entry['selected_c']}")
        per_label = entry.get("per_label_auroc") or {}
        if per_label:
            formatted = ", ".join(f"{k} {v:.4f}" for k, v in per_label.items())
            lines.append(f"- AUROC por classe: {formatted}")
        code = entry.get("code", {})
        if code:
            tag = code.get("git") or f"src {code.get('src_sha256')}"
            lines.append(f"- Código: {tag}")
        lines.append(f"- Métricas completas: `{entry['artifacts']['metrics']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", help="path to a metrics.json")
    parser.add_argument("--label", help="short identifier for the run")
    parser.add_argument("--question", help="what this run was meant to answer")
    parser.add_argument("--finding", help="what the numbers showed, in one or two sentences")
    parser.add_argument("--view", help="series, image, graph, or a combination")
    parser.add_argument("--family", action="append", help="feature family (repeatable)")
    parser.add_argument("--leads", nargs="+", help="leads the views were computed on")
    parser.add_argument("--params", help="JSON object with the operator parameters")
    parser.add_argument(
        "--rerender",
        action="store_true",
        help="only regenerate RESULTS.md from the existing log",
    )
    args = parser.parse_args()
    if not args.rerender and not (args.metrics and args.label):
        parser.error("--metrics and --label are required unless --rerender is given")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not args.rerender:
        entry = build_entry(args)
        with LOG_PATH.open("a") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Appended {entry['label']!r} to {LOG_PATH}")

    entries = [json.loads(line) for line in LOG_PATH.read_text().splitlines() if line.strip()]
    TABLE_PATH.write_text(render_table(entries))
    print(f"Rendered {len(entries)} experiment(s) into {TABLE_PATH}")


if __name__ == "__main__":
    main()
