#!/usr/bin/env python3
"""Create a dependency-free SVG gallery of representative PTB-XL records.

The script reads the official metadata and the 100 Hz WFDB files directly.  It
selects one clean, single-superclass example for each PTB-XL diagnostic
superclass and plots leads II, V1 and V5 on the same physical scale.
"""

from __future__ import annotations

import argparse
import ast
import csv
import html
import math
import re
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path


SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
CLASS_NAMES = {
    "NORM": "Normal",
    "MI": "Infarto do miocardio",
    "STTC": "Alteracao de ST/T",
    "CD": "Disturbio de conducao",
    "HYP": "Hipertrofia",
}
LEADS_TO_PLOT = ("II", "V1", "V5")
PLOT_Y_LIMIT = 4.0
NOISE_COLUMNS = (
    "baseline_drift",
    "static_noise",
    "burst_noise",
    "electrodes_problems",
)


@dataclass
class Record:
    metadata: dict[str, str]
    diagnostic_codes: list[str]
    diagnostic_classes: set[str]
    dat_path: Path
    hea_path: Path


@dataclass
class Signal:
    sampling_rate: int
    samples: int
    leads: dict[str, list[float]]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data" / "ptb-xl",
        help="Directory containing ptbxl_database.csv and the flat *_lr files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "artifacts" / "ptbxl_examples.svg",
        help="SVG file to create.",
    )
    parser.add_argument(
        "--png-output",
        type=Path,
        default=project_root / "artifacts" / "ptbxl_examples.png",
        help="PNG file to create with Matplotlib (use an empty value to skip).",
    )
    return parser.parse_args()


def load_statement_classes(data_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with (data_dir / "scp_statements.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            code = row.get("", "")
            diagnostic_class = row.get("diagnostic_class", "")
            if row.get("diagnostic") == "1.0" and diagnostic_class:
                result[code] = diagnostic_class
    return result


def expected_dat_size(header_path: Path) -> int | None:
    try:
        fields = header_path.read_text(encoding="utf-8").splitlines()[0].split()
        signals, samples = int(fields[1]), int(fields[3])
    except (IndexError, OSError, ValueError):
        return None
    return signals * samples * 2  # WFDB format 16: signed 16-bit integers.


def load_candidates(data_dir: Path, class_by_code: dict[str, str]) -> list[Record]:
    records: list[Record] = []
    with (data_dir / "ptbxl_database.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            basename = Path(row["filename_lr"]).name
            dat_path = data_dir / f"{basename}.dat"
            hea_path = data_dir / f"{basename}.hea"
            if not dat_path.is_file() or not hea_path.is_file():
                continue
            expected = expected_dat_size(hea_path)
            if expected is None or dat_path.stat().st_size != expected:
                continue
            try:
                scp_codes = ast.literal_eval(row["scp_codes"])
            except (SyntaxError, ValueError):
                continue
            diagnostic_codes = [code for code in scp_codes if code in class_by_code]
            diagnostic_classes = {class_by_code[code] for code in diagnostic_codes}
            if diagnostic_classes:
                records.append(
                    Record(
                        metadata=row,
                        diagnostic_codes=diagnostic_codes,
                        diagnostic_classes=diagnostic_classes,
                        dat_path=dat_path,
                        hea_path=hea_path,
                    )
                )
    return records


def selection_score(record: Record, target: str) -> tuple[int, int, int, int]:
    row = record.metadata
    no_reported_noise = all(not row.get(column, "").strip() for column in NOISE_COLUMNS)
    human_validated = row.get("validated_by_human") == "True"
    only_target_class = record.diagnostic_classes == {target}
    # Use the lowest ECG id only as a deterministic final tiebreaker.
    return (
        int(only_target_class),
        int(no_reported_noise),
        int(human_validated),
        -int(row["ecg_id"]),
    )


def select_examples(records: list[Record]) -> list[Record]:
    selected: list[Record] = []
    for target in SUPERCLASSES:
        candidates = [r for r in records if target in r.diagnostic_classes]
        if not candidates:
            raise RuntimeError(
                f"No complete downloaded record is available for superclass {target}."
            )
        selected.append(max(candidates, key=lambda r: selection_score(r, target)))
    return selected


GAIN_PATTERN = re.compile(
    r"(?P<gain>[+-]?(?:\d+(?:\.\d*)?|\.\d+))"
    r"(?:\((?P<baseline>[+-]?\d+)\))?/(?P<unit>\S+)"
)


def read_signal(record: Record) -> Signal:
    lines = record.hea_path.read_text(encoding="utf-8").splitlines()
    first = lines[0].split()
    signal_count, sampling_rate, sample_count = (
        int(first[1]),
        int(float(first[2])),
        int(first[3]),
    )

    lead_specs: list[tuple[str, float, int]] = []
    for line in lines[1 : signal_count + 1]:
        fields = line.split()
        match = GAIN_PATTERN.fullmatch(fields[2])
        if not match:
            raise ValueError(f"Unsupported gain specification in {record.hea_path}: {fields[2]}")
        lead_specs.append(
            (
                fields[-1],
                float(match.group("gain")),
                int(match.group("baseline") or 0),
            )
        )

    digital = array("h")
    digital.frombytes(record.dat_path.read_bytes())
    if sys.byteorder != "little":
        digital.byteswap()
    if len(digital) != signal_count * sample_count:
        raise ValueError(f"Unexpected data length in {record.dat_path}")

    leads: dict[str, list[float]] = {}
    for lead_index, (lead_name, gain, baseline) in enumerate(lead_specs):
        leads[lead_name] = [
            (digital[i * signal_count + lead_index] - baseline) / gain
            for i in range(sample_count)
        ]
    return Signal(sampling_rate, sample_count, leads)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def fmt_age(value: str) -> str:
    try:
        number = float(value)
    except ValueError:
        return "n/d"
    return str(int(number)) if math.isfinite(number) else "n/d"


def shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def trace_path(
    values: list[float], x: float, y: float, width: float, height: float, y_limit: float
) -> str:
    points: list[str] = []
    last = len(values) - 1
    for index, value in enumerate(values):
        px = x + width * index / last
        py = y + height / 2 - (value / y_limit) * height / 2
        command = "M" if index == 0 else "L"
        points.append(f"{command}{px:.2f},{py:.2f}")
    return " ".join(points)


def make_svg(examples: list[Record], signals: list[Signal]) -> str:
    width = 1600
    top = 110
    row_height = 230
    bottom = 70
    height = top + len(examples) * row_height + bottom
    info_width = 275
    plot_left = 315
    gap = 18
    plot_width = (width - plot_left - 38 - 2 * gap) / 3
    plot_height = 160
    y_limit = PLOT_Y_LIMIT

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Inter,Segoe UI,Arial,sans-serif;fill:#17212b}",
        ".title{font-size:30px;font-weight:700}",
        ".subtitle{font-size:15px;fill:#52606d}",
        ".class{font-size:22px;font-weight:700}",
        ".meta{font-size:14px;fill:#394b59}",
        ".lead{font-size:15px;font-weight:700;fill:#243b53}",
        ".axis{font-size:11px;fill:#6b7c93}",
        ".trace{fill:none;stroke:#0b4f6c;stroke-width:1.35;stroke-linejoin:round}",
        ".minor{stroke:#f8dfe3;stroke-width:.7}",
        ".major{stroke:#e9aab3;stroke-width:1}",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#f5f7fa"/>',
        '<text x="38" y="46" class="title">PTB-XL em cinco exemplos</text>',
        '<text x="38" y="74" class="subtitle">ECG de 12 derivações, 10 s a 100 Hz — três derivações exibidas na mesma escala (±4 mV)</text>',
        '<text x="38" y="96" class="subtitle">Seleção automática: um registro completo, sem ruído anotado e com apenas uma superclasse diagnóstica.</text>',
        "<defs>",
    ]

    for row_index in range(len(examples)):
        plot_y = top + row_index * row_height + 42
        for col_index in range(3):
            x = plot_left + col_index * (plot_width + gap)
            parts.append(
                f'<clipPath id="clip-{row_index}-{col_index}"><rect x="{x:.2f}" y="{plot_y:.2f}" width="{plot_width:.2f}" height="{plot_height:.2f}"/></clipPath>'
            )
    parts.append("</defs>")

    for row_index, (target, record, signal) in enumerate(
        zip(SUPERCLASSES, examples, signals)
    ):
        row_y = top + row_index * row_height
        plot_y = row_y + 42
        parts.append(
            f'<rect x="24" y="{row_y + 8}" width="1552" height="214" rx="12" fill="#ffffff" stroke="#d9e2ec"/>'
        )
        parts.append(
            f'<rect x="24" y="{row_y + 8}" width="8" height="214" rx="4" fill="#d1495b"/>'
        )
        parts.append(f'<text x="48" y="{row_y + 43}" class="class">{esc(target)}</text>')
        parts.append(
            f'<text x="48" y="{row_y + 66}" class="meta">{esc(CLASS_NAMES[target])}</text>'
        )
        metadata = record.metadata
        parts.append(
            f'<text x="48" y="{row_y + 93}" class="meta">ECG #{esc(metadata["ecg_id"])} · idade {esc(fmt_age(metadata["age"]))}</text>'
        )
        parts.append(
            f'<text x="48" y="{row_y + 114}" class="meta">sexo={esc(metadata["sex"])} · fold {esc(metadata["strat_fold"])}</text>'
        )
        codes = ", ".join(record.diagnostic_codes)
        parts.append(
            f'<text x="48" y="{row_y + 139}" class="meta">SCP: {esc(shorten(codes, 27))}</text>'
        )
        report = shorten(metadata.get("report", ""), 31)
        parts.append(f'<text x="48" y="{row_y + 164}" class="meta">Laudo original:</text>')
        parts.append(f'<text x="48" y="{row_y + 185}" class="meta">{esc(report)}</text>')

        for col_index, lead in enumerate(LEADS_TO_PLOT):
            x = plot_left + col_index * (plot_width + gap)
            values = signal.leads[lead]
            parts.append(f'<text x="{x:.2f}" y="{row_y + 32}" class="lead">{lead}</text>')
            parts.append(
                f'<rect x="{x:.2f}" y="{plot_y:.2f}" width="{plot_width:.2f}" height="{plot_height:.2f}" fill="#fffdfd" stroke="#cbd5e1"/>'
            )
            # ECG-paper-like grid: 0.2 s / 0.1 mV minor, 1 s / 0.5 mV major.
            for tick in range(51):
                gx = x + plot_width * tick / 50
                css_class = "major" if tick % 5 == 0 else "minor"
                parts.append(
                    f'<line x1="{gx:.2f}" y1="{plot_y:.2f}" x2="{gx:.2f}" y2="{plot_y + plot_height:.2f}" class="{css_class}"/>'
                )
            for tick in range(51):
                gy = plot_y + plot_height * tick / 50
                css_class = "major" if tick % 5 == 0 else "minor"
                parts.append(
                    f'<line x1="{x:.2f}" y1="{gy:.2f}" x2="{x + plot_width:.2f}" y2="{gy:.2f}" class="{css_class}"/>'
                )
            path = trace_path(values, x, plot_y, plot_width, plot_height, y_limit)
            parts.append(
                f'<path d="{path}" class="trace" clip-path="url(#clip-{row_index}-{col_index})"/>'
            )
            parts.append(
                f'<text x="{x:.2f}" y="{plot_y + plot_height + 16:.2f}" class="axis">0 s</text>'
            )
            parts.append(
                f'<text x="{x + plot_width - 22:.2f}" y="{plot_y + plot_height + 16:.2f}" class="axis">10 s</text>'
            )

    footer_y = height - 32
    parts.append(
        f'<text x="38" y="{footer_y}" class="subtitle">Cada linha é uma instância real; a classe vem das anotações SCP-ECG, não de uma predição. Escala vertical idêntica em todos os painéis.</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def make_png(examples: list[Record], signals: list[Signal], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    figure = plt.figure(figsize=(16, 12), facecolor="#f5f7fa")
    grid = figure.add_gridspec(
        len(examples),
        4,
        left=0.035,
        right=0.985,
        top=0.90,
        bottom=0.07,
        hspace=0.58,
        wspace=0.18,
        width_ratios=(0.82, 1.45, 1.45, 1.45),
    )
    figure.suptitle(
        "PTB-XL em cinco exemplos",
        x=0.035,
        y=0.975,
        ha="left",
        fontsize=23,
        fontweight="bold",
        color="#17212b",
    )
    figure.text(
        0.035,
        0.942,
        "ECG de 12 derivações, 10 s a 100 Hz — derivações II, V1 e V5 na mesma escala (±4 mV)",
        ha="left",
        fontsize=11,
        color="#52606d",
    )
    figure.text(
        0.035,
        0.922,
        "Uma instância real por superclasse; seleção automática entre registros completos, sem ruído anotado e com uma única superclasse.",
        ha="left",
        fontsize=9.5,
        color="#52606d",
    )

    for row_index, (target, record, signal) in enumerate(
        zip(SUPERCLASSES, examples, signals)
    ):
        metadata = record.metadata
        info_axis = figure.add_subplot(grid[row_index, 0])
        info_axis.set_facecolor("white")
        info_axis.set_xticks([])
        info_axis.set_yticks([])
        for spine in info_axis.spines.values():
            spine.set_color("#d9e2ec")
        info_axis.text(
            0.06,
            0.84,
            target,
            transform=info_axis.transAxes,
            fontsize=17,
            fontweight="bold",
            color="#d1495b",
        )
        info_axis.text(
            0.06,
            0.67,
            CLASS_NAMES[target],
            transform=info_axis.transAxes,
            fontsize=10.5,
            fontweight="bold",
            color="#243b53",
        )
        details = (
            f"ECG #{metadata['ecg_id']}  ·  idade {fmt_age(metadata['age'])}\n"
            f"sexo={metadata['sex']}  ·  fold {metadata['strat_fold']}\n"
            f"SCP: {shorten(', '.join(record.diagnostic_codes), 27)}\n\n"
            f"Laudo original:\n{shorten(metadata.get('report', ''), 38)}"
        )
        info_axis.text(
            0.06,
            0.53,
            details,
            transform=info_axis.transAxes,
            va="top",
            fontsize=8.7,
            linespacing=1.35,
            color="#394b59",
        )

        times = [index / signal.sampling_rate for index in range(signal.samples)]
        for column_index, lead in enumerate(LEADS_TO_PLOT, start=1):
            axis = figure.add_subplot(grid[row_index, column_index])
            axis.set_facecolor("#fffdfd")
            axis.plot(times, signal.leads[lead], color="#0b4f6c", linewidth=0.85)
            axis.set_xlim(0, 10)
            axis.set_ylim(-PLOT_Y_LIMIT, PLOT_Y_LIMIT)
            axis.set_title(lead, loc="left", fontsize=10.5, fontweight="bold", pad=3)
            axis.xaxis.set_major_locator(MultipleLocator(1.0))
            axis.xaxis.set_minor_locator(MultipleLocator(0.2))
            axis.yaxis.set_major_locator(MultipleLocator(0.5))
            axis.yaxis.set_minor_locator(MultipleLocator(0.1))
            axis.grid(which="major", color="#e9aab3", linewidth=0.55)
            axis.grid(which="minor", color="#f8dfe3", linewidth=0.35)
            axis.tick_params(axis="both", labelsize=7, colors="#6b7c93", length=2)
            if column_index == 1:
                axis.set_ylabel("mV", fontsize=8, color="#52606d")
            else:
                axis.set_yticklabels([])
            if row_index == len(examples) - 1:
                axis.set_xlabel("tempo (s)", fontsize=8, color="#52606d")
            for spine in axis.spines.values():
                spine.set_color("#cbd5e1")

    figure.text(
        0.035,
        0.025,
        "A classe vem das anotações SCP-ECG, não de uma predição. sexo=0/1 preserva a codificação original do dataset.",
        ha="left",
        fontsize=9,
        color="#52606d",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    args = parse_args()
    class_by_code = load_statement_classes(args.data_dir)
    records = load_candidates(args.data_dir, class_by_code)
    examples = select_examples(records)
    signals = [read_signal(record) for record in examples]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(make_svg(examples, signals), encoding="utf-8")
    make_png(examples, signals, args.png_output)
    for target, record in zip(SUPERCLASSES, examples):
        print(f"{target}: ECG {record.metadata['ecg_id']} ({record.dat_path.name})")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.png_output}")


if __name__ == "__main__":
    main()
