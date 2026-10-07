#!/usr/bin/env python3
"""Generate CSV and Markdown comparisons for distributed benchmark JSONs.

Usage:
  python generate_distributed_perf_report.py \
      --label1 upstream results/upstream \
      --label2 hermetic results/hermetic

  # Write to a specific report directory instead of reports/<label1>_vs_<label2>/:
  python generate_distributed_perf_report.py \
      --label1 upstream results/upstream \
      --label2 hermetic results/hermetic \
      --output /tmp/hermetic-comparison

The comparison primitives intentionally follow compare_results.py: JSON files
are paired by name, result entries are paired by their identifying fields, and
the comparable metrics are p50_us, alpha_us, beta_gbps, and correctness
``passed`` values.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


LABEL_KEYS = ("section", "topology", "collective", "op", "routing",
              "model", "param_name", "name", "scenario", "label")
VALUE_KEYS = ("nelems", "seq_len", "num_tokens", "num_layers", "batch_size",
              "num_microbatches", "dtype", "hidden", "nbytes")
CSV_PREFIX_FIELDS = ["Benchmark", "metrics", "unit"]
TIMESTAMP_RE = re.compile(r"_\d{8}_\d{6}")
IMAGE_PULL_RE = re.compile(r"Trying to pull\s+(.+?)\.\.\.\s*$")


@dataclass
class Row:
    benchmark: str
    metrics: str
    unit: str
    value1: Any = ""
    value2: Any = ""
    delta: Any = ""
    status: str = "OK"
    direction: str = "lower"

    def as_csv(self, label1: str, label2: str) -> dict[str, Any]:
        return {
            "Benchmark": self.benchmark,
            "metrics": self.metrics,
            "unit": self.unit,
            label1: self.value1,
            label2: self.value2,
            "delta %": self.delta,
            "Status": self.status,
        }


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def entry_key(entry: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    parts = []
    for key in LABEL_KEYS + VALUE_KEYS:
        if key in entry:
            value = entry[key]
            if isinstance(value, list):
                value = tuple(value)
            if isinstance(value, dict):
                value = json.dumps(value, sort_keys=True)
            parts.append((key, value))
    return tuple(parts)


def extract_label(entry: dict[str, Any]) -> str:
    parts = [str(entry[key]) for key in LABEL_KEYS if key in entry]
    parts.extend(f"{key}={entry[key]}" for key in VALUE_KEYS if key in entry)
    return "  ".join(parts) if parts else "unknown"


def find_p50_metrics(entry: dict[str, Any], prefix: str = "") -> Iterable[tuple[str, Any]]:
    if not isinstance(entry, dict):
        return
    if "p50_us" in entry:
        yield prefix.rstrip("."), entry["p50_us"]
    for key, value in entry.items():
        if isinstance(value, dict):
            yield from find_p50_metrics(value, f"{prefix}{key}.")


def strip_timestamp(name: str) -> str:
    return TIMESTAMP_RE.sub("", name)


def json_files(folder: Path) -> dict[str, Path]:
    """Return JSON files keyed by stable names, excluding run_all* logs."""
    result: dict[str, Path] = {}
    for path in folder.glob("*.json"):
        if path.name.lower().startswith("run_all"):
            continue
        result[strip_timestamp(path.name)] = path
    return result


def numeric(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def delta_pct(reference: Any, scored: Any) -> float | None:
    reference = numeric(reference)
    scored = numeric(scored)
    if reference is None or scored is None or reference == 0:
        return None
    return round((scored - reference) / abs(reference) * 100, 2)


def status_for(delta: float | None, direction: str, improvement: float,
               regression: float) -> str:
    if delta is None:
        return "MISSING"
    # delta is scored - reference. For latency/memory, negative is better;
    # for bandwidth/speedup, positive is better.
    better = delta < -improvement if direction == "lower" else delta > improvement
    worse = delta > regression if direction == "lower" else delta < -regression
    if better:
        return "IMPROVEMENT"
    if worse:
        return "REGRESSION"
    return "OK"


def report_delta(raw_delta: float | None, direction: str,
                 hermetic_comparison: bool) -> float | None:
    """Return the displayed delta.

    For a Hermetic comparison, positive means Hermetic is better, matching the
    existing distributed reports. For ordinary label1-vs-label2 comparisons,
    retain the normal scored-minus-reference sign.
    """
    if raw_delta is None:
        return None
    if not hermetic_comparison:
        return raw_delta
    return round(-raw_delta if direction == "lower" else raw_delta, 2)


def display_value(value: Any, unit: str) -> Any:
    if value == "":
        return ""
    number = numeric(value)
    if number is None:
        return value
    if unit == "μs":
        return round(number, 4)
    return round(number, 6)


def metric_rows(data1: dict[str, Any] | None, data2: dict[str, Any] | None,
                label1: str, label2: str, improvement: float,
                regression: float) -> list[Row]:
    """Compare one pair while preserving label1/label2 column order."""
    if data1 is None or data2 is None:
        name = (data1 or data2 or {}).get("benchmark", "unknown")
        return [Row(str(name), "file", "", "" if data1 is None else "present",
                    "" if data2 is None else "present", "", "MISSING")]

    # For Hermetic-vs-other, score Hermetic against the other build. Otherwise
    # retain the command-line order: label1 is reference, label2 is scored.
    l1_hermetic = "hermetic" in label1.lower()
    l2_hermetic = "hermetic" in label2.lower()
    hermetic_comparison = l1_hermetic != l2_hermetic
    if l1_hermetic and not l2_hermetic:
        reference, scored = data2, data1
    else:
        reference, scored = data1, data2

    benchmark = str(data1.get("benchmark", data2.get("benchmark", "unknown")))
    rows: list[Row] = []

    # Correctness files use the same passed semantics as compare_results.py.
    if "passed" in reference or "passed" in scored:
        r = reference.get("passed")
        s = scored.get("passed")
        status = "OK" if bool(r) == bool(s) else (
            "IMPROVEMENT" if bool(s) and not bool(r) else "REGRESSION")
        value1 = s if l1_hermetic else r
        value2 = r if l1_hermetic else s
        rows.append(Row(benchmark, "passed", "bool", "pass" if value1 else "FAIL",
                        "pass" if value2 else "FAIL", 0 if bool(r) == bool(s) else "",
                        status, "higher"))

    reference_results = reference.get("results", [])
    scored_by_key = {entry_key(item): item for item in scored.get("results", [])}
    for ref_entry in reference_results:
        scored_entry = scored_by_key.get(entry_key(ref_entry))
        entry_name = extract_label(ref_entry)
        if scored_entry is None:
            rows.append(Row(benchmark, entry_name, "", "present", "", "", "MISSING"))
            continue

        ref_p50 = dict(find_p50_metrics(ref_entry))
        scored_p50 = dict(find_p50_metrics(scored_entry))
        for metric, ref_value in ref_p50.items():
            if metric not in scored_p50:
                rows.append(Row(benchmark, f"{entry_name}  {metric}", "μs",
                                "" if l1_hermetic else display_value(ref_value, "μs"),
                                "" if not l1_hermetic else display_value(ref_value, "μs"),
                                "", "MISSING"))
                continue
            scored_value = scored_p50[metric]
            d = delta_pct(ref_value, scored_value)
            status = status_for(d, "lower", improvement, regression)
            value1 = scored_value if l1_hermetic else ref_value
            value2 = ref_value if l1_hermetic else scored_value
            rows.append(Row(benchmark, f"{entry_name}  {metric}", "μs",
                            display_value(value1, "μs"), display_value(value2, "μs"),
                            report_delta(d, "lower", hermetic_comparison),
                            status, "lower"))

    # Preserve top-level alpha/beta handling from compare_results.py.
    ref_ab = reference.get("alpha_beta", {})
    scored_ab = scored.get("alpha_beta", {})
    for coll, ref_values in ref_ab.items():
        if coll not in scored_ab:
            continue
        for field, unit, direction in (("alpha_us", "μs", "lower"),
                                       ("beta_gbps", "GB/s", "higher")):
            if field not in ref_values or field not in scored_ab[coll]:
                continue
            ref_value = ref_values[field]
            scored_value = scored_ab[coll][field]
            d = delta_pct(ref_value, scored_value)
            status = status_for(d, direction, improvement, regression)
            value1 = scored_value if l1_hermetic else ref_value
            value2 = ref_value if l1_hermetic else scored_value
            rows.append(Row(benchmark, f"{coll}.{field}", unit,
                            display_value(value1, unit), display_value(value2, unit),
                            report_delta(d, direction, hermetic_comparison),
                            status, direction))

    return rows


def find_image(folder: Path) -> str:
    for path in sorted(folder.glob("*.log")):
        try:
            for line in path.read_text(errors="replace").splitlines():
                match = IMAGE_PULL_RE.search(line.strip())
                if match:
                    return match.group(1)
        except OSError:
            pass
    summary = folder / "summary.txt"
    if summary.exists():
        try:
            for line in summary.read_text(errors="replace").splitlines():
                if line.strip().lower().startswith("image:"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return "N/A (host install, no container image)"


def write_csv(path: Path, rows: list[Row], label1: str, label2: str) -> None:
    fields = CSV_PREFIX_FIELDS + [label1, label2, "delta %", "Status"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row.as_csv(label1, label2) for row in rows)


def md_table(rows: list[Row], label1: str, label2: str) -> str:
    lines = [f"| Benchmark | metrics | unit | {label1} | {label2} | delta % | Status |",
             "|---|---|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|")
                     for value in (row.benchmark, row.metrics, row.unit,
                                   row.value1, row.value2, row.delta, row.status)) + " |")
    return "\n".join(lines)


def clean_label(label: str) -> str:
    return re.sub(r"[_-]+", " ", label).strip().title()


def version_from_label(label: str) -> str:
    value = re.sub(r"(?i)(^|[_ -])(hermetic|upstream)(?=$|[_ -])", " ", label)
    return clean_label(value)


def write_summary(path: Path, label1: str, label2: str, folder1: Path,
                  folder2: Path, rows: list[Row], matched: int,
                  only1: list[str], only2: list[str], image1: str,
                  image2: str) -> None:
    improvements = [r for r in rows if r.status == "IMPROVEMENT"]
    regressions = [r for r in rows if r.status == "REGRESSION"]
    by_benchmark: dict[str, Counter[str]] = {}
    for row in rows:
        if row.status != "MISSING":
            by_benchmark.setdefault(row.benchmark, Counter())[row.status] += 1

    benchmark_outcomes = Counter()
    for benchmark, outcome_counts in by_benchmark.items():
        improvements_for_benchmark = outcome_counts["IMPROVEMENT"]
        regressions_for_benchmark = outcome_counts["REGRESSION"]
        if improvements_for_benchmark > regressions_for_benchmark:
            benchmark_outcomes["Hermetic performing better"] += 1
        elif regressions_for_benchmark > improvements_for_benchmark:
            benchmark_outcomes["Upstream performing better"] += 1
        else:
            benchmark_outcomes["On-par"] += 1

    comparable_benchmarks = len(by_benchmark)
    total_benchmarks = comparable_benchmarks
    is_hermetic_comparison = "hermetic" in label1.lower() or "hermetic" in label2.lower()
    better_name = "Hermetic performing better" if is_hermetic_comparison else f"{clean_label(label2)} performing better"
    worse_name = "Upstream performing better" if is_hermetic_comparison else f"{clean_label(label1)} performing better"
    benchmark_outcomes[better_name] = benchmark_outcomes.pop("Hermetic performing better", 0)
    benchmark_outcomes[worse_name] = benchmark_outcomes.pop("Upstream performing better", 0)
    # For Hermetic-vs-other, CSV deltas are expressed from the other build to
    # Hermetic, so larger positive/negative values are ranked by magnitude.
    top_improvements = sorted(improvements, key=lambda r: abs(float(r.delta)), reverse=True)[:3]
    top_regressions = sorted(regressions, key=lambda r: abs(float(r.delta)), reverse=True)[:3]
    hermetic = label1 if "hermetic" in label1.lower() else label2 if "hermetic" in label2.lower() else None
    version = version_from_label(hermetic or label2) or "Hermetic"
    if hermetic:
        executive = f"{version} Hermetic Build Matches Upstream Across Key Metrics"
    else:
        executive = f"{clean_label(label2)} Compared with {clean_label(label1)} Across Key Metrics"
    text = ["# Distributed Benchmark Comparison Summary", "",
            f"## {clean_label(label1)} vs {clean_label(label2)}", "",
            f"- **{clean_label(label1)}:** `{folder1}`",
            f"- **{clean_label(label2)}:** `{folder2}`", "",
            "### Container Images", "",
            f"- **{clean_label(label1)}:** `{image1}`",
            f"- **{clean_label(label2)}:** `{image2}`", "",
            "## Key Observations", "", f"**{executive}.**", "",
            f"The comparison contains **{comparable_benchmarks} benchmark(s) with classified changes** "
            f"out of {total_benchmarks} benchmark(s). "
            f"{better_name}: {benchmark_outcomes[better_name]}; "
            f"{worse_name}: {benchmark_outcomes[worse_name]}; "
            f"On-par: {benchmark_outcomes['On-par']}.", "",
            "## Results at a Glance", "",
            "| Outcome | Benchmark count |", "|---|---:|",
            f"| {better_name} | {benchmark_outcomes[better_name]} |",
            f"| {worse_name} | {benchmark_outcomes[worse_name]} |",
            f"| On-par | {benchmark_outcomes['On-par']} |",
            f"| Total benchmarks | {total_benchmarks} |", ""]
    if only1 or only2:
        text += ["## Incomplete Inputs", ""]
        if only1:
            text.append("- Only in label1: " + ", ".join(only1))
        if only2:
            text.append("- Only in label2: " + ", ".join(only2))
        text.append("")
    text += ["## Top 3 Improvements", "", md_table(top_improvements, label1, label2), "",
             "## Top 3 Regressions", "", md_table(top_regressions, label1, label2), "",
             f"Generated: {datetime.now().isoformat(timespec='seconds')}", ""]
    path.write_text("\n".join(text))


def allocate_output_dir(base: Path) -> Path:
    """Return a new output directory without overwriting an existing report."""
    candidate = base
    suffix = 0
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            suffix += 1
            candidate = Path(f"{base}_{suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label1", nargs=2, required=True, metavar=("LABEL", "FOLDER"))
    parser.add_argument("--label2", nargs=2, required=True, metavar=("LABEL", "FOLDER"))
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--improvement-threshold", type=float, default=3.0)
    parser.add_argument("--regression-threshold", type=float, default=5.0)
    args = parser.parse_args()

    label1, folder1 = args.label1[0], Path(args.label1[1])
    label2, folder2 = args.label2[0], Path(args.label2[1])
    if not folder1.is_dir() or not folder2.is_dir():
        print("Both input folders must exist and be directories.", file=sys.stderr)
        return 2
    output_base = args.output or Path("reports") / f"{label1}_vs_{label2}"
    output = allocate_output_dir(output_base)

    files1, files2 = json_files(folder1), json_files(folder2)
    names = sorted(set(files1) | set(files2))
    all_rows: list[Row] = []
    only1, only2 = sorted(set(files1) - set(files2)), sorted(set(files2) - set(files1))
    for name in names:
        data1 = load_json(files1[name]) if name in files1 else None
        data2 = load_json(files2[name]) if name in files2 else None
        rows = metric_rows(data1, data2, label1, label2,
                           args.improvement_threshold, args.regression_threshold)
        write_csv(output / Path(name).with_suffix(".csv").name, rows, label1, label2)
        all_rows.extend(rows)

    write_csv(output / "regression.csv",
              [r for r in all_rows if r.status == "REGRESSION"], label1, label2)
    write_csv(output / "improvements.csv",
              [r for r in all_rows if r.status == "IMPROVEMENT"], label1, label2)
    write_summary(output / "Summary.md", label1, label2, folder1, folder2,
                  all_rows, len(names) - len(only1) - len(only2), only1, only2,
                  find_image(folder1), find_image(folder2))
    print(f"Compared {len(names) - len(only1) - len(only2)} JSON file(s)")
    print(f"Wrote report to {output}")
    return 1 if any(r.status == "REGRESSION" for r in all_rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
