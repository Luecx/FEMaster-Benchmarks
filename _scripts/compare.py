#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
import sys

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table
from scipy.stats import pearsonr, spearmanr, wasserstein_distance


console = Console()


@dataclass(frozen=True)
class FieldKey:
    loadcase: str
    name: str
    occurrence: int = 0

    def label(self) -> str:
        suffix = f"#{self.occurrence + 1}" if self.occurrence else ""
        return f"LC={self.loadcase}::{self.name}{suffix}"


@dataclass
class ResField:
    key: FieldKey
    domain: str
    index_cols: int
    values: np.ndarray
    indices: list[tuple[str, ...]] | None


@dataclass
class Metric:
    field: str
    quantity: str
    rows: int
    rel_l2: float
    rel_linf: float
    w1_rel: float
    pearson: float
    spearman: float
    q99_rms: float
    max_change_rel: float


@dataclass
class Comparison:
    reference: Path
    candidate: Path
    metrics: list[Metric]
    missing_in_reference: list[str]
    missing_in_candidate: list[str]
    row_mismatches: list[str]


def _parse_header(line: str) -> dict[str, str]:
    result: dict[str, str] = {}

    for part in line.split(",")[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        result[key.strip().upper()] = value.strip()

    return result


def read_res(path: str | Path) -> dict[FieldKey, ResField]:
    path = Path(path)
    fields: dict[FieldKey, ResField] = {}
    occurrences: defaultdict[tuple[str, str], int] = defaultdict(int)

    loadcase = "0"
    current_header: dict[str, str] | None = None
    current_rows: list[list[str]] = []

    def finish_field() -> None:
        nonlocal current_header, current_rows

        if current_header is None:
            return

        name = current_header.get("NAME", "UNKNOWN")
        domain = current_header.get("TYPE", "UNKNOWN")
        index_cols = int(current_header.get("INDEX_COLS", "0"))
        value_cols = int(
            current_header.get(
                "VALUE_COLS",
                current_header.get("COLS", "0"),
            )
        )
        expected_rows = int(current_header.get("ROWS", str(len(current_rows))))

        occurrence = occurrences[(loadcase, name)]
        occurrences[(loadcase, name)] += 1
        key = FieldKey(loadcase, name, occurrence)

        if expected_rows != len(current_rows):
            raise ValueError(
                f"{path}: {key.label()} declares {expected_rows} rows, "
                f"but {len(current_rows)} were read."
            )

        indices: list[tuple[str, ...]] | None = [] if index_cols else None
        values: list[list[float]] = []

        for row in current_rows:
            if len(row) < index_cols:
                raise ValueError(f"{path}: malformed row in {key.label()}: {row}")

            if indices is not None:
                indices.append(tuple(row[:index_cols]))

            numeric = row[index_cols:]
            if value_cols and len(numeric) != value_cols:
                raise ValueError(
                    f"{path}: {key.label()} expected {value_cols} value columns, "
                    f"got {len(numeric)}."
                )

            values.append([float(value) for value in numeric])

        if values:
            array = np.asarray(values, dtype=float)
            if array.ndim == 1:
                array = array[:, None]
        else:
            array = np.empty((0, value_cols), dtype=float)

        fields[key] = ResField(
            key=key,
            domain=domain,
            index_cols=index_cols,
            values=array,
            indices=indices,
        )

        current_header = None
        current_rows = []

    with path.open("r", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()

            if not line:
                continue

            if line.startswith("LC "):
                finish_field()
                loadcase = line.split(maxsplit=1)[1].strip()
                continue

            if line.startswith("FIELD"):
                finish_field()
                current_header = _parse_header(line)
                current_rows = []
                continue

            if line.startswith("END FIELD"):
                finish_field()
                continue

            if current_header is not None:
                current_rows.append(line.split())

    finish_field()
    return fields


def _occurrence_keys(
    indices: list[tuple[str, ...]],
) -> list[tuple[tuple[str, ...], int]]:
    counts: defaultdict[tuple[str, ...], int] = defaultdict(int)
    result: list[tuple[tuple[str, ...], int]] = []

    for index in indices:
        occurrence = counts[index]
        counts[index] += 1
        result.append((index, occurrence))

    return result


def align_fields(
    reference: ResField,
    candidate: ResField,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    if reference.values.shape[1] != candidate.values.shape[1]:
        raise ValueError(
            f"{reference.key.label()}: component count differs "
            f"({reference.values.shape[1]} vs {candidate.values.shape[1]})."
        )

    if reference.index_cols != candidate.index_cols:
        raise ValueError(
            f"{reference.key.label()}: INDEX_COLS differs "
            f"({reference.index_cols} vs {candidate.index_cols})."
        )

    if reference.indices is None and candidate.indices is None:
        n = min(len(reference.values), len(candidate.values))
        missing_ref = max(0, len(candidate.values) - len(reference.values))
        missing_candidate = max(0, len(reference.values) - len(candidate.values))
        return reference.values[:n], candidate.values[:n], missing_ref, missing_candidate

    assert reference.indices is not None
    assert candidate.indices is not None

    ref_keys = _occurrence_keys(reference.indices)
    cand_keys = _occurrence_keys(candidate.indices)

    ref_map = {key: i for i, key in enumerate(ref_keys)}
    cand_map = {key: i for i, key in enumerate(cand_keys)}

    common = [key for key in ref_keys if key in cand_map]
    missing_ref = len(set(cand_keys) - set(ref_keys))
    missing_candidate = len(set(ref_keys) - set(cand_keys))

    ref_idx = [ref_map[key] for key in common]
    cand_idx = [cand_map[key] for key in common]

    return (
        reference.values[ref_idx],
        candidate.values[cand_idx],
        missing_ref,
        missing_candidate,
    )


def _component_names(field_name: str, count: int) -> list[str]:
    upper = field_name.upper()

    if count == 6 and "STRESS" in upper:
        return ["S11", "S22", "S33", "S23", "S13", "S12"]

    if count == 6 and "STRAIN" in upper:
        return ["E11", "E22", "E33", "E23", "E13", "E12"]

    if count == 3 and "DISPLACEMENT" in upper:
        return ["U1", "U2", "U3"]

    return [f"C{i}" for i in range(count)]


def _tensor_from_voigt(values: np.ndarray, engineering_shear: bool) -> np.ndarray:
    tensor = np.zeros((len(values), 3, 3), dtype=float)
    tensor[:, 0, 0] = values[:, 0]
    tensor[:, 1, 1] = values[:, 1]
    tensor[:, 2, 2] = values[:, 2]

    shear_scale = 0.5 if engineering_shear else 1.0
    tensor[:, 1, 2] = tensor[:, 2, 1] = shear_scale * values[:, 3]
    tensor[:, 0, 2] = tensor[:, 2, 0] = shear_scale * values[:, 4]
    tensor[:, 0, 1] = tensor[:, 1, 0] = shear_scale * values[:, 5]
    return tensor


def quantities(field_name: str, values: np.ndarray) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}

    for i, name in enumerate(_component_names(field_name, values.shape[1])):
        result[name] = values[:, i]

    if values.size:
        result["ALL_VALUES_FLATTENED"] = values.reshape(-1)

    if values.shape[1] > 1:
        result["ROW_L2_MAGNITUDE"] = np.linalg.norm(values, axis=1)

    upper = field_name.upper()
    is_stress = values.shape[1] == 6 and "STRESS" in upper
    is_strain = values.shape[1] == 6 and "STRAIN" in upper

    if not (is_stress or is_strain):
        return result

    tensor = _tensor_from_voigt(values, engineering_shear=is_strain)
    trace = np.trace(tensor, axis1=1, axis2=2)
    mean = trace / 3.0
    eye = np.eye(3)
    dev = tensor - mean[:, None, None] * eye[None, :, :]
    frob = np.linalg.norm(tensor, axis=(1, 2))
    dev_frob = np.linalg.norm(dev, axis=(1, 2))
    principal = np.linalg.eigvalsh(tensor)[:, ::-1]

    result["DERIVED_TENSOR_FROBENIUS"] = frob
    result["DERIVED_TRACE"] = trace
    result["DERIVED_MEAN_NORMAL"] = mean
    result["DERIVED_DEVIATORIC_FROBENIUS"] = dev_frob
    result["DERIVED_PRINCIPAL_1"] = principal[:, 0]
    result["DERIVED_PRINCIPAL_2"] = principal[:, 1]
    result["DERIVED_PRINCIPAL_3"] = principal[:, 2]

    if is_stress:
        result["DERIVED_VON_MISES"] = math.sqrt(1.5) * dev_frob
        result["DERIVED_J2"] = 0.5 * dev_frob**2
    else:
        result["DERIVED_EQUIVALENT_DEVIATORIC_STRAIN"] = math.sqrt(2.0 / 3.0) * dev_frob

    return result


def _safe_corr(a: np.ndarray, b: np.ndarray, method: str) -> float:
    if len(a) < 2:
        return math.nan

    a_std = float(np.std(a))
    b_std = float(np.std(b))

    if a_std == 0.0 or b_std == 0.0:
        return 1.0 if np.array_equal(a, b) else math.nan

    if method == "pearson":
        return float(pearsonr(a, b).statistic)

    return float(spearmanr(a, b).statistic)


def compare_quantity(
    field: str,
    name: str,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> Metric:
    reference = np.asarray(reference, dtype=float).reshape(-1)
    candidate = np.asarray(candidate, dtype=float).reshape(-1)

    finite = np.isfinite(reference) & np.isfinite(candidate)
    reference = reference[finite]
    candidate = candidate[finite]

    if not len(reference):
        return Metric(
            field=field,
            quantity=name,
            rows=0,
            rel_l2=math.nan,
            rel_linf=math.nan,
            w1_rel=math.nan,
            pearson=math.nan,
            spearman=math.nan,
            q99_rms=math.nan,
            max_change_rel=math.nan,
        )

    diff = candidate - reference
    rms_ref = float(np.sqrt(np.mean(reference**2)))
    l2_ref = float(np.linalg.norm(reference))
    linf_ref = float(np.max(np.abs(reference)))
    eps = np.finfo(float).eps

    rel_l2 = float(np.linalg.norm(diff) / max(l2_ref, eps))
    rel_linf = float(np.max(np.abs(diff)) / max(linf_ref, eps))
    w1_rel = float(wasserstein_distance(reference, candidate) / max(rms_ref, eps))

    q99_ref = float(np.quantile(reference, 0.99))
    q99_cand = float(np.quantile(candidate, 0.99))
    q99_rms = (q99_cand - q99_ref) / max(rms_ref, eps)

    max_ref = float(np.max(np.abs(reference)))
    max_cand = float(np.max(np.abs(candidate)))
    max_change_rel = (max_cand - max_ref) / max(max_ref, eps)

    return Metric(
        field=field,
        quantity=name,
        rows=len(reference),
        rel_l2=rel_l2,
        rel_linf=rel_linf,
        w1_rel=w1_rel,
        pearson=_safe_corr(reference, candidate, "pearson"),
        spearman=_safe_corr(reference, candidate, "spearman"),
        q99_rms=q99_rms,
        max_change_rel=max_change_rel,
    )


def compare_files(reference: str | Path, candidate: str | Path) -> Comparison:
    reference = Path(reference)
    candidate = Path(candidate)

    ref_fields = read_res(reference)
    cand_fields = read_res(candidate)

    ref_keys = set(ref_fields)
    cand_keys = set(cand_fields)

    missing_in_reference = sorted(key.label() for key in cand_keys - ref_keys)
    missing_in_candidate = sorted(key.label() for key in ref_keys - cand_keys)

    metrics: list[Metric] = []
    row_mismatches: list[str] = []

    for key in sorted(ref_keys & cand_keys, key=lambda x: (x.loadcase, x.name, x.occurrence)):
        ref = ref_fields[key]
        cand = cand_fields[key]

        ref_values, cand_values, missing_ref, missing_cand = align_fields(ref, cand)

        if missing_ref or missing_cand:
            row_mismatches.append(
                f"{key.label()}: matched={len(ref_values)}, "
                f"only_candidate={missing_ref}, only_reference={missing_cand}"
            )

        ref_quantities = quantities(key.name, ref_values)
        cand_quantities = quantities(key.name, cand_values)

        for quantity in ref_quantities:
            if quantity not in cand_quantities:
                continue

            metrics.append(
                compare_quantity(
                    key.label(),
                    quantity,
                    ref_quantities[quantity],
                    cand_quantities[quantity],
                )
            )

    return Comparison(
        reference=reference,
        candidate=candidate,
        metrics=metrics,
        missing_in_reference=missing_in_reference,
        missing_in_candidate=missing_in_candidate,
        row_mismatches=row_mismatches,
    )


def _fmt(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "-"

    if abs(value) < 5e-13:
        return "0"

    return f"{value:.{digits}g}"


def create_table(comparison: Comparison) -> Table:
    table = Table(
        title=f"{comparison.reference.name} → {comparison.candidate.name}",
        box=box.SIMPLE,
        show_lines=False,
        header_style="bold",
    )

    table.add_column("Field")
    table.add_column("Quantity")
    table.add_column("relL2", justify="right")
    table.add_column("relLinf", justify="right")
    table.add_column("W1rel", justify="right")
    table.add_column("Pearson", justify="right")
    table.add_column("Spearman", justify="right")
    table.add_column("q99/RMS", justify="right")
    table.add_column("Δmax", justify="right")

    for metric in comparison.metrics:
        table.add_row(
            metric.field,
            metric.quantity,
            _fmt(metric.rel_l2),
            _fmt(metric.rel_linf),
            _fmt(metric.w1_rel),
            _fmt(metric.pearson),
            _fmt(metric.spearman),
            _fmt(metric.q99_rms),
            _fmt(metric.max_change_rel),
        )

    return table


def print_comparison(comparison: Comparison) -> None:
    console.print(create_table(comparison))

    if comparison.missing_in_reference:
        console.print(
            "[yellow]Fields only in candidate:[/yellow] "
            + ", ".join(comparison.missing_in_reference)
        )

    if comparison.missing_in_candidate:
        console.print(
            "[yellow]Fields only in reference:[/yellow] "
            + ", ".join(comparison.missing_in_candidate)
        )

    for mismatch in comparison.row_mismatches:
        console.print(f"[yellow]Row mismatch:[/yellow] {mismatch}")


def primary_metrics(comparison: Comparison) -> list[Metric]:
    """
    Return one representative quantity per field.

    Stress fields use von Mises, strain fields use equivalent deviatoric strain,
    and all other fields use the flattened field values.
    """
    grouped: defaultdict[str, list[Metric]] = defaultdict(list)

    for metric in comparison.metrics:
        grouped[metric.field].append(metric)

    selected: list[Metric] = []

    for field, rows in grouped.items():
        upper = field.upper()

        preferred = None
        if "STRESS" in upper:
            preferred = "DERIVED_VON_MISES"
        elif "STRAIN" in upper:
            preferred = "DERIVED_EQUIVALENT_DEVIATORIC_STRAIN"
        else:
            preferred = "ALL_VALUES_FLATTENED"

        metric = next((row for row in rows if row.quantity == preferred), None)
        if metric is None:
            metric = next((row for row in rows if row.quantity == "ALL_VALUES_FLATTENED"), None)
        if metric is None and rows:
            metric = rows[0]

        if metric is not None:
            selected.append(metric)

    return selected


def aggregate(comparison: Comparison) -> dict[str, float | int]:
    metrics = primary_metrics(comparison)

    def finite_values(attribute: str) -> list[float]:
        return [
            float(getattr(metric, attribute))
            for metric in metrics
            if math.isfinite(float(getattr(metric, attribute)))
        ]

    rel_l2 = finite_values("rel_l2")
    w1 = finite_values("w1_rel")
    pearson = finite_values("pearson")
    spearman = finite_values("spearman")
    q99 = finite_values("q99_rms")
    max_change = finite_values("max_change_rel")

    return {
        "fields": len(metrics),
        "max_rel_l2": max(rel_l2, default=math.nan),
        "max_w1_rel": max(w1, default=math.nan),
        "min_pearson": min(pearson, default=math.nan),
        "min_spearman": min(spearman, default=math.nan),
        "max_abs_q99_rms": max((abs(x) for x in q99), default=math.nan),
        "max_abs_max_change": max((abs(x) for x in max_change), default=math.nan),
    }


def gate_failures(
    comparison: Comparison,
    max_w1: float,
    min_spearman: float,
    max_q99: float,
) -> list[str]:
    failures: list[str] = []

    if comparison.missing_in_reference or comparison.missing_in_candidate or comparison.row_mismatches:
        failures.append("field/row mismatch")

    for metric in primary_metrics(comparison):
        if math.isfinite(metric.w1_rel) and metric.w1_rel > max_w1:
            failures.append(
                f"{metric.field}::{metric.quantity}: W1rel={metric.w1_rel:.4g} > {max_w1:.4g}"
            )

        if math.isfinite(metric.spearman) and metric.spearman < min_spearman:
            failures.append(
                f"{metric.field}::{metric.quantity}: Spearman={metric.spearman:.4g} < {min_spearman:.4g}"
            )

        if math.isfinite(metric.q99_rms) and abs(metric.q99_rms) > max_q99:
            failures.append(
                f"{metric.field}::{metric.quantity}: |q99/RMS|={abs(metric.q99_rms):.4g} > {max_q99:.4g}"
            )

    return failures


def _escape_md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def write_github_summary(
    comparison: Comparison,
    failures: list[str] | None = None,
) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return

    lines = [
        "# FEMaster Result Comparison",
        "",
        f"Reference: `{_escape_md(str(comparison.reference))}`  ",
        f"Candidate: `{_escape_md(str(comparison.candidate))}`",
        "",
        "| Field | Quantity | relL2 | relLinf | W1rel | Pearson | Spearman | q99/RMS | Δmax |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for metric in comparison.metrics:
        lines.append(
            f"| {_escape_md(metric.field)} "
            f"| {_escape_md(metric.quantity)} "
            f"| {_fmt(metric.rel_l2)} "
            f"| {_fmt(metric.rel_linf)} "
            f"| {_fmt(metric.w1_rel)} "
            f"| {_fmt(metric.pearson)} "
            f"| {_fmt(metric.spearman)} "
            f"| {_fmt(metric.q99_rms)} "
            f"| {_fmt(metric.max_change_rel)} |"
        )

    if failures:
        lines += ["", "## Gate failures", ""]
        lines += [f"- {_escape_md(failure)}" for failure in failures]

    with open(path, "a") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two FEMaster .res files.")
    parser.add_argument("reference", help="Reference .res file")
    parser.add_argument("candidate", help="Candidate .res file")
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Return exit code 1 when comparison thresholds are exceeded",
    )
    parser.add_argument("--max-w1", type=float, default=0.12)
    parser.add_argument("--min-spearman", type=float, default=0.94)
    parser.add_argument("--max-q99", type=float, default=0.20)
    args = parser.parse_args()

    comparison = compare_files(args.reference, args.candidate)
    print_comparison(comparison)

    failures = (
        gate_failures(comparison, args.max_w1, args.min_spearman, args.max_q99)
        if args.gate
        else []
    )

    if failures:
        console.print("\n[red bold]Gate failed[/red bold]")
        for failure in failures:
            console.print(f"[red]- {failure}[/red]")

    write_github_summary(comparison, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
