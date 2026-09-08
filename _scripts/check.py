#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gzip
import os
import re
import sys
import tempfile

from dataclasses import dataclass
from pathlib import Path

import yaml
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from common import (
    ROOT,
    SolverRun,
    effective_time,
    escape_markdown,
    failure_detail,
    format_int,
    format_time,
    github_summary_path,
    require_case,
    require_solver,
    run_solver,
)


DEFAULT_RTOL = 1.0e-2
DEFAULT_ATOL = 1.0e-8

console = Console()


@dataclass
class CheckResult:
    name: str
    run: SolverRun | None
    passed: bool
    detail: str = ""
    error: str | None = None


def read_result_maxima(path: Path) -> dict[str, float]:
    fields: dict[str, float] = {}
    current_field: str | None = None

    open_text = gzip.open if path.suffix == ".gz" else open

    with open_text(path, "rt") as stream:
        for line in stream:
            if line.startswith("FIELD"):
                match = re.search(r"NAME=([^,]+)", line)
                if not match:
                    raise RuntimeError(f"Could not parse FIELD line:\n{line}")

                current_field = match.group(1).strip()

                if current_field in fields:
                    raise RuntimeError(
                        f"Field '{current_field}' occurs more than once. "
                        "The target format currently assumes unique field names."
                    )

                fields[current_field] = 0.0
                continue

            if line.startswith("END FIELD"):
                current_field = None
                continue

            if current_field is None:
                continue

            parts = line.split()
            if len(parts) <= 1:
                continue

            try:
                values = [float(value) for value in parts[1:]]
            except ValueError:
                continue

            if values:
                fields[current_field] = max(
                    fields[current_field],
                    max(abs(value) for value in values),
                )

    if not fields:
        raise RuntimeError(f"No result fields found in '{path}'.")

    return fields


def read_target(path: Path) -> dict:
    with path.open("rt") as stream:
        data = yaml.safe_load(stream)

    if not isinstance(data, dict):
        raise RuntimeError(
            f"Invalid target file '{path}': expected a YAML mapping."
        )

    results = data.get("results")
    if not isinstance(results, dict) or not results:
        raise RuntimeError(
            f"Invalid target file '{path}': "
            "missing non-empty 'results' mapping."
        )

    return results


def compare_results(result_path: Path, target_path: Path) -> str | None:
    result = read_result_maxima(result_path)
    targets = read_target(target_path)

    for field_name, field_target in targets.items():
        if field_name not in result:
            return f"{field_name}: field missing from result"

        if not isinstance(field_target, dict):
            return f"{field_name}: target must be a mapping"

        unknown_metrics = set(field_target) - {"max_abs"}
        if unknown_metrics:
            metrics = ", ".join(sorted(unknown_metrics))
            return f"{field_name}: unsupported target metric(s): {metrics}"

        max_abs = field_target.get("max_abs")
        if max_abs is None:
            return f"{field_name}: missing max_abs target"

        if isinstance(max_abs, (int, float)):
            ref_value = float(max_abs)
            rtol = DEFAULT_RTOL
            atol = DEFAULT_ATOL
        elif isinstance(max_abs, dict):
            if "value" not in max_abs:
                return f"{field_name}: max_abs target is missing 'value'"

            ref_value = float(max_abs["value"])
            rtol = float(max_abs.get("rtol", DEFAULT_RTOL))
            atol = float(max_abs.get("atol", DEFAULT_ATOL))
        else:
            return f"{field_name}: invalid max_abs target"

        value = result[field_name]
        error = abs(value - ref_value)
        tolerance = atol + rtol * abs(ref_value)

        if error > tolerance:
            relative_error = (
                error / abs(ref_value)
                if ref_value != 0.0
                else None
            )
            relative_text = (
                f"{relative_error:.2%}"
                if relative_error is not None
                else "n/a"
            )

            return (
                f"{field_name}: maximum magnitude "
                f"{value:.6e} vs {ref_value:.6e}, "
                f"abs={error:.2e}, "
                f"rel={relative_text}, "
                f"tol={tolerance:.2e}"
            )

    return None


def check_case(
    solver: str | Path,
    name: str,
    *,
    ncpus: int = 1,
    work_root: Path | None = None,
) -> CheckResult:
    solver = require_solver(solver)
    case = require_case(name)
    target = case / "target.yaml"

    if not target.is_file():
        return CheckResult(
            name=name,
            run=None,
            passed=False,
            error=f"Target not found: {target}",
        )

    own_temp = None
    if work_root is None:
        own_temp = tempfile.TemporaryDirectory(prefix="femaster_check_")
        work_root = Path(own_temp.name)

    workdir = Path(work_root) / name

    try:
        run = run_solver(
            solver,
            name,
            workdir,
            ncpus=ncpus,
            runs=1,
        )
        detail = compare_results(run.result, target)

        return CheckResult(
            name=name,
            run=run,
            passed=detail is None,
            detail=detail or "",
        )

    except Exception as error:
        return CheckResult(
            name=name,
            run=None,
            passed=False,
            error=failure_detail(workdir / "model.log", error),
        )

    finally:
        if own_temp is not None:
            own_temp.cleanup()


def create_table(result: CheckResult) -> Table:
    table = Table(
        title="FEMaster Result Check",
        box=box.SIMPLE,
        header_style="bold",
    )

    table.add_column("Model")
    table.add_column("Analysis")
    table.add_column("Nodes", justify="right")
    table.add_column("DOFs", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Status")

    if result.run is None:
        table.add_row(
            result.name,
            "-",
            "-",
            "-",
            "-",
            Text(
                f"FAIL\n{result.error or result.detail}",
                style="red",
            ),
        )
    else:
        meta = result.run.meta
        status = (
            Text("PASS", style="green")
            if result.passed
            else Text(f"FAIL\n{result.detail}", style="red")
        )

        table.add_row(
            result.name,
            meta["step_type"] or "-",
            format_int(meta["num_nodes"]),
            format_int(meta["num_dofs"]),
            format_time(effective_time(result.run)),
            status,
        )

    return table


def write_github_summary(result: CheckResult, solver: Path) -> None:
    path = github_summary_path()
    if not path:
        return

    if result.run is None:
        analysis = nodes = dofs = timing = "-"
        status = "FAIL — " + escape_markdown(
            result.error or result.detail
        )
    else:
        meta = result.run.meta
        analysis = meta["step_type"] or "-"
        nodes = format_int(meta["num_nodes"])
        dofs = format_int(meta["num_dofs"])
        timing = format_time(effective_time(result.run))
        status = (
            "PASS"
            if result.passed
            else "FAIL — " + escape_markdown(result.detail)
        )

    lines = [
        "# FEMaster Result Check",
        "",
        f"Solver: `{escape_markdown(str(solver))}`",
        "",
        "| Model | Analysis | Nodes | DOFs | Time | Status |",
        "|---|---|---:|---:|---:|---|",
        f"| {escape_markdown(result.name)} "
        f"| {analysis} "
        f"| {nodes} "
        f"| {dofs} "
        f"| {timing} "
        f"| {status} |",
    ]

    with open(path, "a") as stream:
        stream.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one FEMaster benchmark and check it against target.yaml."
    )
    parser.add_argument("solver", help="Path to FEMaster executable")
    parser.add_argument("case", help="Benchmark case directory name")
    parser.add_argument(
        "--ncpus",
        type=int,
        default=1,
        help="Threads used by the solver (default: 1)",
    )
    args = parser.parse_args()

    solver = require_solver(args.solver)
    result = check_case(
        solver,
        args.case,
        ncpus=args.ncpus,
    )

    console.print(create_table(result))
    write_github_summary(result, solver)

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
