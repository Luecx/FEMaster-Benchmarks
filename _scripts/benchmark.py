#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import tempfile

from dataclasses import dataclass
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from common import (
    DEFAULT_THREADS,
    SolverRun,
    effective_time,
    escape_markdown,
    failure_detail,
    format_time,
    github_summary_path,
    require_case,
    require_solver,
    run_solver,
)


console = Console()


@dataclass
class TimedRun:
    run: SolverRun | None
    error: str | None = None


@dataclass
class BenchmarkResult:
    name: str
    candidate: dict[int, TimedRun]
    reference: dict[int, TimedRun] | None = None


def _run_config(
    solver: Path,
    name: str,
    work_root: Path,
    label: str,
    threads: int,
    runs: int,
) -> TimedRun:
    workdir = work_root / label / f"{threads}T" / name

    try:
        return TimedRun(
            run=run_solver(
                solver,
                name,
                workdir,
                ncpus=threads,
                runs=runs,
            )
        )
    except Exception as error:
        return TimedRun(
            run=None,
            error=failure_detail(workdir / "model.log", error),
        )


def benchmark_case(
    candidate_solver: str | Path,
    name: str,
    *,
    reference_solver: str | Path | None = None,
    runs: int = 3,
    threads: tuple[int, ...] = DEFAULT_THREADS,
    work_root: Path | None = None,
) -> BenchmarkResult:
    candidate_solver = require_solver(candidate_solver)
    require_case(name)

    reference = (
        require_solver(reference_solver)
        if reference_solver is not None
        else None
    )

    own_temp = None
    if work_root is None:
        own_temp = tempfile.TemporaryDirectory(prefix="femaster_benchmark_")
        work_root = Path(own_temp.name)

    try:
        candidate_results = {
            ncpus: _run_config(
                candidate_solver,
                name,
                work_root,
                "candidate",
                ncpus,
                runs,
            )
            for ncpus in threads
        }

        reference_results = None
        if reference is not None:
            reference_results = {
                ncpus: _run_config(
                    reference,
                    name,
                    work_root,
                    "reference",
                    ncpus,
                    runs,
                )
                for ncpus in threads
            }

        return BenchmarkResult(
            name=name,
            candidate=candidate_results,
            reference=reference_results,
        )

    finally:
        if own_temp is not None:
            own_temp.cleanup()


def _time(result: TimedRun | None) -> float | None:
    return effective_time(result.run) if result and result.run else None


def _delta(reference: float | None, candidate: float | None) -> float | None:
    if reference is None or candidate is None or reference == 0.0:
        return None
    return candidate / reference - 1.0


def _speedup(results: dict[int, TimedRun]) -> float | None:
    t1 = _time(results.get(1))
    t4 = _time(results.get(4))
    if t1 is None or t4 is None or t4 == 0.0:
        return None
    return t1 / t4


def _fmt_delta(value: float | None) -> str:
    return "-" if value is None else f"{value:+.1%}"


def _fmt_speedup(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}x"


def _first_error(result: BenchmarkResult) -> str | None:
    groups = [result.candidate]
    if result.reference is not None:
        groups.append(result.reference)

    for group in groups:
        for timed in group.values():
            if timed.run is None:
                return timed.error or "solver failed"

    return None


def create_table(result: BenchmarkResult) -> Table:
    two_versions = result.reference is not None

    table = Table(
        title="FEMaster Benchmark",
        box=box.SIMPLE,
        header_style="bold",
    )

    table.add_column("Model")

    if not two_versions:
        table.add_column("1T", justify="right")
        table.add_column("4T", justify="right")
        table.add_column("Speedup", justify="right")
    else:
        table.add_column("Ref 1T", justify="right")
        table.add_column("New 1T", justify="right")
        table.add_column("Δ1T", justify="right")
        table.add_column("Ref 4T", justify="right")
        table.add_column("New 4T", justify="right")
        table.add_column("Δ4T", justify="right")
        table.add_column("Ref scale", justify="right")
        table.add_column("New scale", justify="right")

    table.add_column("Status")

    error = _first_error(result)
    status = (
        Text("PASS", style="green")
        if error is None
        else Text(f"FAIL\n{error}", style="red")
    )

    c1 = _time(result.candidate.get(1))
    c4 = _time(result.candidate.get(4))

    if not two_versions:
        table.add_row(
            result.name,
            format_time(c1),
            format_time(c4),
            _fmt_speedup(_speedup(result.candidate)),
            status,
        )
    else:
        assert result.reference is not None
        r1 = _time(result.reference.get(1))
        r4 = _time(result.reference.get(4))

        table.add_row(
            result.name,
            format_time(r1),
            format_time(c1),
            _fmt_delta(_delta(r1, c1)),
            format_time(r4),
            format_time(c4),
            _fmt_delta(_delta(r4, c4)),
            _fmt_speedup(_speedup(result.reference)),
            _fmt_speedup(_speedup(result.candidate)),
            status,
        )

    return table


def write_github_summary(
    result: BenchmarkResult,
    candidate_solver: Path,
    reference_solver: Path | None,
    runs: int,
) -> None:
    path = github_summary_path()
    if not path:
        return

    lines = [
        "# FEMaster Benchmark",
        "",
        f"Candidate: `{escape_markdown(str(candidate_solver))}`  ",
        f"Runs per configuration: `{runs}`",
    ]

    if reference_solver is not None:
        lines.insert(
            3,
            f"Reference: `{escape_markdown(str(reference_solver))}`  ",
        )

    lines.append("")

    error = _first_error(result)
    state = "PASS" if error is None else "FAIL — " + escape_markdown(error)

    c1 = _time(result.candidate.get(1))
    c4 = _time(result.candidate.get(4))

    if result.reference is None:
        lines += [
            "| Model | 1T | 4T | Speedup | Status |",
            "|---|---:|---:|---:|---|",
            f"| {escape_markdown(result.name)} "
            f"| {format_time(c1)} "
            f"| {format_time(c4)} "
            f"| {_fmt_speedup(_speedup(result.candidate))} "
            f"| {state} |",
        ]
    else:
        r1 = _time(result.reference.get(1))
        r4 = _time(result.reference.get(4))

        lines += [
            "| Model | Ref 1T | New 1T | Δ1T | Ref 4T | New 4T | Δ4T | Ref scale | New scale | Status |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            f"| {escape_markdown(result.name)} "
            f"| {format_time(r1)} "
            f"| {format_time(c1)} "
            f"| {_fmt_delta(_delta(r1, c1))} "
            f"| {format_time(r4)} "
            f"| {format_time(c4)} "
            f"| {_fmt_delta(_delta(r4, c4))} "
            f"| {_fmt_speedup(_speedup(result.reference))} "
            f"| {_fmt_speedup(_speedup(result.candidate))} "
            f"| {state} |",
        ]

    with open(path, "a") as stream:
        stream.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one FEMaster case with 1T/4T. "
            "Optionally compare against a reference solver."
        )
    )
    parser.add_argument("solver", help="Candidate FEMaster executable")
    parser.add_argument("case", help="Benchmark case directory name")
    parser.add_argument(
        "--reference",
        help="Optional reference FEMaster executable",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Runs per model and thread count (default: 3)",
    )
    args = parser.parse_args()

    candidate_solver = require_solver(args.solver)
    reference_solver = (
        require_solver(args.reference)
        if args.reference
        else None
    )

    result = benchmark_case(
        candidate_solver,
        args.case,
        reference_solver=reference_solver,
        runs=args.runs,
    )

    console.print(create_table(result))
    write_github_summary(
        result,
        candidate_solver,
        reference_solver,
        args.runs,
    )

    return 1 if _first_error(result) else 0


if __name__ == "__main__":
    sys.exit(main())
