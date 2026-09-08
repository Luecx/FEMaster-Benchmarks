#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import tempfile

from pathlib import Path

from rich import box
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from benchmark import (
    BenchmarkResult,
    _delta,
    _first_error,
    _fmt_delta,
    _fmt_speedup,
    _speedup,
    _time,
    benchmark_case,
)
from common import (
    discover_cases,
    escape_markdown,
    format_time,
    github_summary_path,
    require_solver,
)


console = Console()


def create_table(
    results: dict[str, BenchmarkResult],
    two_versions: bool,
    running: str | None = None,
) -> Table:
    table = Table(
        title="FEMaster Benchmarks",
        box=box.SIMPLE,
        show_lines=False,
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

    for name in sorted(results):
        result = results[name]
        error = _first_error(result)
        state = (
            Text("PASS", style="green")
            if error is None
            else Text(f"FAIL\n{error}", style="red")
        )

        c1 = _time(result.candidate.get(1))
        c4 = _time(result.candidate.get(4))

        if not two_versions:
            table.add_row(
                name,
                format_time(c1),
                format_time(c4),
                _fmt_speedup(_speedup(result.candidate)),
                state,
            )
        else:
            assert result.reference is not None
            r1 = _time(result.reference.get(1))
            r4 = _time(result.reference.get(4))

            table.add_row(
                name,
                format_time(r1),
                format_time(c1),
                _fmt_delta(_delta(r1, c1)),
                format_time(r4),
                format_time(c4),
                _fmt_delta(_delta(r4, c4)),
                _fmt_speedup(_speedup(result.reference)),
                _fmt_speedup(_speedup(result.candidate)),
                state,
            )

    if running is not None:
        width = 4 if not two_versions else 9
        table.add_row(
            running,
            *(["-"] * (width - 1)),
            Text("RUNNING", style="yellow"),
        )

    return table


def write_github_summary(
    results: dict[str, BenchmarkResult],
    candidate_solver: Path,
    reference_solver: Path | None,
    runs: int,
) -> None:
    path = github_summary_path()
    if not path:
        return

    two_versions = reference_solver is not None

    lines = [
        "# FEMaster Benchmarks",
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

    if not two_versions:
        lines += [
            "| Model | 1T | 4T | Speedup | Status |",
            "|---|---:|---:|---:|---|",
        ]
    else:
        lines += [
            "| Model | Ref 1T | New 1T | Δ1T | Ref 4T | New 4T | Δ4T | Ref scale | New scale | Status |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]

    passed = 0

    for name in sorted(results):
        result = results[name]
        error = _first_error(result)
        state = "PASS" if error is None else "FAIL — " + escape_markdown(error)

        if error is None:
            passed += 1

        c1 = _time(result.candidate.get(1))
        c4 = _time(result.candidate.get(4))

        if not two_versions:
            lines.append(
                f"| {escape_markdown(name)} "
                f"| {format_time(c1)} "
                f"| {format_time(c4)} "
                f"| {_fmt_speedup(_speedup(result.candidate))} "
                f"| {state} |"
            )
        else:
            assert result.reference is not None
            r1 = _time(result.reference.get(1))
            r4 = _time(result.reference.get(4))

            lines.append(
                f"| {escape_markdown(name)} "
                f"| {format_time(r1)} "
                f"| {format_time(c1)} "
                f"| {_fmt_delta(_delta(r1, c1))} "
                f"| {format_time(r4)} "
                f"| {format_time(c4)} "
                f"| {_fmt_delta(_delta(r4, c4))} "
                f"| {_fmt_speedup(_speedup(result.reference))} "
                f"| {_fmt_speedup(_speedup(result.candidate))} "
                f"| {state} |"
            )

    total = len(results)
    lines += [
        "",
        f"**Status:** {'PASS' if passed == total else 'FAIL'} ({passed}/{total})",
    ]

    with open(path, "a") as stream:
        stream.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark all FEMaster cases with 1T/4T. "
            "Optionally compare against a reference solver."
        )
    )
    parser.add_argument("solver", help="Candidate FEMaster executable")
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
    two_versions = reference_solver is not None

    cases = discover_cases()
    if not cases:
        console.print("[red]No benchmark cases found.[/red]")
        return 1

    results: dict[str, BenchmarkResult] = {}

    with tempfile.TemporaryDirectory(prefix="femaster_benchmark_all_") as temp:
        work_root = Path(temp)

        with Live(
            create_table(results, two_versions),
            console=console,
            refresh_per_second=4,
            transient=False,
        ) as live:
            for name in cases:
                live.update(
                    create_table(
                        results,
                        two_versions,
                        running=name,
                    )
                )

                results[name] = benchmark_case(
                    candidate_solver,
                    name,
                    reference_solver=reference_solver,
                    runs=args.runs,
                    work_root=work_root,
                )

                live.update(create_table(results, two_versions))

    write_github_summary(
        results,
        candidate_solver,
        reference_solver,
        args.runs,
    )

    return 1 if any(_first_error(result) for result in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
