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

from check import CheckResult, check_case
from common import (
    discover_cases,
    effective_time,
    escape_markdown,
    format_int,
    format_time,
    github_summary_path,
    require_solver,
)


console = Console()


def create_table(
    results: dict[str, CheckResult],
    running: str | None = None,
) -> Table:
    table = Table(
        title="FEMaster Result Checks",
        box=box.SIMPLE,
        show_lines=False,
        header_style="bold",
    )

    table.add_column("Model")
    table.add_column("Analysis")
    table.add_column("Nodes", justify="right")
    table.add_column("DOFs", justify="right")
    table.add_column("Constraint")
    table.add_column("Constraints", justify="right")
    table.add_column("NNZ", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Status")

    for name in sorted(results):
        result = results[name]

        if result.run is None:
            table.add_row(
                name,
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                Text(
                    f"FAIL\n{result.error or result.detail}",
                    style="red",
                ),
            )
            continue

        meta = result.run.meta
        status = (
            Text("PASS", style="green")
            if result.passed
            else Text(f"FAIL\n{result.detail}", style="red")
        )

        table.add_row(
            name,
            meta["step_type"] or "-",
            format_int(meta["num_nodes"]),
            format_int(meta["num_dofs"]),
            meta["constraint_method"] or "-",
            format_int(meta["num_constraints"]),
            format_int(meta["num_nonzeros"]),
            format_time(effective_time(result.run)),
            status,
        )

    if running is not None:
        table.add_row(
            running,
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            Text("RUNNING", style="yellow"),
        )

    return table


def write_github_summary(
    results: dict[str, CheckResult],
    solver: Path,
    ncpus: int,
) -> None:
    path = github_summary_path()
    if not path:
        return

    lines = [
        "# FEMaster Result Checks",
        "",
        f"Solver: `{escape_markdown(str(solver))}`  ",
        f"Threads: `{ncpus}`",
        "",
        "| Model | Analysis | Nodes | DOFs | Constraint | Constraints | NNZ | Time | Status |",
        "|---|---|---:|---:|---|---:|---:|---:|---|",
    ]

    passed = 0

    for name in sorted(results):
        result = results[name]

        if result.run is None:
            lines.append(
                f"| {escape_markdown(name)} | - | - | - | - | - | - | - "
                f"| FAIL — {escape_markdown(result.error or result.detail)} |"
            )
            continue

        meta = result.run.meta
        if result.passed:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL — " + escape_markdown(result.detail)

        lines.append(
            f"| {escape_markdown(name)} "
            f"| {meta['step_type'] or '-'} "
            f"| {format_int(meta['num_nodes'])} "
            f"| {format_int(meta['num_dofs'])} "
            f"| {meta['constraint_method'] or '-'} "
            f"| {format_int(meta['num_constraints'])} "
            f"| {format_int(meta['num_nonzeros'])} "
            f"| {format_time(effective_time(result.run))} "
            f"| {status} |"
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
        description="Run all FEMaster benchmarks and check target.yaml files."
    )
    parser.add_argument("solver", help="Path to FEMaster executable")
    parser.add_argument(
        "--ncpus",
        type=int,
        default=1,
        help="Threads used by the solver (default: 1)",
    )
    args = parser.parse_args()

    solver = require_solver(args.solver)
    cases = discover_cases()

    if not cases:
        console.print("[red]No benchmark cases found.[/red]")
        return 1

    results: dict[str, CheckResult] = {}

    with tempfile.TemporaryDirectory(prefix="femaster_check_all_") as temp:
        work_root = Path(temp)

        with Live(
            create_table(results),
            console=console,
            refresh_per_second=4,
            transient=False,
        ) as live:
            for name in cases:
                live.update(create_table(results, name))

                results[name] = check_case(
                    solver,
                    name,
                    ncpus=args.ncpus,
                    work_root=work_root,
                )

                live.update(create_table(results))

    write_github_summary(results, solver, args.ncpus)

    return 0 if all(result.passed for result in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
