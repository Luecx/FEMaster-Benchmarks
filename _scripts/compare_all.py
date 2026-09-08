#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import sys
import tempfile

from dataclasses import dataclass
from pathlib import Path

from rich import box
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from common import (
    discover_cases,
    escape_markdown,
    failure_detail,
    github_summary_path,
    require_solver,
    run_solver,
)
from compare import (
    Comparison,
    aggregate,
    compare_files,
    gate_failures,
    print_comparison,
)


console = Console()


@dataclass
class CaseResult:
    name: str
    comparison: Comparison | None
    failures: list[str]
    error: str | None = None


def _fmt(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "-"

    if abs(value) < 5e-13:
        return "0"

    return f"{value:.{digits}g}"


def _status(result: CaseResult, gate: bool) -> Text:
    if result.comparison is None:
        return Text("FAIL", style="red bold")

    if gate and result.failures:
        return Text("FAIL", style="red bold")

    data = aggregate(result.comparison)
    changed = (
        math.isfinite(float(data["max_rel_l2"]))
        and float(data["max_rel_l2"]) > 5e-13
    )

    return (
        Text("CHANGED", style="yellow")
        if changed
        else Text("IDENTICAL", style="green")
    )


def create_table(
    results: dict[str, CaseResult],
    gate: bool,
    running: tuple[str, str] | None = None,
) -> Table:
    table = Table(
        title="FEMaster Output Comparison",
        box=box.SIMPLE,
        show_lines=False,
        header_style="bold",
    )

    table.add_column("Model")
    table.add_column("Fields", justify="right")
    table.add_column("max relL2", justify="right")
    table.add_column("max W1rel", justify="right")
    table.add_column("min Pearson", justify="right")
    table.add_column("min Spearman", justify="right")
    table.add_column("max |q99/RMS|", justify="right")
    table.add_column("max |Δmax|", justify="right")
    table.add_column("Status")

    for name in sorted(results):
        result = results[name]

        if result.comparison is None:
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
                    f"FAIL\n{result.error or '; '.join(result.failures)}",
                    style="red",
                ),
            )
            continue

        data = aggregate(result.comparison)
        state = _status(result, gate)

        if gate and result.failures:
            state.append("\n" + result.failures[0], style="red")

        table.add_row(
            name,
            str(data["fields"]),
            _fmt(float(data["max_rel_l2"])),
            _fmt(float(data["max_w1_rel"])),
            _fmt(float(data["min_pearson"])),
            _fmt(float(data["min_spearman"])),
            _fmt(float(data["max_abs_q99_rms"])),
            _fmt(float(data["max_abs_max_change"])),
            state,
        )

    if running is not None:
        name, phase = running
        table.add_row(
            name,
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            Text(f"RUNNING {phase}", style="yellow"),
        )

    return table


def write_github_summary(
    results: dict[str, CaseResult],
    reference_solver: Path,
    candidate_solver: Path,
    ncpus: int,
    gate: bool,
) -> None:
    path = github_summary_path()
    if not path:
        return

    lines = [
        "# FEMaster Output Comparison",
        "",
        f"Reference: `{escape_markdown(str(reference_solver))}`  ",
        f"Candidate: `{escape_markdown(str(candidate_solver))}`  ",
        f"Threads: `{ncpus}`",
        "",
        "| Model | Fields | max relL2 | max W1rel | min Pearson | min Spearman | max \\|q99/RMS\\| | max \\|Δmax\\| | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    identical = 0
    changed = 0
    failed = 0

    for name in sorted(results):
        result = results[name]

        if result.comparison is None:
            detail = result.error or "; ".join(result.failures)
            lines.append(
                f"| {escape_markdown(name)} | - | - | - | - | - | - | - "
                f"| FAIL — {escape_markdown(detail)} |"
            )
            failed += 1
            continue

        data = aggregate(result.comparison)
        max_rel_l2 = float(data["max_rel_l2"])
        is_changed = (
            math.isfinite(max_rel_l2)
            and max_rel_l2 > 5e-13
        )

        if gate and result.failures:
            state = "FAIL — " + "; ".join(result.failures)
            failed += 1
        elif is_changed:
            state = "CHANGED"
            changed += 1
        else:
            state = "IDENTICAL"
            identical += 1

        lines.append(
            f"| {escape_markdown(name)} "
            f"| {data['fields']} "
            f"| {_fmt(max_rel_l2)} "
            f"| {_fmt(float(data['max_w1_rel']))} "
            f"| {_fmt(float(data['min_pearson']))} "
            f"| {_fmt(float(data['min_spearman']))} "
            f"| {_fmt(float(data['max_abs_q99_rms']))} "
            f"| {_fmt(float(data['max_abs_max_change']))} "
            f"| {escape_markdown(state)} |"
        )

    total = len(results)
    lines += [
        "",
        f"**Total:** {total}  ",
        f"**Identical:** {identical}  ",
        f"**Changed:** {changed}  ",
        f"**Failed:** {failed}",
    ]

    with open(path, "a") as stream:
        stream.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run all FEMaster benchmarks with two solver executables "
            "and compare their .res outputs."
        )
    )
    parser.add_argument("reference_solver", help="Reference FEMaster executable")
    parser.add_argument("candidate_solver", help="Candidate FEMaster executable")
    parser.add_argument(
        "--ncpus",
        type=int,
        default=1,
        help="Threads per solver run (default: 1)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed field comparisons after the summary",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Fail when W1/Spearman/q99 thresholds are exceeded",
    )
    parser.add_argument("--max-w1", type=float, default=0.12)
    parser.add_argument("--min-spearman", type=float, default=0.94)
    parser.add_argument("--max-q99", type=float, default=0.20)
    args = parser.parse_args()

    reference_solver = require_solver(args.reference_solver)
    candidate_solver = require_solver(args.candidate_solver)

    cases = discover_cases()
    if not cases:
        console.print("[red]No benchmark cases found.[/red]")
        return 1

    results: dict[str, CaseResult] = {}

    with tempfile.TemporaryDirectory(prefix="femaster_compare_all_") as temp:
        work_root = Path(temp)

        with Live(
            create_table(results, args.gate),
            console=console,
            refresh_per_second=4,
            transient=False,
        ) as live:
            for name in cases:
                ref_dir = work_root / "reference" / name
                cand_dir = work_root / "candidate" / name

                try:
                    live.update(
                        create_table(
                            results,
                            args.gate,
                            (name, "reference"),
                        )
                    )
                    reference_run = run_solver(
                        reference_solver,
                        name,
                        ref_dir,
                        ncpus=args.ncpus,
                        runs=1,
                    )

                    live.update(
                        create_table(
                            results,
                            args.gate,
                            (name, "candidate"),
                        )
                    )
                    candidate_run = run_solver(
                        candidate_solver,
                        name,
                        cand_dir,
                        ncpus=args.ncpus,
                        runs=1,
                    )

                    live.update(
                        create_table(
                            results,
                            args.gate,
                            (name, "compare"),
                        )
                    )
                    comparison = compare_files(
                        reference_run.result,
                        candidate_run.result,
                    )

                    failures = (
                        gate_failures(
                            comparison,
                            args.max_w1,
                            args.min_spearman,
                            args.max_q99,
                        )
                        if args.gate
                        else []
                    )

                    results[name] = CaseResult(
                        name=name,
                        comparison=comparison,
                        failures=failures,
                    )

                except Exception as error:
                    log = (
                        cand_dir / "model.log"
                        if (cand_dir / "model.log").exists()
                        else ref_dir / "model.log"
                    )
                    results[name] = CaseResult(
                        name=name,
                        comparison=None,
                        failures=["execution/comparison failed"],
                        error=failure_detail(log, error),
                    )

                live.update(create_table(results, args.gate))

        if args.verbose:
            for name in cases:
                result = results[name]
                if result.comparison is None:
                    continue

                console.rule(name)
                print_comparison(result.comparison)

    write_github_summary(
        results,
        reference_solver,
        candidate_solver,
        args.ncpus,
        args.gate,
    )

    hard_failure = any(
        result.comparison is None
        for result in results.values()
    )
    gate_failure = (
        args.gate
        and any(result.failures for result in results.values())
    )

    return 1 if hard_failure or gate_failure else 0


if __name__ == "__main__":
    sys.exit(main())
