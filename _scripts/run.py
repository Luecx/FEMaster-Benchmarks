#!/usr/bin/env python3

import argparse
import os
import sys

from pathlib import Path

from rich import box
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from run_case import run_case


ROOT = Path(__file__).parent.parent.absolute()
THREADS = (1, 4)

console = Console()


def format_time(value):
    return "-" if value is None else f"{value:.2f} s"


def failure_detail(name, error):
    log = ROOT / name / "model.log"

    if not log.exists():
        return str(error)

    lines = [
        line.strip()
        for line in log.read_text(errors="replace").splitlines()
        if line.strip()
    ]

    for line in reversed(lines):
        if "[ERROR]" in line or line.startswith("Error:"):
            return line

    return str(error)


def format_status(r1, r4):
    if r1["passed"] and r4["passed"]:
        return Text("PASS", style="green")

    status = Text("FAIL", style="red bold")

    if not r1["passed"]:
        status.append(f"\n1T: {r1['detail']}", style="red")

    if not r4["passed"]:
        status.append(f"\n4T: {r4['detail']}", style="red")

    return status


def create_table(results, running=None):
    table = Table(
        title="FEMaster Benchmarks",
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
    table.add_column("1T", justify="right")
    table.add_column("4T", justify="right")
    table.add_column("Speedup", justify="right")
    table.add_column("Status")

    for name, model in results.items():
        if 1 not in model or 4 not in model:
            continue

        r1 = model[1]
        r4 = model[4]
        meta = r1["meta"] or r4["meta"]

        t1 = r1["meta"]["time"] if r1["meta"] else None
        t4 = r4["meta"]["time"] if r4["meta"] else None
        speedup = t1 / t4 if t1 and t4 else None

        table.add_row(
            name,
            meta["step_type"] if meta else "-",
            f"{meta['num_nodes']:,}" if meta and meta["num_nodes"] is not None else "-",
            f"{meta['num_dofs']:,}" if meta and meta["num_dofs"] is not None else "-",
            meta["constraint_method"] if meta and meta["constraint_method"] else "-",
            f"{meta['num_constraints']:,}" if meta and meta["num_constraints"] is not None else "-",
            f"{meta['num_nonzeros']:,}" if meta and meta["num_nonzeros"] is not None else "-",
            format_time(t1),
            format_time(t4),
            f"{speedup:.2f}x" if speedup else "-",
            format_status(r1, r4),
        )

    if running:
        name, threads = running

        table.add_row(
            name,
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-" if threads == 1 else format_time(results[name][1]["meta"]["time"]),
            "-",
            "-",
            Text(f"RUNNING {threads}T", style="yellow"),
        )

    return table


def write_github_summary(results):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return

    lines = [
        "# FEMaster Benchmarks",
        "",
        "| Model | Analysis | Nodes | DOFs | Constraint | Constraints | NNZ | 1T | 4T | Speedup | Status |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]

    for name, model in results.items():
        r1 = model[1]
        r4 = model[4]
        meta = r1["meta"] or r4["meta"]

        t1 = r1["meta"]["time"] if r1["meta"] else None
        t4 = r4["meta"]["time"] if r4["meta"] else None
        speedup = t1 / t4 if t1 and t4 else None

        if r1["passed"] and r4["passed"]:
            status = "PASS"
        else:
            details = []

            if not r1["passed"]:
                details.append(f"1T: {r1['detail']}")

            if not r4["passed"]:
                details.append(f"4T: {r4['detail']}")

            status = "FAIL — " + "; ".join(details)
            status = status.replace("|", "\\|").replace("\n", " ")

        lines.append(
            f"| {name} "
            f"| {meta['step_type'] if meta else '-'} "
            f"| {meta['num_nodes']:,} "
            f"| {meta['num_dofs']:,} "
            f"| {meta['constraint_method'] or '-'} "
            f"| {meta['num_constraints']:,} "
            f"| {meta['num_nonzeros']:,} "
            f"| {format_time(t1)} "
            f"| {format_time(t4)} "
            f"| {f'{speedup:.2f}x' if speedup else '-'} "
            f"| {status} |"
        )

    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run FEMaster benchmark suite.")
    parser.add_argument("solver", help="Path to FEMaster executable")
    parser.add_argument("--runs", type=int, default=1, help="Runs per model and thread count")
    args = parser.parse_args()

    cases = sorted(
        path.name
        for path in ROOT.iterdir()
        if path.is_dir()
        and not path.name.startswith("_")
        and (path / "model.inp").exists()
    )

    if not cases:
        console.print("[red]No benchmark cases found.[/red]")
        return 1

    results = {name: {} for name in cases}

    with Live(
            create_table(results),
            console=console,
            refresh_per_second=4,
            transient=False,
    ) as live:
        for name in cases:
            for threads in THREADS:
                live.update(create_table(results, (name, threads)))

                try:
                    meta, passed, detail = run_case(
                        args.solver,
                        name,
                        args.runs,
                        threads,
                    )

                    results[name][threads] = {
                        "meta": meta,
                        "passed": passed,
                        "detail": detail or "",
                    }

                except Exception as error:
                    results[name][threads] = {
                        "meta": None,
                        "passed": False,
                        "detail": failure_detail(name, error),
                    }

                live.update(create_table(results))

    write_github_summary(results)

    failed = any(
        not result["passed"]
        for model in results.values()
        for result in model.values()
    )

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())