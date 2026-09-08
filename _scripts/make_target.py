#!/usr/bin/env python3

from __future__ import annotations

import argparse

from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

from check import DEFAULT_ATOL, DEFAULT_RTOL, read_result_maxima


console = Console()


def write_target(
    path: Path,
    fields: dict[str, float],
    rtol: float,
    atol: float,
) -> None:
    with path.open("w", newline="\n") as stream:
        stream.write("results:\n")

        for name, value in fields.items():
            stream.write(f"  {name}:\n")
            stream.write("    max_abs:\n")
            stream.write(f"      value: {value:.16e}\n")
            stream.write(f"      rtol: {rtol:.16e}\n")
            stream.write(f"      atol: {atol:.16e}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate target.yaml from a FEMaster .res file."
    )
    parser.add_argument(
        "result",
        type=Path,
        help="FEMaster .res or .res.gz result file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output YAML. Defaults to target.yaml beside the result.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=DEFAULT_RTOL,
        help=f"Relative tolerance (default: {DEFAULT_RTOL:g})",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=DEFAULT_ATOL,
        help=f"Absolute tolerance (default: {DEFAULT_ATOL:g})",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite an existing target file.",
    )
    args = parser.parse_args()

    result_path = args.result.expanduser().resolve()
    if not result_path.is_file():
        raise FileNotFoundError(result_path)

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else result_path.parent / "target.yaml"
    )

    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"Target already exists: {output_path}\n"
            "Use --force to overwrite it."
        )

    fields = read_result_maxima(result_path)
    write_target(
        output_path,
        fields,
        args.rtol,
        args.atol,
    )

    table = Table(
        title=f"Created {output_path.name}",
        box=box.SIMPLE,
        header_style="bold",
    )
    table.add_column("Field")
    table.add_column("max_abs", justify="right")

    for name, value in fields.items():
        table.add_row(name, f"{value:.8e}")

    console.print(table)
    console.print(f"[green]Created:[/green] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
