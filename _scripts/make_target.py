#!/usr/bin/env python3

import argparse
import gzip
from pathlib import Path
import re


DEFAULT_RTOL = 1.0e-2
DEFAULT_ATOL = 1.0e-8


def read_field_maxima(path: Path) -> dict[str, float]:
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

            # First column is the entity/node identifier.
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a FEMaster benchmark target.yaml from a .res file."
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
        help="Output YAML file. Defaults to target.yaml beside the result file.",
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

    result_path = args.result.resolve()

    if not result_path.is_file():
        raise FileNotFoundError(result_path)

    output_path = (
        args.output.resolve()
        if args.output is not None
        else result_path.parent / "target.yaml"
    )

    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"Target already exists: {output_path}\n"
            "Use --force to overwrite it."
        )

    fields = read_field_maxima(result_path)

    write_target(
        output_path,
        fields,
        args.rtol,
        args.atol,
    )

    print(f"Created: {output_path}")
    print()
    print("Targets:")

    for name, value in fields.items():
        print(f"  {name:<24} {value:.8e}")


if __name__ == "__main__":
    main()