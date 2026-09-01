import pathlib
import os
import re
import subprocess
import sys

import yaml


ROOT = pathlib.Path(__file__).parent.parent.absolute()


def read_input_meta(path, meta):
    text = path.read_text()

    # loadcase type
    match = re.search(r"\*LOADCASE\b[^\n]*\bTYPE\s*=\s*([^,\s]+)", text, re.IGNORECASE)
    if match:
        meta["step_type"] = match.group(1).upper()

    # nodes
    blocks = re.findall(
        r"^\*NODE[^\n]*\n(.*?)(?=^\*)",
        text + "\n*",
        re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    meta["num_nodes"] = sum(
        len([line for line in block.splitlines() if line.strip() and not line.strip().startswith("**")])
        for block in blocks
    )

    # elements
    blocks = re.findall(
        r"^\*ELEMENT([^\n]*)\n(.*?)(?=^\*)",
        text + "\n*",
        re.IGNORECASE | re.MULTILINE | re.DOTALL
    )

    for header, block in blocks:
        match = re.search(r"\bTYPE\s*=\s*([^,\s]+)", header, re.IGNORECASE)
        if not match:
            continue

        element_type = match.group(1).upper()
        num_elements = len([
            line for line in block.splitlines()
            if line.strip() and not line.strip().startswith("**")
        ])

        meta["elements"][element_type] = meta["elements"].get(element_type, 0) + num_elements


def read_log_meta(path, meta):
    text = path.read_text()

    matches = re.findall(r"Assembled C:\s*m=(\d+)\s+n=(\d+)", text)
    if matches:
        meta["num_constraints"] = int(matches[-1][0])
        meta["num_dofs"] = int(matches[-1][1])

    matches = re.findall(r"method\s*:\s*(\S+)", text)
    if matches:
        meta["constraint_method"] = matches[-1].lower()

    matches = re.findall(r"nnz=(\d+)", text)
    if matches:
        meta["num_nonzeros"] = int(matches[-1])

    matches = re.findall(r"\((\d+)\s*ms total\)", text)
    if matches:
        meta["time"] = sum(map(int, matches)) / len(matches) / 1000.0


def read_result_maxima(path):
    """Read the maximum absolute value of every FIELD in a FEMaster .res file."""
    fields = {}
    field = None

    with open(path, "rt") as stream:
        for line in stream:
            if line.startswith("FIELD"):
                match = re.search(r"NAME=([^,]+)", line)
                if not match:
                    raise RuntimeError(f"Could not parse FIELD line:\n{line}")

                field = match.group(1).strip()
                fields[field] = 0.0
                continue

            if line.startswith("END FIELD"):
                field = None
                continue

            if field is None:
                continue

            parts = line.split()
            if len(parts) <= 1:
                continue

            try:
                values = [float(value) for value in parts[1:]]
            except ValueError:
                continue

            if values:
                fields[field] = max(
                    fields[field],
                    max(abs(value) for value in values),
                )

    if not fields:
        raise RuntimeError(f"No result fields found in '{path}'.")

    return fields


def read_target(path):
    """Load and validate the benchmark target YAML."""
    with open(path, "rt") as stream:
        data = yaml.safe_load(stream)

    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid target file '{path}': expected a YAML mapping.")

    results = data.get("results")
    if not isinstance(results, dict) or not results:
        raise RuntimeError(f"Invalid target file '{path}': missing non-empty 'results' mapping.")

    return results


def compare_results(result_path, target_path):
    """
    Compare FEMaster result maxima against explicitly defined YAML targets.

    Only fields listed in target.yaml are checked. Extra fields in model.res are
    intentionally ignored, allowing benchmarks to target only physically relevant
    quantities.
    """
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
            rtol = 1e-2
            atol = 1e-8
        elif isinstance(max_abs, dict):
            if "value" not in max_abs:
                return f"{field_name}: max_abs target is missing 'value'"

            ref_value = float(max_abs["value"])
            rtol = float(max_abs.get("rtol", 1e-2))
            atol = float(max_abs.get("atol", 1e-8))
        else:
            return f"{field_name}: invalid max_abs target"

        value = result[field_name]
        error = abs(value - ref_value)
        tolerance = atol + rtol * abs(ref_value)

        if error > tolerance:
            relative_error = error / abs(ref_value) if ref_value != 0.0 else None
            relative_text = f"{relative_error:.2%}" if relative_error is not None else "n/a"

            return (
                f"{field_name}: maximum magnitude "
                f"{value:.6e} vs {ref_value:.6e}, "
                f"abs={error:.2e}, "
                f"rel={relative_text}, "
                f"tol={tolerance:.2e}"
            )

    return None


def run_case(solver_path, name, num_runs=1, ncpus=1):
    case = ROOT / name
    model = case / "model.inp"
    result = case / "model.res"
    target = case / "target.yaml"
    log = case / "model.log"

    # check if the folder exists
    if not case.is_dir():
        raise FileNotFoundError(case)

    # check if the solver exists
    if not pathlib.Path(solver_path).is_file():
        raise FileNotFoundError("Solver not found: {}".format(solver_path))

    # check if the folder contains all important files
    if not model.exists():
        raise FileNotFoundError("Model not found: {}".format(model))

    if not target.exists():
        raise FileNotFoundError("Benchmark target not found: {}".format(target))

    # remove generated files
    for file in os.listdir(case):
        if not file.endswith((".inp", ".ref", ".ref.gz", ".yaml")):
            os.remove(case / file)

    # run solver
    with open(log, "w") as f:
        for _ in range(num_runs):
            subprocess.check_call(
                [solver_path, model, "--no-frd", "--ncpus", str(ncpus)],
                stdout=f,
                stderr=subprocess.STDOUT,
            )

    if not result.exists():
        raise FileNotFoundError("Solver did not create result file: {}".format(result))

    meta = {
        "time"              : None,
        "step_type"         : None,
        "constraint_method" : None,
        "num_nodes"         : None,
        "num_dofs"          : None,
        "num_constraints"   : None,
        "num_nonzeros"      : None,
        "elements"          : {}
    }

    # extract metadata
    read_input_meta(model, meta)
    read_log_meta(log, meta)

    # compare result against explicit YAML target
    detail = compare_results(result, target)
    passed = detail is None

    # clean generated files
    for file in os.listdir(case):
        if not file.endswith((".inp", ".yaml")):
            os.remove(case / file)

    return meta, passed, detail


if __name__ == "__main__":
    args = sys.argv[1:]
    print(run_case(*args))
