import gzip
import pathlib
import os
import re
import subprocess
import sys

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


def compare_results(result_path, reference_path, rtol=1e-4, atol=1e-8):
    def read(path):
        fields = []
        field = None

        open_text = gzip.open if path.suffix == ".gz" else open

        with open_text(path, "rt") as stream:
            for line in stream:
                if line.startswith("FIELD"):
                    field = {
                        "name": re.search(r"NAME=([^,]+)", line).group(1),
                        "rows": {}
                    }
                    fields.append(field)
                    continue

                if line.startswith("END FIELD") or field is None:
                    continue

                parts = line.split()
                if len(parts) > 1:
                    field["rows"][parts[0]] = [float(x) for x in parts[1:]]

        return fields

    result = read(result_path)
    reference = read(reference_path)

    if len(result) != len(reference):
        return f"field count differs: {len(result)} vs {len(reference)}"

    worst = None

    for field, ref_field in zip(result, reference):
        if field["name"] != ref_field["name"]:
            return f"field differs: {field['name']} vs {ref_field['name']}"

        rows = field["rows"]
        ref_rows = ref_field["rows"]

        new_rows = rows.keys() - ref_rows.keys()
        missing_rows = ref_rows.keys() - rows.keys()

        if new_rows:
            return f"{field['name']}: new rows: {', '.join(sorted(new_rows)[:5])}"

        if missing_rows:
            return f"{field['name']}: missing rows: {', '.join(sorted(missing_rows)[:5])}"

        # use the maximum magnitude of the reference field as numerical scale
        scale = max(
            (
                abs(value)
                for values in ref_rows.values()
                for value in values
            ),
            default=0.0,
        )

        tolerance = atol + rtol * scale

        for row_id, values in rows.items():
            ref_values = ref_rows[row_id]

            if len(values) != len(ref_values):
                return (
                    f"{field['name']} / {row_id}: "
                    f"{len(values)} values vs {len(ref_values)}"
                )

            for col, (value, ref_value) in enumerate(zip(values, ref_values), 1):
                error = abs(value - ref_value)
                ratio = error / tolerance if tolerance > 0.0 else 0.0

                if worst is None or ratio > worst[0]:
                    normalized = error / scale if scale > 0.0 else 0.0

                    worst = (
                        ratio,
                        field["name"],
                        row_id,
                        col,
                        value,
                        ref_value,
                        error,
                        normalized,
                        scale,
                        tolerance,
                    )

    if worst is None or worst[0] <= 1.0:
        return None

    (
        ratio,
        field,
        row_id,
        col,
        value,
        ref_value,
        error,
        normalized,
        scale,
        tolerance,
    ) = worst

    return (
        f"{field} / {row_id} / col {col}: "
        f"{value:.6e} vs {ref_value:.6e}, "
        f"abs={error:.2e}, "
        f"scale={scale:.2e}, "
        f"rel={normalized:.2%}, "
        f"tol={tolerance:.2e} ({ratio:.2f}x)"
    )


def run_case(solver_path, name, num_runs=1, ncpus=1):
    case = ROOT / name
    model = case / "model.inp"
    result = case / "model.res"
    reference = case / "model.res.ref.gz"
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

    if not reference.exists():
        raise FileNotFoundError("Model reference result not found: {}".format(reference))

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

    # compare result against reference
    detail = compare_results(result, reference)
    passed = detail is None

    # clean generated files
    for file in os.listdir(case):
        if not file.endswith((".inp", ".ref", ".ref.gz", ".yaml")):
            os.remove(case / file)

    return meta, passed, detail


if __name__ == "__main__":
    args = sys.argv[1:]
    print(run_case(*args))
