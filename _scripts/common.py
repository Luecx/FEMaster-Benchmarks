#!/usr/bin/env python3

from __future__ import annotations

import os
import re
import shutil
import statistics
import subprocess
import time

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).parent.parent.absolute()
DEFAULT_THREADS = (1, 4)


@dataclass
class SolverRun:
    case: str
    workdir: Path
    model: Path
    result: Path
    log: Path
    meta: dict
    wall_times: list[float]


def discover_cases() -> list[str]:
    return sorted(
        path.name
        for path in ROOT.iterdir()
        if path.is_dir()
        and not path.name.startswith("_")
        and (path / "model.inp").exists()
    )


def require_solver(path: str | Path) -> Path:
    solver = Path(path).expanduser().resolve()

    if not solver.is_file():
        raise FileNotFoundError(f"Solver not found: {solver}")

    return solver


def require_case(name: str) -> Path:
    case = ROOT / name

    if not case.is_dir():
        raise FileNotFoundError(f"Benchmark case not found: {case}")

    model = case / "model.inp"
    if not model.is_file():
        raise FileNotFoundError(f"Model not found: {model}")

    return case


def read_input_meta(path: Path, meta: dict) -> None:
    text = path.read_text(errors="replace")

    match = re.search(
        r"\*LOADCASE\b[^\n]*\bTYPE\s*=\s*([^,\s]+)",
        text,
        re.IGNORECASE,
    )
    if match:
        meta["step_type"] = match.group(1).upper()

    blocks = re.findall(
        r"^\*NODE[^\n]*\n(.*?)(?=^\*)",
        text + "\n*",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    meta["num_nodes"] = sum(
        len([
            line
            for line in block.splitlines()
            if line.strip() and not line.strip().startswith("**")
        ])
        for block in blocks
    )

    blocks = re.findall(
        r"^\*ELEMENT([^\n]*)\n(.*?)(?=^\*)",
        text + "\n*",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )

    for header, block in blocks:
        match = re.search(r"\bTYPE\s*=\s*([^,\s]+)", header, re.IGNORECASE)
        if not match:
            continue

        element_type = match.group(1).upper()
        num_elements = len([
            line
            for line in block.splitlines()
            if line.strip() and not line.strip().startswith("**")
        ])

        meta["elements"][element_type] = (
            meta["elements"].get(element_type, 0) + num_elements
        )


def read_log_meta(path: Path, meta: dict) -> None:
    text = path.read_text(errors="replace")

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

    # Keep the timing definition used by the existing benchmark runner:
    # average all solver-reported "(... ms total)" measurements in the log.
    matches = re.findall(r"\((\d+(?:\.\d+)?)\s*ms total\)", text)
    if matches:
        meta["time"] = statistics.mean(map(float, matches)) / 1000.0


def _fresh_meta() -> dict:
    return {
        "time": None,
        "wall_time": None,
        "step_type": None,
        "constraint_method": None,
        "num_nodes": None,
        "num_dofs": None,
        "num_constraints": None,
        "num_nonzeros": None,
        "elements": {},
    }


def _copy_case(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)

    # Remove only known generated files. Everything else may be an input/include.
    for filename in (
        "model.res",
        "model.log",
        "model.frd",
        "model.vtu",
        "model.pvd",
        "model.vtkhdf",
    ):
        path = destination / filename
        if path.exists():
            path.unlink()


def run_solver(
    solver: str | Path,
    case_name: str,
    workdir: str | Path,
    *,
    ncpus: int = 1,
    runs: int = 1,
) -> SolverRun:
    if ncpus < 1:
        raise ValueError("ncpus must be >= 1")

    if runs < 1:
        raise ValueError("runs must be >= 1")

    solver = require_solver(solver)
    source_case = require_case(case_name)
    workdir = Path(workdir).resolve()

    _copy_case(source_case, workdir)

    model = workdir / "model.inp"
    result = workdir / "model.res"
    log = workdir / "model.log"

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(ncpus)
    env["MKL_NUM_THREADS"] = str(ncpus)
    env["OPENBLAS_NUM_THREADS"] = str(ncpus)

    wall_times: list[float] = []

    with log.open("w") as stream:
        for run_index in range(runs):
            if runs > 1:
                stream.write(
                    f"\n===== BENCHMARK RUN {run_index + 1}/{runs} =====\n"
                )
                stream.flush()

            start = time.perf_counter()

            subprocess.check_call(
                [
                    str(solver),
                    model.name,
                    "--no-frd",
                    "--ncpus",
                    str(ncpus),
                ],
                cwd=workdir,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )

            wall_times.append(time.perf_counter() - start)

    if not result.is_file():
        raise FileNotFoundError(
            f"Solver did not create result file: {result}"
        )

    meta = _fresh_meta()
    read_input_meta(model, meta)
    read_log_meta(log, meta)
    meta["wall_time"] = statistics.mean(wall_times)

    return SolverRun(
        case=case_name,
        workdir=workdir,
        model=model,
        result=result,
        log=log,
        meta=meta,
        wall_times=wall_times,
    )


def effective_time(run: SolverRun | None) -> float | None:
    if run is None:
        return None

    if run.meta.get("time") is not None:
        return float(run.meta["time"])

    if run.meta.get("wall_time") is not None:
        return float(run.meta["wall_time"])

    return None


def failure_detail(log: Path, error: Exception) -> str:
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


def format_time(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f} s"


def format_int(value: int | None) -> str:
    return "-" if value is None else f"{value:,}"


def escape_markdown(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def github_summary_path() -> str | None:
    return os.environ.get("GITHUB_STEP_SUMMARY")
