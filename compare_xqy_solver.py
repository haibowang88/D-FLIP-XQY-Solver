#!/usr/bin/env python3
"""
compare_xqy_solver.py

Compare the feasibility-preserving heuristic:

    d_flip_xqy_dense_cd

against Gurobi's direct nonconvex MIQP model for the multi-floor
cross-dock door assignment problem.

The heuristic parameters are configurable from the command line.

Default heuristic settings
--------------------------

    --heuristic-time-limit 300
    --heuristic-max-iterations 100000
    --heuristic-n1 1
    --heuristic-n2 10
    --heuristic-sequence-count 150
    --heuristic-tabu-tenure 5
    --heuristic-restart-stagnation 5
    --heuristic-rebuild-interval 100000
    --heuristic-seed 1
    --heuristic-random-first

Default Gurobi settings
-----------------------

    --gurobi-time-limit 7200
    --gurobi-threads 0
    --gurobi-seed 1
    --mip-gap 0
    NonConvex = 2

Model
-----

    minimize

        sum_{i,w} FCX[i,w] x[i,w]
      + sum_{j,d} FCY[j,d] y[j,d]
      + sum_{i,w,j,d} Q[(i,w),(j,d)] x[i,w] y[j,d]

    subject to

        sum_w x[i,w] = 1
        sum_i x[i,w] <= 1

        sum_d y[j,d] = 1
        sum_j y[j,d] <= 1

        x[i,w], y[j,d] binary.

Prepared input format
---------------------

    number_of_instances
    x_items x_slots y_items y_slots q_record_count
    <x_items*x_slots x fixed costs>
    <y_items*y_slots y fixed costs>
    q_row q_column coefficient
    ...

The Q row and column indices are one-based flattened assignment-position
indices.

Requirements
------------

    python -m pip install gurobipy

A valid Gurobi license is required.

Examples
--------

Run with the default heuristic parameters:

    python compare_xqy_solver.py \
        --input n100m10-1-cd-xqy.txt \
        --heuristic ./d_flip_xqy_dense_cd \
        --output-dir comparison-default

Change the heuristic parameters:

    python compare_xqy_solver.py \
        --input n100m10-1-cd-xqy.txt \
        --heuristic ./d_flip_xqy_dense_cd \
        --output-dir comparison-custom \
        --heuristic-time-limit 200 \
        --heuristic-max-iterations 100000 \
        --heuristic-n1 1 \
        --heuristic-n2 10 \
        --heuristic-sequence-count 150 \
        --heuristic-tabu-tenure 5 \
        --heuristic-restart-stagnation 5 \
        --heuristic-seed 42

Use the validated heuristic result as a Gurobi MIP start:

    python compare_xqy_solver.py \
        --input n100m10-1-cd-xqy.txt \
        --heuristic ./d_flip_xqy_dense_cd \
        --output-dir comparison-warm \
        --warm-start

Use an existing heuristic result without executing the heuristic:

    python compare_xqy_solver.py \
        --input n100m10-1-cd-xqy.txt \
        --heuristic-result heuristic-result.txt \
        --skip-heuristic \
        --warm-start \
        --output-dir comparison-existing
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import pathlib
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as error:
    raise SystemExit(
        "Error: gurobipy could not be imported.\n"
        "Install it with:\n"
        "    python -m pip install gurobipy\n"
        "A valid Gurobi license is also required."
    ) from error


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class InstanceHeader:
    instance_count: int
    x_items: int
    x_slots: int
    y_items: int
    y_slots: int
    q_record_count: int

    @property
    def x_variable_count(self) -> int:
        return self.x_items * self.x_slots

    @property
    def y_variable_count(self) -> int:
        return self.y_items * self.y_slots


@dataclass
class AssignmentSolution:
    x_assignment: List[int]
    y_assignment: List[int]
    objective: Optional[int] = None


@dataclass
class HeuristicResult:
    status: str
    return_code: Optional[int]
    process_wall_seconds: float

    reported_objective: Optional[int]
    validated_objective: Optional[int]
    objective_valid: bool

    time_to_best_seconds: Optional[float]
    search_seconds: Optional[float]
    elapsed_seconds: Optional[float]
    load_seconds: Optional[float]

    outer_iterations: Optional[int]
    moves: Optional[int]
    evaluated_moves: Optional[int]
    critical_events: Optional[int]
    restarts: Optional[int]
    move_to_best: Optional[int]
    outer_iteration_to_best: Optional[int]

    x_assignment: List[int]
    y_assignment: List[int]

    result_file: str
    log_file: str
    command: str
    error_message: str = ""


@dataclass
class GurobiResult:
    status: str
    status_code: int
    solution_count: int

    objective: Optional[float]
    validated_objective: Optional[int]
    objective_valid: bool

    best_bound: Optional[float]
    mip_gap: Optional[float]

    runtime_seconds: float
    process_wall_seconds: float
    time_to_best_seconds: Optional[float]

    nodes: float
    simplex_iterations: float
    barrier_iterations: int
    work: Optional[float]

    warm_start_used: bool

    x_assignment: List[int]
    y_assignment: List[int]

    log_file: str
    solution_file: str
    error_message: str = ""


# =============================================================================
# Input parsing
# =============================================================================

def numeric_tokens(path: pathlib.Path) -> Iterator[str]:
    """Stream whitespace-separated tokens from a prepared input file."""

    with path.open(
        "r",
        encoding="utf-8",
        errors="strict",
        buffering=8 * 1024 * 1024,
    ) as stream:
        for raw_line in stream:
            line = raw_line.strip()

            if not line:
                continue

            for token in line.split():
                yield token


def parse_integer_token(
    token: str,
    description: str,
) -> int:
    """
    Parse an integer token.

    Integral floating-point representations such as "10.0" are accepted.
    """

    try:
        return int(token)
    except ValueError:
        try:
            value = float(token)
        except ValueError as error:
            raise ValueError(
                f"Invalid integer {description}: {token!r}"
            ) from error

        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(
                f"Nonintegral {description}: {token!r}"
            )

        return int(value)


def next_integer(
    tokens: Iterator[str],
    description: str,
) -> int:
    try:
        token = next(tokens)
    except StopIteration as error:
        raise ValueError(
            f"Unexpected end of file while reading {description}"
        ) from error

    return parse_integer_token(token, description)


def read_instance_prefix(
    path: pathlib.Path,
) -> Tuple[
    InstanceHeader,
    List[int],
    List[int],
    Iterator[str],
]:
    """
    Read the header and fixed costs.

    The returned iterator remains positioned at the first Q record.
    """

    tokens = numeric_tokens(path)

    instance_count = next_integer(
        tokens,
        "instance count",
    )

    if instance_count != 1:
        raise ValueError(
            "This comparison script currently requires exactly one "
            f"instance; the input declares {instance_count}"
        )

    x_items = next_integer(tokens, "x item count")
    x_slots = next_integer(tokens, "x slot count")
    y_items = next_integer(tokens, "y item count")
    y_slots = next_integer(tokens, "y slot count")
    q_record_count = next_integer(
        tokens,
        "Q record count",
    )

    if x_items <= 0 or y_items <= 0:
        raise ValueError(
            "The numbers of x and y items must be positive"
        )

    if x_slots < x_items:
        raise ValueError(
            "x_slots must be at least x_items"
        )

    if y_slots < y_items:
        raise ValueError(
            "y_slots must be at least y_items"
        )

    if q_record_count < 0:
        raise ValueError(
            "The Q record count cannot be negative"
        )

    header = InstanceHeader(
        instance_count=instance_count,
        x_items=x_items,
        x_slots=x_slots,
        y_items=y_items,
        y_slots=y_slots,
        q_record_count=q_record_count,
    )

    x_fixed = [
        next_integer(
            tokens,
            f"x fixed cost {index + 1}",
        )
        for index in range(
            header.x_variable_count
        )
    ]

    y_fixed = [
        next_integer(
            tokens,
            f"y fixed cost {index + 1}",
        )
        for index in range(
            header.y_variable_count
        )
    ]

    return header, x_fixed, y_fixed, tokens


# =============================================================================
# Independent objective validation
# =============================================================================

def validate_assignment_structure(
    header: InstanceHeader,
    solution: AssignmentSolution,
) -> None:
    if len(solution.x_assignment) != header.x_items:
        raise ValueError(
            f"Expected {header.x_items} x assignments, found "
            f"{len(solution.x_assignment)}"
        )

    if len(solution.y_assignment) != header.y_items:
        raise ValueError(
            f"Expected {header.y_items} y assignments, found "
            f"{len(solution.y_assignment)}"
        )

    if len(set(solution.x_assignment)) != header.x_items:
        raise ValueError(
            "The x assignment is not injective"
        )

    if len(set(solution.y_assignment)) != header.y_items:
        raise ValueError(
            "The y assignment is not injective"
        )

    if any(
        slot < 0 or slot >= header.x_slots
        for slot in solution.x_assignment
    ):
        raise ValueError(
            "An x assignment is outside the valid slot range"
        )

    if any(
        slot < 0 or slot >= header.y_slots
        for slot in solution.y_assignment
    ):
        raise ValueError(
            "A y assignment is outside the valid slot range"
        )


def evaluate_assignment(
    input_path: pathlib.Path,
    solution: AssignmentSolution,
) -> int:
    """Independently evaluate a feasible assignment."""

    header, x_fixed, y_fixed, tokens = read_instance_prefix(
        input_path
    )

    validate_assignment_structure(
        header,
        solution,
    )

    objective = 0

    active_x_positions = set()
    active_y_positions = set()

    for item, slot in enumerate(
        solution.x_assignment
    ):
        flat_index = (
            item * header.x_slots + slot
        )

        active_x_positions.add(flat_index)
        objective += x_fixed[flat_index]

    for item, slot in enumerate(
        solution.y_assignment
    ):
        flat_index = (
            item * header.y_slots + slot
        )

        active_y_positions.add(flat_index)
        objective += y_fixed[flat_index]

    for record in range(
        header.q_record_count
    ):
        row = next_integer(
            tokens,
            f"Q row at record {record + 1}",
        )

        column = next_integer(
            tokens,
            f"Q column at record {record + 1}",
        )

        coefficient = next_integer(
            tokens,
            f"Q coefficient at record {record + 1}",
        )

        if row < 1 or row > header.x_variable_count:
            raise ValueError(
                f"Invalid Q row {row} at record {record + 1}"
            )

        if column < 1 or column > header.y_variable_count:
            raise ValueError(
                f"Invalid Q column {column} at record {record + 1}"
            )

        if (
            row - 1 in active_x_positions
            and column - 1 in active_y_positions
        ):
            objective += coefficient

    try:
        unexpected = next(tokens)
    except StopIteration:
        unexpected = None

    if unexpected is not None:
        raise ValueError(
            "Unexpected data after the declared Q records: "
            f"{unexpected!r}"
        )

    return objective


# =============================================================================
# Heuristic output parsing
# =============================================================================

KEY_VALUE_PATTERN = re.compile(
    r"(?:^|\s)([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)"
)


def parse_key_value_header(text: str) -> Dict[str, str]:
    for line in text.splitlines():
        stripped = line.strip()

        if "=" not in stripped:
            continue

        values = {
            match.group(1): match.group(2)
            for match in KEY_VALUE_PATTERN.finditer(stripped)
        }

        if "objective" in values:
            return values

    raise ValueError(
        "No key=value solver result header was found"
    )


def parse_assignment_line(
    text: str,
    name: str,
) -> List[int]:
    match = re.search(
        rf"^{re.escape(name)}=(.*)$",
        text,
        flags=re.MULTILINE,
    )

    if match is None:
        raise ValueError(
            f"Missing {name}= line in the heuristic result"
        )

    payload = match.group(1).strip()

    if not payload:
        return []

    # The C++ solver reports one-based slot numbers.
    return [
        int(token) - 1
        for token in payload.split()
    ]


def optional_int(
    values: Dict[str, str],
    *names: str,
) -> Optional[int]:
    for name in names:
        if name in values:
            return int(values[name])

    return None


def optional_float(
    values: Dict[str, str],
    *names: str,
) -> Optional[float]:
    for name in names:
        if name in values:
            result = float(values[name])

            if not math.isfinite(result):
                raise ValueError(
                    f"Non-finite {name}: {values[name]}"
                )

            return result

    return None


# =============================================================================
# Heuristic command construction
# =============================================================================

def build_heuristic_command(
    arguments: argparse.Namespace,
    input_path: pathlib.Path,
    result_path: pathlib.Path,
) -> List[str]:
    command = [
        str(arguments.heuristic),
        str(input_path),
        str(result_path),
        "--minimize",
        "--time-limit",
        format(
            arguments.heuristic_time_limit,
            ".17g",
        ),
        "--max-iterations",
        str(
            arguments.heuristic_max_iterations
        ),
        "--n1",
        str(arguments.heuristic_n1),
        "--n2",
        str(arguments.heuristic_n2),
        "--sequence-count",
        str(
            arguments.heuristic_sequence_count
        ),
        "--tabu-tenure",
        str(
            arguments.heuristic_tabu_tenure
        ),
        "--restart-stagnation",
        str(
            arguments.heuristic_restart_stagnation
        ),
        "--rebuild-interval",
        str(
            arguments.heuristic_rebuild_interval
        ),
        "--seed",
        str(arguments.heuristic_seed),
    ]

    if arguments.heuristic_identity_first:
        command.append("--identity-first")
    else:
        command.append("--random-first")

    if arguments.heuristic_verify_every_move:
        command.append("--verify-every-move")

    # Optional unparsed arguments permit testing newer solver builds without
    # immediately changing this comparison script.
    for extra_argument in arguments.heuristic_extra_arg:
        command.append(extra_argument)

    return command


def run_heuristic(
    arguments: argparse.Namespace,
    input_path: pathlib.Path,
    output_directory: pathlib.Path,
) -> HeuristicResult:
    result_path = (
        output_directory
        / "heuristic_result.txt"
    )

    log_path = (
        output_directory
        / "heuristic_process.log"
    )

    if arguments.skip_heuristic:
        if arguments.heuristic_result is None:
            raise ValueError(
                "--skip-heuristic requires --heuristic-result"
            )

        result_path = (
            arguments.heuristic_result
        )

        if not result_path.is_file():
            raise ValueError(
                f"Heuristic result does not exist: {result_path}"
            )

        return_code: Optional[int] = None
        process_wall_seconds = 0.0
        command_text = "not executed"
    else:
        heuristic_path = (
            arguments.heuristic
            .expanduser()
            .resolve()
        )

        if not heuristic_path.is_file():
            raise ValueError(
                f"Heuristic executable does not exist: "
                f"{heuristic_path}"
            )

        if not os.access(heuristic_path, os.X_OK):
            raise ValueError(
                f"Heuristic is not executable: "
                f"{heuristic_path}"
            )

        arguments.heuristic = heuristic_path

        command = build_heuristic_command(
            arguments,
            input_path,
            result_path,
        )

        command_text = shlex.join(command)

        started = time.perf_counter()

        try:
            completed = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=arguments.heuristic_external_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            process_wall_seconds = (
                time.perf_counter() - started
            )

            stdout = (
                error.stdout
                if isinstance(error.stdout, str)
                else ""
            )

            stderr = (
                error.stderr
                if isinstance(error.stderr, str)
                else ""
            )

            log_path.write_text(
                "HEURISTIC EXTERNAL TIMEOUT\n\n"
                + stdout
                + "\n\n"
                + stderr,
                encoding="utf-8",
            )

            return HeuristicResult(
                status="timeout",
                return_code=None,
                process_wall_seconds=process_wall_seconds,
                reported_objective=None,
                validated_objective=None,
                objective_valid=False,
                time_to_best_seconds=None,
                search_seconds=None,
                elapsed_seconds=None,
                load_seconds=None,
                outer_iterations=None,
                moves=None,
                evaluated_moves=None,
                critical_events=None,
                restarts=None,
                move_to_best=None,
                outer_iteration_to_best=None,
                x_assignment=[],
                y_assignment=[],
                result_file=str(result_path),
                log_file=str(log_path),
                command=command_text,
                error_message=(
                    "The heuristic exceeded its external timeout"
                ),
            )

        process_wall_seconds = (
            time.perf_counter() - started
        )

        return_code = completed.returncode

        log_path.write_text(
            "--- command ---\n"
            + command_text
            + "\n\n--- stdout ---\n"
            + completed.stdout
            + "\n\n--- stderr ---\n"
            + completed.stderr,
            encoding="utf-8",
        )

        if return_code != 0:
            return HeuristicResult(
                status="failed",
                return_code=return_code,
                process_wall_seconds=process_wall_seconds,
                reported_objective=None,
                validated_objective=None,
                objective_valid=False,
                time_to_best_seconds=None,
                search_seconds=None,
                elapsed_seconds=None,
                load_seconds=None,
                outer_iterations=None,
                moves=None,
                evaluated_moves=None,
                critical_events=None,
                restarts=None,
                move_to_best=None,
                outer_iteration_to_best=None,
                x_assignment=[],
                y_assignment=[],
                result_file=str(result_path),
                log_file=str(log_path),
                command=command_text,
                error_message=(
                    f"Heuristic returned code {return_code}"
                ),
            )

    try:
        result_text = result_path.read_text(
            encoding="utf-8"
        )

        values = parse_key_value_header(
            result_text
        )

        x_assignment = parse_assignment_line(
            result_text,
            "x_assignment",
        )

        y_assignment = parse_assignment_line(
            result_text,
            "y_assignment",
        )

        reported_objective = int(
            values["objective"]
        )

        solution = AssignmentSolution(
            x_assignment=x_assignment,
            y_assignment=y_assignment,
            objective=reported_objective,
        )

        validated_objective = evaluate_assignment(
            input_path,
            solution,
        )

        objective_valid = (
            reported_objective
            == validated_objective
        )

        return HeuristicResult(
            status=(
                "success"
                if objective_valid
                else "validation_failed"
            ),
            return_code=return_code,
            process_wall_seconds=process_wall_seconds,
            reported_objective=reported_objective,
            validated_objective=validated_objective,
            objective_valid=objective_valid,
            time_to_best_seconds=optional_float(
                values,
                "time_to_best_seconds",
            ),
            search_seconds=optional_float(
                values,
                "search_seconds",
            ),
            elapsed_seconds=optional_float(
                values,
                "elapsed_seconds",
            ),
            load_seconds=optional_float(
                values,
                "load_seconds",
            ),
            outer_iterations=optional_int(
                values,
                "outer_iterations",
                "completed_outer_iterations",
            ),
            moves=optional_int(
                values,
                "moves",
                "move_number",
            ),
            evaluated_moves=optional_int(
                values,
                "evaluated_moves",
            ),
            critical_events=optional_int(
                values,
                "critical_events",
            ),
            restarts=optional_int(
                values,
                "restarts",
            ),
            move_to_best=optional_int(
                values,
                "move_to_best",
                "iteration_to_best",
            ),
            outer_iteration_to_best=optional_int(
                values,
                "outer_iteration_to_best",
            ),
            x_assignment=x_assignment,
            y_assignment=y_assignment,
            result_file=str(result_path),
            log_file=str(log_path),
            command=command_text,
            error_message=(
                ""
                if objective_valid
                else (
                    "Reported and independently validated "
                    "heuristic objectives differ"
                )
            ),
        )

    except Exception as error:
        return HeuristicResult(
            status="parse_failed",
            return_code=return_code,
            process_wall_seconds=process_wall_seconds,
            reported_objective=None,
            validated_objective=None,
            objective_valid=False,
            time_to_best_seconds=None,
            search_seconds=None,
            elapsed_seconds=None,
            load_seconds=None,
            outer_iterations=None,
            moves=None,
            evaluated_moves=None,
            critical_events=None,
            restarts=None,
            move_to_best=None,
            outer_iteration_to_best=None,
            x_assignment=[],
            y_assignment=[],
            result_file=str(result_path),
            log_file=str(log_path),
            command=command_text,
            error_message=str(error),
        )


# =============================================================================
# Gurobi utilities
# =============================================================================

def gurobi_status_name(status: int) -> str:
    status_names = {
        GRB.LOADED: "LOADED",
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INPROGRESS: "INPROGRESS",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
        GRB.WORK_LIMIT: "WORK_LIMIT",
        GRB.MEM_LIMIT: "MEM_LIMIT",
    }

    return status_names.get(
        status,
        f"UNKNOWN_{status}",
    )


def write_gurobi_solution(
    solution_path: pathlib.Path,
    header: InstanceHeader,
    result: GurobiResult,
) -> None:
    with solution_path.open(
        "w",
        encoding="utf-8",
    ) as stream:
        stream.write("GUROBI MCDAP SOLUTION\n")
        stream.write("====================\n")
        stream.write(f"status={result.status}\n")
        stream.write(
            f"x_items={header.x_items}\n"
        )
        stream.write(
            f"x_slots={header.x_slots}\n"
        )
        stream.write(
            f"y_items={header.y_items}\n"
        )
        stream.write(
            f"y_slots={header.y_slots}\n"
        )
        stream.write(
            f"objective={result.objective}\n"
        )
        stream.write(
            "validated_objective="
            f"{result.validated_objective}\n"
        )
        stream.write(
            f"objective_valid="
            f"{int(result.objective_valid)}\n"
        )
        stream.write(
            f"best_bound={result.best_bound}\n"
        )
        stream.write(
            f"mip_gap={result.mip_gap}\n"
        )
        stream.write(
            f"runtime_seconds="
            f"{result.runtime_seconds}\n"
        )
        stream.write(
            f"time_to_best_seconds="
            f"{result.time_to_best_seconds}\n"
        )
        stream.write(
            f"warm_start_used="
            f"{int(result.warm_start_used)}\n\n"
        )

        if result.x_assignment:
            stream.write("x_assignment=")
            stream.write(
                " ".join(
                    str(slot + 1)
                    for slot in result.x_assignment
                )
            )
            stream.write("\n")

        if result.y_assignment:
            stream.write("y_assignment=")
            stream.write(
                " ".join(
                    str(slot + 1)
                    for slot in result.y_assignment
                )
            )
            stream.write("\n")


# =============================================================================
# Gurobi model construction and execution
# =============================================================================

def solve_with_gurobi(
    arguments: argparse.Namespace,
    input_path: pathlib.Path,
    output_directory: pathlib.Path,
    heuristic_result: HeuristicResult,
) -> GurobiResult:
    log_path = (
        output_directory / "gurobi.log"
    )

    solution_path = (
        output_directory
        / "gurobi_solution.txt"
    )

    header, x_fixed, y_fixed, tokens = read_instance_prefix(
        input_path
    )

    model = gp.Model("MCDAP_xQy_MIQP")

    model.Params.TimeLimit = (
        arguments.gurobi_time_limit
    )
    model.Params.NonConvex = 2
    model.Params.MIPGap = arguments.mip_gap
    model.Params.Threads = arguments.gurobi_threads
    model.Params.Seed = arguments.gurobi_seed
    model.Params.LogFile = str(log_path)

    if arguments.mip_focus is not None:
        model.Params.MIPFocus = (
            arguments.mip_focus
        )

    if arguments.presolve is not None:
        model.Params.Presolve = (
            arguments.presolve
        )

    if arguments.nodefile_start is not None:
        model.Params.NodefileStart = (
            arguments.nodefile_start
        )

    if arguments.nodefile_dir is not None:
        model.Params.NodefileDir = str(
            arguments.nodefile_dir
        )

    x = model.addVars(
        header.x_items,
        header.x_slots,
        vtype=GRB.BINARY,
        name="x",
    )

    y = model.addVars(
        header.y_items,
        header.y_slots,
        vtype=GRB.BINARY,
        name="y",
    )

    model.addConstrs(
        (
            gp.quicksum(
                x[item, slot]
                for slot in range(
                    header.x_slots
                )
            )
            == 1
            for item in range(
                header.x_items
            )
        ),
        name="assign_x_item",
    )

    model.addConstrs(
        (
            gp.quicksum(
                x[item, slot]
                for item in range(
                    header.x_items
                )
            )
            <= 1
            for slot in range(
                header.x_slots
            )
        ),
        name="capacity_x_slot",
    )

    model.addConstrs(
        (
            gp.quicksum(
                y[item, slot]
                for slot in range(
                    header.y_slots
                )
            )
            == 1
            for item in range(
                header.y_items
            )
        ),
        name="assign_y_item",
    )

    model.addConstrs(
        (
            gp.quicksum(
                y[item, slot]
                for item in range(
                    header.y_items
                )
            )
            <= 1
            for slot in range(
                header.y_slots
            )
        ),
        name="capacity_y_slot",
    )

    linear_objective = gp.LinExpr()

    linear_objective.addTerms(
        x_fixed,
        [
            x[item, slot]
            for item in range(
                header.x_items
            )
            for slot in range(
                header.x_slots
            )
        ],
    )

    linear_objective.addTerms(
        y_fixed,
        [
            y[item, slot]
            for item in range(
                header.y_items
            )
            for slot in range(
                header.y_slots
            )
        ],
    )

    objective = gp.QuadExpr()
    objective += linear_objective

    coefficients: List[int] = []
    x_variables: List[gp.Var] = []
    y_variables: List[gp.Var] = []

    for record in range(
        header.q_record_count
    ):
        row = next_integer(
            tokens,
            f"Q row at record {record + 1}",
        )

        column = next_integer(
            tokens,
            f"Q column at record {record + 1}",
        )

        coefficient = next_integer(
            tokens,
            f"Q coefficient at record {record + 1}",
        )

        if row < 1 or row > header.x_variable_count:
            raise ValueError(
                f"Invalid Q row {row} at record {record + 1}"
            )

        if column < 1 or column > header.y_variable_count:
            raise ValueError(
                f"Invalid Q column {column} at record {record + 1}"
            )

        if coefficient == 0:
            continue

        flat_x = row - 1
        flat_y = column - 1

        x_item, x_slot = divmod(
            flat_x,
            header.x_slots,
        )

        y_item, y_slot = divmod(
            flat_y,
            header.y_slots,
        )

        coefficients.append(coefficient)
        x_variables.append(x[x_item, x_slot])
        y_variables.append(y[y_item, y_slot])

        if len(coefficients) >= arguments.q_chunk_size:
            objective.addTerms(
                coefficients,
                x_variables,
                y_variables,
            )

            coefficients.clear()
            x_variables.clear()
            y_variables.clear()

    if coefficients:
        objective.addTerms(
            coefficients,
            x_variables,
            y_variables,
        )

    try:
        unexpected = next(tokens)
    except StopIteration:
        unexpected = None

    if unexpected is not None:
        raise ValueError(
            "Unexpected data after the final Q record: "
            f"{unexpected!r}"
        )

    model.setObjective(
        objective,
        GRB.MINIMIZE,
    )

    warm_start_used = False

    if arguments.warm_start:
        if (
            heuristic_result.status != "success"
            or not heuristic_result.objective_valid
        ):
            raise ValueError(
                "--warm-start requires a successful, independently "
                "validated heuristic solution"
            )

        warm_solution = AssignmentSolution(
            x_assignment=heuristic_result.x_assignment,
            y_assignment=heuristic_result.y_assignment,
        )

        validate_assignment_structure(
            header,
            warm_solution,
        )

        for item in range(
            header.x_items
        ):
            selected_slot = (
                warm_solution.x_assignment[item]
            )

            for slot in range(
                header.x_slots
            ):
                x[item, slot].Start = (
                    1.0
                    if slot == selected_slot
                    else 0.0
                )

        for item in range(
            header.y_items
        ):
            selected_slot = (
                warm_solution.y_assignment[item]
            )

            for slot in range(
                header.y_slots
            ):
                y[item, slot].Start = (
                    1.0
                    if slot == selected_slot
                    else 0.0
                )

        warm_start_used = True

    model._best_incumbent = math.inf
    model._time_to_best = None
    model._incumbent_count = 0

    def callback(
        callback_model: gp.Model,
        where: int,
    ) -> None:
        if where != GRB.Callback.MIPSOL:
            return

        incumbent = callback_model.cbGet(
            GRB.Callback.MIPSOL_OBJ
        )

        runtime = callback_model.cbGet(
            GRB.Callback.RUNTIME
        )

        callback_model._incumbent_count += 1

        tolerance = (
            1.0e-9
            + 1.0e-12 * abs(incumbent)
        )

        if (
            incumbent
            < callback_model._best_incumbent
            - tolerance
        ):
            callback_model._best_incumbent = incumbent
            callback_model._time_to_best = runtime

    process_started = time.perf_counter()

    try:
        model.optimize(callback)
    except gp.GurobiError as error:
        process_wall_seconds = (
            time.perf_counter() - process_started
        )

        return GurobiResult(
            status="GUROBI_ERROR",
            status_code=-1,
            solution_count=0,
            objective=None,
            validated_objective=None,
            objective_valid=False,
            best_bound=None,
            mip_gap=None,
            runtime_seconds=process_wall_seconds,
            process_wall_seconds=process_wall_seconds,
            time_to_best_seconds=None,
            nodes=0.0,
            simplex_iterations=0.0,
            barrier_iterations=0,
            work=None,
            warm_start_used=warm_start_used,
            x_assignment=[],
            y_assignment=[],
            log_file=str(log_path),
            solution_file=str(solution_path),
            error_message=(
                f"Gurobi error {error.errno}: {error}"
            ),
        )

    process_wall_seconds = (
        time.perf_counter() - process_started
    )

    status_code = model.Status
    status_name = gurobi_status_name(
        status_code
    )

    solution_count = model.SolCount

    objective_value: Optional[float] = None
    validated_objective: Optional[int] = None
    objective_valid = False
    best_bound: Optional[float] = None
    mip_gap: Optional[float] = None

    x_assignment: List[int] = []
    y_assignment: List[int] = []

    try:
        best_bound = model.ObjBound
    except gp.GurobiError:
        best_bound = None

    if solution_count > 0:
        objective_value = model.ObjVal
        mip_gap = model.MIPGap

        for item in range(
            header.x_items
        ):
            selected_slot = max(
                range(header.x_slots),
                key=lambda slot: x[item, slot].X,
            )

            if x[item, selected_slot].X < 0.5:
                raise RuntimeError(
                    f"No selected x slot for item {item}"
                )

            x_assignment.append(selected_slot)

        for item in range(
            header.y_items
        ):
            selected_slot = max(
                range(header.y_slots),
                key=lambda slot: y[item, slot].X,
            )

            if y[item, selected_slot].X < 0.5:
                raise RuntimeError(
                    f"No selected y slot for item {item}"
                )

            y_assignment.append(selected_slot)

        validated_objective = evaluate_assignment(
            input_path,
            AssignmentSolution(
                x_assignment=x_assignment,
                y_assignment=y_assignment,
            ),
        )

        tolerance = (
            1.0e-6
            + 1.0e-9
            * abs(validated_objective)
        )

        objective_valid = (
            abs(
                objective_value
                - validated_objective
            )
            <= tolerance
        )

    try:
        work = model.Work
    except (AttributeError, gp.GurobiError):
        work = None

    result = GurobiResult(
        status=status_name,
        status_code=status_code,
        solution_count=solution_count,
        objective=objective_value,
        validated_objective=validated_objective,
        objective_valid=objective_valid,
        best_bound=best_bound,
        mip_gap=mip_gap,
        runtime_seconds=model.Runtime,
        process_wall_seconds=process_wall_seconds,
        time_to_best_seconds=model._time_to_best,
        nodes=model.NodeCount,
        simplex_iterations=model.IterCount,
        barrier_iterations=model.BarIterCount,
        work=work,
        warm_start_used=warm_start_used,
        x_assignment=x_assignment,
        y_assignment=y_assignment,
        log_file=str(log_path),
        solution_file=str(solution_path),
        error_message="",
    )

    write_gurobi_solution(
        solution_path,
        header,
        result,
    )

    return result


# =============================================================================
# Comparison output
# =============================================================================

def heuristic_parameter_dictionary(
    arguments: argparse.Namespace,
) -> Dict[str, object]:
    return {
        "time_limit_seconds": (
            arguments.heuristic_time_limit
        ),
        "max_iterations": (
            arguments.heuristic_max_iterations
        ),
        "n1": arguments.heuristic_n1,
        "n2": arguments.heuristic_n2,
        "sequence_count": (
            arguments.heuristic_sequence_count
        ),
        "tabu_tenure": (
            arguments.heuristic_tabu_tenure
        ),
        "restart_stagnation": (
            arguments.heuristic_restart_stagnation
        ),
        "rebuild_interval": (
            arguments.heuristic_rebuild_interval
        ),
        "seed": arguments.heuristic_seed,
        "initialization": (
            "identity"
            if arguments.heuristic_identity_first
            else "random"
        ),
        "verify_every_move": (
            arguments.heuristic_verify_every_move
        ),
        "extra_arguments": (
            arguments.heuristic_extra_arg
        ),
    }


def write_comparison_outputs(
    arguments: argparse.Namespace,
    input_path: pathlib.Path,
    output_directory: pathlib.Path,
    heuristic: HeuristicResult,
    gurobi: GurobiResult,
) -> None:
    comparison_json = (
        output_directory
        / "comparison.json"
    )

    comparison_csv = (
        output_directory
        / "comparison.csv"
    )

    heuristic_objective = (
        heuristic.validated_objective
    )

    gurobi_objective = (
        gurobi.validated_objective
    )

    absolute_difference: Optional[float] = None
    relative_difference: Optional[float] = None
    better_method = "unavailable"

    if (
        heuristic_objective is not None
        and gurobi_objective is not None
    ):
        # Positive means Gurobi obtained a lower minimization objective.
        absolute_difference = (
            heuristic_objective
            - gurobi_objective
        )

        relative_difference = (
            absolute_difference
            / max(
                1.0,
                abs(float(gurobi_objective)),
            )
        )

        if heuristic_objective < gurobi_objective:
            better_method = "heuristic"
        elif gurobi_objective < heuristic_objective:
            better_method = "gurobi"
        else:
            better_method = "tie"

    payload = {
        "timestamp_utc": (
            dt.datetime.now(dt.timezone.utc)
            .isoformat(timespec="seconds")
        ),
        "input_file": str(input_path),
        "heuristic_parameters": (
            heuristic_parameter_dictionary(
                arguments
            )
        ),
        "gurobi_parameters": {
            "time_limit_seconds": (
                arguments.gurobi_time_limit
            ),
            "threads": arguments.gurobi_threads,
            "seed": arguments.gurobi_seed,
            "mip_gap": arguments.mip_gap,
            "mip_focus": arguments.mip_focus,
            "presolve": arguments.presolve,
            "nonconvex": 2,
            "warm_start": arguments.warm_start,
            "q_chunk_size": arguments.q_chunk_size,
        },
        "heuristic": asdict(heuristic),
        "gurobi": asdict(gurobi),
        "comparison": {
            "heuristic_objective": (
                heuristic_objective
            ),
            "gurobi_objective": (
                gurobi_objective
            ),
            "heuristic_minus_gurobi": (
                absolute_difference
            ),
            "relative_difference": (
                relative_difference
            ),
            "better_method": better_method,
        },
    }

    comparison_json.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    fields = [
        "method",
        "status",
        "objective",
        "validated_objective",
        "objective_valid",
        "runtime_seconds",
        "time_to_best_seconds",
        "best_bound",
        "mip_gap",
        "warm_start",
        "solution_count",
        "nodes",
        "heuristic_time_limit",
        "heuristic_max_iterations",
        "heuristic_n1",
        "heuristic_n2",
        "heuristic_sequence_count",
        "heuristic_tabu_tenure",
        "heuristic_restart_stagnation",
        "heuristic_rebuild_interval",
        "heuristic_seed",
        "heuristic_minus_gurobi",
        "relative_difference",
        "better_method",
    ]

    common_parameters = {
        "heuristic_time_limit": (
            arguments.heuristic_time_limit
        ),
        "heuristic_max_iterations": (
            arguments.heuristic_max_iterations
        ),
        "heuristic_n1": (
            arguments.heuristic_n1
        ),
        "heuristic_n2": (
            arguments.heuristic_n2
        ),
        "heuristic_sequence_count": (
            arguments.heuristic_sequence_count
        ),
        "heuristic_tabu_tenure": (
            arguments.heuristic_tabu_tenure
        ),
        "heuristic_restart_stagnation": (
            arguments.heuristic_restart_stagnation
        ),
        "heuristic_rebuild_interval": (
            arguments.heuristic_rebuild_interval
        ),
        "heuristic_seed": (
            arguments.heuristic_seed
        ),
        "heuristic_minus_gurobi": (
            absolute_difference
        ),
        "relative_difference": (
            relative_difference
        ),
        "better_method": better_method,
    }

    with comparison_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerow({
            "method": "d_flip_xqy_dense_cd",
            "status": heuristic.status,
            "objective": (
                heuristic.reported_objective
            ),
            "validated_objective": (
                heuristic.validated_objective
            ),
            "objective_valid": int(
                heuristic.objective_valid
            ),
            "runtime_seconds": (
                heuristic.search_seconds
                if heuristic.search_seconds is not None
                else heuristic.process_wall_seconds
            ),
            "time_to_best_seconds": (
                heuristic.time_to_best_seconds
            ),
            "best_bound": "",
            "mip_gap": "",
            "warm_start": "",
            "solution_count": "",
            "nodes": "",
            **common_parameters,
        })

        writer.writerow({
            "method": "gurobi_miqp",
            "status": gurobi.status,
            "objective": gurobi.objective,
            "validated_objective": (
                gurobi.validated_objective
            ),
            "objective_valid": int(
                gurobi.objective_valid
            ),
            "runtime_seconds": (
                gurobi.runtime_seconds
            ),
            "time_to_best_seconds": (
                gurobi.time_to_best_seconds
            ),
            "best_bound": gurobi.best_bound,
            "mip_gap": gurobi.mip_gap,
            "warm_start": int(
                gurobi.warm_start_used
            ),
            "solution_count": (
                gurobi.solution_count
            ),
            "nodes": gurobi.nodes,
            **common_parameters,
        })

    print()
    print("Comparison summary")
    print("==================")
    print(
        "Heuristic parameters: "
        f"time={arguments.heuristic_time_limit}s, "
        f"iterations={arguments.heuristic_max_iterations}, "
        f"N1={arguments.heuristic_n1}, "
        f"N2={arguments.heuristic_n2}, "
        f"S={arguments.heuristic_sequence_count}, "
        f"tenure={arguments.heuristic_tabu_tenure}, "
        f"restart={arguments.heuristic_restart_stagnation}, "
        f"seed={arguments.heuristic_seed}"
    )
    print(
        f"Heuristic status:    {heuristic.status}"
    )
    print(
        f"Heuristic objective: "
        f"{heuristic.validated_objective}"
    )
    print(
        f"Heuristic TTB:       "
        f"{heuristic.time_to_best_seconds}"
    )
    print(
        f"Gurobi status:       {gurobi.status}"
    )
    print(
        f"Gurobi objective:    "
        f"{gurobi.validated_objective}"
    )
    print(
        f"Gurobi best bound:   {gurobi.best_bound}"
    )
    print(
        f"Gurobi MIP gap:      {gurobi.mip_gap}"
    )
    print(
        f"Gurobi TTB:          "
        f"{gurobi.time_to_best_seconds}"
    )
    print(
        f"Heuristic - Gurobi:  "
        f"{absolute_difference}"
    )
    print(
        f"Better result:       {better_method}"
    )
    print()
    print(f"CSV:  {comparison_csv}")
    print(f"JSON: {comparison_json}")
    print(f"Gurobi log: {gurobi.log_file}")


# =============================================================================
# Command-line options
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare configurable d_flip_xqy_dense_cd settings "
            "against Gurobi MIQP."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=pathlib.Path(
            "n100m10-1-cd-xqy.txt"
        ),
        help="Prepared CD-xQy input file",
    )

    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path(
            "comparison-n100m10-1"
        ),
        help="Output directory",
    )

    # -------------------------------------------------------------------------
    # Heuristic executable and result handling
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--heuristic",
        type=pathlib.Path,
        default=pathlib.Path(
            "./d_flip_xqy_dense_cd"
        ),
        help="Path to d_flip_xqy_dense_cd",
    )

    parser.add_argument(
        "--heuristic-result",
        type=pathlib.Path,
        help=(
            "Existing heuristic result file, used with "
            "--skip-heuristic"
        ),
    )

    parser.add_argument(
        "--skip-heuristic",
        action="store_true",
        help="Do not execute the heuristic",
    )

    parser.add_argument(
        "--heuristic-external-timeout",
        type=float,
        default=None,
        help=(
            "External process timeout. When omitted, it is set to "
            "heuristic-time-limit plus 300 seconds."
        ),
    )

    # -------------------------------------------------------------------------
    # Configurable heuristic parameters
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--heuristic-time-limit",
        type=float,
        default=300.0,
        help="Heuristic internal time limit in seconds",
    )

    parser.add_argument(
        "--heuristic-max-iterations",
        type=int,
        default=100_000,
        help="Heuristic Algorithm 3 outer-iteration limit",
    )

    parser.add_argument(
        "--heuristic-n1",
        type=int,
        default=1,
        help="Heuristic N1 parameter",
    )

    parser.add_argument(
        "--heuristic-n2",
        type=int,
        default=10,
        help="Heuristic N2 parameter",
    )

    parser.add_argument(
        "--heuristic-sequence-count",
        type=int,
        default=150,
        help="Number of random-key sequences",
    )

    parser.add_argument(
        "--heuristic-tabu-tenure",
        type=int,
        default=5,
        help="Position-level tabu tenure",
    )

    parser.add_argument(
        "--heuristic-restart-stagnation",
        type=int,
        default=5,
        help=(
            "Restart after this many stagnant outer iterations; "
            "zero disables restarts"
        ),
    )

    parser.add_argument(
        "--heuristic-rebuild-interval",
        type=int,
        default=100_000,
        help=(
            "Exact derivative rebuild interval in atomic moves; "
            "zero disables periodic rebuilds"
        ),
    )

    parser.add_argument(
        "--heuristic-seed",
        type=int,
        default=1,
        help="Heuristic random seed",
    )

    initialization_group = (
        parser.add_mutually_exclusive_group()
    )

    initialization_group.add_argument(
        "--heuristic-random-first",
        dest="heuristic_identity_first",
        action="store_false",
        help="Use a random initial feasible assignment",
    )

    initialization_group.add_argument(
        "--heuristic-identity-first",
        dest="heuristic_identity_first",
        action="store_true",
        help="Use the identity assignment initially",
    )

    parser.set_defaults(
        heuristic_identity_first=False
    )

    parser.add_argument(
        "--heuristic-verify-every-move",
        action="store_true",
        help=(
            "Pass --verify-every-move to the heuristic. "
            "This is computationally expensive."
        ),
    )

    parser.add_argument(
        "--heuristic-extra-arg",
        action="append",
        default=[],
        metavar="ARG",
        help=(
            "Append an additional raw heuristic argument. "
            "Repeat this option for multiple arguments."
        ),
    )

    # -------------------------------------------------------------------------
    # Gurobi parameters
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--gurobi-time-limit",
        type=float,
        default=7200.0,
        help="Gurobi optimization time limit in seconds",
    )

    parser.add_argument(
        "--gurobi-threads",
        type=int,
        default=0,
        help=(
            "Gurobi thread count; zero lets Gurobi choose"
        ),
    )

    parser.add_argument(
        "--gurobi-seed",
        type=int,
        default=1,
        help="Gurobi random seed",
    )

    parser.add_argument(
        "--mip-gap",
        type=float,
        default=0.0,
        help=(
            "Target relative MIP gap. Zero requests proof of "
            "optimality unless another stopping condition is reached."
        ),
    )

    parser.add_argument(
        "--warm-start",
        action="store_true",
        help=(
            "Use the independently validated heuristic assignment "
            "as a Gurobi MIP start"
        ),
    )

    parser.add_argument(
        "--mip-focus",
        type=int,
        choices=(0, 1, 2, 3),
        default=None,
        help="Optional Gurobi MIPFocus value",
    )

    parser.add_argument(
        "--presolve",
        type=int,
        choices=(-1, 0, 1, 2),
        default=None,
        help="Optional Gurobi Presolve value",
    )

    parser.add_argument(
        "--nodefile-start",
        type=float,
        default=None,
        help="Optional Gurobi NodefileStart value in GB",
    )

    parser.add_argument(
        "--nodefile-dir",
        type=pathlib.Path,
        default=None,
        help="Optional Gurobi node-file directory",
    )

    parser.add_argument(
        "--q-chunk-size",
        type=int,
        default=100_000,
        help=(
            "Quadratic terms added to Gurobi per Python chunk"
        ),
    )

    return parser.parse_args()


def validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    arguments.input = (
        arguments.input.expanduser().resolve()
    )

    arguments.output_dir = (
        arguments.output_dir
        .expanduser()
        .resolve()
    )

    arguments.heuristic = (
        arguments.heuristic
        .expanduser()
        .resolve()
    )

    if arguments.heuristic_result is not None:
        arguments.heuristic_result = (
            arguments.heuristic_result
            .expanduser()
            .resolve()
        )

    if arguments.nodefile_dir is not None:
        arguments.nodefile_dir = (
            arguments.nodefile_dir
            .expanduser()
            .resolve()
        )

    if not arguments.input.is_file():
        raise SystemExit(
            f"Input file does not exist: {arguments.input}"
        )

    if arguments.heuristic_time_limit <= 0.0:
        raise SystemExit(
            "--heuristic-time-limit must be positive"
        )

    if arguments.heuristic_max_iterations <= 0:
        raise SystemExit(
            "--heuristic-max-iterations must be positive"
        )

    if arguments.heuristic_n1 <= 0:
        raise SystemExit(
            "--heuristic-n1 must be positive"
        )

    if arguments.heuristic_n2 < arguments.heuristic_n1:
        raise SystemExit(
            "--heuristic-n2 must be at least --heuristic-n1"
        )

    if arguments.heuristic_sequence_count <= 0:
        raise SystemExit(
            "--heuristic-sequence-count must be positive"
        )

    if arguments.heuristic_tabu_tenure < 0:
        raise SystemExit(
            "--heuristic-tabu-tenure cannot be negative"
        )

    if arguments.heuristic_restart_stagnation < 0:
        raise SystemExit(
            "--heuristic-restart-stagnation cannot be negative"
        )

    if arguments.heuristic_rebuild_interval < 0:
        raise SystemExit(
            "--heuristic-rebuild-interval cannot be negative"
        )

    if arguments.heuristic_seed < 0:
        raise SystemExit(
            "--heuristic-seed cannot be negative"
        )

    if arguments.heuristic_external_timeout is None:
        arguments.heuristic_external_timeout = (
            arguments.heuristic_time_limit
            + 300.0
        )

    if arguments.heuristic_external_timeout <= 0.0:
        raise SystemExit(
            "--heuristic-external-timeout must be positive"
        )

    if arguments.gurobi_time_limit <= 0.0:
        raise SystemExit(
            "--gurobi-time-limit must be positive"
        )

    if arguments.gurobi_threads < 0:
        raise SystemExit(
            "--gurobi-threads cannot be negative"
        )

    if arguments.gurobi_seed < 0:
        raise SystemExit(
            "--gurobi-seed cannot be negative"
        )

    if arguments.mip_gap < 0.0:
        raise SystemExit(
            "--mip-gap cannot be negative"
        )

    if arguments.q_chunk_size <= 0:
        raise SystemExit(
            "--q-chunk-size must be positive"
        )

    if (
        arguments.skip_heuristic
        and arguments.heuristic_result is None
    ):
        raise SystemExit(
            "--skip-heuristic requires --heuristic-result"
        )


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    arguments = parse_arguments()
    validate_arguments(arguments)

    arguments.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("MCDAP xQy heuristic/Gurobi comparison")
    print("====================================")
    print(f"Input: {arguments.input}")
    print()
    print("Heuristic parameters")
    print("--------------------")
    print(
        f"time limit:          "
        f"{arguments.heuristic_time_limit} seconds"
    )
    print(
        f"max iterations:      "
        f"{arguments.heuristic_max_iterations}"
    )
    print(
        f"N1:                  "
        f"{arguments.heuristic_n1}"
    )
    print(
        f"N2:                  "
        f"{arguments.heuristic_n2}"
    )
    print(
        f"sequence count:      "
        f"{arguments.heuristic_sequence_count}"
    )
    print(
        f"tabu tenure:         "
        f"{arguments.heuristic_tabu_tenure}"
    )
    print(
        f"restart stagnation:  "
        f"{arguments.heuristic_restart_stagnation}"
    )
    print(
        f"rebuild interval:    "
        f"{arguments.heuristic_rebuild_interval}"
    )
    print(
        f"seed:                "
        f"{arguments.heuristic_seed}"
    )
    print(
        f"initialization:      "
        + (
            "identity"
            if arguments.heuristic_identity_first
            else "random"
        )
    )
    print()
    print(
        f"Gurobi time limit:   "
        f"{arguments.gurobi_time_limit} seconds"
    )
    print(
        f"Gurobi warm start:   "
        f"{arguments.warm_start}"
    )

    print()
    print("Running or loading heuristic result...")

    heuristic_result = run_heuristic(
        arguments,
        arguments.input,
        arguments.output_dir,
    )

    print(
        f"Heuristic status:    "
        f"{heuristic_result.status}"
    )
    print(
        f"Heuristic objective: "
        f"{heuristic_result.validated_objective}"
    )

    if (
        arguments.warm_start
        and (
            heuristic_result.status != "success"
            or not heuristic_result.objective_valid
        )
    ):
        raise SystemExit(
            "Error: the heuristic result could not be validated, "
            "so it cannot be supplied as a Gurobi MIP start.\n"
            f"Details: {heuristic_result.error_message}"
        )

    print()
    print("Building and running Gurobi MIQP...")

    gurobi_result = solve_with_gurobi(
        arguments,
        arguments.input,
        arguments.output_dir,
        heuristic_result,
    )

    write_comparison_outputs(
        arguments,
        arguments.input,
        arguments.output_dir,
        heuristic_result,
        gurobi_result,
    )

    if heuristic_result.status != "success":
        return 2

    if (
        gurobi_result.solution_count > 0
        and not gurobi_result.objective_valid
    ):
        return 2

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except gp.GurobiError as error:
        raise SystemExit(
            f"Gurobi error {error.errno}: {error}"
        )
    except (
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        raise SystemExit(
            f"Error: {error}"
        )