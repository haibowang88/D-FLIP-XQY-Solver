#!/usr/bin/env python3
"""
compare_cd_gurobi.py

Compare:

    d_flip_xqy_dense_cd

against Gurobi's nonconvex MIQP solver on a prepared CD-xQy instance.

Default test instance:
    n100m10-1-cd-xqy.txt

Heuristic settings:
    --time-limit 300
    --max-iterations 100000
    --n1 1
    --n2 10
    --sequence-count 150
    --tabu-tenure 5
    --restart-stagnation 5

Gurobi settings:
    TimeLimit = 7200 seconds
    NonConvex = 2
    MIPGap = configurable
    Threads = configurable

Mathematical model
------------------

    minimize

        sum_{i,w} FCX[i,w] x[i,w]
      + sum_{j,d} FCY[j,d] y[j,d]
      + sum_{i,w,j,d} Q[(i,w),(j,d)] x[i,w] y[j,d]

    subject to

        sum_w x[i,w] = 1                 for every x item i
        sum_i x[i,w] <= 1                for every x slot w

        sum_d y[j,d] = 1                 for every y item j
        sum_j y[j,d] <= 1                for every y slot d

        x[i,w], y[j,d] are binary.

Input format
------------

    number_of_instances
    x_items x_slots y_items y_slots q_record_count
    <x_items*x_slots x fixed costs>
    <y_items*y_slots y fixed costs>
    q_row q_column coefficient
    ...

The flattened Q record indices are one-based.

Requirements
------------

Install Gurobi and its Python package, and ensure that a valid license
is available:

    python -m pip install gurobipy

Example
-------

Cold-start Gurobi comparison:

    python compare_cd_gurobi.py \
        --input n100m10-1-cd-xqy.txt \
        --heuristic ./d_flip_xqy_dense_cd \
        --output-dir comparison-n100m10-1

Use the heuristic assignment as a Gurobi MIP start:

    python compare_cd_gurobi.py \
        --input n100m10-1-cd-xqy.txt \
        --heuristic ./d_flip_xqy_dense_cd \
        --output-dir comparison-n100m10-1-warm \
        --warm-start

Skip the heuristic when an existing heuristic result is supplied:

    python compare_cd_gurobi.py \
        --input n100m10-1-cd-xqy.txt \
        --heuristic-result existing-result.txt \
        --output-dir comparison-existing \
        --skip-heuristic \
        --warm-start
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
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterator, List, Optional, Sequence, TextIO, Tuple

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as error:
    raise SystemExit(
        "Error: gurobipy is not installed or cannot be imported.\n"
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
    moves: Optional[int]
    evaluated_moves: Optional[int]
    result_file: str
    log_file: str
    command: str
    x_assignment: List[int]
    y_assignment: List[int]
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
# Tokenized input
# =============================================================================

def numeric_tokens(path: pathlib.Path) -> Iterator[str]:
    """
    Stream whitespace-separated tokens.

    The prepared solver format contains integer coefficients and indices.
    Blank lines are ignored.
    """

    with path.open(
        "r",
        encoding="utf-8",
        errors="strict",
        buffering=8 * 1024 * 1024,
    ) as stream:
        for line_number, raw_line in enumerate(stream, start=1):
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

    Values such as '10.0' are accepted only if they are mathematically
    integral.
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


def read_header_and_fixed_costs(
    path: pathlib.Path,
) -> Tuple[
    InstanceHeader,
    List[int],
    List[int],
    Iterator[str],
]:
    """
    Open and begin parsing an instance.

    Note:
        The returned iterator keeps the generator's underlying file open until
        it is exhausted. This function is used only while constructing the
        Gurobi model.
    """

    tokens = numeric_tokens(path)

    instance_count = next_integer(
        tokens,
        "instance count",
    )

    if instance_count != 1:
        raise ValueError(
            "This comparison script currently requires exactly one "
            f"instance, but the file declares {instance_count}"
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
            "The item counts must be positive"
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


def read_header(path: pathlib.Path) -> InstanceHeader:
    tokens = numeric_tokens(path)

    instance_count = next_integer(
        tokens,
        "instance count",
    )

    x_items = next_integer(tokens, "x item count")
    x_slots = next_integer(tokens, "x slot count")
    y_items = next_integer(tokens, "y item count")
    y_slots = next_integer(tokens, "y slot count")
    q_record_count = next_integer(
        tokens,
        "Q record count",
    )

    return InstanceHeader(
        instance_count=instance_count,
        x_items=x_items,
        x_slots=x_slots,
        y_items=y_items,
        y_slots=y_slots,
        q_record_count=q_record_count,
    )


# =============================================================================
# Independent objective validation
# =============================================================================

def evaluate_assignment(
    path: pathlib.Path,
    solution: AssignmentSolution,
) -> int:
    """
    Independently evaluate a feasible assignment from the prepared input.

    Duplicate Q records are naturally summed because every record is added.
    """

    tokens = numeric_tokens(path)

    instance_count = next_integer(
        tokens,
        "instance count",
    )

    if instance_count != 1:
        raise ValueError(
            "Objective validation requires one instance"
        )

    x_items = next_integer(tokens, "x item count")
    x_slots = next_integer(tokens, "x slot count")
    y_items = next_integer(tokens, "y item count")
    y_slots = next_integer(tokens, "y slot count")
    q_record_count = next_integer(
        tokens,
        "Q record count",
    )

    if len(solution.x_assignment) != x_items:
        raise ValueError(
            f"Expected {x_items} x assignments, found "
            f"{len(solution.x_assignment)}"
        )

    if len(solution.y_assignment) != y_items:
        raise ValueError(
            f"Expected {y_items} y assignments, found "
            f"{len(solution.y_assignment)}"
        )

    if len(set(solution.x_assignment)) != x_items:
        raise ValueError(
            "The x assignment is not injective"
        )

    if len(set(solution.y_assignment)) != y_items:
        raise ValueError(
            "The y assignment is not injective"
        )

    if any(
        slot < 0 or slot >= x_slots
        for slot in solution.x_assignment
    ):
        raise ValueError(
            "An x slot is outside its valid range"
        )

    if any(
        slot < 0 or slot >= y_slots
        for slot in solution.y_assignment
    ):
        raise ValueError(
            "A y slot is outside its valid range"
        )

    x_variable_count = x_items * x_slots
    y_variable_count = y_items * y_slots

    x_fixed = [
        next_integer(
            tokens,
            f"x fixed cost {index + 1}",
        )
        for index in range(x_variable_count)
    ]

    y_fixed = [
        next_integer(
            tokens,
            f"y fixed cost {index + 1}",
        )
        for index in range(y_variable_count)
    ]

    objective = 0

    for item, slot in enumerate(
        solution.x_assignment
    ):
        objective += x_fixed[
            item * x_slots + slot
        ]

    for item, slot in enumerate(
        solution.y_assignment
    ):
        objective += y_fixed[
            item * y_slots + slot
        ]

    active_x_positions = {
        item * x_slots + slot
        for item, slot in enumerate(
            solution.x_assignment
        )
    }

    active_y_positions = {
        item * y_slots + slot
        for item, slot in enumerate(
            solution.y_assignment
        )
    }

    for record in range(q_record_count):
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

        if row < 1 or row > x_variable_count:
            raise ValueError(
                f"Invalid Q row {row} at record {record + 1}"
            )

        if column < 1 or column > y_variable_count:
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
            f"Unexpected token after the final Q record: "
            f"{unexpected!r}"
        )

    return objective


# =============================================================================
# Heuristic execution and parsing
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
            for match in KEY_VALUE_PATTERN.finditer(
                stripped
            )
        }

        if "objective" in values:
            return values

    raise ValueError(
        "No solver result header containing objective= was found"
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

    # The C++ solver writes one-based slot numbers. Convert to zero-based.
    values = [
        int(token) - 1
        for token in payload.split()
    ]

    return values


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
        source_path = (
            arguments.heuristic_result
            if arguments.heuristic_result is not None
            else result_path
        )

        if not source_path.is_file():
            raise ValueError(
                "--skip-heuristic requires an existing "
                "--heuristic-result file"
            )

        result_path = source_path
        process_wall_seconds = 0.0
        return_code: Optional[int] = None
        command_text = "not executed"
    else:
        heuristic_path = (
            arguments.heuristic.expanduser().resolve()
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

        command = [
            str(heuristic_path),
            str(input_path),
            str(result_path),
            "--minimize",
            "--time-limit",
            "300",
            "--max-iterations",
            "100000",
            "--n1",
            "1",
            "--n2",
            "10",
            "--sequence-count",
            "150",
            "--tabu-tenure",
            "5",
            "--restart-stagnation",
            "5",
            "--seed",
            str(arguments.seed),
        ]

        if arguments.heuristic_verify:
            command.append("--verify-every-move")

        command_text = " ".join(command)

        started = time.perf_counter()

        try:
            completed = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=arguments.heuristic_timeout,
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
                "HEURISTIC TIMEOUT\n\n"
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
                moves=None,
                evaluated_moves=None,
                result_file=str(result_path),
                log_file=str(log_path),
                command=command_text,
                x_assignment=[],
                y_assignment=[],
                error_message=(
                    "The heuristic exceeded its external timeout"
                ),
            )

        process_wall_seconds = (
            time.perf_counter() - started
        )

        return_code = completed.returncode

        log_path.write_text(
            "--- stdout ---\n"
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
                moves=None,
                evaluated_moves=None,
                result_file=str(result_path),
                log_file=str(log_path),
                command=command_text,
                x_assignment=[],
                y_assignment=[],
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

        time_to_best = (
            float(values["time_to_best_seconds"])
            if "time_to_best_seconds" in values
            else None
        )

        search_seconds = (
            float(values["search_seconds"])
            if "search_seconds" in values
            else None
        )

        moves = (
            int(values["moves"])
            if "moves" in values
            else None
        )

        evaluated_moves = (
            int(values["evaluated_moves"])
            if "evaluated_moves" in values
            else None
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
            time_to_best_seconds=time_to_best,
            search_seconds=search_seconds,
            moves=moves,
            evaluated_moves=evaluated_moves,
            result_file=str(result_path),
            log_file=str(log_path),
            command=command_text,
            x_assignment=x_assignment,
            y_assignment=y_assignment,
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
            moves=None,
            evaluated_moves=None,
            result_file=str(result_path),
            log_file=str(log_path),
            command=command_text,
            x_assignment=[],
            y_assignment=[],
            error_message=str(error),
        )


# =============================================================================
# Gurobi status
# =============================================================================

def gurobi_status_name(status: int) -> str:
    names = {
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

    return names.get(
        status,
        f"UNKNOWN_{status}",
    )


# =============================================================================
# Gurobi model
# =============================================================================

def build_and_solve_gurobi(
    arguments: argparse.Namespace,
    input_path: pathlib.Path,
    output_directory: pathlib.Path,
    heuristic_result: HeuristicResult,
) -> GurobiResult:
    log_path = (
        output_directory
        / "gurobi.log"
    )

    solution_path = (
        output_directory
        / "gurobi_solution.txt"
    )

    header, x_fixed, y_fixed, tokens = (
        read_header_and_fixed_costs(input_path)
    )

    model = gp.Model("MCDAP_xQy_MIQP")

    # -------------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------------

    model.Params.TimeLimit = arguments.gurobi_time_limit
    model.Params.NonConvex = 2
    model.Params.MIPGap = arguments.mip_gap
    model.Params.Threads = arguments.gurobi_threads
    model.Params.Seed = arguments.gurobi_seed
    model.Params.LogFile = str(log_path)

    if arguments.nodefile_start is not None:
        model.Params.NodefileStart = (
            arguments.nodefile_start
        )

    if arguments.nodefile_dir is not None:
        model.Params.NodefileDir = str(
            arguments.nodefile_dir
        )

    if arguments.presolve is not None:
        model.Params.Presolve = arguments.presolve

    if arguments.mip_focus is not None:
        model.Params.MIPFocus = arguments.mip_focus

    # -------------------------------------------------------------------------
    # Variables
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Assignment constraints
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Linear objective
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Quadratic objective
    #
    # Add the terms in chunks. This avoids retaining all one million
    # coefficient/variable triples in Python lists simultaneously.
    # -------------------------------------------------------------------------

    coefficient_chunk: List[int] = []
    x_variable_chunk: List[gp.Var] = []
    y_variable_chunk: List[gp.Var] = []

    x_variable_count = (
        header.x_variable_count
    )

    y_variable_count = (
        header.y_variable_count
    )

    chunk_size = arguments.q_chunk_size

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

        if row < 1 or row > x_variable_count:
            raise ValueError(
                f"Invalid Q row {row} at record "
                f"{record + 1}"
            )

        if column < 1 or column > y_variable_count:
            raise ValueError(
                f"Invalid Q column {column} at record "
                f"{record + 1}"
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

        coefficient_chunk.append(
            coefficient
        )

        x_variable_chunk.append(
            x[x_item, x_slot]
        )

        y_variable_chunk.append(
            y[y_item, y_slot]
        )

        if len(coefficient_chunk) >= chunk_size:
            objective.addTerms(
                coefficient_chunk,
                x_variable_chunk,
                y_variable_chunk,
            )

            coefficient_chunk.clear()
            x_variable_chunk.clear()
            y_variable_chunk.clear()

    if coefficient_chunk:
        objective.addTerms(
            coefficient_chunk,
            x_variable_chunk,
            y_variable_chunk,
        )

    try:
        unexpected = next(tokens)
    except StopIteration:
        unexpected = None

    if unexpected is not None:
        raise ValueError(
            f"Unexpected token after the final Q record: "
            f"{unexpected!r}"
        )

    model.setObjective(
        objective,
        GRB.MINIMIZE,
    )

    # -------------------------------------------------------------------------
    # Optional heuristic MIP start
    # -------------------------------------------------------------------------

    warm_start_used = False

    if arguments.warm_start:
        if (
            heuristic_result.status != "success"
            or not heuristic_result.objective_valid
        ):
            raise ValueError(
                "--warm-start requires a successful, independently "
                "validated heuristic result"
            )

        if (
            len(heuristic_result.x_assignment)
            != header.x_items
        ):
            raise ValueError(
                "The heuristic x assignment has the wrong length"
            )

        if (
            len(heuristic_result.y_assignment)
            != header.y_items
        ):
            raise ValueError(
                "The heuristic y assignment has the wrong length"
            )

        for item in range(
            header.x_items
        ):
            selected_slot = (
                heuristic_result.x_assignment[item]
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
                heuristic_result.y_assignment[item]
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

    # -------------------------------------------------------------------------
    # Time-to-Best callback
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Optimize
    # -------------------------------------------------------------------------

    optimize_started = time.perf_counter()

    try:
        model.optimize(callback)
    except gp.GurobiError as error:
        runtime = (
            time.perf_counter()
            - optimize_started
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
            runtime_seconds=runtime,
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

    status_code = model.Status
    status_name = gurobi_status_name(
        status_code
    )

    solution_count = model.SolCount

    objective_value: Optional[float] = None
    best_bound: Optional[float] = None
    mip_gap: Optional[float] = None
    validated_objective: Optional[int] = None
    objective_valid = False

    x_assignment: List[int] = []
    y_assignment: List[int] = []

    if solution_count > 0:
        objective_value = model.ObjVal
        best_bound = model.ObjBound
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

            x_assignment.append(
                selected_slot
            )

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

            y_assignment.append(
                selected_slot
            )

        assignment_solution = AssignmentSolution(
            x_assignment=x_assignment,
            y_assignment=y_assignment,
        )

        validated_objective = evaluate_assignment(
            input_path,
            assignment_solution,
        )

        objective_tolerance = (
            1.0e-6
            + 1.0e-9
            * abs(validated_objective)
        )

        objective_valid = (
            abs(
                objective_value
                - validated_objective
            )
            <= objective_tolerance
        )

        write_gurobi_solution(
            solution_path=solution_path,
            header=header,
            status=status_name,
            objective=objective_value,
            validated_objective=validated_objective,
            best_bound=best_bound,
            mip_gap=mip_gap,
            runtime=model.Runtime,
            time_to_best=model._time_to_best,
            x_assignment=x_assignment,
            y_assignment=y_assignment,
        )

    else:
        try:
            best_bound = model.ObjBound
        except gp.GurobiError:
            best_bound = None

    try:
        work = model.Work
    except (AttributeError, gp.GurobiError):
        work = None

    return GurobiResult(
        status=status_name,
        status_code=status_code,
        solution_count=solution_count,
        objective=objective_value,
        validated_objective=validated_objective,
        objective_valid=objective_valid,
        best_bound=best_bound,
        mip_gap=mip_gap,
        runtime_seconds=model.Runtime,
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


def write_gurobi_solution(
    *,
    solution_path: pathlib.Path,
    header: InstanceHeader,
    status: str,
    objective: float,
    validated_objective: int,
    best_bound: float,
    mip_gap: float,
    runtime: float,
    time_to_best: Optional[float],
    x_assignment: Sequence[int],
    y_assignment: Sequence[int],
) -> None:
    with solution_path.open(
        "w",
        encoding="utf-8",
    ) as stream:
        stream.write("GUROBI MCDAP SOLUTION\n")
        stream.write("====================\n")
        stream.write(f"status={status}\n")
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
            f"objective={objective:.17g}\n"
        )
        stream.write(
            f"validated_objective="
            f"{validated_objective}\n"
        )
        stream.write(
            f"best_bound={best_bound:.17g}\n"
        )
        stream.write(
            f"mip_gap={mip_gap:.17g}\n"
        )
        stream.write(
            f"runtime_seconds={runtime:.9f}\n"
        )
        stream.write(
            "time_to_best_seconds="
            + (
                f"{time_to_best:.9f}"
                if time_to_best is not None
                else "NA"
            )
            + "\n\n"
        )

        stream.write("x_assignment=")
        stream.write(
            " ".join(
                str(slot + 1)
                for slot in x_assignment
            )
        )
        stream.write("\n")

        stream.write("y_assignment=")
        stream.write(
            " ".join(
                str(slot + 1)
                for slot in y_assignment
            )
        )
        stream.write("\n")


# =============================================================================
# Comparison output
# =============================================================================

def finite_or_none(
    value: Optional[float],
) -> Optional[float]:
    if value is None:
        return None

    if not math.isfinite(value):
        return None

    return value


def write_comparison_outputs(
    output_directory: pathlib.Path,
    input_path: pathlib.Path,
    heuristic: HeuristicResult,
    gurobi: GurobiResult,
    arguments: argparse.Namespace,
) -> None:
    json_path = (
        output_directory
        / "comparison.json"
    )

    csv_path = (
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
        # Positive value means Gurobi found a lower minimization objective.
        absolute_difference = (
            heuristic_objective
            - gurobi_objective
        )

        relative_difference = (
            absolute_difference
            / max(1.0, abs(float(gurobi_objective)))
        )

        if gurobi_objective < heuristic_objective:
            better_method = "gurobi"
        elif heuristic_objective < gurobi_objective:
            better_method = "heuristic"
        else:
            better_method = "tie"

    payload = {
        "timestamp_utc": (
            dt.datetime.now(dt.timezone.utc)
            .isoformat(timespec="seconds")
        ),
        "input_file": str(input_path),
        "heuristic_settings": {
            "time_limit_seconds": 300,
            "max_iterations": 100000,
            "n1": 1,
            "n2": 10,
            "sequence_count": 150,
            "tabu_tenure": 5,
            "restart_stagnation": 5,
            "seed": arguments.seed,
        },
        "gurobi_settings": {
            "time_limit_seconds": (
                arguments.gurobi_time_limit
            ),
            "mip_gap": arguments.mip_gap,
            "threads": arguments.gurobi_threads,
            "seed": arguments.gurobi_seed,
            "nonconvex": 2,
            "warm_start": arguments.warm_start,
            "presolve": arguments.presolve,
            "mip_focus": arguments.mip_focus,
        },
        "heuristic": asdict(heuristic),
        "gurobi": asdict(gurobi),
        "comparison": {
            "heuristic_objective": (
                heuristic_objective
            ),
            "gurobi_objective": gurobi_objective,
            "heuristic_minus_gurobi": (
                absolute_difference
            ),
            "relative_difference": (
                relative_difference
            ),
            "better_method": better_method,
        },
    }

    json_path.write_text(
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
        "best_bound",
        "mip_gap",
        "runtime_seconds",
        "time_to_best_seconds",
        "warm_start",
        "solution_count",
        "nodes",
        "heuristic_minus_gurobi",
        "relative_difference",
        "better_method",
    ]

    with csv_path.open(
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
            "objective": heuristic.reported_objective,
            "validated_objective": (
                heuristic.validated_objective
            ),
            "objective_valid": int(
                heuristic.objective_valid
            ),
            "best_bound": "",
            "mip_gap": "",
            "runtime_seconds": (
                heuristic.search_seconds
                if heuristic.search_seconds is not None
                else heuristic.process_wall_seconds
            ),
            "time_to_best_seconds": (
                heuristic.time_to_best_seconds
            ),
            "warm_start": "",
            "solution_count": "",
            "nodes": "",
            "heuristic_minus_gurobi": (
                absolute_difference
            ),
            "relative_difference": (
                relative_difference
            ),
            "better_method": better_method,
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
            "best_bound": gurobi.best_bound,
            "mip_gap": gurobi.mip_gap,
            "runtime_seconds": (
                gurobi.runtime_seconds
            ),
            "time_to_best_seconds": (
                gurobi.time_to_best_seconds
            ),
            "warm_start": int(
                gurobi.warm_start_used
            ),
            "solution_count": (
                gurobi.solution_count
            ),
            "nodes": gurobi.nodes,
            "heuristic_minus_gurobi": (
                absolute_difference
            ),
            "relative_difference": (
                relative_difference
            ),
            "better_method": better_method,
        })

    print()
    print("Comparison")
    print("==========")

    print(
        "Heuristic objective: "
        f"{heuristic_objective}"
    )

    print(
        "Heuristic Time-to-Best: "
        f"{heuristic.time_to_best_seconds}"
    )

    print(
        "Gurobi status: "
        f"{gurobi.status}"
    )

    print(
        "Gurobi objective: "
        f"{gurobi_objective}"
    )

    print(
        "Gurobi best bound: "
        f"{gurobi.best_bound}"
    )

    print(
        "Gurobi MIP gap: "
        f"{gurobi.mip_gap}"
    )

    print(
        "Gurobi Time-to-Best: "
        f"{gurobi.time_to_best_seconds}"
    )

    print(
        "Heuristic - Gurobi objective: "
        f"{absolute_difference}"
    )

    print(
        "Better result: "
        f"{better_method}"
    )

    print()
    print(f"Comparison CSV:  {csv_path}")
    print(f"Comparison JSON: {json_path}")
    print(
        f"Gurobi solution: {gurobi.solution_file}"
    )
    print(f"Gurobi log:      {gurobi.log_file}")


# =============================================================================
# Command-line interface
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare d_flip_xqy_dense_cd with Gurobi MIQP "
            "on a prepared MCDAP xQy instance."
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
            "Existing heuristic result, used with "
            "--skip-heuristic"
        ),
    )

    parser.add_argument(
        "--skip-heuristic",
        action="store_true",
        help="Do not execute the heuristic",
    )

    parser.add_argument(
        "--heuristic-timeout",
        type=float,
        default=600.0,
        help=(
            "External heuristic timeout. The internal search "
            "limit remains fixed at 300 seconds."
        ),
    )

    parser.add_argument(
        "--heuristic-verify",
        action="store_true",
        help=(
            "Pass --verify-every-move to the heuristic; "
            "this can be very expensive"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path(
            "comparison-n100m10-1"
        ),
        help="Output directory",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Heuristic random seed",
    )

    parser.add_argument(
        "--gurobi-time-limit",
        type=float,
        default=7200.0,
        help="Gurobi time limit in seconds",
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
            "optimality unless the time limit is reached."
        ),
    )

    parser.add_argument(
        "--warm-start",
        action="store_true",
        help=(
            "Use the validated heuristic assignment as a "
            "Gurobi MIP start"
        ),
    )

    parser.add_argument(
        "--mip-focus",
        type=int,
        choices=(0, 1, 2, 3),
        default=None,
        help=(
            "Optional Gurobi MIPFocus setting. "
            "1 emphasizes finding feasible solutions; "
            "2 emphasizes proving optimality; "
            "3 emphasizes improving the bound."
        ),
    )

    parser.add_argument(
        "--presolve",
        type=int,
        choices=(-1, 0, 1, 2),
        default=None,
        help=(
            "Optional Gurobi Presolve setting; "
            "-1 automatic, 0 off, 1 conservative, 2 aggressive"
        ),
    )

    parser.add_argument(
        "--nodefile-start",
        type=float,
        default=None,
        help=(
            "Optional Gurobi NodefileStart in GB"
        ),
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
            "Number of quadratic terms passed to Gurobi "
            "per Python chunk"
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
        arguments.output_dir.expanduser().resolve()
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
            f"Input file does not exist: "
            f"{arguments.input}"
        )

    if arguments.gurobi_time_limit <= 0.0:
        raise SystemExit(
            "--gurobi-time-limit must be positive"
        )

    if arguments.heuristic_timeout <= 0.0:
        raise SystemExit(
            "--heuristic-timeout must be positive"
        )

    if arguments.gurobi_threads < 0:
        raise SystemExit(
            "--gurobi-threads cannot be negative"
        )

    if arguments.mip_gap < 0.0:
        raise SystemExit(
            "--mip-gap cannot be negative"
        )

    if arguments.q_chunk_size <= 0:
        raise SystemExit(
            "--q-chunk-size must be positive"
        )

    if arguments.seed < 0:
        raise SystemExit(
            "--seed cannot be negative"
        )

    if arguments.gurobi_seed < 0:
        raise SystemExit(
            "--gurobi-seed cannot be negative"
        )

    if (
        arguments.warm_start
        and arguments.skip_heuristic
        and arguments.heuristic_result is None
    ):
        raise SystemExit(
            "--warm-start with --skip-heuristic requires "
            "--heuristic-result"
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

    print("MCDAP xQy comparison")
    print("====================")
    print(f"Input: {arguments.input}")
    print(
        "Heuristic settings: "
        "300 seconds, 100000 max iterations, "
        "N1=1, N2=10, S=150, tenure=5, "
        "restart-stagnation=5"
    )
    print(
        f"Gurobi time limit: "
        f"{arguments.gurobi_time_limit} seconds"
    )
    print(
        f"Gurobi warm start: "
        f"{arguments.warm_start}"
    )

    print()
    print("Running heuristic...")

    heuristic_result = run_heuristic(
        arguments,
        arguments.input,
        arguments.output_dir,
    )

    print(
        f"Heuristic status: "
        f"{heuristic_result.status}"
    )

    print(
        f"Heuristic objective: "
        f"{heuristic_result.validated_objective}"
    )

    if (
        arguments.warm_start
        and heuristic_result.status != "success"
    ):
        raise SystemExit(
            "Error: the heuristic result could not be "
            "validated, so it cannot be used as a MIP start.\n"
            f"Details: {heuristic_result.error_message}"
        )

    print()
    print("Building and running Gurobi MIQP...")

    gurobi_result = build_and_solve_gurobi(
        arguments,
        arguments.input,
        arguments.output_dir,
        heuristic_result,
    )

    write_comparison_outputs(
        output_directory=arguments.output_dir,
        input_path=arguments.input,
        heuristic=heuristic_result,
        gurobi=gurobi_result,
        arguments=arguments,
    )

    if (
        heuristic_result.status != "success"
        or not gurobi_result.objective_valid
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