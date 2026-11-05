#!/usr/bin/env python3
"""
data_prepare.py

Convert cross-dock fixed-cost and interaction files into the input format
used by:

    d_flip_xqy_dense_cd.cpp
    d_flip_xqy_sparse_cd.cpp

Mathematical model
------------------

    minimize

        sum_{i,s} FCX[i,s] x[i,s]
      + sum_{j,t} FCY[j,t] y[j,t]
      + sum_{i,s,j,t} Q[(i,s),(j,t)] x[i,s] y[j,t]

    subject to

        sum_s x[i,s] = 1              for every x-side item i
        sum_i x[i,s] <= 1             for every x-side slot s

        sum_t y[j,t] = 1              for every y-side item j
        sum_j y[j,t] <= 1             for every y-side slot t

        x[i,s], y[j,t] in {0,1}

Output format
-------------

    number_of_instances
    x_items x_slots y_items y_slots number_of_q_records
    <x_items*x_slots fixed x costs>
    <y_items*y_slots fixed y costs>
    q_row q_column coefficient
    ...

The flattened assignment indices are:

    x_position = x_item * x_slots + x_slot
    y_position = y_item * y_slots + y_slot

Q record indices in the generated file are one-based.

Default dimension inference
---------------------------

If explicit dimensions are not supplied, the script requires each
fixed-cost count to be a perfect square:

    len(fcx) = x_items * x_slots, with x_items == x_slots
    len(fcy) = y_items * y_slots, with y_items == y_slots

For the attached n100m10-1 files, this should infer:

    x_items = 100
    x_slots = 100
    y_items = 10
    y_slots = 10

The script validates the actual counts before writing output.

Q ordering
----------

The default is row-major:

    Q[0,0], Q[0,1], ..., Q[0,ny-1],
    Q[1,0], Q[1,1], ..., Q[1,ny-1],
    ...

Use:

    --q-order column-major

if the source file instead stores complete columns consecutively.

Use:

    --transpose

to exchange the x and y sides and write Q^T.

Examples
--------

Automatic dimension inference:

    python3 data_prepare.py \
        --fcx n100m10-1-fcx.txt \
        --fcy n100m10-1-fcy.txt \
        --qxy n100m10-1-qxy.txt \
        --output n100m10-1-cd-xqy.txt

Explicit dimensions:

    python3 data_prepare.py \
        --fcx n100m10-1-fcx.txt \
        --fcy n100m10-1-fcy.txt \
        --qxy n100m10-1-qxy.txt \
        --x-items 100 \
        --x-slots 100 \
        --y-items 10 \
        --y-slots 10 \
        --output n100m10-1-cd-xqy.txt

Keep zero Q coefficients as explicit records:

    python3 data_prepare.py ... --keep-zeros

Transpose the instance:

    python3 data_prepare.py ... --transpose
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import tempfile
from dataclasses import dataclass
from typing import Iterator, Optional, TextIO, Tuple


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class AssignmentShape:
    """Numbers of entities and available assignment slots."""

    items: int
    slots: int

    @property
    def variable_count(self) -> int:
        return self.items * self.slots


@dataclass(frozen=True)
class FileStatistics:
    """Numeric-value statistics for one source file."""

    value_count: int
    nonzero_count: int
    minimum: float
    maximum: float


# =============================================================================
# Numeric input
# =============================================================================

def strip_comment(line: str) -> str:
    """
    Remove supported trailing comments.

    Supported comment prefixes are '#' and '//'. This does not treat a minus
    sign or scientific-notation character as a comment.
    """

    hash_position = line.find("#")
    slash_position = line.find("//")

    positions = [
        position
        for position in (hash_position, slash_position)
        if position >= 0
    ]

    if positions:
        line = line[:min(positions)]

    return line


def normalize_numeric_text(text: str) -> str:
    """
    Permit whitespace-separated or comma-separated scalar files.

    A comma is interpreted as a field separator, not a decimal separator.
    """

    return text.replace(",", " ")


def numeric_tokens(path: pathlib.Path) -> Iterator[float]:
    """
    Stream finite floating-point values from a text file.

    Blank lines and trailing comments are ignored. Every remaining token must
    be a valid finite number.
    """

    with path.open(
        "r",
        encoding="utf-8",
        errors="strict",
        buffering=1024 * 1024,
    ) as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = strip_comment(raw_line)
            line = normalize_numeric_text(line).strip()

            if not line:
                continue

            for token_number, token in enumerate(
                line.split(),
                start=1,
            ):
                try:
                    value = float(token)
                except ValueError as error:
                    raise ValueError(
                        f"{path}: invalid numeric token {token!r} "
                        f"at line {line_number}, field {token_number}"
                    ) from error

                if not math.isfinite(value):
                    raise ValueError(
                        f"{path}: non-finite value {token!r} "
                        f"at line {line_number}, field {token_number}"
                    )

                yield value


def inspect_numeric_file(path: pathlib.Path) -> FileStatistics:
    """Count and summarize numeric values without retaining them in memory."""

    value_count = 0
    nonzero_count = 0
    minimum = math.inf
    maximum = -math.inf

    for value in numeric_tokens(path):
        value_count += 1

        if value != 0.0:
            nonzero_count += 1

        minimum = min(minimum, value)
        maximum = max(maximum, value)

    if value_count == 0:
        raise ValueError(
            f"{path}: file contains no numeric values"
        )

    return FileStatistics(
        value_count=value_count,
        nonzero_count=nonzero_count,
        minimum=minimum,
        maximum=maximum,
    )


# =============================================================================
# Dimension handling
# =============================================================================

def infer_square_shape(
    value_count: int,
    side_name: str,
) -> AssignmentShape:
    """
    Infer an equal number of entities and slots from a perfect-square count.
    """

    dimension = math.isqrt(value_count)

    if dimension * dimension != value_count:
        raise ValueError(
            f"Cannot infer the {side_name}-side dimensions from "
            f"{value_count:,} fixed-cost values because the count is not "
            f"a perfect square. Supply --{side_name}-items and "
            f"--{side_name}-slots explicitly."
        )

    return AssignmentShape(
        items=dimension,
        slots=dimension,
    )


def determine_shape(
    value_count: int,
    side_name: str,
    items: Optional[int],
    slots: Optional[int],
) -> AssignmentShape:
    """Validate explicit dimensions or infer a square assignment shape."""

    if items is None and slots is None:
        return infer_square_shape(
            value_count,
            side_name,
        )

    if items is None or slots is None:
        raise ValueError(
            f"--{side_name}-items and --{side_name}-slots must be "
            "supplied together"
        )

    if items <= 0:
        raise ValueError(
            f"--{side_name}-items must be positive"
        )

    if slots <= 0:
        raise ValueError(
            f"--{side_name}-slots must be positive"
        )

    if slots < items:
        raise ValueError(
            f"--{side_name}-slots ({slots}) must be at least "
            f"--{side_name}-items ({items}) so a feasible injective "
            "assignment exists"
        )

    shape = AssignmentShape(
        items=items,
        slots=slots,
    )

    if shape.variable_count != value_count:
        raise ValueError(
            f"The requested {side_name}-side shape "
            f"{items} x {slots} requires "
            f"{shape.variable_count:,} fixed costs, but the file "
            f"contains {value_count:,}"
        )

    return shape


def checked_matrix_size(
    row_count: int,
    column_count: int,
) -> int:
    """Calculate the required number of dense Q values."""

    if row_count <= 0 or column_count <= 0:
        raise ValueError(
            "Q dimensions must be positive"
        )

    return row_count * column_count


# =============================================================================
# Number formatting
# =============================================================================

def format_number(value: float) -> str:
    """
    Format values compactly while retaining double-precision information.
    """

    if value == 0.0:
        return "0"

    if value.is_integer():
        return str(int(value))

    return format(value, ".17g")


# =============================================================================
# Output helpers
# =============================================================================

def write_scalar_file(
    output: TextIO,
    source_path: pathlib.Path,
    expected_count: int,
    label: str,
) -> None:
    """Copy one fixed-cost scalar per output line."""

    observed_count = 0

    for value in numeric_tokens(source_path):
        observed_count += 1

        if observed_count > expected_count:
            raise ValueError(
                f"{source_path}: contains more than the expected "
                f"{expected_count:,} {label} values"
            )

        output.write(format_number(value))
        output.write("\n")

    if observed_count != expected_count:
        raise ValueError(
            f"{source_path}: expected {expected_count:,} {label} values "
            f"but found {observed_count:,}"
        )


def decode_q_position(
    flat_index: int,
    row_count: int,
    column_count: int,
    q_order: str,
) -> Tuple[int, int]:
    """Convert a flat source position into zero-based matrix coordinates."""

    if q_order == "row-major":
        row = flat_index // column_count
        column = flat_index % column_count
    elif q_order == "column-major":
        row = flat_index % row_count
        column = flat_index // row_count
    else:
        raise ValueError(
            f"Unsupported Q order: {q_order}"
        )

    return row, column


def write_q_records(
    output: TextIO,
    q_path: pathlib.Path,
    source_rows: int,
    source_columns: int,
    q_order: str,
    keep_zeros: bool,
    transpose: bool,
    progress_interval: int,
) -> Tuple[int, int]:
    """
    Convert flat Q values into indexed records.

    Returns:
        (number of source values, number of records written)
    """

    expected_count = checked_matrix_size(
        source_rows,
        source_columns,
    )

    observed_count = 0
    records_written = 0

    for flat_index, coefficient in enumerate(
        numeric_tokens(q_path)
    ):
        observed_count += 1

        if observed_count > expected_count:
            raise ValueError(
                f"{q_path}: contains more than the expected "
                f"{expected_count:,} Q coefficients"
            )

        if not keep_zeros and coefficient == 0.0:
            continue

        row, column = decode_q_position(
            flat_index=flat_index,
            row_count=source_rows,
            column_count=source_columns,
            q_order=q_order,
        )

        if transpose:
            output_row = column
            output_column = row
        else:
            output_row = row
            output_column = column

        # The C++ input convention is one-based.
        output.write(
            f"{output_row + 1} "
            f"{output_column + 1} "
            f"{format_number(coefficient)}\n"
        )

        records_written += 1

        if (
            progress_interval > 0
            and observed_count % progress_interval == 0
        ):
            percentage = 100.0 * observed_count / expected_count

            print(
                f"\rWriting Q: {observed_count:,}/"
                f"{expected_count:,} values "
                f"({percentage:.1f}%), "
                f"{records_written:,} records",
                end="",
                flush=True,
            )

    if progress_interval > 0:
        print()

    if observed_count != expected_count:
        raise ValueError(
            f"{q_path}: Q dimension mismatch: expected "
            f"{expected_count:,} coefficients for a "
            f"{source_rows:,} x {source_columns:,} matrix, "
            f"but found {observed_count:,}"
        )

    return observed_count, records_written


# =============================================================================
# Argument parsing
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert cross-dock FCX, FCY, and QXY files to the "
            "constrained CD-xQy solver format."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--fcx",
        type=pathlib.Path,
        required=True,
        help=(
            "Fixed-cost file for flattened x-side "
            "(item, slot) assignment positions"
        ),
    )

    parser.add_argument(
        "--fcy",
        type=pathlib.Path,
        required=True,
        help=(
            "Fixed-cost file for flattened y-side "
            "(item, slot) assignment positions"
        ),
    )

    parser.add_argument(
        "--qxy",
        type=pathlib.Path,
        required=True,
        help=(
            "Flat dense interaction matrix between x and y "
            "assignment positions"
        ),
    )

    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Destination CD-xQy input file",
    )

    parser.add_argument(
        "--x-items",
        type=int,
        help=(
            "Number of x-side entities, such as origins; "
            "inferred from FCX when omitted"
        ),
    )

    parser.add_argument(
        "--x-slots",
        type=int,
        help=(
            "Number of x-side assignment slots, such as unloading doors; "
            "inferred from FCX when omitted"
        ),
    )

    parser.add_argument(
        "--y-items",
        type=int,
        help=(
            "Number of y-side entities, such as destinations; "
            "inferred from FCY when omitted"
        ),
    )

    parser.add_argument(
        "--y-slots",
        type=int,
        help=(
            "Number of y-side assignment slots, such as loading doors; "
            "inferred from FCY when omitted"
        ),
    )

    parser.add_argument(
        "--q-order",
        choices=("row-major", "column-major"),
        default="row-major",
        help="Flattening order used by the source QXY file",
    )

    parser.add_argument(
        "--transpose",
        action="store_true",
        help=(
            "Exchange the x and y sides and write the transposed "
            "interaction matrix"
        ),
    )

    zero_group = parser.add_mutually_exclusive_group()

    zero_group.add_argument(
        "--keep-zeros",
        action="store_true",
        help="Write zero-valued Q coefficients as explicit records",
    )

    zero_group.add_argument(
        "--omit-zeros",
        action="store_true",
        help=(
            "Omit zero-valued Q coefficients; this is the default "
            "and is recommended for the sparse solver"
        ),
    )

    parser.add_argument(
        "--progress",
        type=int,
        default=1_000_000,
        metavar="N",
        help=(
            "Report progress after every N source Q values; "
            "use zero to disable"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing destination file",
    )

    return parser.parse_args()


# =============================================================================
# Validation
# =============================================================================

def validate_input_file(
    path: pathlib.Path,
    label: str,
) -> pathlib.Path:
    resolved = path.expanduser().resolve()

    if not resolved.exists():
        raise ValueError(
            f"{label} file does not exist: {resolved}"
        )

    if not resolved.is_file():
        raise ValueError(
            f"{label} path is not a regular file: {resolved}"
        )

    return resolved


def validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    if arguments.progress < 0:
        raise ValueError(
            "--progress must be zero or a positive integer"
        )


# =============================================================================
# Conversion
# =============================================================================

def prepare_conversion(
    arguments: argparse.Namespace,
) -> Tuple[
    pathlib.Path,
    pathlib.Path,
    pathlib.Path,
    pathlib.Path,
    AssignmentShape,
    AssignmentShape,
    FileStatistics,
    FileStatistics,
    FileStatistics,
    int,
]:
    """Inspect source files and validate all dimensions."""

    validate_arguments(arguments)

    fcx_path = validate_input_file(
        arguments.fcx,
        "FCX",
    )

    fcy_path = validate_input_file(
        arguments.fcy,
        "FCY",
    )

    qxy_path = validate_input_file(
        arguments.qxy,
        "QXY",
    )

    output_path = arguments.output.expanduser().resolve()

    if output_path.exists() and not arguments.force:
        raise ValueError(
            f"Output file already exists: {output_path}. "
            "Use --force to replace it."
        )

    # Prevent accidentally replacing a source file.
    for source_path in (fcx_path, fcy_path, qxy_path):
        if output_path == source_path:
            raise ValueError(
                "The output file must be different from every input file"
            )

    print("Inspecting FCX...", flush=True)
    fcx_statistics = inspect_numeric_file(fcx_path)

    print("Inspecting FCY...", flush=True)
    fcy_statistics = inspect_numeric_file(fcy_path)

    x_shape = determine_shape(
        value_count=fcx_statistics.value_count,
        side_name="x",
        items=arguments.x_items,
        slots=arguments.x_slots,
    )

    y_shape = determine_shape(
        value_count=fcy_statistics.value_count,
        side_name="y",
        items=arguments.y_items,
        slots=arguments.y_slots,
    )

    expected_q_count = checked_matrix_size(
        fcx_statistics.value_count,
        fcy_statistics.value_count,
    )

    print(
        "Inspecting QXY "
        f"(expecting {expected_q_count:,} values)...",
        flush=True,
    )

    qxy_statistics = inspect_numeric_file(qxy_path)

    if qxy_statistics.value_count != expected_q_count:
        raise ValueError(
            "QXY dimension mismatch: FCX contains "
            f"{fcx_statistics.value_count:,} assignment positions and "
            f"FCY contains {fcy_statistics.value_count:,}, so QXY must "
            f"contain {expected_q_count:,} coefficients. The source file "
            f"contains {qxy_statistics.value_count:,}."
        )

    q_record_count = (
        qxy_statistics.value_count
        if arguments.keep_zeros
        else qxy_statistics.nonzero_count
    )

    return (
        fcx_path,
        fcy_path,
        qxy_path,
        output_path,
        x_shape,
        y_shape,
        fcx_statistics,
        fcy_statistics,
        qxy_statistics,
        q_record_count,
    )


def convert(arguments: argparse.Namespace) -> pathlib.Path:
    (
        fcx_path,
        fcy_path,
        qxy_path,
        output_path,
        x_shape,
        y_shape,
        fcx_statistics,
        fcy_statistics,
        qxy_statistics,
        q_record_count,
    ) = prepare_conversion(arguments)

    if arguments.transpose:
        output_x_shape = y_shape
        output_y_shape = x_shape

        output_x_fixed_path = fcy_path
        output_y_fixed_path = fcx_path

        output_x_fixed_count = fcy_statistics.value_count
        output_y_fixed_count = fcx_statistics.value_count
    else:
        output_x_shape = x_shape
        output_y_shape = y_shape

        output_x_fixed_path = fcx_path
        output_y_fixed_path = fcy_path

        output_x_fixed_count = fcx_statistics.value_count
        output_y_fixed_count = fcy_statistics.value_count

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path: Optional[pathlib.Path] = None

    try:
        # The temporary file is created in the destination directory so the
        # final os.replace operation is atomic on the same filesystem.
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            text=True,
        )

        temporary_path = pathlib.Path(temporary_name)

        with os.fdopen(
            descriptor,
            mode="w",
            encoding="utf-8",
            buffering=8 * 1024 * 1024,
        ) as output:
            # One converted problem.
            output.write("1\n")

            output.write(
                f"{output_x_shape.items} "
                f"{output_x_shape.slots} "
                f"{output_y_shape.items} "
                f"{output_y_shape.slots} "
                f"{q_record_count}\n"
            )

            print(
                f"Writing {output_x_fixed_count:,} x fixed costs...",
                flush=True,
            )

            write_scalar_file(
                output=output,
                source_path=output_x_fixed_path,
                expected_count=output_x_fixed_count,
                label="x fixed-cost",
            )

            print(
                f"Writing {output_y_fixed_count:,} y fixed costs...",
                flush=True,
            )

            write_scalar_file(
                output=output,
                source_path=output_y_fixed_path,
                expected_count=output_y_fixed_count,
                label="y fixed-cost",
            )

            print(
                f"Writing {q_record_count:,} indexed Q records...",
                flush=True,
            )

            source_values, records_written = write_q_records(
                output=output,
                q_path=qxy_path,
                source_rows=fcx_statistics.value_count,
                source_columns=fcy_statistics.value_count,
                q_order=arguments.q_order,
                keep_zeros=arguments.keep_zeros,
                transpose=arguments.transpose,
                progress_interval=arguments.progress,
            )

            if source_values != qxy_statistics.value_count:
                raise RuntimeError(
                    "QXY changed between validation and conversion"
                )

            if records_written != q_record_count:
                raise RuntimeError(
                    "Q record count changed between validation "
                    f"and conversion: expected {q_record_count:,}, "
                    f"wrote {records_written:,}"
                )

            output.flush()
            os.fsync(output.fileno())

        os.replace(
            temporary_path,
            output_path,
        )

        temporary_path = None

    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass

    print()
    print("Conversion completed successfully.")
    print(f"Output: {output_path}")
    print(
        f"Source FCX: {fcx_statistics.value_count:,} values, "
        f"range [{format_number(fcx_statistics.minimum)}, "
        f"{format_number(fcx_statistics.maximum)}]"
    )
    print(
        f"Source FCY: {fcy_statistics.value_count:,} values, "
        f"range [{format_number(fcy_statistics.minimum)}, "
        f"{format_number(fcy_statistics.maximum)}]"
    )
    print(
        f"Source QXY: {qxy_statistics.value_count:,} values, "
        f"{qxy_statistics.nonzero_count:,} nonzero"
    )
    print(
        f"Output x side: {output_x_shape.items} items x "
        f"{output_x_shape.slots} slots = "
        f"{output_x_shape.variable_count:,} assignment positions"
    )
    print(
        f"Output y side: {output_y_shape.items} items x "
        f"{output_y_shape.slots} slots = "
        f"{output_y_shape.variable_count:,} assignment positions"
    )
    print(f"Output Q records: {q_record_count:,}")
    print(f"Source Q order: {arguments.q_order}")
    print(f"Transposed: {'yes' if arguments.transpose else 'no'}")
    print(
        "Zero coefficients: "
        + (
            "retained"
            if arguments.keep_zeros
            else "omitted"
        )
    )

    return output_path


# =============================================================================
# Entry point
# =============================================================================

def main() -> int:
    arguments = parse_arguments()
    convert(arguments)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Error: {error}")