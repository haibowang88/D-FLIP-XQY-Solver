# D-FLIP-XQY-Solver

# Data source

#Compare:
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
