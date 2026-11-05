# D-FLIP-XQY-Solver

# Data source: Wang, Haibo, 2019, "Multi-floor Cross-docking Instances", https://doi.org/10.18738/T8/LQBTFF, Texas Data Repository, V2

This data sets for multi-floor cross-dock door assignment problems, the size of Q matrix ranges from 1000 by 1000 to 90000 by 90000 <br>
For each input file, we have two dimensional matrix. For example, n100m10-1-fcx.txt file has a 100 by 100 matrix with the values of fcx.<br>
n100m10-1-fcy.txt has a 10 by 10 matrix with the values of fcy.<br>
n100m10-1-qxy.txt has a 1000 by 1000 matrix with the values of qxy.<br>
There are 10 instances for each problem size and the large instances of 40000 by 40000 and 90000 by 90000 data files will split into multiple files<br>
After downloading the split files, you could type this:<br>
copy /b file1 + file2 + file3 + file4 filetogether<br>
then type:<br>
gunzip filetogether<br>
ref:<a href="https://www.researchgate.net/profile/Haibo-Wang/publication/382249951_TRE-June24pdf/data/66941afe3e0edb1e0fe4b219/TRE-June24.pdf">“The Multi-floor Cross-dock Door Assignment Problem: Rising Challenges for the New Trend in Logistics Industry”, Transportation Research Part E: Logistics and Transportation Review, 132, 30-47, 2019</a>
<br>
# Prepare the input data
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


Compare:
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
