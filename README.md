# D-FLIP-XQY-Solver

# Data source: Wang, Haibo, 2019, "Multi-floor Cross-docking Instances", https://doi.org/10.18738/T8/LQBTFF, Texas Data Repository, V2

This data sets for multi-floor cross-dock door assignment problems, the size of Q matrix ranges from 1000 by 1000 to 90000 by 90000
For each input file, we have two dimensional matrix. For example, n100m10-1-fcx.txt file has a 100 by 100 matrix with the values of fcx.
n100m10-1-fcy.txt has a 10 by 10 matrix with the values of fcy.
n100m10-1-qxy.txt has a 1000 by 1000 matrix with the values of qxy.
There are 10 instances for each problem size and the large instances of 40000 by 40000 and 90000 by 90000 data files will split into multiple files
After downloading the split files, you could type this:
copy /b file1 + file2 + file3 + file4 filetogether
then type:
gunzip filetogether
ref:“The Multi-floor Cross-dock Door Assignment Problem: Rising Challenges for the New Trend in Logistics Industry”, Transportation Research Part E: Logistics and Transportation Review, 132, 30-47, 2019

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
