#!/bin/bash



mkdir -p logs


for run in {5..10}; do

    DIM=800
    python3 lattice_challenge.py ${DIM} --threads 32 --gpus 1 2>&1 | tee logs/SIS_${DIM}_${run}.log

    DIM=750
    python3 lattice_challenge.py ${DIM} --threads 32 --gpus 1 2>&1 | tee logs/SIS_${DIM}_${run}.log

    DIM=700
    python3 lattice_challenge.py ${DIM} --threads 32 --gpus 1 2>&1 | tee logs/SIS_${DIM}_${run}.log

done
# DIM=800
# python3 lattice_challenge.py ${DIM} --threads 32 --gpus 1 2>&1 | tee logs/SIS_${DIM}2.log

# DIM=750
# python3 lattice_challenge.py ${DIM} --threads 32 --gpus 1 2>&1 | tee logs/SIS_${DIM}2.log

# DIM=700
# python3 lattice_challenge.py ${DIM} --threads 32 --gpus 1 2>&1 | tee logs/SIS_${DIM}2.log


# DIM=800
# python3 lattice_challenge.py ${DIM} --threads 32 --gpus 1 2>&1 | tee logs/SIS_${DIM}3.log

# DIM=750
# python3 lattice_challenge.py ${DIM} --threads 32 --gpus 1 2>&1 | tee logs/SIS_${DIM}3.log

# DIM=700
# python3 lattice_challenge.py ${DIM} --threads 32 --gpus 1 2>&1 | tee logs/SIS_${DIM}3.log