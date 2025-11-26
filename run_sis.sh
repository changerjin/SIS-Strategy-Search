#!/bin/bash



mkdir -p logs


DIM=800
python3 lattice_challenge.py ${DIM} --threads 32 --gpus 1 2>&1 | tee logs/SIS_${DIM}.log

DIM=750
python3 lattice_challenge.py ${DIM} --threads 32 --gpus 1 2>&1 | tee logs/SIS_${DIM}.log

DIM=700
python3 lattice_challenge.py ${DIM} --threads 32 --gpus 1 2>&1 | tee logs/SIS_${DIM}.log


DIM=800
python3 lattice_challenge.py ${DIM} --threads 32 --gpus 1 2>&1 | tee logs/SIS_${DIM}2.log

DIM=750
python3 lattice_challenge.py ${DIM} --threads 32 --gpus 1 2>&1 | tee logs/SIS_${DIM}2.log

DIM=700
python3 lattice_challenge.py ${DIM} --threads 32 --gpus 1 2>&1 | tee logs/SIS_${DIM}2.log


DIM=800
python3 lattice_challenge.py ${DIM} --threads 32 --gpus 1 2>&1 | tee logs/SIS_${DIM}3.log

DIM=750
python3 lattice_challenge.py ${DIM} --threads 32 --gpus 1 2>&1 | tee logs/SIS_${DIM}3.log

DIM=700
python3 lattice_challenge.py ${DIM} --threads 32 --gpus 1 2>&1 | tee logs/SIS_${DIM}3.log