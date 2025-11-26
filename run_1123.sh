
python3 test_pump_time.py 900 --threads 32 --gpus 1 2>&1 | tee logs/Test_Pump_Average_Time_900.log


DIM=900
python3 lattice_challenge.py ${DIM} --threads 32 --gpus 1 2>&1 | tee logs/SIS_${DIM}_2.log
