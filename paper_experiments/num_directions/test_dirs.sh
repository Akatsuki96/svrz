#!/bin/bash


l_values=(1 10 20 30 40 50)
m_values=(25 50 100 150)

for m in "${m_values[@]}"; do
    for l in "${l_values[@]}"; do
        nohup python3 changing_num_directions.py ${l} ${m} 1  "/data/mrando/svrz_results/num_directions" &
    done
done