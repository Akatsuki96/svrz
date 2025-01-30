#!/bin/bash


l_values=(1 10 20 30 40 50)
m_values=(50 100 150)


if [[ -z "$1" ]] ; then
    echo -e "\nPlease call '$0 <output directory>' to run this command!\n"
else
    for m in "${m_values[@]}"; do
        for l in "${l_values[@]}"; do
            nohup python3 changing_num_directions.py ${l} ${m} 1 ${1} &
        done
    done
fi