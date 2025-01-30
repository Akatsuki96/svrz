#!/bin/bash


l_values=(1 10 25 50)

opt_names=("opsvrz" "zo_psvrg_coord" "zo_psvrg_gaus" "zo_psvrg_sph" "zo_pspider_coord" "zo_pspider" "rspgf")

if [[ -z "$1" || -z "$2" ]] ; then
    echo -e "\nPlease call '$0 <output directory> <number of inner loop iterations>' to run this command!\n"
else
    for opt_name in "${opt_names[@]}"; do
        for l in "${l_values[@]}"; do
            if [[ ${opt_name} != 'rspgf' || ( ${opt_name} = 'rspgf' && ${l} = 1 ) ]]; then
                nohup python3 least_squares.py ${opt_name} 10 1 50 ${2} ${l} ${1} &
            fi 
        done
    done
fi