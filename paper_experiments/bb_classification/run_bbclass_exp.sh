#!/bin/bash


l_values=(10 5 2 1)

#opt_names=("opsvrz" "zo_psvrg_coord" "zo_psvrg_gaus") # "zo_psvrg_sph" "zo_pspider_coord" "zo_pspider" "rspgf")
opt_names=("zo_psvrg_sph" "zo_pspider_coord" "zo_pspider" "rspgf")

if [[ -z "$1" || -z "$2" || -z "$3" || -z "$4" ]] ; then
    echo -e "\nPlease call '$0 <dataset name> <data directory> <number of inner loop iterations> <output directory>' to run this command!\n"
else
    for opt_name in "${opt_names[@]}"; do
        for l in "${l_values[@]}"; do
            if [[ (${opt_name} != 'rspgf' && ${opt_name} != "zo_pspider_coord" && ${opt_name} != "zo_psvrg_coord") || ( ${opt_name} = 'rspgf' && ${l} = 1 ) || ( ${opt_name} = 'zo_pspider_coord' && ${l} = 1 ) || ( ${opt_name} = 'zo_psvrg_coord' && ${l} = 1 ) ]]; then
                #echo "Launched ${opt_name} ${l}"
                nohup python3 bb_classification.py ${1} ${2} ${opt_name} ${l} ${3} ${4} &
            fi 
        done
    done
fi