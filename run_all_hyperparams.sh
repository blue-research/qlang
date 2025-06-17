#!/bin/bash

ANSATZ=0
TIME=$(date +"%Y%m%d_%H%M%S")

for n_layers in 0 1 2 3 4; do
    for n_single_qubit_params in 0 1 2 3 4; do
        echo "Running experiment with ansatz=$ANSATZ, n_layers=$n_layers, n_single_qubit_params=$n_single_qubit_params"
        python exp_hyperparams.py $ANSATZ $n_layers $n_single_qubit_params $TIME
    done
done
