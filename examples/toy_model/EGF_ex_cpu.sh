#!/bin/bash
# $1 : path to DeepGEM

args=(
    # Input
    --trc0 ./ex_synth_trace.npy
    --egf0 ./ex_synth_egf.npy
    --synthetics
    --stf_true ./ex_synth_stf_true.npy # if synthetics True
    --gf_true ./ex_synth_egf_true.npy  # if synthetics True
    # Kernel and STF size
    --stf_size 40
    --num_egf 1
    # Training & optimization
    --num_epochs 15
    --num_subepochsE 70 
    --num_subepochsM 15 
    --seqfrac 4  # < stf_size//2
    # Rates 
    --Elr 1e-2
    --Mlr 1e-4
    --data_sigma 8e-7
    --stf0_sigma 5e-1
    # Output
    --save_every 10
    --print_every 5 
    --dir ./output/
    # Misc
    --output
    )

python3 $1GEMDeconvEgf.py "${args[@]}"

# eof
