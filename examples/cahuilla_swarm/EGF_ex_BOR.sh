#!/bin/bash

datadir=./data
gemdir=$1

# Declare arrays
declare -a sta=(BOR)
declare -a numegf=(4)

arraylength=${#sta[@]}

for (( i=0; i<${arraylength}; i++ ));
do
    outdir=./out/${sta[$i]}/
    mkdir -p $outdir
    
    python3 $gemdir/GEMDeconvEgf.py \
        -dir $outdir \
        -trc0 $datadir/${sta[$i]}_trc_P_keep.mseed \
         -egf0 $datadir/${sta[$i]}_multi_gf_P_keep.mseed \
         --num_egf ${numegf[$i]} \
         --num_epochs 9 \
         --num_subepochsE 100 \
         --num_subepochsM 40 \
         --EMFull \
         --output \
         --seqfrac 15 \
         --data_sigma 1e-6 \
         --Elr 3e-3 \
         --Mlr 7e-5 \
         --stf_dur 2 \
         --egf_multi_weight 2e1 
done
