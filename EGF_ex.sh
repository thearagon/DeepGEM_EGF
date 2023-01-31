#!/bin/bash

args=(
    # Input
    --trc /home/ragon/projet/egf/DeepGEM/EGFdata/38245496_PALA_m4_trc.npy
    --egf /home/ragon/projet/egf/DeepGEM/EGFdata/PALA_38242792_m2_trc.npy
    --stf0 /home/ragon/projet/egf/DeepGEM/EGFdata/a0_m1_rec0_stf.npy
#     --synthetics
#     --stf_true /home/ragon/projet/egf/DeepGEM/EGFdata/a0_m1_rec0_stf_true.npy # if synthetics True
#     --gf_true /home/ragon/projet/egf/DeepGEM/EGFdata/a0_m1_rec0_gf.npy  # if synthetics True
    # Kernel and STF size
    --num_layers 3 
    --stf_size 40
    --num_egf 1
    # Training & optimization
    --btsize 1024
    --num_epochs 50
    --num_subepochsE 400 
    --num_subepochsM 400 
    --EMFull 
    --x_rand
    --padding_mode zeros 
    --seqfrac 4  # < stf_size//2
    # Rates 
    --Elr 1e-5
    --Mlr 1e-3
    --data_sigma 5e-5  # weight on MSE loss with trace, def: 5e-5
    # E step priors, every weight must scale with data_sigma
    --px_init_weight 0   # weight on init STF
#     --px_weight 5e4 0. 5e3  # weight for priors: list, [boundaries = (1/sigma)/2, area=0., TV = (1/sigma)/2e1]
#     --logdet_weight 5e3   # weight on q_theta, the larger the larger the posterior uncertainty on STF, = (1/sigma)/2e1
    # M step priors
#     --phi_weight 1e-1  # weight on MSE loss with randome STF and init GF, def: 1e-1
    --prior_phi_weight 5e2  # weight on init GF, = (1/sigma)/2e3
#     --kernel_norm_weight 1e2 # + weight on TV, has to be <= prior_phi_weight, = (1/sigma)/1e3
    # Output
    --save_every 500 
    --print_every 500 
    --dir /home/ragon/projet/egf/res/38245496_37195604_PALA_stf40
    # Misc
    --dv cuda:7
#     --multidv 2
    --output
    )

python3 GEMDeconvEgf.py "${args[@]}"

# eof
