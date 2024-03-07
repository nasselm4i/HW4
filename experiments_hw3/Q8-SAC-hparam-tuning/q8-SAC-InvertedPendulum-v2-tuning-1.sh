#!/bin/bash

# Set variables
entropy_coeff=0.1
n_iter=150000
# Construct exp_name dynamically
exp_name="q8_sac_entropy_${entropy_coeff}"

# Use variables in the command
python run_hw3_ql.py env.exp_name=${exp_name} alg.rl_alg=sac env.env_name=InvertedPendulum-v2 alg.entropy_coeff=$entropy_coeff env.atari=False double_q=False alg.n_iter=$n_iter

# Tuning Hyperparameters (diff entropy coeff)

echo -e '\a'