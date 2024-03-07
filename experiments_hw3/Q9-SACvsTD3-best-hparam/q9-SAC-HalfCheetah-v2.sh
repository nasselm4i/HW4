#!/bin/bash

# Set variables - Best Param
entropy_coeff=0.2
n_iter=500000
# Construct exp_name dynamically
exp_name="q9_sac_hard_entropy_${entropy_coeff}"

# Use variables in the command
python run_hw3_ql.py env.exp_name=${exp_name} alg.rl_alg=sac env.env_name=HalfCheetah-v2 alg.entropy_coeff=$entropy_coeff env.atari=False  alg.n_iter=$n_iter

# Evaluate SAC compared to TD3 using best params

echo -e '\a'