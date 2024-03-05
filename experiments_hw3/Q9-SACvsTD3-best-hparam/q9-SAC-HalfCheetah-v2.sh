#!/bin/bash

# Set variables - Best Param
entropy_coeff=0.3

# Construct exp_name dynamically
exp_name="q9_sac_hard_entropy_${entropy_coeff}"

# Use variables in the command
python run_hw3_ql.py env.exp_name=${exp_name} alg.rl_alg=sac env.env_name=InvertedPendulum-v2 alg.entropy_coeff=$entropy_coeff env.atari=False double_q=False

# Evaluate SAC compared to TD3 using best params 