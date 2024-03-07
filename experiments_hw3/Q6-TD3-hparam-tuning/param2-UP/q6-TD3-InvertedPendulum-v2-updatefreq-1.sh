#!/bin/bash

# Set variables
td3_target_policy_noise=0.5
td3_target_policy_noise_clip=0.5
num_agent_train_steps_per_iter=1
n_iter=150000

# Construct exp_name dynamically
exp_name="q6_td3_shapeS_rho${td3_target_policy_noise}_up${num_agent_train_steps_per_iter}"

# Use variables in the command
python run_hw3_ql.py env.exp_name=${exp_name} alg.rl_alg=td3 env.env_name=InvertedPendulum-v2 alg.td3_target_policy_noise=$td3_target_policy_noise alg.td3_target_policy_noise_clip=$td3_target_policy_noise_clip num_agent_train_steps_per_iter=$num_agent_train_steps_per_iter alg.n_iter=$n_iter env.atari=False double_q=False

# Tuning update freq

echo -e '\a'
