#!/bin/bash

# Set variables
td3_target_policy_noise=0.2
td3_target_policy_noise_clip=0.2
n_iter=300_000
# Construct exp_name dynamically
exp_name="q7_td3_hard_rho${td3_target_policy_noise}_up${num_agent_train_steps_per_iter}"

# Use variables in the command
python run_hw3_ql.py env.exp_name=${exp_name} alg.rl_alg=td3 env.env_name=HalfCheetah-v2 alg.td3_target_policy_noise=$td3_target_policy_noise alg.td3_target_policy_noise_clip=$td3_target_policy_noise_clip env.atari=False alg.n_iter=$n_iter

# Evaluate TD3 Compared to DDPG
# Execute TD3 HalfCheetah with the best hyperparameters from the Q6 tuning

echo -e '\a'