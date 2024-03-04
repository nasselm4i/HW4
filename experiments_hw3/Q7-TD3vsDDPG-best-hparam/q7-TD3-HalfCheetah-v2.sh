#!/bin/bash
python run_hw3_ql.py env.exp_name=q7_td3_hard alg.rl_alg=td3 env.env_name=HalfCheetah-v2 env.atari=false

# Evaluate TD3 Compared to DDPG
# Execute TD3 HalfCheetah with the best hyperparameters from the Q6 tuning