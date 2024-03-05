#!/bin/bash

# Set variables - Best Params
num_agent_train_steps_per_iter=2
learning_rate=0.0001

# Construct exp_name dynamically
exp_name="q5_ddpg_hard_up${num_agent_train_steps_per_iter}_lr${learning_rate}"

# Use variables in the command
python run_hw3_ql.py env.exp_name=${exp_name} alg.rl_alg=ddpg env.env_name=HalfCheetah-v2 env.atari=false alg.num_agent_train_steps_per_iter=$num_agent_train_steps_per_iter alg.learning_rate=$learning_rate alg.double_q=false

# Training with best hyperparameters on HalCheetah-v2
