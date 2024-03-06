#!/bin/bash

# Set variables
num_agent_train_steps_per_iter=2
learning_rate=0.000001

# Construct exp_name dynamically
exp_name="q4_ddpg_up${num_agent_train_steps_per_iter}_lr${learning_rate}"

# Use variables in the command
python run_hw3_ql.py env.exp_name=${exp_name} alg.rl_alg=ddpg env.env_name=InvertedPendulum-v2 alg.num_agent_train_steps_per_iter=$num_agent_train_steps_per_iter alg.learning_rate=$learning_rate env.atari=false alg.double_q=false

# Tuning Learning Rate