#!/bin/bash
python run_hw3_ql.py env.exp_name=q4_ddpg alg.rl_alg=ddpg env.env_name=InvertedPendulum-v2 env.atari=false

# Tuning Learning Rate