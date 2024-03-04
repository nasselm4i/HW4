#!/bin/bash
python run_hw3_ql.py env.exp_name=q8_sac alg.rl_alg=sac env.env_name=InvertedPendulum-v2 env.atari=false

# Tuning Hyperparameters (diff entropy coeff)