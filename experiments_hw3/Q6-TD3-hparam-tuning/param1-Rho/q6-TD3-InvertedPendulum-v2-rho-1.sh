#!/bin/bash
python run_hw3_ql.py env.exp_name=q6_td3_shapeS_rhoR alg.rl_alg=td3 env.env_name=InvertedPendulum-v2 env.atari=false

# Tuning rho