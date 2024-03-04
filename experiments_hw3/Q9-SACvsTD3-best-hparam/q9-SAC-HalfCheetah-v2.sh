#!/bin/bash
python run_hw3_ql.py env.exp_name=q9_sac_hard alg.rl_alg=sac env.env_name=HalfCheetah-v2 env.atari=false


# Evaluate SAC compared to TD3 using best params 