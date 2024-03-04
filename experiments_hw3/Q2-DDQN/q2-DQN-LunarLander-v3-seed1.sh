#!/bin/bash
# python run_hw3_ql.py env.env_name=LunarLander-v3 env.exp_name=q2_dqn_1 logging.seed=1

python run_hw3_ql.py env.env_name=MsPacman-v0 env.exp_name=q2_dqn_1 +logging.seed=1 alg.double_q=False

# Running seed 1