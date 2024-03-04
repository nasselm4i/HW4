#!/bin/bash
# python run_hw3_ql.py env.env_name=MsPacman-v0 env.exp_name=q1 alg.double_q=False

# python run_hw3_ql.py env.env_name=LunarLander-v3 env.exp_name=q1 alg.double_q=False

python run_hw3_ql.py env.env_name=MsPacman-v0 env.exp_name=q1 alg.double_q=False alg.use_gpu=True