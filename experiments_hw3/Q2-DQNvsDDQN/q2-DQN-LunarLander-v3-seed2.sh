#!/bin/bash
n_iter=

python run_hw3_ql.py env.env_name=LunarLander-v3 env.exp_name=q2_dqn_2 logging.random_seed=2 alg.double_q=False alg.n_iter=$n_iter


# Running seed 2

echo -e '\a'