#!/bin/bash
n_iter=

python run_hw3_ql.py env.env_name=LunarLander-v3 env.exp_name=q2_doubledqn_2 alg.double_q=True logging.random_seed=2 alg.n_iter=$n_iter


# Running seed 2

echo -e '\a'