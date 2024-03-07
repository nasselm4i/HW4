#!/bin/bash
n_iter=

python run_hw3_ql.py env.env_name=LunarLander-v3 env.exp_name=q3_hparam2 alg.double_q=False alg.learning_rate=0.0001 alg.n_iter=$n_iter


#  Q Learning sensitivity with Learning Rate

echo -e '\a'