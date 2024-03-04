#!/bin/bash
python run_hw3_ql.py env.exp_name=q5_ddpg_hard_up500_lr0.001  alg.rl_alg=ddpg env.env_name=HalfCheetah-v2 env.atari=false alg.num_agent_train_steps_per_iter=5 alg.learning_rate=0.00001 alg.double_q=false


# Training with best hyperparameters on HalCheetah-v2
