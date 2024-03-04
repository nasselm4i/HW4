#!/bin/bash

# Define experiments with their configurations
declare -a experiments=(
    "q1-DQN-MsPacman-v0:env.env_name=MsPacman-v0 env.exp_name=q1"
    "q2-DDQN-LunarLander-v3-seed1:env.env_name=LunarLander-v3 env.exp_name=q2_dqn_1 logging.seed=1"
    "q2-DDQN-LunarLander-v3-seed2:env.env_name=LunarLander-v3 env.exp_name=q2_dqn_2 logging.seed=2"
    "q2-DDQN-LunarLander-v3-seed3:env.env_name=LunarLander-v3 env.exp_name=q2_dqn_3 logging.seed=3"
    "q2-DoubleDQN-LunarLander-v3-seed1:env.env_name=LunarLander-v3 env.exp_name=q2_doubledqn_1 alg.double_q=true logging.seed=1"
    "q2-DoubleDQN-LunarLander-v3-seed2:env.env_name=LunarLander-v3 env.exp_name=q2_doubledqn_2 alg.double_q=true logging.seed=2"
    "q2-DoubleDQN-LunarLander-v3-seed3:env.env_name=LunarLander-v3 env.exp_name=q2_doubledqn_3 alg.double_q=true logging.seed=3"
    # To modify the hyperparameters, add them to the experiment configuration
    "q3-DoubleDQN-LunarLander-v3-hparam1:env.env_name=LunarLander-v3 env.exp_name=q3_hparam1"
    "q3-DoubleDQN-LunarLander-v3-hparam2:env.env_name=LunarLander-v3 env.exp_name=q3_hparam2"
    "q3-DoubleDQN-LunarLander-v3-hparam3:env.env_name=LunarLander-v3 env.exp_name=q3_hparam3"

    # Try different learning rates

    "q4-DDPG-InvertedPendulum-v2:env.exp_name=q4_ddpg_up1_lr0.001 alg.rl_alg=ddpg env.env_name=InvertedPendulum-v2 env.atari=false alg.num_critic_updates_per_agent_update=1 alg.learning_rate=0.001" # 1e-3
    "q4-DDPG-InvertedPendulum-v2:env.exp_name=q4_ddpg_up1_lr0.0001 alg.rl_alg=ddpg env.env_name=InvertedPendulum-v2 env.atari=false alg.num_critic_updates_per_agent_update=1 alg.learning_rate=0.0001" # 1e-4
    "q4-DDPG-InvertedPendulum-v2:env.exp_name=q4_ddpg_up1_lr0.00001 alg.rl_alg=ddpg env.env_name=InvertedPendulum-v2 env.atari=false alg.num_critic_updates_per_agent_update=1 alg.learning_rate=0.00001" # 1e-5

    # Try Different update frequencies for training the policies
    "q4-DDPG-InvertedPendulum-v2:env.exp_name=q4_ddpg_up1_lr0.001 alg.rl_alg=ddpg env.env_name=InvertedPendulum-v2 env.atari=false alg.num_critic_updates_per_agent_update=1 alg.learning_rate=0.001" # 1 update
    "q4-DDPG-InvertedPendulum-v2:env.exp_name=q4_ddpg_up2_lr0.001 alg.rl_alg=ddpg env.env_name=InvertedPendulum-v2 env.atari=false alg.num_critic_updates_per_agent_update=2 alg.learning_rate=0.001" # 2 updates
    "q4-DDPG-InvertedPendulum-v2:env.exp_name=q4_ddpg_up4_lr0.001 alg.rl_alg=ddpg env.env_name=InvertedPendulum-v2 env.atari=false alg.num_critic_updates_per_agent_update=4 alg.learning_rate=0.001" # 4 updates

    # After you have completed the parameter tuning on the simpler InvertedPendulum-v2 environment, use those parameters to train a model on the more difficult HalfCheetah-v2 environment.

    # "q5-DDPG-HalfCheetah-v2:env.exp_name=q5_ddpg_hard alg.rl_alg=ddpg env.env_name=HalfCheetah-v2 env.atari=false"

    #Question 6: TD3 tuning
    # Again, the hyperparameters for this new algorithm need to be tuned as well using InvertedPendulum-v2. Try different values for the noise being added to the target policy ρ when computing the target values. Also try different Q-Function network structures. Start with trying different values for ρ.
    
    # Noise 0.05 env.exp_name=q6_td3_shape<s>_rho<r> 

    "q6-TD3-InvertedPendulum-v2:env.exp_name=q6_td3_shape1_rho0.05 alg.rl_alg=td3 env.env_name=InvertedPendulum-v2 env.atari=false alg.noise=0.05" # 1 update

    # Noise 0.1 env.exp_name=q6_td3_shape1_rho<r>

    "q6-TD3-InvertedPendulum-v2:env.exp_name=q6_td3_shape1_rho0.1 alg.rl_alg=td3 env.env_name=InvertedPendulum-v2 env.atari=false alg.noise=0.1" # 1 update

    # Noise 0.2 env.exp_name=q6_td3_shape1_rho<r>

    "q6-TD3-InvertedPendulum-v2:env.exp_name=q6_td3_shape1_rho0.2 alg.rl_alg=td3 env.env_name=InvertedPendulum-v2 env.atari=false alg.noise=0.2" # 1 update

    # try different update frequencies for training the policies

    "q6-TD3-InvertedPendulum-v2:env.exp_name=q6_td3_shape1_rho0.05 alg.rl_alg=td3 env.env_name=InvertedPendulum-v2 env.atari=false alg.noise=0.05" # 1 update

    "q6-TD3-InvertedPendulum-v2:env.exp_name=q6_td3_shape2_rho0.05 alg.rl_alg=td3 env.env_name=InvertedPendulum-v2 env.atari=false alg.noise=0.05 alg.policy_update_freq=2" # 2 updates

    "q6-TD3-InvertedPendulum-v2:env.exp_name=q6_td3_shape4_rho0.05 alg.rl_alg=td3 env.env_name=InvertedPendulum-v2 env.atari=false alg.noise=0.05 alg.policy_update_freq=4" # 4 updates

    ########################################

    "q7-TD3-HalfCheetah-v2:env.exp_name=q7_td3_hard alg.rl_alg=td3 env.env_name=HalfCheetah-v2 env.atari=false"

    #Question 8: SAC tuning
    # The hyperparameters for this new algorithm need to be tuned as well using InvertedPendulum-v2. Try different values for the entropy coefficient α. Start with trying different values for α. Bonus: Add linear annealing to alpha (the entropy coefficient).

    # Alpha 0.1 env.exp_name=q8_sac_alpha0.1

    "q8-SAC-InvertedPendulum-v2:env.exp_name=q8_sac_alpha0.1 alg.rl_alg=sac env.env_name=InvertedPendulum-v2 env.atari=false alg.alpha=0.1" # 1 update

    # Alpha 0.2 env.exp_name=q8_sac_alpha0.2

    "q8-SAC-InvertedPendulum-v2:env.exp_name=q8_sac_alpha0.2 alg.rl_alg=sac env.env_name=InvertedPendulum-v2 env.atari=false alg.alpha=0.2" # 1 update

    # Alpha 0.3 env.exp_name=q8_sac_alpha0.3

    "q8-SAC-InvertedPendulum-v2:env.exp_name=q8_sac_alpha0.3 alg.rl_alg=sac env.env_name=InvertedPendulum-v2 env.atari=false alg.alpha=0.3" # 1 update

    # evaluate SAC compared to TD3. Using the best parameter setting from Q6 train SAC on the more difficult environment used for Q5.

    "q9-SAC-HalfCheetah-v2:env.exp_name=q9_sac_hard alg.rl_alg=sac env.env_name=HalfCheetah-v2 env.atari=false"
)

# Generate a .sh file for each experiment
for exp in "${experiments[@]}" ; do
    exp_name="${exp%%:*}"
    exp_config="${exp##*:}"
    script_name="${exp_name// /-}.sh"
    
    echo "Generating script: $script_name"
    
    # Create the .sh file
    echo "#!/bin/bash" > "$script_name"
    echo "python run_hw3_ql.py $exp_config" >> "$script_name"
    
    # Make the script executable
    chmod +x "$script_name"
done

echo "All experiment scripts have been generated."
