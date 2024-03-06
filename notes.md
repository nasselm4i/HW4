Expected eval_return_average for each of the following environments:


MsPacman-v0: 1500 (or 40 when clipped) in 1M iterations
kwargs = {
            'learning_starts': 50000,
            'target_update_freq': 10000,
            'replay_buffer_size': int(1e5),
            'num_timesteps': int(1e6),
            'q_func': create_atari_q_network,
            'learning_freq': 4,
            'grad_norm_clipping': 10,
            'input_shape': (84, 84, 4),
            'env_wrappers': wrap_deepmind,
            'frame_history_len': 4,
            'gamma': 0.99,
        }
LunarLander: 150 reward after 350k timesteps
kwargs = {
            'optimizer_spec': lander_optimizer(),
            'q_func': create_lander_q_network,
            'replay_buffer_size': 50000,
            'batch_size': 32,
            'gamma': 1.00,
            'learning_starts': 1000,
            'learning_freq': 1,
            'frame_history_len': 1,
            'target_update_freq': 3000,
            'grad_norm_clipping': 10,
            'lander': True,
            'num_timesteps': 500000,
            'env_wrappers': lunar_empty_wrapper
        }

InvertedPendulum: 200



HalfCheetah-v2: 150

Keeping up with the latest research often means adding new types of programming questions to the assignments. If you find issues with the code and provide a solution, you can receive bonus points. Also, adding features that help the class can also result in bonus points for the assignment.