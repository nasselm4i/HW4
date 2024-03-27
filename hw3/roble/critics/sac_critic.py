from .ddpg_critic import DDPGCritic
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn
import copy
import numpy as np

from hw1.roble.infrastructure import pytorch_util as ptu
from hw3.roble.policies.MLP_policy import ConcatMLP


class SACCritic(DDPGCritic):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, actor, **kwargs):
        super().__init__(actor,  **kwargs)
        
    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """
            Update the parameters of the critic.
            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories
            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                ac_na: length: sum_of_path_lengths. The action taken at the current step.
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end
            returns:
                nothing
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n).float()

        # Compute the current Q-values
        qa_t_values = self._q_net(ob_no, ac_na)
        
        # Generate actions and log probabilities for next observations using the actor policy
        # Note: Since `MLPPolicyStochastic` handles tensor conversion internally within `get_action`, it's directly used here.
        next_actions, log_prob_next = self._actor.get_action_log_prob(next_ob_no)  # Assuming this method exists or similar implementation
        next_actions = ptu.from_numpy(next_actions)
        
        # Compute the Q-values for next states and actions using the target Q network
        qa_tp1_values = self._q_net_target(next_ob_no, next_actions)
        
        # Entropy bonus calculation needs the log probability of next actions, 
        # which should ideally be provided by the policy itself.
        entropy_bonus = self._actor.entropy_coeff * log_prob_next

<<<<<<< HEAD
        # Compute the target values incorporating the entropy bonus
        targets = reward_n + self._gamma * (1 - terminal_n) * (qa_tp1_values - entropy_bonus).detach()
=======
        # TODO add the entropy term to the Q-values
        ## Hint: you will need the use the lob_prob function from the distribution of the actor policy
        ## Hint: use the self.hparams['alg']['sac_entropy_coeff'] value for the entropy term
        qa_tp1_values_reg = TODO

        # TODO compute targets for minimizing Bellman error
        # HINT: as you saw in lecture, this would be:
            #currentReward + self._gamma * qValuesOfNextTimestep * (not terminal)
        target = TODO
        target = target.detach()

        loss = self._loss(q_t_values, target)
>>>>>>> upstream/main

        # Compute the loss and update the critic network
        loss =self._loss(qa_t_values, targets)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
<<<<<<< HEAD

=======
>>>>>>> upstream/main
        return {
            'Training Loss': ptu.to_numpy(loss),
        }

    def update_target_network(self):
        pass

