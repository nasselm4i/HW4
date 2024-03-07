import abc
import itertools
import numpy as np
import torch
import hw1.roble.util.class_util as classu

from hw1.roble.infrastructure import pytorch_util as ptu
from hw1.roble.policies.base_policy import BasePolicy 
from hw1.roble.policies.MLP_policy import MLPPolicy 
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import distributions

class ConcatMLP(MLPPolicy):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, dim=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self._dim)
        return super().forward(flat_inputs, **kwargs)

class MLPPolicyDeterministic(MLPPolicy):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, *args, **kwargs):
        kwargs['deterministic'] = True
        super().__init__(*args, **kwargs)
        
    def update(self, observations, q_fun):
        # TODO: update the policy and return the loss
        ## Hint you will need to use the q_fun for the loss
        ## Hint: do not update the parameters for q_fun in the loss
        observations = ptu.from_numpy(observations)
    
        # Forward pass through the policy to get the actions
        
        q_values = q_fun.qa_values(observations)
        loss = -q_values.mean() # Loss
        
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        
        return {"Actor Loss": loss.item()}
    
class MLPPolicyStochastic(MLPPolicy):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, entropy_coeff, *args, **kwargs):
        kwargs['deterministic'] = False
        super().__init__(*args, **kwargs)
        self.entropy_coeff = entropy_coeff

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if isinstance(obs, np.ndarray):
            observation = obs
            if len(obs.shape) > 1:
                observation = obs
            else:
                observation = obs[None]
            observation = ptu.from_numpy(observation)
        elif isinstance(obs, torch.Tensor):
            observation = obs
        else:
            raise TypeError("Input must be a np.ndarray or torch.Tensor")

        action = self(observation)
        if not self._deterministic:
            action = action.sample()  # Use rsample() if you need gradients to flow through this operation
        return ptu.to_numpy(action)
    
    
    def get_action_log_prob(self, obs: np.ndarray):
        """
        Generates an action for the given observation(s) and returns the action
        and its log probability under the current policy.

        Args:
            obs (np.ndarray): The observation(s) using which the action is to be generated.

        Returns:
            np.ndarray: The action(s) sampled from the policy.
            torch.Tensor: The log probability of the sampled action(s).
        """
        if isinstance(obs, np.ndarray):
            observation = obs
            if len(obs.shape) > 1:
                observation = obs
            else:
                observation = obs[None]
            observation = ptu.from_numpy(observation)
        elif isinstance(obs, torch.Tensor):
            observation = obs
        else:
            raise TypeError("Input must be a np.ndarray or torch.Tensor")

        action_distribution = self(observation)
        actions = action_distribution.sample()  # Sample actions
        log_probs = action_distribution.log_prob(actions)  # Compute log probabilities

        return ptu.to_numpy(actions), log_probs
        
    def update(self, observations, q_fun):
        # TODO: update the policy and return the loss
        ## Hint you will need to use the q_fun for the loss
        ## Hint: do not update the parameters for q_fun in the loss
        ## Hint: you will have to add the entropy term to the loss using self.entropy_coeff
        observations = ptu.from_numpy(observations)
        action_distribution = self(observations)
        actions = action_distribution.rsample()  # Parametrization trick
        log_probs = action_distribution.log_prob(actions).sum(axis=-1)
        q_values = q_fun(observations, actions)
        if q_values.shape[1] > 1:
            q_values, _ = q_values.min(dim=1)
        loss = -(q_values - self.entropy_coeff * log_probs).mean()
        
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return {"Actor Loss": loss.item()}
    
#####################################################