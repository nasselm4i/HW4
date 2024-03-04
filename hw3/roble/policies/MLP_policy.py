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
        loss = -q_values.mean()
        
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        
        return {"Loss": loss.item()}
    
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
        # TODO: sample actions from the gaussian distribrution given by MLPPolicy policy when providing the observations.
        # Hint: make sure to use the reparameterization trick to sample from the distribution
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        observation = ptu.from_numpy(observation)
        action_distribution = self(observation)
        action = action_distribution.rsample()  
        return ptu.to_numpy(action)
        
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
        return {"Loss": loss.item()}
    
#####################################################