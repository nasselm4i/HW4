import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        obs = np.array(obs)
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        # print("self.critic")
        # print(self.critic)
        ## TODO return the action that maximizes the Q-value
        # at the current observation as the output
        actions = self.critic.qa_values(observation)
        action = np.argmax(actions)
        # print("LIST OF ACTIONS")
        # print(actions)
        return action