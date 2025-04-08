from stable_baselines3.common.policies import BasePolicy
import torch
import torch.nn as nn
import numpy as np

class StaticPolicy(BasePolicy):
    '''
    Policy that selects non-migration actions only.
    '''
    def __init__(self, observation_space, action_space):
        super(StaticPolicy, self).__init__(observation_space, action_space)

    # public facing, to ensure observations are properly converted to tensors.
    def predict(self, observation, state=None, mask=None, deterministic=True, episode_start=None):

        # simply try to find a non migration action, by finding the argmax of the observation
        # print("observation: ", observation)
        # print("type(observation): ", type(observation))
        # then convert that to switch and controller (col, row index)

        # convert to action space format

        obs_tensor = torch.as_tensor(observation, dtype=torch.float32)
        action = self._predict(obs_tensor.unsqueeze(0), deterministic)
        # print("action: ", action)
        return action.cpu().numpy(), state
    
    # private facing, used for raw processing.
    def _predict(self, obs: torch.Tensor, deterministic: bool = True):
        c_dim = 2
        s_dim = 3

        # matrix = obs.view(int(c_dim ** 0.5), -1)

        # print("obs: ", obs.shape)

        max_idx = torch.argmax(obs)
        # max_idx = torch.argmax(matrix)  # Find index of max value in flattened 2D matrix
        row, col = divmod(max_idx.item(), obs.shape[s_dim])  # Convert to (row, col) indices

        # print("row: ", row, ", col: ", col)
        # print([row*obs.shape[s_dim] + col])
        
        return torch.tensor([row*obs.shape[s_dim] + col], dtype=torch.int32)