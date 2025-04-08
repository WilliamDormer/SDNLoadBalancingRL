from stable_baselines3.common.policies import BasePolicy
import torch
import torch.nn as nn
import numpy as np

class RandomPolicy(BasePolicy):
    def __init__(self, observation_space, action_space):
        super(RandomPolicy, self).__init__(observation_space, action_space)

    # public facing, to ensure observations are properly converted to tensors.
    def predict(self, observation, state=None, mask=None, deterministic=True, episode_start=None):
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32)
        action = self._predict(obs_tensor.unsqueeze(0), deterministic)
        # print("action: ", action)
        return action.cpu().numpy(), state
    
    # private facing, used for raw processing.
    def _predict(self, obs: torch.Tensor, deterministic: bool = False):
        # we don't respect deterministic here. 
        # print("obs.size: ", obs.shape)
        # rand_row = torch.round(torch.rand(1) * obs.size(2))
        # rand_col = torch.round(torch.rand(1) * obs.size(3))

        rand_row = torch.randint(0, obs.size(2), (1,))
        rand_col = torch.randint(0, obs.size(3), (1,))
        # print("rand_row: ", rand_row)
        # print("rand_col: ", rand_col)
        return torch.tensor([rand_row.item()*obs.size(3) + rand_col.item()], dtype=torch.int32)