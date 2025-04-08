from stable_baselines3.common.policies import BasePolicy
import torch
import torch.nn as nn
import numpy as np

class ThresholdPolicy(BasePolicy):
    '''
    Policy that selects migration actions based on a threshold of the controller load balancing rate.
    '''
    def __init__(self, observation_space, action_space, threshold, capacities):
        super(ThresholdPolicy, self).__init__(observation_space, action_space)
        self.threshold = threshold
        self.capacities = torch.tensor(capacities)

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

        m = obs.size(c_dim)
        n = obs.size(s_dim)

        # Compute the load ratios by dividing by capacities
        L = obs.sum(s_dim)
        L = L.squeeze() # compress the unnecessary dimensions
        # print("L shape: ", L.shape)
        # print("L: ", L)


        B = L / self.capacities

        # print("B shape: ", B.shape)
        # print("B: ", B)

        # Sort the CLR (controller load ratios) of each controller, and track the original indices
        sorted_controllers = torch.argsort(B)

        # print("sorted_controllers: ", sorted_controllers)

        assert len(self.capacities) >= 2  # if we have less than 2 controllers, it doesn't make sense to migrate them.
        most_overloaded = sorted_controllers[-1]  # these are indexed starting at 0.
        least_loaded = sorted_controllers[0]

        # print("most_overloaded: ", most_overloaded)
        # print("least_loaded: ", least_loaded)

        # Migration decision logic
        clr_threshold = self.threshold
        if clr_threshold < 0: # then we are going to dynamically pick it.
            clr_threshold = (B[most_overloaded] + B[least_loaded])/2.0

        if B[most_overloaded] > clr_threshold and B[least_loaded] < clr_threshold:
            # Select the switch from the overloaded controller with the highest traffic
            switch_loads = obs[0][0][most_overloaded]  # Load per switch for the overloaded controller
            switch_to_migrate = torch.argmax(switch_loads)  # Switch with highest traffic

            # print(f"Switch {switch_to_migrate.item() + 1} selected for migration to Controller {least_loaded.item() + 1}")

            # Compute the action (index of the switch to migrate)
            a = switch_to_migrate.item() + least_loaded.item() * n
            return torch.tensor([a], dtype=torch.int32)
        else:
            # print("No migration needed.") 
            max_idx = torch.argmax(obs)
            # max_idx = torch.argmax(matrix)  # Find index of max value in flattened 2D matrix
            row, col = divmod(max_idx.item(), obs.shape[s_dim])  # Convert to (row, col) indices

            # print("row: ", row, ", col: ", col)
            # print([row*obs.shape[s_dim] + col])
            
            return torch.tensor([row*obs.shape[s_dim] + col], dtype=torch.int32)
        