'''
This file allows you to view the archiecture of the model.

There are two parts:
1. Feature Extractor: extracts features fro high dimensional observations, for instance a CNN that extracts features from images. feature_extractor_class.
2. Network that maps features to actions. controlled by net_arch parameter

The default network architecture used by SB3 depends on the algorithm and observation space. 

https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
'''


import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN

from model_classes.cnn_dqn_model import DQN_CNN
import network_env

class ActionOffsetWrapper(gym.Wrapper): # this is needed because our action space starts at 1, but DQN only supports starting at 0.
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(env.action_space.n)
    
    def step(self, action):
        return self.env.step(action + 1) # shift actions.

env = ActionOffsetWrapper(
    gym.make(
    'network_env/NetworkSim-v0',
    render_mode="human",
    num_controllers = 4,
    num_switches = 26,
    max_rate = 5000,
    # gc_ip = "192.168.56.101",
    gc_ip = "127.0.0.1", # for fake_sim
    gc_port = "8000",
    step_time = 0.5, # for the fake_sim
    reward_function = "paper"
    )
    )

policy_kwargs = dict(
    features_extractor_class=DQN_CNN,
    features_extractor_kwargs=dict(features_dim=128),
)

model = DQN(
    "MlpPolicy",
    env, 
    verbose=1, 
    policy_kwargs=policy_kwargs
    )

print("model architecture: \n",model.policy)