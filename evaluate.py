# this is the example of using a DQN provided by stable baselines3
# we need to modify this for our use with a custom environment for our network simulator

# TODO we also need to evaluate if this will fit with our slower, more complex simulator.

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import network_env
import json

from model_classes.static_model import StaticPolicy
# from model_classes.threshold_model import ThresholdPolicy
from model_classes.random_model import RandomPolicy
from model_classes.threshold_model import ThresholdPolicy

import requests

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-mf", "--model_file", help="the model to evaluate, can be 'static', 'threshol', 'random', or 'DQN'", required=True)
parser.add_argument("-me", "--model_to_evaluate", required=True)
parser.add_argument("-s", "--steps", help="The number of iterations to test. ", required=True, type=int)
parser.add_argument("-rf", "--reward_function", help="The reward function to use.", required=True)
parser.add_argument("-t", "--threshold", type=float, default = 0.5, help="The threshold for the threshold based evaluation")
parser.add_argument("-r", "--render_mode", default=None, help="The render mode, either None or human")
parser.add_argument("-c", "--num_controllers", type=int, default=4)
parser.add_argument("-ns", "--num_switches", type=int, default=26)
args = parser.parse_args()

model_file = args.model_file
model_to_evaluate = args.model_to_evaluate
max_steps = args.steps
reward_function = args.reward_function
threshold = args.threshold
render_mode = args.render_mode
num_controllers = args.num_controllers
num_switches = args.num_switches


# model_to_evaluate = "static" # "static", "threshold", "DQN", "random"

# all policies assume that actions space starts at 0, because that makes more sense, then this adjusts it for what the network sim expects.
class ActionOffsetWrapper(gym.Wrapper): # this is needed because our action space starts at 1, but DQN only supports starting at 0.
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(env.action_space.n)
    
    def step(self, action):
        return self.env.step(action + 1) # shift actions.

gc_ip = "127.0.0.1"
gc_port = "8000"

env = None
if model_to_evaluate != "PPO":
    env = gym.make(
        'network_env/NetworkSim-v0',
        render_mode=render_mode, # can set this to human if you want to capture the movements for a video or something.
        num_controllers = num_controllers,
        num_switches = num_switches,
        max_rate = 5000,
        # gc_ip = "192.168.56.101",
        gc_ip = gc_ip, # for fake_sim
        gc_port = gc_port,
        step_time = 0.5, # for the fake_sim
        fast_mode = True,
        reward_function= reward_function
    )
    # configure the environment
    env = ActionOffsetWrapper(env)
else: 
    env = gym.make(
    'network_env/NetworkSim-v0',
    # render_mode="human",
    render_mode = None, # no rendering. Way faster.
    num_controllers = num_controllers,
    num_switches = num_switches,
    max_rate = 5000,
    # gc_ip = "192.168.56.101",
    gc_ip = "127.0.0.1", # for fake_sim
    gc_port = "8000",
    step_time = 0.5, # for the fake_sim, when running in non fast-mode
    fast_mode = True, # for no delays, it uses a simulated delay
    alternate_action_space = True,
    reward_function = "explore"
    )

url = f"http://{gc_ip}:{gc_port}/" + "capacities"
response = requests.get(url)
capacities = None
if response.status_code == 200:
    # get the state and return it
    json_data = response.json()  # Convert response to a dictionary
    capacities = json_data["data"]  # Convert list back to NumPy array
    response.close()
else:
    raise Exception("Failed to retreive capacities from network.")

print("capacities: ", capacities)


# used to allow the simulation to terminate for evaluation
class IterationLimitWrapper(gym.Wrapper):
    def __init__(self, env, max_steps=100):
        super().__init__(env)
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self, **kwargs):
        self.current_step = 0  # Reset step counter
        # print("calling reset from iteration wrapper")
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, truncated, done, info = self.env.step(action)
        self.current_step += 1

        # Force done when max_steps is reached
        if self.current_step >= self.max_steps:
            done = True  # Manually set done
            truncated = True

        return observation, reward, done, truncated, info
    
env = IterationLimitWrapper(env, max_steps=max_steps)

print("in evaluate calling reset")
env.reset() # reset the environment before training.

model = None

if model_to_evaluate == "static":
    model = StaticPolicy(env.action_space, env.observation_space)
elif model_to_evaluate == "threshold":
    model = ThresholdPolicy(env.action_space, env.observation_space, threshold, capacities=capacities)
    pass
elif model_to_evaluate == "random":
    model = RandomPolicy(env.action_space, env.observation_space)
    pass
elif model_to_evaluate == "DQN":
    # default to the DQN model.
    # model = DQN(
    #     "MlpPolicy",
    #     env, 
    #     verbose=1, 
    #     exploration_fraction = 0, # not sure this is correct, but i don't want it exploring, choosing optimal
    # )
    model = DQN.load(model_file, env=env)
elif model_to_evaluate == "PPO":
    model = PPO.load(model_file, env=env)
elif model_to_evaluate == "DQN_CNN":
    # TODO figure out why I commented this out? do you need this to run?
    # policy_kwargs = dict(
    #     features_extractor_class=DQN_CNN,
    #     features_extractor_kwargs=dict(features_dim=128),
    # )
    # model = DQN(
    #     "MlpPolicy",
    #     env, 
    #     verbose=1, 
    #     exploration_fraction = 0, # not sure this is correct, but i don't want it exploring, choosing optimal
    #     policy_kwargs=policy_kwargs
    # )
    model = DQN.load(model_file, env=env)
elif model_to_evaluate == "PPO":
    model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=2048 # the size of a batch
)
else: 
    raise ValueError("Didn't provide a valid value for model")


# evaluating the agent
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
print("mean reward: ", mean_reward)
print("std_reward: ", std_reward)

# obs, info = env.reset()
# print("got here")
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()