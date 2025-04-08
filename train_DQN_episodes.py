# this is the example of using a DQN provided by stable baselines3
# we need to modify this for our use with a custom environment for our network simulator

# TODO we also need to evaluate if this will fit with our slower, more complex simulator.
# TODO see if there's a way to get this to run instantly, instead of waiting 2 seconds for the time module to run.


import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import network_env
import json

from model_classes.cnn_dqn_model import DQN_CNN

from evaluate_class import Evaluate


import sys

import argparse

parser = argparse.ArgumentParser()
# required arguments
parser.add_argument("-rf", "--reward_function", required=True, help="The reward function to use, should match one in the network environment")
parser.add_argument("-t", "--timesteps", type=int, required=True, help="The number of iterations to train")
parser.add_argument("-m", "--model_name", required=True, help="The name of the model save file.")
parser.add_argument("-nn", "--nn_type", required=True, help="What kind of NN you want, can be MLP or CNN")
# optional arguments
parser.add_argument('-r', "--run_name", default="test", required=False)
parser.add_argument("-tl", "--tensorboard_log", default = "./logs/")
parser.add_argument("-c", "--num_controllers", type=int, required=False, default=4)
parser.add_argument("-s", "--num_switches", type=int, required=False, default=26)
parser.add_argument("--gc_ip", default='127.0.0.1', type=str)
parser.add_argument("--gc_port", default="8000", type=str)
parser.add_argument("--max_rate", type=int, default=5000)
parser.add_argument("--step_time", type=float, default=0.5)
parser.add_argument("-f", "--fast_mode", type=bool, default=True)
parser.add_argument('-fe', "--final_eps", type=float, default=0.05)
parser.add_argument('-ef', "--exploration_fraction", type=float, default=0.1)
parser.add_argument("-ie", "--initial_eps", type=float, default=1)
parser.add_argument("-e", "--episode_length", type=int, default=100)
args = parser.parse_args()

print("args: ", args)

# logging parameters
run_name = args.run_name
tensorboard_log = args.tensorboard_log

episode_length = args.episode_length

# environment arguments: 
gc_ip : str = args.gc_ip # for fake_sim
gc_port : str = args.gc_port
reward_function = args.reward_function
num_controllers = args.num_controllers
num_switches = args.num_switches
max_rate = args.max_rate
step_time = args.step_time
fast_mode = args.fast_mode

# DQN arguments: 
learning_rate = 0.01 #  0.001 #0.0001 # TODO try changing this from the default to what they have in paper 0.01
model_name = args.model_name
total_timesteps = args.timesteps

# also the way they used these is completely different. Check paper. 
exploration_fraction = args.exploration_fraction
exploration_initial_eps = args.initial_eps
exploration_final_eps = args.final_eps
learning_starts=0
tau = 1
gamma = 0.2 # TODO try making this low? they used 0.2 in the paper. #when using 0.999 it actually tried to move, here it doesn't. # doesn't seem to have much impact, i tried 3 values on all sides. It does impact the loss though.
buffer_size= 1000000
nn_type = args.nn_type

# run_name = "test_paper_reward_10000"
# tensorboard_log = "./logs/fake_sim/"
# tensorboard --logdir ./tensor_board_log/run_name/
# tensorboard --logdir ./test_train_3/
# tensorboard --logdir ./logs/fake_sim/   


class ActionOffsetWrapper(gym.Wrapper): # this is needed because our action space starts at 1, but DQN only supports starting at 0.

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(env.action_space.n)
    
    def step(self, action):
        return self.env.step(action + 1) # shift actions.

print("making env")

env = ActionOffsetWrapper(
    gym.make(
    'network_env/NetworkSim-v0',
    # render_mode="human",
    render_mode = None, # no rendering. Way faster.
    num_controllers = num_controllers,
    num_switches = num_switches,
    max_rate = max_rate,
    # gc_ip = "192.168.56.101",
    gc_ip = gc_ip, # for fake_sim
    gc_port = gc_port,
    step_time = step_time, # for the fake_sim, when running in non fast-mode
    fast_mode = fast_mode, # for no delays, it uses a simulated delay
    reward_function = reward_function
    )
)

print("env created")

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
            print("end of episode")

        return observation, reward, done, truncated, info
    
env = IterationLimitWrapper(env, max_steps=episode_length)

from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback): # for logging in tensorboard. 
    def __init__(self, verbose=0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        # called at each step
        # print(self.locals)
        # print("\n\n\n\n")

        if (self.num_timesteps % 1 == 0):
            # for custom values:
            # self.logger.record("random_value", value)

            # record the relevant metrics: # TODO fix the rest of these metrics reporting.
            if "loss" in self.locals:
                loss = self.locals["loss"]
                self.logger.record("train/loss", loss)
            if "rewards" in self.locals:
                rewards = self.locals['rewards']
                # print("length of rewards: ", len(rewards))
                self.logger.record("train/rewards", rewards[0]) #rewards is only a 1 element array.
            if "D_t" in self.locals:
                D_t = self.locals['D_t']
                self.logger.record("train/controller_load_balancing_rate", D_t)
            if "B_bar" in self.locals:
                B_bar = self.locals['B_bar']
                self.logger.record("train/average_load", B_bar)

            self.logger.dump(self.num_timesteps)

        return True


# they say there are different options for policy here, including (MlpPolicy, and CnnPolicy), maybe more. I'll use the later because that's similar to what the paper used. 
model = DQN(
    "MlpPolicy",
    env, 
    learning_rate=learning_rate,
    learning_starts=learning_starts, # using something other than 0 for this this seems to have improved things?
    tau = tau, # default 1
    verbose=0, 
    tensorboard_log=tensorboard_log,
    exploration_fraction = exploration_fraction, # defult 0.1 # so far 0.5 has done the best
    exploration_initial_eps = exploration_initial_eps, # default 1 # initial value of epsilon
    exploration_final_eps = exploration_final_eps, # default 0.05 # final value of epsilon after exploration_fraction timesteps.
    gamma = gamma, # default 0.99 
    buffer_size= buffer_size, # size of the experience buffer, default : 1,000,000
    target_update_interval=episode_length
    ) # save the training information in a log.) 
# this is actually training the agent and displaying the progress bar while doing so. 
assert total_timesteps > episode_length

print("beginning training:")
model.learn(total_timesteps=total_timesteps,
            log_interval=1, # this logs after each run (not each iteration)
            callback= CustomCallback(), # this logs at an iteration frequency
            progress_bar=True,
            tb_log_name=run_name
            )
model.save(f"./saves/{model_name}")

print("beginning evaluation")

evaluator = Evaluate(model, env, episode_length, -1, [12000, 10000, 10000, 12000])
evaluator.evaluate()

# # del model # remove to demonstrate saving and loading

# # this is just showing how to use the trained agent. 
# # model = DQN.load("dqn_network", env=env)

# # used to allow the simulation to terminate for evaluation
# class IterationLimitWrapper(gym.Wrapper):
#     def __init__(self, env, max_steps=100):
#         super().__init__(env)
#         self.max_steps = max_steps
#         self.current_step = 0

#     def reset(self, **kwargs):
#         self.current_step = 0  # Reset step counter
#         return self.env.reset(**kwargs)

#     def step(self, action):
#         observation, reward, truncated, done, info = self.env.step(action)
#         self.current_step += 1

#         # Force done when max_steps is reached
#         if self.current_step >= self.max_steps:
#             done = True  # Manually set done
#             truncated = True

#         return observation, reward, done, truncated, info
    
# wrapped_env = IterationLimitWrapper(env, max_steps=100)

# # evaluating the agent
# # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1)
# mean_reward, std_reward = evaluate_policy(model, wrapped_env, n_eval_episodes=1)
# print("mean reward: ", mean_reward)
# print("std_reward: ", std_reward)

# obs, info = env.reset()
# print("got here")
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()