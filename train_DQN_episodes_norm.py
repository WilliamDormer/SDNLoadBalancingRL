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

import requests
import os


import sys

import argparse

parser = argparse.ArgumentParser()
# required arguments
parser.add_argument("-rf", "--reward_function", required=True, help="The reward function to use, should match one in the network environment")
parser.add_argument("-t", "--timesteps", type=int, required=True, help="The number of iterations to train")
parser.add_argument("-m", "--model_name", required=True, help="The name of the model save file.")
parser.add_argument("-nn", "--nn_type", required=True, help="What kind of NN you want, can be MLP or CNN")
parser.add_argument("-tu", "--target_update_frequency", type=int, required=True, help="the target update interval (after how many environment steps to update the target network), must be higher than the episode length.")
parser.add_argument("-c", "--num_controllers", type=int, required=True)
parser.add_argument("-s", "--num_switches", type=int, required=True)
# optional arguments
parser.add_argument("--normalize", default=True, type=bool)
parser.add_argument('-r', "--run_name", default="test", required=False)
parser.add_argument("-tl", "--tensorboard_log", default = "./logs/")
parser.add_argument("--gc_ip", default='127.0.0.1', type=str)
parser.add_argument("--gc_port", default="8000", type=str)
parser.add_argument("--max_rate", type=int, default=5000)
parser.add_argument("--step_time", type=float, default=5)
parser.add_argument("-f", "--fast_mode", type=bool, default=True)
parser.add_argument('-fe', "--final_eps", type=float, default=0.05)
parser.add_argument('-ef', "--exploration_fraction", type=float, default=0.1)
parser.add_argument("-ie", "--initial_eps", type=float, default=1)
parser.add_argument("-e", "--episode_length", type=int, default=10)
parser.add_argument("-re", "--render_mode", default=None)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--train_freq", default=4, type=int, help="Update the model every train_freq steps.")
parser.add_argument("--buffer_size", default=10000, type=int)
parser.add_argument("--gamma", type=float, default=0.2)
parser.add_argument("--tau", default=1.0, type=float)
parser.add_argument("--learning_starts", default=0, type=int)
parser.add_argument("--learning_rate", default=0.01, type=float)
parser.add_argument("--gradient_steps", default=1, type=int)
parser.add_argument("--max_grad_norm", type=float, default=10.0)
parser.add_argument("--stats_window_size", default=100, type=int)
parser.add_argument("--seed", default=None, type=int)
args = parser.parse_args()

print("args: ", args)

save_path = f"./saves/{args.model_name}"

# save all arguments into a file for reference. 
param_log_path = os.path.splitext(save_path)[0] + "_params.txt"

# Save all arguments to the parameter log file
with open(param_log_path, "w") as f:
    for arg, value in vars(args).items():
        f.write(f"{arg}:{value}\n")



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
render_mode = args.render_mode
target_update_frequency = args.target_update_frequency
normalize = args.normalize

# DQN arguments: 
learning_rate = args.learning_rate #  0.001 #0.0001 # TODO try changing this from the default to what they have in paper 0.01
model_name = args.model_name
total_timesteps = args.timesteps
# also the way they used these is completely different. Check paper. 
exploration_fraction = args.exploration_fraction
exploration_initial_eps = args.initial_eps
exploration_final_eps = args.final_eps
learning_starts=args.learning_starts
tau = args.tau
gamma = args.gamma # TODO try making this low? they used 0.2 in the paper. #when using 0.999 it actually tried to move, here it doesn't. # doesn't seem to have much impact, i tried 3 values on all sides. It does impact the loss though.
buffer_size= args.buffer_size # 1000000
nn_type = args.nn_type
batch_size = args.batch_size
train_freq = args.train_freq
gradient_steps = args.gradient_steps
max_grad_norm = args.max_grad_norm
stats_window_size = args.stats_window_size
seed = args.seed

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
    render_mode = render_mode,
    num_controllers = num_controllers,
    num_switches = num_switches,
    max_rate = max_rate,
    # gc_ip = "192.168.56.101",
    gc_ip = gc_ip, # for fake_sim
    gc_port = gc_port,
    step_time = step_time, # for the fake_sim, when running in non fast-mode
    fast_mode = fast_mode, # for no delays, it uses a simulated delay
    reward_function = reward_function,
    normalize = normalize,
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
            if "infos" in self.locals:
                info = self.locals["infos"][0]

                if "L" in info:
                    L = info["L"]
                    self.logger.record("train/L", L) # Histogram
                
                if "B" in info:
                    B = info["B"]
                    self.logger.record("train/B", B) # Histogram
                
                if "B_bar" in info:
                    B_bar = info["B_bar"]
                    self.logger.record("train/B_bar", B_bar) # scalar

                if "D_t" in info:
                    D_t = info["D_t"]
                    self.logger.record("train/D_t", D_t) # 
                
                if "D_t_diff" in info:
                    D_t_diff = info["D_t_diff"]
                    self.logger.record("train/D_t_diff", D_t_diff)

            self.logger.dump(self.num_timesteps)

        return True


# they say there are different options for policy here, including (MlpPolicy, and CnnPolicy), maybe more. I'll use the later because that's similar to what the paper used. 
model = DQN(
    "MlpPolicy",
    env, 
    learning_rate=learning_rate,
    buffer_size= buffer_size, # size of the experience buffer, default : 1,000,000
    learning_starts=learning_starts, # using something other than 0 for this this seems to have improved things?
    batch_size = batch_size,
    tau = tau, # default 1
    gamma = gamma, # default 0.99 
    train_freq = train_freq,
    gradient_steps = gradient_steps,
    target_update_interval=target_update_frequency,
    exploration_fraction = exploration_fraction, # defult 0.1 # so far 0.5 has done the best
    exploration_initial_eps = exploration_initial_eps, # default 1 # initial value of epsilon
    exploration_final_eps = exploration_final_eps, # default 0.05 # final value of epsilon after exploration_fraction timesteps.
    max_grad_norm = max_grad_norm,
    stats_window_size = stats_window_size,
    tensorboard_log=tensorboard_log,
    verbose=0, 
    seed = seed,
    ) # save the training information in a log.) 
# this is actually training the agent and displaying the progress bar while doing so. 
assert total_timesteps > episode_length

# num_episodes = total_timesteps / episode_length 

print("beginning training:")
model.learn(total_timesteps=total_timesteps,
            log_interval=1, # this logs after each run (not each iteration)
            callback= CustomCallback(), # this logs at an iteration frequency
            progress_bar=True,
            tb_log_name=run_name
            )
model.save(save_path)

print("beginning evaluation")

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

# print("capacities: ", capacities)

# evaluator = Evaluate(model, env, episode_length, -1, [12000, 10000, 10000, 12000])
evaluator = Evaluate(model, env, episode_length, -1, capacities)
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