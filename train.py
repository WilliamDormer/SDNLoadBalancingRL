# this is the example of using a DQN provided by stable baselines3
# we need to modify this for our use with a custom environment for our network simulator

# TODO we also need to evaluate if this will fit with our slower, more complex simulator.

import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

import network_env

run_name = "test_train"
tensorboard_log = "./logs/fake_sim/"
# tensorboard --logdir ./tensor_board_log/run_name/
# tensorboard --logdir ./test_train_3/

env = gym.make(
    'network_env/NetworkSim-v0',
    render_mode="human",
    num_controllers = 4,
    num_switches = 26,
    max_rate = 5000,
    # gc_ip = "192.168.56.101",
    gc_ip = "127.0.0.1", # for fake_sim
    gc_port = "8000",
    step_time = 0.5, # for the fake_sim
    )
# env = gym.make("CartPole-v1", render_mode="human")

# instantiate the agent, using DQN from paper "Playing Atari with Deep Learning"

# TODO I might have to manually call self.logger.dump(self.num_timesteps) in the callback, because the state isn't being written.
from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback): # for logging in tensorboard. 
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.num_timesteps = 1
    
    def _on_step(self) -> bool:
        # called at each step
        if (self.num_timesteps % 2 == 0):
            self.logger.dump(self.num_timesteps)
            # for custom values:
            # self.logger.record("random_value", value)
            self.num_timesteps = 0
        self.num_timesteps += 1
        return True

 
# they say there are different options for policy here, including (MlpPolicy, and CnnPolicy), maybe more. I'll use the later because that's similar to what the paper used. 
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log) # save the training information in a log.) 
# this is actually training the agent and displaying the progress bar while doing so. 
model.learn(total_timesteps=8,
            log_interval=4,
            callback= CustomCallback(),
            progress_bar=True,
            tb_log_name=run_name
            ) #TODO this is sampling outside of the sample space.. maybe because of DummyVecEnv
model.save("dqn_network")

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