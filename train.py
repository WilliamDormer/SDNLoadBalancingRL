# this is the example of using a DQN provided by stable baselines3
# we need to modify this for our use with a custom environment for our network simulator

# TODO we also need to evaluate if this will fit with our slower, more complex simulator.

import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

import network_env

env = gym.make(
    'network_env/NetworkSim-v0',
    render_mode="human",
    num_controllers = 4,
    num_switches = 26,
    max_rate = 5000,
    gc_ip = "192.168.56.101",
    gc_port = "8000",
    step_time = 5,
    )
# env = gym.make("CartPole-v1", render_mode="human")

# instantiate the agent, using DQN from paper "Playing Atari with Deep Learning"

# they say there are different options for policy here, including (MlpPolicy, and CnnPolicy), maybe more. I'll use the later because that's similar to what the paper used. 
model = DQN("MlpPolicy", env, verbose=1) 
# this is actually training the agent and displaying the progress bar while doing so. 
model.learn(total_timesteps=4, log_interval=4, progress_bar=True)
model.save("dqn_network")

del model # remove to demonstrate saving and loading

# this is just showing how to use the trained agent. 
model = DQN.load("dqn_network", env=env)

# evaluating the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1)

obs, info = env.reset()
print("got here")
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()