from model_classes.DDQN import main
import gymnasium as gym
from gymnasium import spaces
import torch
import requests
from stable_baselines3.common.evaluation import evaluate_policy
from model_classes.DDQN import QNetworkCNN, evaluate

import sys

# baselines to compare against
from model_classes.static_model import StaticPolicy
from model_classes.random_model import RandomPolicy
from model_classes.threshold_model import ThresholdPolicy

import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Train DDQN on NetworkSim-v0")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--render_mode", type=str, default=None, help="Render mode for the environment")
    parser.add_argument("--num_controllers", type=int, required=True, help="Number of controllers")
    parser.add_argument("--num_switches", type=int, required=True, help="Number of switches")
    parser.add_argument("--max_rate", type=int, default=5000, help="Maximum rate")
    parser.add_argument("--gc_ip", type=str, default="127.0.0.1", help="GC IP address")
    parser.add_argument("--gc_port", type=str, default="8000", help="GC port")
    parser.add_argument("--step_time", type=int, default=5, help="Step time")
    parser.add_argument("--fast_mode", type = bool, default=True, help="Enable fast mode")
    parser.add_argument("--reward_function", type=str, default="paper", help="Reward function to use")
    parser.add_argument("--episode_length", type=int, default=10, help="Maximum episode length")
    parser.add_argument("--normalize", type=bool, required=True, help="whether to normalize observation states to the range 0 to 1" )

    parser.add_argument("--repeats", default=100, type=int)

    args = parser.parse_args()

    save_path = args.save_path
    render_mode = args.render_mode
    num_controllers = args.num_controllers
    num_switches = args.num_switches
    max_rate = args.max_rate
    gc_ip = args.gc_ip
    gc_port = args.gc_port
    step_time = args.step_time
    fast_mode = args.fast_mode
    reward_function = args.reward_function
    episode_length = args.episode_length
    normalize = args.normalize

    repeats = args.repeats

    class ActionOffsetWrapper(gym.Wrapper): # this is needed because our action space starts at 1, but DQN only supports starting at 0.
        def __init__(self, env):
            super().__init__(env)
            self.action_space = spaces.Discrete(env.action_space.n)
        
        def step(self, action):
            return self.env.step(action + 1) # shift actions.

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

            # print("current step in Iteration Wrapper: ", self.current_step)

            # Force done when max_steps is reached
            if self.current_step >= self.max_steps:
                done = True  # Manually set done
                truncated = True
                # print("end of episode")

            return observation, reward, done, truncated, info
    
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
            normalize=normalize
            )
    )

    env = IterationLimitWrapper(env, max_steps=episode_length)
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device: ", device)

    # load the model from the save. 
    model = QNetworkCNN(action_dim=env.action_space.n, num_controllers=num_controllers, num_switches=num_switches).to(device)
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()

    # run the model evaluation. 
    avg_reward = evaluate(model, env, repeats)
    print("avg model reward: ", avg_reward)
    # print("trying to close environment")
    env.close()

    # get capacities
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

    # evaluate the baselines
    threshold = -1
    n_eval_episodes = repeats
    model = StaticPolicy(env.action_space, env.observation_space)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
    print(f"from StaticPolicy: mean reward: {mean_reward}, std_reward: {std_reward}")
    model = ThresholdPolicy(env.action_space, env.observation_space, threshold, capacities=capacities)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
    print(f"from ThresholdPolicy: mean reward: {mean_reward}, std_reward: {std_reward}")
    model = RandomPolicy(env.action_space, env.observation_space)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
    print(f"from RandomPolicy: mean reward: {mean_reward}, std_reward: {std_reward}")