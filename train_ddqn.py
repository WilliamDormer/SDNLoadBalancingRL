from model_classes.DDQN import main
import gymnasium as gym
from gymnasium import spaces
import torch
import requests
from stable_baselines3.common.evaluation import evaluate_policy
import argparse
import os
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train DDQN on NetworkSim-v0")
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
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the model")
    parser.add_argument("--normalize", type=bool, required=True, help="whether to normalize observation states to the range 0 to 1" )

    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--learning_rate", default=0.001 ,type=float)
    parser.add_argument("--min_episodes", default=20, type=int)
    parser.add_argument("--eps", default=1, type=int)
    parser.add_argument("--eps_decay", default=0.998, type=float)
    parser.add_argument("--eps_min", default=0.01, type=float)
    parser.add_argument("--update_step", default=10, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--update_repeats", default=50, type=int)
    parser.add_argument("--num_episodes", default=3000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max_memory_size", default=5000, type=int)
    parser.add_argument("--lr_gamma", default=1, type=int)
    parser.add_argument("--lr_step", default=100, type=int)
    parser.add_argument("--measure_step", default=100, type=int)
    parser.add_argument("--measure_repeats", default=100, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--cnn", default=True, type=bool)
    parser.add_argument("--horizon", default=np.inf, type=float)
    parser.add_argument("--render", default=False, type=bool)
    parser.add_argument("--render_step", default=50, type=int)
    args = parser.parse_args()

    # save all arguments into a file for reference. 
    param_log_path = os.path.splitext(args.save_path)[0] + "_params.txt"

    # Save all arguments to the parameter log file
    with open(param_log_path, "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}:{value}\n")

    gamma = args.gamma
    learning_rate = args.learning_rate
    min_episodes = args.min_episodes
    eps = args.eps
    eps_decay = args.eps_decay
    eps_min=args.eps_min
    update_step=args.update_step
    batch_size= args.batch_size
    update_repeats=args.update_repeats
    num_episodes=args.num_episodes
    seed = args.seed
    max_memory_size = args.max_memory_size
    lr_gamma = args.lr_gamma
    lr_step = args.lr_step
    measure_step = args.measure_step
    measure_repeats= args.measure_repeats
    hidden_dim = args.hidden_dim
    cnn = args.cnn
    horizon = args.horizon
    render = args.render
    render_step = args.render_step

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
    save_path = args.save_path
    normalize = args.normalize

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
                normalize = normalize
            )
    )

    env = IterationLimitWrapper(env, max_steps=episode_length)
    # don't use horizon, since you need the iteration limit for the evaluation function. 
    model, performance = main(
        env=env,
        gamma=gamma,
        lr=learning_rate,
        min_episodes=min_episodes,
        eps=eps,
        eps_decay=eps_decay,
        eps_min = eps_min,
        update_step = update_step,
        batch_size = batch_size,
        update_repeats = update_repeats,
        num_episodes = num_episodes,
        seed = seed,
        max_memory_size = max_memory_size,
        lr_gamma = lr_gamma,
        lr_step = lr_step,
        measure_step = measure_step,
        measure_repeats = measure_repeats,
        hidden_dim = hidden_dim,
        cnn = cnn,
        horizon = horizon,
        render=render,
        render_step = render_step,
        num_controllers = num_controllers,
        num_switches = num_switches
    )
    print("performance: ", performance)
    # save the model
    torch.save(model.state_dict(), save_path)
    # evaluate against the baselines

    

