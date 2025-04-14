from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import numpy as np
import gymnasium as gym
from gymnasium import Env

# baselines to compare against
from model_classes.static_model import StaticPolicy
from model_classes.random_model import RandomPolicy
from model_classes.threshold_model import ThresholdPolicy


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

class Evaluate:
    '''
    This class designed to run evaluation on the given model, to make it easier to see if training is successful.
    '''
    def __init__(self, model:BasePolicy, environment:Env, max_steps ,threshold, capacities, n_eval_episodes = 5):
        self.model = model # the model to evaluate
        # max_steps is the maximum number of steps of the environment before termination.

        # wrap the environment in an iteration limit. 
        self.env = IterationLimitWrapper(environment, max_steps=max_steps)

        # for threshold based evaluation
        self.threshold = threshold # the threshold value to use
        self.capacities = capacities # the capacities of the controllers.

        self.n_eval_episodes = n_eval_episodes # the number of evalulation episodes. Important because it makes it so that randomness doesn't influence things. 

    def evaluate(self):

        # evaluate the given model
        mean_reward, std_reward = evaluate_policy(self.model, self.env, n_eval_episodes=self.n_eval_episodes)
        print(f"from model: mean reward: {mean_reward}, std_reward: {std_reward}")

        # evaluate the baselines
        model = StaticPolicy(self.env.action_space, self.env.observation_space)
        mean_reward, std_reward = evaluate_policy(model, self.env, n_eval_episodes=1)
        print(f"from StaticPolicy: mean reward: {mean_reward}, std_reward: {std_reward}")
        model = ThresholdPolicy(self.env.action_space, self.env.observation_space, self.threshold, capacities=self.capacities)
        mean_reward, std_reward = evaluate_policy(model, self.env, n_eval_episodes=1)
        print(f"from ThresholdPolicy: mean reward: {mean_reward}, std_reward: {std_reward}")
        model = RandomPolicy(self.env.action_space, self.env.observation_space)
        mean_reward, std_reward = evaluate_policy(model, self.env, n_eval_episodes=self.n_eval_episodes)
        print(f"from RandomPolicy: mean reward: {mean_reward}, std_reward: {std_reward}")