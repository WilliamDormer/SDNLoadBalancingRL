import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import network_env


run_name = "test_train"
tensorboard_log = "./logs/fake_sim/PPO/"


# class ActionOffsetWrapper(gym.Wrapper): # this is needed because our action space starts at 1, but DQN only supports starting at 0.
#     def __init__(self, env):
#         super().__init__(env)
#         self.action_space = spaces.Discrete(env.action_space.n)
    
#     def step(self, action):
#         return self.env.step(action + 1) # shift actions.


# env = ActionOffsetWrapper(
#     gym.make(
#     'network_env/NetworkSim-v0',
#     # render_mode="human",
#     render_mode = None, # no rendering. Way faster.
#     num_controllers = 4,
#     num_switches = 26,
#     max_rate = 5000,
#     # gc_ip = "192.168.56.101",
#     gc_ip = "127.0.0.1", # for fake_sim
#     gc_port = "8000",
#     step_time = 0.5, # for the fake_sim, when running in non fast-mode
#     fast_mode = True, # for no delays, it uses a simulated delay
#     alternate_action_space = True,
#     reward_function = "explore"
#     )
# )


env = gym.make(
    'network_env/NetworkSim-v0',
    # render_mode="human",
    render_mode = None, # no rendering. Way faster.
    num_controllers = 4,
    num_switches = 26,
    max_rate = 5000,
    # gc_ip = "192.168.56.101",
    gc_ip = "127.0.0.1", # for fake_sim
    gc_port = "8000",
    step_time = 0.5, # for the fake_sim, when running in non fast-mode
    fast_mode = True, # for no delays, it uses a simulated delay
    alternate_action_space = True,
    reward_function = "explore"
    )

from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback): # for logging in tensorboard. 
    def __init__(self, verbose=0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        # called at each step
        # print(self.locals)
        # print("\n\n\n\n")

        if (self.num_timesteps % 5 == 0):
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


policy_kwargs = dict(
    net_arch=[256, 256, 128]  # More layers and larger sizes
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=0,
    tensorboard_log=tensorboard_log,
    n_steps=2048, # the size of a batch
    ent_coef=0.1,
    gamma = 0.2,
    gae_lambda=0.98,
    policy_kwargs=policy_kwargs,
    batch_size=2
)
model.learn( 
    total_timesteps=2048*5,
    progress_bar=True,
    callback=CustomCallback(),
    tb_log_name=run_name
)
model.save("ppo_cartpole")