import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
from torch import nn
import random
import torch.nn.functional as F
import collections
from torch.optim.lr_scheduler import StepLR

import sys

import network_env

"""
Implementation of Double DQN for gym environments with discrete action space.
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device: ", device)

"""
The Q-Network has as input a state s and outputs the state-action values q(s,a_1), ..., q(s,a_n) for all n actions.
"""
class QNetwork(nn.Module):
    def __init__(self, action_dim, state_dim, hidden_dim):
        super(QNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, inp):

        x1 = F.leaky_relu(self.fc_1(inp))
        x1 = F.leaky_relu(self.fc_2(x1))
        x1 = self.fc_3(x1)

        return x1


"""
If the observations are images we use CNNs.
"""
class QNetworkCNN(nn.Module):
    def __init__(self, action_dim, num_controllers, num_switches):
        super(QNetworkCNN, self).__init__()

        # self.conv_1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        # self.conv_2 = nn.Conv2d(32, 64, kernel_size=4, stride=3)
        # self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.fc_1 = nn.Linear(8960, 512)
        # self.fc_2 = nn.Linear(512, action_dim)

        #TODO see if you can modify this to use appropriate kernel sizes
        # If you're using a small enough space it might not make sense to use a CNN at all. 

        self.num_switches = num_switches
        self.num_controllers = num_controllers

        # print("num_controllers: ", num_controllers)
        # print("num_switches: ", num_switches)

        min_dim = min(num_controllers, num_switches)

        # print("min_dim: ", min_dim)

        k1 = min(4, min_dim)
        # print("k1: ", k1)
        s1 = min(2, k1)

        k2 = min(4, max(1, (min_dim - k1) // s1))
        # print("k2: ", k2)
        s2 = min(2, k2)

        k3 = min(3, max(1, (num_controllers - k1 - k2) // max(1, s1 * s2)))
        # print("k3: ", k3)

        self.conv_1 = nn.Conv2d(1, 32, kernel_size=k1, stride=s1)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=k2, stride=s2)
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=k3, stride=1)
        # self.fc_1 = nn.Linear(8960, 512)
        self.fc_1 = nn.Linear(64, 512) # for 5x5
        # self.fc_1 = nn.Linear(256, 512) # for 7x7
        # self.fc_1 = nn.Linear(768, 512) # for 4 x 26 (janos)
        self.fc_2 = nn.Linear(512, action_dim)

        

    def forward(self, inp):
        # print("shape of inp: ", inp.shape)

        # if it only has size of 2, then batch size is 1
        # if it has size of 3, then the batch size is the first dimension. 

        if inp.dim() == 2:
            # unsqueeze to add a channel and batch dimension
            inp = inp.unsqueeze(0)
            inp = inp.unsqueeze(0)
        elif inp.dim() == 3:
            # set the channel and batch appropriately
            # it's passed in as batch_size x num_controllers x num_switches or something along those lines. 
            inp = inp.view((inp.shape[0], 1, self.num_controllers, self.num_switches)) # modified this to use one channel, but keep the batch_size
        else:
            raise ValueError("issue with the inp dimension in forward pass.")
        
        x1 = F.relu(self.conv_1(inp))
        x1 = F.relu(self.conv_2(x1))
        x1 = F.relu(self.conv_3(x1))
        x1 = torch.flatten(x1, 1)
        x1 = F.leaky_relu(self.fc_1(x1))
        x1 = self.fc_2(x1)

        return x1


"""
memory to save the state, action, reward sequence from the current episode. 
"""
class Memory:
    def __init__(self, len):
        self.rewards = collections.deque(maxlen=len)
        self.state = collections.deque(maxlen=len)
        self.action = collections.deque(maxlen=len)
        self.is_done = collections.deque(maxlen=len)

    def update(self, state, action, reward, done):
        # if the episode is finished we do not save to new state. Otherwise we have more states per episode than rewards
        # and actions whcih leads to a mismatch when we sample from memory.
        if not done:
            self.state.append(state)
        self.action.append(action)
        self.rewards.append(reward)
        self.is_done.append(done)

    def sample(self, batch_size):
        """
        sample "batch_size" many (state, action, reward, next state, is_done) datapoints.
        """
        n = len(self.is_done)
        idx = random.sample(range(0, n-1), batch_size)

        state = np.array(self.state)
        action = np.array(self.action)
        return torch.Tensor(state)[idx].to(device), torch.LongTensor(action)[idx].to(device), \
               torch.Tensor(state)[1+np.array(idx)].to(device), torch.Tensor(self.rewards)[idx].to(device), \
               torch.Tensor(self.is_done)[idx].to(device)

    def reset(self):
        self.rewards.clear()
        self.state.clear()
        self.action.clear()
        self.is_done.clear()


def select_action(model, env, state, eps):
    
    # print("state: ", state)
    state = torch.Tensor(state).to(device)
    # print('\nstate: ', state)
    with torch.no_grad():
        values = model(state)

    # select a random action wih probability eps
    if random.random() <= eps:
        # print("selecting random")
        action = np.random.randint(0, env.action_space.n)
    else:
        # print("selecting argmax")
        # print("values: ", values.cpu().numpy())
        # print("shape of values: ", values.cpu().numpy().shape)
        action = np.argmax(values.cpu().numpy())
    
    # print("ACTION(s): ", action)

    return action


def train(batch_size, current, target, optim, memory, gamma):

    states, actions, next_states, rewards, is_done = memory.sample(batch_size)

    q_values = current(states)

    next_q_values = current(next_states)
    next_q_state_values = target(next_states)

    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = rewards + gamma * next_q_value * (1 - is_done)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()

    optim.zero_grad()
    loss.backward()
    optim.step()


def evaluate(Qmodel, env, repeats):
    """
    Runs a greedy policy with respect to the current Q-Network for "repeats" many episodes. Returns the average
    episode reward.
    """
    Qmodel.eval()
    perform = 0
    # print("running evaluation")
    for _ in range(repeats):
        # print("starting repeat")
        state, info = env.reset()
        done = False
        while not done:
            state = torch.Tensor(state).to(device)
            with torch.no_grad():
                values = Qmodel(state)
            action = np.argmax(values.cpu().numpy())
            state, reward, done, _, info = env.step(action)
            # no need to normalize here since we don't use the state 
            # print("done: ", done) # for some reason, it's not ever getting a done signal, even though the environment has an iteration wrapper. 
            perform += reward
    Qmodel.train()
    return perform/repeats


def update_parameters(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


def main(env, gamma=0.99, lr=1e-3, min_episodes=20, eps=1, eps_decay=0.998, eps_min=0.01, update_step=10, batch_size=64, update_repeats=50,
         num_episodes=3000, seed=42, max_memory_size=5000, lr_gamma=1, lr_step=100, measure_step=100,
         measure_repeats=100, hidden_dim=64, cnn=False, horizon=np.inf, render=True, render_step=50, num_controllers = 5, num_switches = 5):
    """

    Remark: Convergence is slow. Wait until around episode 2500 to see good performance.

    :param gamma: reward discount factor
    :param lr: learning rate for the Q-Network
    :param min_episodes: we wait "min_episodes" many episodes in order to aggregate enough data before starting to train
    :param eps: probability to take a random action during training
    :param eps_decay: after every episode "eps" is multiplied by "eps_decay" to reduces exploration over time
    :param eps_min: minimal value of "eps"
    :param update_step: after "update_step" many episodes the Q-Network is trained "update_repeats" many times with a
    batch of size "batch_size" from the memory.
    :param batch_size: see above
    :param update_repeats: see above
    :param num_episodes: the number of episodes played in total
    :param seed: random seed for reproducibility
    :param max_memory_size: size of the replay memory
    :param lr_gamma: learning rate decay for the Q-Network
    :param lr_step: every "lr_step" episodes we decay the learning rate
    :param measure_step: every "measure_step" episode the performance is measured
    :param measure_repeats: the amount of episodes played in to asses performance
    :param hidden_dim: hidden dimensions for the Q_network
    :param env_name: name of the gym environment
    :param cnn: set to "True" when using environments with image observations like "Pong-v0"
    :param horizon: number of steps taken in the environment before terminating the episode (prevents very long episodes)
    :param render: if "True" renders the environment every "render_step" episodes
    :param render_step: see above
    :return: the trained Q-Network and the measured performances
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    _, _ = env.reset(seed = 42) # the newer pattern for seeding
    # env.seed(seed) # outdated seeding pattern

    # print("env.action_space: ", env.action_space)
    # print("env.action_space.n: ", env.action_space.n)

    if cnn:
        Q_1 = QNetworkCNN(action_dim=env.action_space.n, num_controllers=num_controllers, num_switches=num_switches).to(device)
        Q_2 = QNetworkCNN(action_dim=env.action_space.n, num_controllers=num_controllers, num_switches=num_switches).to(device)
    else:
        Q_1 = QNetwork(action_dim=env.action_space.n, state_dim=env.observation_space.shape[0],
                       hidden_dim=hidden_dim).to(device)
        Q_2 = QNetwork(action_dim=env.action_space.n, state_dim=env.observation_space.shape[0],
                       hidden_dim=hidden_dim).to(device)
    # transfer parameters from Q_1 to Q_2
    update_parameters(Q_1, Q_2)

    # print("Q_1: ", Q_1)
    # sys.exit(0)

    # we only train Q_1
    for param in Q_2.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(Q_1.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    memory = Memory(max_memory_size)
    performance = []

    for episode in range(num_episodes):
        # print("starting episode")
        # display the performance
        if (episode % measure_step == 0) and episode >= min_episodes:
            print("recording performance")
            performance.append([episode, evaluate(Q_1, env, measure_repeats)]) #TODO it's getting stuck here for a long time.
            print("Episode: ", episode)
            print("rewards: ", performance[-1][1])
            print("lr: ", scheduler.get_last_lr()[0])
            print("eps: ", eps)

        # print("resetting state")
        state, info = env.reset()
        # print("state before memory: ", state)
        # print("appending memory")
        memory.state.append(state)
        # print("state after memory: ", state)

        done = False
        i = 0
        # print("collecting experiences")
        while not done:
            i += 1
            # print("selecting action")
            action = select_action(Q_2, env, state, eps)
            # state, reward, done, _ = env.step(action)
            state, reward, done, _, info = env.step(action)

            if i > horizon:
                done = True

            # render the environment if render == True
            if render and episode % render_step == 0:
                env.render()

            # save state, action, reward sequence
            memory.update(state, action, reward, done)

        if episode >= min_episodes and episode % update_step == 0:
            # print("performing update")
            for _ in range(update_repeats):
                train(batch_size, Q_1, Q_2, optimizer, memory, gamma)

            # transfer new parameter from Q_1 to Q_2
            update_parameters(Q_1, Q_2)

        # update learning rate and eps
        scheduler.step()
        eps = max(eps*eps_decay, eps_min)

    return Q_1, performance


if __name__ == '__main__':
    
    # env_name='CartPole-v1'
    # env = gym.make(
    #     'CartPole-v1',
    #     render_mode = "human"
    # )

    class ActionOffsetWrapper(gym.Wrapper): # this is needed because our action space starts at 1, but DQN only supports starting at 0.
        def __init__(self, env):
            super().__init__(env)
            self.action_space = spaces.Discrete(env.action_space.n)
        
        def step(self, action):
            return self.env.step(action + 1) # shift actions.

    render_mode = None
    num_controllers = 3
    num_switches = 3
    max_rate = 5000
    gc_ip = '127.0.0.1'
    gc_port = "8000"
    step_time = 5
    fast_mode = True
    reward_function = "paper"
    episode_length = 10

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
            reward_function = reward_function
            )
    )

    env = IterationLimitWrapper(env, max_steps=episode_length)
    # don't use horizon, since you need the iteration limit for the evaluation function. 
    model, performance = main(env, cnn=True)
    print("performance: ", performance)