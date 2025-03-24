'''
This is a simple threshold based model using the environment. It's designed to run 
based on simple rules instead of a complex RL algorithm for choosing the migration actions
This is intended to act as a baseline for comparision. 

it uses a simple threshold based on the controller load ratio. 
'''

import gymnasium as gym
import network_env
import sys
import numpy as np
import time

print("running script")

# Initialize the environment
env = gym.make(
    'network_env/NetworkSim-v0',
    render_mode="human",
    num_controllers=4,
    num_switches=26,
    max_rate=5000,
    # gc_ip="192.168.56.101", # for network sim
    gc_ip = "127.0.0.1", # for fake_sim
    gc_port="8000",
    step_time=1, # from experience, needs to be longer than 5. It takes more than 5 just for the switch to migrate properly.
)

if hasattr(env, "unwrapped"): #allows us to access the get_capacities method.
    env = env.unwrapped


capacities = env.get_capacities()
clr_threshold = 0.7 # the controller load ratio threshold. # seems like 0.35 is about what you'll get in terms of max load ratio in a short run of the network. 

print("capacities: ", capacities)

# first it polls the network for the capacities of each of the controllers
# then in the step function it returns the state matrix
# use that to compute the controller load and the controller load ratio
# then make decision based on threshold from that. 
# it attempts to move the most loaded controller to the least loaded controller, and so on. 

# Simple threshold-based agent
def simple_threshold_agent(state):
    """
    A simple threshold-based agent that makes decisions based on the state.
    This example assumes that 'state' contains certain metrics to make decisions on.
    Adjust these conditions based on the actual state representation of the environment.
    """
    
    # print("state: ", state)

    m = len(state) # the number of controllers
    n = len(state[0]) # the number of switches

    # use the state to compute the controller load ratios: 
    # print("state: ", state)
    # print("type of state: ", type(state))

    # compute the load ratios
    L = np.sum(state, axis=1)
    # compute Bh(t) by dividing each by uh (capacities)
    B = L / capacities
    # print("load_ratios: ", B)

    # sort the CLR of each controller (keeping track of the original controller indices)
    sorted_controllers = np.argsort(B)

    assert len(capacities) >= 2 # if we have less than 2 controllers, it doesn't make sense to migrate them.
    most_overloaded = sorted_controllers[-1] # these are indexed starting at 0.
    least_loaded = sorted_controllers[0] 

    print(f"Most Overloaded Controller: {most_overloaded+1} (Load Ratio: {B[most_overloaded]})")
    print(f"Least Loaded Controller: {least_loaded+1} (Load Ratio: {B[least_loaded]})")

    # if the largest CLR is above threshold, and the lowest is below the threshold, then that's the migration target.
    if B[most_overloaded] > clr_threshold and B[least_loaded] < clr_threshold:
        # Select the switch from the overloaded controller with the highest traffic
        switch_loads = state[most_overloaded]  # Load per switch for the overloaded controller
        switch_to_migrate = np.argmax(switch_loads)  # Switch with highest traffic

        print(f"Switch {switch_to_migrate+1} selected for migration to Controller {least_loaded+1}")

        # following the convention in the paper. 
        # w = least_loaded
        # e = switch_to_migrate

        # compute the action
        a = (switch_to_migrate + 1) + least_loaded * n

        return a

    print("No migration needed.")
    # find a non-migration action: 
    # max_index = np.unravel_index(np.argmax(state), state.shape)

    # print("Max value:", state[max_index])
    # print("Max index (x, y):", max_index)
    # a =  int(max_index[0]) * m + int(max_index[1])

    # ask the system for the current ownership of the switches
    # pick the first row that's not empty

    print("env.switches_by_controller: ", env.switches_by_controller)
    a = None
    for i in range(len(env.switches_by_controller)):
        if len(env.switches_by_controller[i]) != 0:
            # then we can find a non-migration action here
            # a = i * m + (env.switches_by_controller[i][0] - 1) # because the switches stored here are 1 indexed
            first_switch = env.switches_by_controller[i][0]
            a = i * n + first_switch
            break

    # which gives a non-migration action in the range 0 to nxm - 1
    a += 1 # to get into the 1 - mxn range. 
    return a  # No migration required

# Main loop for interacting with the environment
num_episodes = 1  # Number of episodes to run
iterations_per_episode = 10

for episode in range(num_episodes):
    # state, info = env.reset()
    state = env._get_obs()
    env._get_switches_by_controller()
    print("Waiting 2 seconds to get controller switch mapping to initialize.")
    time.sleep(2)
    done = False
    total_reward = 0
    
    for i in range(iterations_per_episode):
        # Get the action from the threshold-based agent
        action = simple_threshold_agent(state)

        print("action: ", action)
        
        # Take the action in the environment
        state, reward, done, truncated, info = env.step(action)
        
        # Update the total reward
        total_reward += reward
        print("reward: ", reward)
        print("total_reward: ", total_reward)
        
        # Optionally render the environment
        env.render()
    
    print(f"Episode {episode + 1} finished with total reward: {total_reward}")
    
# Close the environment after all episodes
env.close()