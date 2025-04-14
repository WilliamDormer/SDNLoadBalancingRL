import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests
import time
import pygame

from .rewards import RewardGenerator

class NetworkEnv(gym.Env):
    # Human render mode will show what controllers the switches belong to in real time.
    # None will simply run with no GUI.
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, num_controllers, num_switches, max_rate, gc_ip, gc_port, reward_function, step_time=5, fast_mode=False, reset_timeout = 35, window_size=512, migration_cost = 5, render_mode=None, alternate_action_space = False, normalize=False):
        '''
        num_controllers: the number of controllers
        num_switches: the number of switches
        max_rate: used as the max value of the observation space box.
        gc_ip: a string holding the IP address of the global controller
        gc_port: a string holding the flask application port on the global controller 
        step_time: a float indicating the number of second to wait after executing a migration action to wait before reporting the reward and observations etc.
        window_size: the size of the pygame window for human rendering.
        fast_mode: Whether to use fast mode. This simulates the delay instead of timing it. Effectively sets the step_time to 0.
        migration_cost: The approximate cost to migrate a switch, represented as an int.
        '''
        self.m = num_controllers
        self.n = num_switches
        self.max_rate = max_rate
        self.gc_ip = gc_ip
        self.gc_port = gc_port
        self.step_time = step_time
        self.window_size = window_size
        self.reset_timeout = reset_timeout
        self.fast_mode = fast_mode
        self.migration_cost = migration_cost
        self.reward_function = reward_function
        self.alternate_action_space = alternate_action_space
        self.normalize = normalize

        self.gc_base_url = f"http://{gc_ip}:{self.gc_port}/"
        print("url: ", self.gc_base_url)

        self.session = requests.Session()

        print("created session")
        
        # define the observation space

        # the system state is defined by the matrix St, which is an m x n matrix, storing the message request rates.
        self.observation_space = spaces.Box(
            low=0, high=max_rate, shape=(self.m,self.n), dtype=np.float32
        )

        # define the action space, 1, 2, ... mxn
        # move switch to controller, of which n of them are non-migration actions.
        self.action_space = spaces.Discrete(n=self.m * self.n, start=1)
        if alternate_action_space:
            # print(self.m)
            # print(self.n)
            print("using other action space")
            self.action_space = spaces.MultiDiscrete([self.n+1, self.m]) # first selects a switch or nothing, second is the target controller.

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        print("about to get capacities")

        # get the capacities of each controller from the network.
        self.capacities = self.get_capacities()

        print("got capacities")

        self.switches_by_controller = self._get_switches_by_controller()

        self.rg = RewardGenerator(self.capacities, self.m)

        if reward_function == "penalize_poor_inaction":
            self.reward_function = self.rg.penalize_poor_inaction_reward
        elif reward_function == "paper":
            self.reward_function = self.rg.paper_reward
        elif reward_function == "penalize_and_encourage":
            self.reward_function = self.rg.penalize_and_encourage_reward
        elif reward_function == "binary":
            self.reward_function = self.rg.binary_reward
        elif reward_function == "explore":
            self.reward_function = self.rg.encourage_explore
        elif reward_function == "balance":
            self.reward_function = self.rg.balance
        elif reward_function == "custom":
            self.reward_function = self.rg.custom_reward
        else: 
            raise ValueError("No reward function selected")


        '''
        If human-rendering is used 'self.window' will be a reference to the window that 
        we draw to. 'self.clock' will be a clock that is used to ensure that the environmet
        is rendered at the correct framerate in human mode. They will remain "none" until human-mode
        is used for the first time
        '''
        self.window = None
        self.clock = None

        print("completed init")

    def _get_switches_by_controller(self):
        '''
        function that polls the global controller for the switch configuration. Should be called at initialization, then on a migrate action.
        '''
        try:
            data = {
            }
            url = self.gc_base_url + "switches_by_controller"
            response = self.session.get(url, json=data)
            if response.status_code == 200:
                # get the state and return it
                json_data = response.json()  # Convert response to a dictionary

                # update self.switches_by_controller
                # print("switches_by_controller: ", json_data["data"])
                response.close()
                return json_data['data'] # this will be a list of lists, where the rows are the controllers (starting 0,1,2,3), and each holds the switches it controls (indexed starting at 1)
            else:
                response.close()
                raise Exception("Failed to retreive switches by controller from network.")
        except Exception as e:
            print("there was an error in _get_switches_by_controller: ", e)


    def get_capacities(self):
        '''
        function that polls the global controller for the capacities of each controller.
        should be computed using CBench before this point. 
        '''
        # make a request to the network simulator to retrieve the state
        data = {
        }
        url = self.gc_base_url + "capacities"
        response = self.session.get(url)
        if response.status_code == 200:
            # get the state and return it
            json_data = response.json()  # Convert response to a dictionary
            array = np.array(json_data["data"])  # Convert list back to NumPy array
            response.close()
            return array
        else:
            raise Exception("Failed to retreive capacities from network.")


    def _get_obs(self):
        '''
        helper function to get an observation from the environment. 
        '''

        # make a request to the network simulator to retrieve the state
        data = {

        }
        url = self.gc_base_url + "state"
        response = self.session.get(url, json=data)
        if response.status_code == 200:
            # get the state and return it
            json_data = response.json()  # Convert response to a dictionary
            array = np.array(json_data["state"])  # Convert list back to NumPy array
            # convert it to the observation space 
            array = np.array(array, dtype=np.float32)

            if self.normalize:
                # normalize the array, need to do this in reset too.
                array = np.clip(array / self.max_rate, 0.0, 1.0)

            # print("self.normalize: ", self.normalize)
            # print("array: ", array)

            response.close()
            return array
        else:
            response.close()
            raise Exception("Failed to retreive state from network.")
        

    def _get_info(self):
        '''
        optional method for providing data that is returend by step and reset.
        '''
        info = {}
        return info

    def reset(self, seed=None, options=None):
        '''
        Method called to initiate a new episode.
        '''
        super().reset(seed=seed)

        response = self.session.post(self.gc_base_url + 'reset', timeout=self.reset_timeout)
        if response.status_code != 200:
            error = response.json()
            e = error["error"]
            raise Exception(f"Failed to reset network, error: ({e})")
        
        # print("in reset: ")
        # print(type(response))

        json_data = response.json()  # Convert response to a dictionary
        observation = np.array(json_data["state"])  # Convert list back to NumPy array
        # convert it to the observation space 
        observation = np.array(observation, dtype=np.float32)
        if self.normalize:
                # normalize the array, need to do this in reset too.
                observation = np.clip(observation / self.max_rate, 0.0, 1.0)

        response.close()

        # get the updated migrated switch positions: 
        self.switches_by_controller = self._get_switches_by_controller() 

        # reset the reward generator.
        self.rg.reset(observation)

        # print("reset called, here's the new switches by controller: ", self.switches_by_controller)

        if self.render_mode == "human":
            self._render_frame()

        return observation, self._get_info()

    def step(self, action):
        '''
        This takes the chosen action in the network (aka the switch migration decision)
        it returns: 
        observation: what the new state is
        reward: what the reward value from taking that action was
        terminated: boolean indicating whether it finished successfully
        truncated: boolean indiciating if it was cut short. 
        info: information.
        '''

        # print("called 'step' with action: ", action)

        # start_time = time.time() # just used for checking time to run function.
        
        # compute the migration action that was selected. 
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}. Must be in {self.action_space}")

        
        if self.alternate_action_space == False:
            e = int((action-1) % self.n) + 1
            w = int(np.ceil(action / self.n))

            # send the migration action to the global controller.
            
            # the action space starts at 1, so it will be the controller id for target_controller.
            data = {
                "target_controller" : w, # range 1 to m
                "switch" : e # range 1 to n
            }

            # changing it to the format that step expects in the wrapper
            # data = {
            #     "action" : [w, e]
            # }
            if not(data["target_controller"] > 0 and data['target_controller'] <= self.m and data['switch'] > 0 and data["switch"] <= self.n):
                print("Invalid action selected:", action)
                print("target_controller: ", w)
                print("switch: ", e)
            assert(data["target_controller"] > 0 and data['target_controller'] <= self.m and data['switch'] > 0 and data["switch"] <= self.n)
            # print(f"migrate request data: {data}")

            # com_start = time.time()
            response = self.session.post(self.gc_base_url + 'migrate', json=data)
            migrate = False
            
            if response.status_code == 200: # non migration action
                pass
            elif response.status_code == 201: # migration action
                migrate = True
            else: 
                response.close()
                raise Exception("Failed to execute migration action with global controller.")
            response.close()

            # if migrate:
            #     print("should be migrating")

            # wait some time to allow the network to adjust to the change    
            if not self.fast_mode: 
                time.sleep(self.step_time)

            # get the updated migrated switch positions: 
            self.switches_by_controller = self._get_switches_by_controller() 
            # get the observation
            observation = self._get_obs()
            # print("observation: ", observation)

            # print("communication overhead in step: ", time.time() - com_start) # about 0.01 seconds overhead for communication. so not much.

            # compute the reward

            # reward = self._paper_reward(observation, migrate)
            # # reward = self._penalize_poor_inaction_reward(observation, migrate)
            reward, info = self.reward_function(observation, migrate)
            # print("reward: ", reward)
            # print("info: ", info)

            if self.render_mode == "human":
                self._render_frame()

            # print("step time: ", time.time() - start_time)

            # print("\n")

            return observation, reward, False, False, info
        
        else:
            # alternative action space:
            # print("action: ", action)
            if action[0] == 0:
                # then we are not migrating

                # wait some time to allow the network to adjust to the change    
                if not self.fast_mode: 
                    time.sleep(self.step_time)

                # get the updated migrated switch positions: 
                self.switches_by_controller = self._get_switches_by_controller() 
                # get the observation
                observation = self._get_obs()
                print("observation: ", observation)

                # print("communication overhead in step: ", time.time() - com_start) # about 0.01 seconds overhead for communication. so not much.

                # compute the reward

                # reward = self._paper_reward(observation, migrate)
                # # reward = self._penalize_poor_inaction_reward(observation, migrate)
                reward = self.reward_function(observation, False)

                if self.render_mode == "human":
                    self._render_frame()
                
                info = self._get_info()

                # print("step time: ", time.time() - start_time)

                # print("\n")

                return observation, reward, False, False, info
            else:
                switch = action[0]
                target_controller = action[1]

                data = {
                    "target_controller" : int(target_controller+1), # range 1 to m
                    "switch" : int(switch) # range 1 to n, first is already the 0
                }

                response = self.session.post(self.gc_base_url + 'migrate', json=data)
                migrate = False
                
                if response.status_code == 200: # non migration action
                    pass
                elif response.status_code == 201: # migration action
                    migrate = True
                else: 
                    response.close()
                    raise Exception("Failed to execute migration action with global controller.")
                response.close()

                # wait some time to allow the network to adjust to the change    
                if not self.fast_mode: 
                    time.sleep(self.step_time)

                # get the updated migrated switch positions: 
                self.switches_by_controller = self._get_switches_by_controller() 
                # get the observation
                observation = self._get_obs()

                # print("communication overhead in step: ", time.time() - com_start) # about 0.01 seconds overhead for communication. so not much.

                # compute the reward

                # reward = self._paper_reward(observation, migrate)
                # # reward = self._penalize_poor_inaction_reward(observation, migrate)
                reward = self.reward_function(observation, migrate)

                if self.render_mode == "human":
                    self._render_frame()
                
                info = self._get_info()

                # print("step time: ", time.time() - start_time)

                # print("\n")

                return observation, reward, False, False, info


    def render(self):
        if self.render_mode == "human":
            return self._render_frame()
        
    def _render_frame(self):
        # use PyGame to render the network information. 
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        # we want to display a grid/matrix of m (number of controllers) by n(num controllers)
        # and for each switch, draw it in the correct controller spot. 
        
        #TODO expand on this 

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        
        pix_square_size = self.window_size / max(self.m, self.n)  # Square size based on grid dimensions

        # Draw controllers (grid cells)
        for i in range(self.m):
            for j in range(self.n):
                pygame.draw.rect(
                    canvas,
                    (200, 200, 200),
                    pygame.Rect(
                        j * pix_square_size, i * pix_square_size, pix_square_size, pix_square_size
                    ),
                    width=3  # Grid lines
                )

        for controller, switches in enumerate(self.switches_by_controller): # controller is indexed starting at 1, but the switches are not, they start at 0.
            # print("drawing for controller")
            for switch in switches:
                # print("drawing for switch")
                pygame.draw.circle(
                    canvas,
                    (0, 0, 255),  # Blue color for switches
                    (((switch-1) + 0.5) * pix_square_size, ((controller) + 0.5) * pix_square_size),
                    pix_square_size / 4,  # Size of switch
                )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            time.sleep(1)
        # else:  # rgb_array
        #     return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def __del__(self):
        self.session.close()