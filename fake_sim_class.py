
# server libraries.
from flask import Flask, request, jsonify
from waitress import serve
import time

import sys
import signal
import copy

import threading

import numpy as np

import random




class FakeSim:
    def __init__(self, generation_mode:str, num_switches:int, num_controllers:int, capacities:list, initial_sbc:list, base_rates:list, fast_mode:bool, simulated_delay = None, fluctuation_amplitudes : list | None = None, period : float | None = None, random_init = False):

        print("init called")

        # input checking
        assert len(base_rates) == num_switches
        assert len(capacities) == num_controllers
        assert len(initial_sbc) == num_controllers

        self.fast_mode = fast_mode # tells if the simulator is running in fast mode or slow mode. 


        # hook into the exit signal for clean shutdown
        signal.signal(signal.SIGINT, self.shutdown_handler)
        signal.signal(signal.SIGTERM, self.shutdown_handler)

        self.switch_lock = threading.Lock()
        self.initial_sbc = initial_sbc
        self.sbc = copy.deepcopy(self.initial_sbc) # switches_by_controller

        self.random_init = random_init

        self.gen = TrafficGenerator(
            generation_mode=generation_mode, 
            num_switches=num_switches, 
            num_controllers=num_controllers, 
            sbc=self.sbc, capacities=capacities,
            base_rates=base_rates, 
            fast_mode=fast_mode, 
            switch_lock=self.switch_lock, 
            simulated_delay=simulated_delay, 
            fluctuation_amplitudes=fluctuation_amplitudes, 
            period=period
            )

        app = Flask(__name__)

        # define endpoint behaviour
        @app.route("/migrate", methods=["POST"])
        def post_migrate():
            try:
                # parse the incoming json data
                data = request.get_json()

                # check if the data is valid
                if not data:
                    return (
                        jsonify({"error": "No data provided"}),
                        400,
                    )  # Bad Request if no data

                # You can add your processing logic here
                # For example, checking required fields:
                if "target_controller" not in data or "switch" not in data:
                    return (
                        jsonify({"error": "missing required fields"}),
                        400,
                    )  # Bad Request if missing required field
                
                # "target_controller" and "switch" are indexed starting at 1. 
                with self.switch_lock:
                    # check if the migration is a non-migration action (aka the controller already controls the switch. )
                    if data["switch"] in self.sbc[data["target_controller"]-1]:
                        # print("Non-migration action called")
                        return "", 200
                    
                    # execute the migration command

                    print(f"Received migrate request for\ntarget_controller: {data['target_controller']}, switch: {data['switch']}")

                    # print("before migration: ", switches_by_controller)

                    c = data["target_controller"]
                    s = data['switch']
                    assert(c > 0 and c <= num_controllers and s > 0 and s <= num_switches)
                    # move the switch

                    # print("sbc before migrate: \n", self.sbc)

                    # find the controller that it use to be in:
                    for i in range(len(self.sbc)):
                        if data['switch'] in self.sbc[i] and i != data['target_controller'] - 1 :
                            # then we need to remove it 
                            self.sbc[i].remove(data["switch"])
                        elif i == data["target_controller"] - 1:
                            # then it's the one that needs to be added to
                            self.sbc[i].append(data["switch"])
                    
                    # print("sbc after migrate: \n", self.sbc ) # This is working.
                    
                    # print("after migration: ", switches_by_controller)

                    # Return a success response with a 201 status code (Created) if successful
                    # print("migration action")
                    return "", 201
            except Exception as e:
                # handle unexpected errors:
                print(e)
                return (
                    jsonify({"error": f"Internal server error: {str(e)}"}),
                    500,
                )  # Internal Server Error on exceptions
        
        @app.route("/state", methods=["GET"])
        def get_state():
            try:
                # collect the up to date state matrix.
                state_matrix = self.gen.generate_state()
                data = jsonify({"state": state_matrix.tolist()})
                # Return a success response with a 201 status code (Created) if successful
                return data, 200
            except Exception as e:
                # handle unexpected errors:
                print(e)
                return (
                    jsonify({"error": f"Internal server error: {str(e)}"}),
                    500,
                )  # Internal Server Error on exceptions

        @app.route("/capacities", methods=["GET"])
        def get_capacities():
            try:
                print("sending capacities")
                data = jsonify({"data": capacities})

                # Return a success response with a 201 status code (Created) if successful
                return data, 200
            except Exception as e:
                # handle unexpected errors:
                print(e)
                return (
                    jsonify({"error": f"Internal server error: {str(e)}"}),
                    500,
                )  # Internal Server Error on exceptions

        @app.route("/switches_by_controller", methods=["GET"])
        def get_switch_by_controller():
            """
            This function reports which controller owns each switch.
            It is intended to be called by the pytorch code via http
            """
            # print("reporting switch configuration to deep learning system")
            try:
                # print(f"reporting switch configuration to deep learning system: \n{switches_by_controller}")
                # print("switches_by_controller from /switchesbycontroller: ", switches_by_controller)
                data = jsonify({"data": self.sbc})
                # data = jsonify({"data": self.controller_switch_mapping})

                # Return a success response with a 201 status code (Created) if successful
                return data, 200
            except Exception as e:
                # handle unexpected errors:
                return (
                    jsonify({"error": f"Internal server error: {str(e)}"}),
                    500,
                )  # Internal Server Error on exceptions

        @app.route("/reset", methods=["POST"])
        def post_reset():
            try:
                # reset the network 
                print("reset called: ")
                # time.sleep(1)
                # print("switches_by_controller in reset: ", switches_by_controller)
                # self.sbc = copy.deepcopy(initial_sbc) # TODO need to change this because it's creating a new list, and then the TrafficGenerator no longer has the reference. 
                self.sbc.clear()
                if self.random_init:
                    # generate a random initialization
                    # for each switch, pick a controller randomly to join.
                    new_sbc = [ [] for _ in range(num_controllers)]
                    # print("new_sbc: ", new_sbc)
                    for switch in range(1,num_switches + 1):
                        # pick a controller randomly. 
                        controller = random.randint(0, num_controllers-1)
                        # print("controller: ", controller)
                        new_sbc[controller].append(switch)
                    self.sbc.extend(new_sbc)
                else:
                    self.sbc.extend(copy.deepcopy(self.initial_sbc))
                # print("initial sbc", self.initial_sbc)
                # print("sbc after reset: ", self.sbc)

                # print("switches_by_controller in reset2: ", switches_by_controller)
                start_time = time.time()
                prev_time = start_time
                if not fast_mode: 
                    time.sleep(2) # give it a second to start generating some values. #TODO make this not a magic number
                observation = self.gen.generate_state().tolist()
                # print("observation: ", observation)
                data = jsonify({"state": observation})
                burst_state = False
                
                burst_state_per_controller = [False]  * len(capacities)
                # print("data: ", data)
                return data, 200
            except Exception as e:
                print(f"error: {e}", file=sys.stderr)
                return (jsonify({"error": f"Internal server error: {str(e)}"}), 500)


        serve(app, host='0.0.0.0', port=8000, threads=1)
        
        
    def shutdown_handler(self, sig, frame):
        print("\n[INFO] Server shutdown requested via Ctrl+C")
        sys.exit(0)



class TrafficGenerator: 
    '''
    This class controls the various traffic generation regimes. 
    '''
    def __init__(self, generation_mode, num_switches:int, num_controllers:int, capacities:list, sbc:list, base_rates:list, fast_mode:bool, switch_lock, simulated_delay=None, max_rate=5000, fluctuation_amplitudes : list | None = None, period : float | None = None):
        
        if fast_mode and simulated_delay == None:
            raise ValueError("must supply a simulated_delay if using fast_mode")
        self.fast_mode = fast_mode
        self.simulated_delay = simulated_delay

        self.generation_mode = generation_mode

        if generation_mode == "poisson":
            if fluctuation_amplitudes == None:
                raise ValueError("if using poisson generation, fluctuation_amplitudes cannot be None")
            if period == None:
                raise ValueError("period cannot be None if using poisson generation")
            
            self.fluctuation_amplitudes = fluctuation_amplitudes
            self.period = period
            print("generating poisson traffic")
        elif generation_mode == "base_rate":
            print("generating base_rate traffic")

        self.base_rates = base_rates
        self.num_controllers = num_controllers
        self.num_switches = num_switches
        self.sbc = sbc #TODO check that this is a reference not a copy
        self.capacities = capacities
        
        self.max_rate = max_rate

        # Thread management
        self.switch_lock = switch_lock

        # begin timing:
        self.prev_time = time.time()
        self.start_time = self.prev_time

        pass

    def generate_state(self):

        # print("sbc in generate state: ", self.sbc)
        
        gm = self.generation_mode

        # determine the interval.
        cur_time = None
        interval = None

        if not self.fast_mode:
            # for normal operation
            cur_time = time.time()
            interval = cur_time - self.prev_time
        else:
            cur_time = self.prev_time + self.simulated_delay
            interval = self.simulated_delay

        # TODO change this,
        base_rate_multiplier = 100
        
        state_matrix = None

        if gm == "poisson": # the one from the paper

            P = self.period # period in seconds for the sinusoid. 
            T_end = cur_time - self.start_time
            T_begin = self.prev_time - self.start_time
            A = self.fluctuation_amplitudes
            A = np.array(A)
            # lambda_0 = base_rates * base_rate_multiplier
            # lambda_0 = self.base_rates
            lambda_0 = np.array(self.base_rates)

            # print("P: ", P)
            # print("A: ", A)
            # print("Type of A[0]: ", type(A[0]))
            # print("T_end: ", T_end)
            # print("T_begin: ", T_begin)
            # print("lambda_0: ", lambda_0)

            # find the expected number of events using sinusoid.
            # find number of expected events as the start and end times, then subtract to find the expected number of events in the interval.
            expected_events_begin = lambda_0 * T_begin - lambda_0 * A * (P / (2 * np.pi)) * (np.cos(2 * np.pi * T_begin / P) - 1)
            expected_events_end = lambda_0 * T_end - lambda_0 * A * (P / (2 * np.pi)) * (np.cos(2 * np.pi * T_end / P) - 1)
            expected_in_duration = (expected_events_end - expected_events_begin )
            # clip so that it's always positive or zero.
            expected_in_duration = np.clip(expected_in_duration, a_min=0, a_max=None)
            # generate the new randomized number via poisson distribution.
            possion_in_duration = np.random.poisson(expected_in_duration)
            # convert the total number of events into an interval
            poisson_rate = possion_in_duration / interval
            # clip the poisson rate into the valid interval.
            poisson_rate = np.clip(poisson_rate, a_min=0, a_max=self.max_rate) # this does slow it down a bit.
            state_matrix = np.zeros(shape=(self.num_controllers, self.num_switches))

            with self.switch_lock:
                for c in range(self.num_controllers):
                    for s in self.sbc[c]:
                        # add the relevant rate to the state matrix
                        state_matrix[c, s-1] = poisson_rate[s-1]

        elif gm == "burst":
            raise ValueError("not implemented in this sim")
        elif gm == "burst_per_controller":
            raise ValueError("not implemented in this sim")
        elif gm == "base_rate":
            with self.switch_lock:
                # print("switches_by_controller from base rate: ", switches_by_controller)
                state_matrix = np.zeros((self.num_controllers, self.num_switches))
                for c in range(self.num_controllers):
                    for s in self.sbc[c]:
                        state_matrix[c, s-1] = self.base_rates[s-1] * interval #TODO is using interval right here? 
        elif gm == "sinusoid":
            raise ValueError("not implemented in this sim")
        else:
            raise ValueError("missing valid parameter for generation_mode")
        
        # update time.
        self.prev_time = cur_time
        # print("state_matrix")
        # print(state_matrix)
        # time.sleep(1)
        # print(state_matrix)
        return state_matrix
    



