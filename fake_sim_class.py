
# server libraries.
from flask import Flask, request, jsonify
from waitress import serve
import time

import sys
import signal
import copy

import threading

import numpy as np




class FakeSim:
    def __init__(self, generation_mode:str, num_switches:int, num_controllers:int, capacities:list, initial_sbc:list, base_rates:list, fast_mode:bool, simulated_delay = None):

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
        self.sbc = copy.deepcopy(initial_sbc) # switches_by_controller

        self.gen = TrafficGenerator(generation_mode=generation_mode, num_switches=num_switches, num_controllers=num_controllers, sbc=self.sbc, capacities=capacities, base_rates=base_rates, fast_mode=fast_mode, switch_lock=self.switch_lock, simulated_delay=simulated_delay)

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

                    # find the controller that it use to be in:
                    for i in range(len(self.sbc)):
                        if data['switch'] in self.sbc[i] and i != data['target_controller'] - 1 :
                            # then we need to remove it 
                            self.sbc[i].remove(data["switch"])
                        elif i == data["target_controller"] - 1:
                            # then it's the one that needs to be added to
                            self.sbc[i].append(data["switch"])
                    
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
                switches_by_controller = copy.deepcopy(initial_sbc)
                
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
    def __init__(self, generation_mode, num_switches:int, num_controllers:int, capacities:list, sbc:list, base_rates:list, fast_mode:bool, switch_lock, simulated_delay=None):
        
        if fast_mode and simulated_delay == None:
            raise ValueError("must supply a simulated_delay if using fast_mode")
        self.fast_mode = fast_mode
        self.simulated_delay = simulated_delay

        self.generation_mode = generation_mode
        self.base_rates = base_rates
        self.num_controllers = num_controllers
        self.num_switches = num_switches
        self.sbc = sbc #TODO check that this is a reference not a copy
        self.capacities = capacities
        

        # Thread management
        self.switch_lock = switch_lock

        # begin timing:
        self.prev_time = time.time()
        self.start_time = self.prev_time

        pass

    def generate_state(self):
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

        if gm == "poisson":
            raise ValueError("not implemented in this sim")
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
        return state_matrix
    



