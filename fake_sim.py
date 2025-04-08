'''
The purpose of this file is to make a simulator capable of running much faster than the 
mininet network based simulator, relying explicitly on the assumption of poisson traffic
being generated for flow messages. This assumption is questionable, so by compairing the 
model trained on this system with the real network evluation, we'll be able to tell how
viable that assumption is, as well as possible try other generation schemes.
'''

from flask import Flask, request, jsonify
import requests
import copy
import sys
import time
import numpy as np
import logging

import threading

from waitress import serve

import argparse


# logger = logging.getLogger(__name__)
global prev_time
global start_time
global total_switches
global num_controllers
global capacities
global initial_sbc
global switches_by_controller
global burst_state
global burst_state_per_controller

burst_state = False

prev_time = time.time()
start_time = prev_time # will help determine where in the time window we are.

# define inputs and constants:
# total_switches = 2 # 26
total_switches = 4
# num_controllers = 2  # 4
num_controllers = 4
capacities = [12000, 10000, 10000, 12000]
capacities = capacities[:num_controllers] # limit based on number of controllers.
# capacities = [12000, 10000]
# initial_sbc = [[1,2,3,4,5,6,7],[8,9,10,11,12,13],[14,15,16,17,18,19],[20,21,22,23,24,25,26]]
# initial_sbc = [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26],[],[],[]]
initial_sbc = [[1,2,3,4],[],[],[]]
# initial_sbc = [[1],[2],[3],[4]]
# initial_sbc = initial_sbc[:num_controllers]
switches_by_controller = copy.deepcopy(initial_sbc)
burst_state_per_controller = [False]  * len(capacities)
 
base_rates = np.array([
            2,
            2,
            8,
            6,
            8,
            8,
            10,
            3,
            8,
            6,
            9,
            6,
            9,
            10,
            2,
            6,
            3,
            4,
            8,
            10,
            5,
            9,
            5,
            8,
            1,
            10,
        ])

base_rates = base_rates[:total_switches]

# base_rates = np.array([
#             4,
#             2,
#         ])

fluctuation_amplitudes = np.array([
            0.06,
            0.64,
            0.41,
            0.56,
            0.72,
            0.45,
            0.74,
            0.79,
            0.25,
            0.32,
            0.32,
            0.68,
            0.49,
            0.95,
            0.93,
            0.34,
            0.17,
            0.85,
            0.68,
            0.73,
            0.95,
            0.64,
            0.97,
            0.32,
            0.93,
            0.11,
        ])

fluctuation_amplitudes = fluctuation_amplitudes[:total_switches]

# fluctuation_amplitudes = np.array([
#             0.06,
#             0.64,
#         ])

# period_hours = 24.0 * 60 * 60
period_hours = 5 * 60
# total_hours = 24.0
# flow_duration = 10
time_scale = 60 # idk what the units of this is? 


def main(generation_mode, fast_mode = False, simulated_delay=5):
    '''
    fast_mode: Doesn't run live durations, instead uses a fixed duration for step function. 
    simulated_delay: the duration of the simulated time intervals. 
    '''

    switch_lock = threading.Lock()

    app = Flask(__name__)
    # log = logging.getLogger('werkzeug') # suppress flask http prints.
    # log.disabled = True

    @app.route("/migrate", methods=["POST"])
    def post_migrate():
        global switches_by_controller
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
            with switch_lock:
                # check if the migration is a non-migration action (aka the controller already controls the switch. )
                if data["switch"] in switches_by_controller[data["target_controller"]-1]:
                    # print("Non-migration action called")
                    return "", 200
                
                # execute the migration command

                print(f"Received migrate request for\ntarget_controller: {data['target_controller']}, switch: {data['switch']}")

                # print("before migration: ", switches_by_controller)

                c = data["target_controller"]
                s = data['switch']
                assert(c > 0 and c <= num_controllers and s > 0 and s <= total_switches)
                # move the switch

                # find the controller that it use to be in:
                for i in range(len(switches_by_controller)):
                    if data['switch'] in switches_by_controller[i] and i != data['target_controller'] - 1 :
                        # then we need to remove it 
                        switches_by_controller[i].remove(data["switch"])
                    elif i == data["target_controller"] - 1:
                        # then it's the one that needs to be added to
                        switches_by_controller[i].append(data["switch"])
                
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
            state_matrix = update_state(generation_mode)
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
        global capacities
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
        global switches_by_controller
        """
        This function reports which controller owns each switch.
        It is intended to be called by the pytorch code via http
        """
        # print("reporting switch configuration to deep learning system")
        try:
            # print(f"reporting switch configuration to deep learning system: \n{switches_by_controller}")
            # print("switches_by_controller from /switchesbycontroller: ", switches_by_controller)
            data = jsonify({"data": switches_by_controller})
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
        # explicitly use the global versions of these variables.
        global switches_by_controller
        global prev_time
        global start_time
        global burst_state
        global burst_state_per_controller
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
            observation = update_state(generation_mode).tolist()
            # print("observation: ", observation)
            data = jsonify({"state": observation})
            burst_state = False
            
            burst_state_per_controller = [False]  * len(capacities)
            # print("data: ", data)
            return data, 200
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return (jsonify({"error": f"Internal server error: {str(e)}"}), 500)

    def update_state(generation_mode):
        global prev_time # use the global variable here
        global start_time
        global num_controllers
        global switches_by_controller
        global total_switches
        global burst_state
        global burst_state_per_controller
        '''
        In this program, we assume that each switch generates poisson traffic
        to its respective domain controller. (with sinusoidal fluctuation.)
        
        This calculates the number of flows for each controller within a given 
        interval (as opposed to finding the time points as was done in simulation.)
        
        '''

        # compute the time since last running

        # print("called update state")
        cur_time = None
        interval = None
        if not fast_mode: # for normal mode
            print("\n\n")
            cur_time = time.time()
            interval = cur_time - prev_time
        else: # for fast mode
            cur_time = prev_time + simulated_delay
            interval = simulated_delay

        base_rate_multiplier = 100 #TODO FOR TEST ONLY 
        state_matrix = None
        if generation_mode == "poisson":

            # print("interval: ", interval)

            # the expected number of events in time T is given by 
            # E = integral(0 to T) lambda(t) dt
            # where lambda(t) is a time-dependent rate function.

            # given the periodic fluctuation: 
            # lambda(t) = lambda0 (1 + A sin (2 pi t / P)) #for a phase shift:  lambda_t = lambda_0 * (1 + A * np.sin(2 * np.pi * t / P + phase_shift))
            # where lambda0 is the base rate
            # A is the fluctuation amplitude, 0 to 1
            # P is the period in hours #TODO there should be a random phase as well, but we didn't do that in the simulator yet. 
            # t is the time in hours

            # integral simplifies to: 
            # E = lambda0 * T + lambda0 A integral (0,T) sin(2 pi t / P) dt
            # which solves to: 
            # E = lambda0 * T - lambda0 A P / 2 PI * [cos(2 pi T/ P) - 1]
            
            P = period_hours # 
            T_end = cur_time - start_time
            T_begin = prev_time - start_time

            A = fluctuation_amplitudes
            lambda_0 = base_rates * base_rate_multiplier


            # compute the number of expected events at the start and end times, and then subtract.
            expected_events_begin = lambda_0 * T_begin - lambda_0 * A * (P / (2 * np.pi)) * (np.cos(2 * np.pi * T_begin / P) - 1)
            expected_events_end = lambda_0 * T_end - lambda_0 * A * (P / (2 * np.pi)) * (np.cos(2 * np.pi * T_end / P) - 1)

            # print("expected_events_begin: ",  expected_events_begin)
            # print("expected_events_end: ",  expected_events_end)
            
            
            expected_in_duration = (expected_events_end - expected_events_begin )

            # print("expected_in_duration: ", expected_in_duration)

            # # Compute expected event count
            # expected_events_start = lambda_0 * T - lambda_0 * A * (P / (2 * np.pi)) * (np.cos(2 * np.pi * T / P) - 1)

            # then perturb with poisson process.
            expected_in_duration = np.clip(expected_in_duration, a_min=0, a_max=None)
            # print("expected in duration: ", expected_in_duration)

            possion_in_duration = np.random.poisson(expected_in_duration)
            # possion_in_duration = np.reshape(possion_in_duration, (4, -1))

            # print("poission in duration: ", possion_in_duration)

            # then compute the rate from this

            poisson_rate = possion_in_duration / interval

            # print("poisson rate: ", poisson_rate)

            # compute the state form. 

            # clip to the valid dimension of action space
            poisson_rate = np.clip(poisson_rate, a_min=0, a_max=5000) # this does slow it down a bit.

            state_matrix = np.zeros(shape=(num_controllers, total_switches))

            for c in range(num_controllers):
                for s in switches_by_controller[c]:
                    # add the relevant rate to the state matrix
                    state_matrix[c, s-1] = poisson_rate[s-1]

            # print("\nstate_matrix: ", state_matrix)
        elif generation_mode == "burst":
            '''
            For ML inference like traffic. 

            Key design choices for ML inference traffic: 

            Bursty Traffic Pattern:
                Uses a two-state model (burst/quiet) instead of sinusoidal patterns
                Better matches real ML inference patterns observed in production systems
                Incorporates:
                    burst_duration: Average time in high-traffic state (not used yet)
                    quiet_duration: Average time in low-traffic state (not used yet)
                    high_rate_multiplier: Magnitude of traffic spikes

            Markov-Modulated Poisson Process:
                State transitions follow Markov chain probabilities
                More realistic than simple periodic patterns for ML workloads
                transition_prob controls burst persistence

            Parameterization:
                Tunable parameters match observable ML workload characteristics

            Can be calibrated to specific use cases (batch processing, real-time APIs, etc.)
            '''

            print("using burst traffic")

            # Parameters specific to ML inference patterns
            burst_duration = 15*60  # 15 minutes in seconds
            quiet_duration =  45*60  # 45 minutes in seconds
            high_rate_multiplier = 2
            transition_prob =  0.3  # Probability of switching state
            
            # Time calculations
            interval = cur_time - prev_time
            T_end = cur_time - start_time
            T_begin = prev_time - start_time
            
            # Base rate and burst modulation
            base_lambda = base_rates  # Base request rate (requests/sec)
            burst_state = False  # Persistent state
            
            print("starting state transition logic")

            # State transition logic
            if np.random.rand() < transition_prob:
                burst_state = not burst_state

            print("new burst state: ", burst_state)
                
            # Calculate effective lambda for this interval
            if burst_state:
                effective_lambda = base_lambda * high_rate_multiplier
            else:
                effective_lambda = base_lambda

            print("effective_lambda: ", effective_lambda)
                
            # Expected events in interval (integral of Poisson rate)
            expected_events = effective_lambda * interval
            
            # Generate Poisson-distributed events
            observed_events = np.random.poisson(expected_events)
            
            # Calculate empirical rate (requests/sec)
            traffic_rate = observed_events / interval if interval > 0 else 0

            traffic_rate *= base_rate_multiplier 

            print("traffic_rate: ", traffic_rate)
            
            # Build state matrix (same structure as original)
            state_matrix = np.zeros((num_controllers, total_switches))
            for c in range(num_controllers):
                for s in switches_by_controller[c]:
                    state_matrix[c, s-1] = traffic_rate[s-1]  # Using same indexing logic

            print(state_matrix)
        elif generation_mode == "burst_per_controller":
             # Time calculations
            interval = cur_time - prev_time  # Time interval between steps
            T_end = cur_time - start_time    # Time elapsed since the start of the simulation
            T_begin = prev_time - start_time  # Time elapsed for the previous step
            
            high_rate_multiplier = 2
            transition_prob =  0.3  # Probability of switching state

            for i in range(len(burst_state_per_controller)):
                # State transition logic
                if np.random.rand() < transition_prob:
                    burst_state_per_controller[i] = not burst_state_per_controller[i]

            # Initialize traffic rate matrix
            state_matrix = np.zeros((num_controllers, total_switches))

            # Iterate through each controller
            for c in range(num_controllers):
                # Base rate for this controller

                burst_state = burst_state_per_controller[c]  # Use the burst state from the list
                
                for s in switches_by_controller[c]:
                    base_lambda = base_rates[s-1]  # Base request rate (per switch, adjust by -1 for indexing)
                    
                    # Calculate effective lambda for this switch based on controller's burst state
                    if burst_state:
                        effective_lambda = base_lambda * high_rate_multiplier
                    else:
                        effective_lambda = base_lambda

                    # Expected events in the time interval (Poisson rate integral)
                    expected_events = effective_lambda * interval

                    # Generate Poisson-distributed events
                    observed_events = np.random.poisson(expected_events)

                    # Calculate traffic rate (requests/sec)
                    traffic_rate = observed_events / interval if interval > 0 else 0

                    # Modulate the traffic rate (if applicable)
                    traffic_rate *= base_rate_multiplier
                    
                    # Debug info for burst state and traffic rate
                    # print(f"Controller {c}: Switch {s}, Burst state: {burst_state}, Effective Lambda: {effective_lambda}, Traffic rate: {traffic_rate}")

                    # Populate the state matrix for the current controller's switch
                    state_matrix[c, s-1] = traffic_rate  # Adjust for 1-based indexing in switches

            print(burst_state_per_controller)
            print(state_matrix)
        elif generation_mode == "base_rate":
            '''
            only generates at the base rate (no noise)
            ''' 
            # Build state matrix (same structure as original)
            
            # print("num_controllers: ", num_controllers)
            # print("total_switches: ", total_switches)
            # print("switches_by_controller: ", switches_by_controller)
            with switch_lock:
                # print("switches_by_controller from base rate: ", switches_by_controller)
                state_matrix = np.zeros((num_controllers, total_switches))
                for c in range(num_controllers):
                    for s in switches_by_controller[c]:
                        state_matrix[c, s-1] = base_rates[s-1] 
            
            # print(state_matrix)
        elif generation_mode == "sinusoid": 

            P = period_hours # 
            T_end = cur_time - start_time
            T_begin = prev_time - start_time

            A = fluctuation_amplitudes
            lambda_0 = base_rates * base_rate_multiplier


            # compute the number of expected events at the start and end times, and then subtract.
            expected_events_begin = lambda_0 * T_begin - lambda_0 * A * (P / (2 * np.pi)) * (np.cos(2 * np.pi * T_begin / P) - 1)
            expected_events_end = lambda_0 * T_end - lambda_0 * A * (P / (2 * np.pi)) * (np.cos(2 * np.pi * T_end / P) - 1)

            # print("expected_events_begin: ",  expected_events_begin)
            # print("expected_events_end: ",  expected_events_end)
            
            
            expected_in_duration = (expected_events_end - expected_events_begin )

            # print("expected_in_duration: ", expected_in_duration)

            # # Compute expected event count
            # expected_events_start = lambda_0 * T - lambda_0 * A * (P / (2 * np.pi)) * (np.cos(2 * np.pi * T / P) - 1)

            # then perturb with poisson process.
            expected_in_duration = np.clip(expected_in_duration, a_min=0, a_max=None)
            # print("expected in duration: ", expected_in_duration)

            rate = expected_in_duration / interval

            # compute the state form. 

            # clip to the valid dimension of action space
            poisson_rate = np.clip(rate, a_min=0, a_max=5000) # this does slow it down a bit.

            state_matrix = np.zeros(shape=(num_controllers, total_switches))
            # print("got here in state")
            for c in range(num_controllers):
                for s in switches_by_controller[c]:
                    # add the relevant rate to the state matrix
                    state_matrix[c, s-1] = poisson_rate[s-1]

        else: 
            raise ValueError("missing valid parameter for generation_mode")

        # update time.
        prev_time = cur_time
        # print("state_matrix")
        # print(state_matrix)
        # time.sleep(1)
        return state_matrix

    serve(app, host="0.0.0.0", port=8000, threads=1)
    # app.run(host="0.0.0.0", port=8000)


if __name__=="__main__":

    # parse arguments from command line:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fast_mode", help="Enable Fast Mode", type=bool,  required=False, default=False)
    parser.add_argument("-d", "--delay", help = "the simulated delay when using fastmode", required=False, default=5)
    parser.add_argument("-g", "--generation_mode", help="alters the way that traffic is generated", default="poisson")

    args = parser.parse_args()

    if args.fast_mode == True: 
        print(f"running simulator in fast mode, with simulated delay: {args.delay}")
    else:
        print(f"Running simulator in standard mode (non fast mode)")

    main(fast_mode=args.fast_mode, simulated_delay=args.delay, generation_mode=args.generation_mode)





    
    




