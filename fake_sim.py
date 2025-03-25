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


# logger = logging.getLogger(__name__)
global prev_time
global start_time
global total_switches
global num_controllers
global capacities
global initial_sbc
global switches_by_controller

prev_time = time.time()
start_time = prev_time # will help determine where in the time window we are.

# define inputs and constants:
total_switches = 26
num_controllers = 4
capacities = [12000, 10000, 10000, 12000]
initial_sbc = [[1,2,3,4,5,6,7],[8,9,10,11,12,13],[14,15,16,17,18,19],[20,21,22,23,24,25,26]]
switches_by_controller = copy.deepcopy(initial_sbc)

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

period_hours = 24.0
total_hours = 24.0
flow_duration = 10
time_scale = 60 # idk what the units of this is? 


def main():

    app = Flask(__name__)

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
            
            # check if the migration is a non-migration action (aka the controller already controls the switch. )
            if data["switch"] in switches_by_controller[data["target_controller"]-1]:
                print("Non-migration action called")
                return "", 200
            
            # execute the migration command

            print(f"Received migrate request for\ntarget_controller: {data['target_controller']}\nswitch: {data['switch']}")
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

            # Return a success response with a 201 status code (Created) if successful
            return "", 200
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
            state_matrix = update_state()
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
            # TODO update the state matrix first, instead of polling regularly.
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
            # TODO update the state matrix first, instead of polling regularly.
            print(f"reporting switch configuration to deep learning system: \n{switches_by_controller}")
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

    @app.route("/reset", methods=["POST"]) #TODO in the full network sim, it returns the state here, but I don't implement that in this case, should I? 
    def post_reset():
        # explicitly use the global versions of these variables.
        global switches_by_controller
        global prev_time
        global start_time
        try:
            # reset the network 
            switches_by_controller = copy.deepcopy(initial_sbc)
            start_time = time.time()
            prev_time = start_time
            time.sleep(2) # give it a second to start generating some values. 
            observation = update_state().tolist()
            # print("observation: ", observation)
            data = jsonify({"state": observation})

            # print("data: ", data)
            return data, 200
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return (jsonify({"error": f"Internal server error: {str(e)}"}), 500)

    def update_state():
        global prev_time # use the global variable here
        global start_time
        global num_controllers
        global switches_by_controller
        global total_switches
        '''
        In this program, we assume that each switch generates poisson traffic
        to its respective domain controller. (with sinusoidal fluctuation.)
        
        This calculates the number of flows for each controller within a given 
        interval (as opposed to finding the time points as was done in simulation.)
        
        '''

        # compute the time since last running

        # print("called update state")
        print("\n\n")
        cur_time = time.time()
        interval = cur_time - prev_time
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
        

        #TODO FOR TEST ONLY 
        base_rate_multiplier = 100

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

        state_matrix = np.zeros(shape=(num_controllers, total_switches))

        for c in range(num_controllers):
            for s in switches_by_controller[c]:
                # add the relevant rate to the state matrix
                state_matrix[c, s-1] = poisson_rate[s-1]

        print("state_matrix: ", state_matrix)

        # update time.
        prev_time = cur_time
        return state_matrix

    app.run(host="0.0.0.0", port=8000)


if __name__=="__main__":
    main()





    
    




