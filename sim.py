from fake_sim_class import FakeSim
import argparse
from ast import literal_eval

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', "--generation_mode", type=str, required=True)
    parser.add_argument('-s', "--num_switches", type=int, required=True)
    parser.add_argument('-c', "--num_controllers", type=int, required=True)
    parser.add_argument('-ca', "--capacities", type=literal_eval, required=True)
    parser.add_argument('-is', "--initial_sbc", type=literal_eval, required=True)
    parser.add_argument('-br', "--base_rates", type=literal_eval, required=True)
    parser.add_argument('-f', "--fast_mode", type=bool, required=True)
    parser.add_argument('-sd', "--simulated_delay", type=float, required=True)
    

    args = parser.parse_args()

    generation_mode = args.generation_mode
    num_switches = args.num_switches
    num_controllers = args.num_controllers
    capacities = args.capacities
    initial_sbc = args.initial_sbc
    base_rates = args.base_rates
    fast_mode = args.fast_mode
    simulated_delay = args.simulated_delay

    print(args)

    print("running simulator")

    sim = FakeSim(
        generation_mode=generation_mode,
        num_switches=num_switches, 
        num_controllers=num_controllers, 
        capacities=capacities, 
        initial_sbc=initial_sbc, 
        base_rates = base_rates, 
        fast_mode=fast_mode,
        simulated_delay=simulated_delay
    )

    # sim = FakeSim(
    #     generation_mode="base_rate",
    #     num_switches=2, 
    #     num_controllers=2, 
    #     capacities=[1000,1000], 
    #     initial_sbc=[[1,2],[]], 
    #     base_rates = [10,20], 
    #     fast_mode=True,
    #     simulated_delay=5
    # )

