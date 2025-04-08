from fake_sim_class import FakeSim

if __name__ == "__main__":

    print("running simulator")

    sim = FakeSim(
        generation_mode="base_rate",
        num_switches=2, 
        num_controllers=2, 
        capacities=[1000,1000], 
        initial_sbc=[[1,2],[]], 
        base_rates = [10,20], 
        fast_mode=True,
        simulated_delay=5
    )

