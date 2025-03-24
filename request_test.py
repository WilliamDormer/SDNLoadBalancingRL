import requests
import numpy as np

gc_ip = "127.0.0.1" # for fake_sim
gc_port="8000"

gc_base_url = f"http://{gc_ip}:{gc_port}/"

# # TEST MIGRATE REQUEST

# data = {
#     "target_controller" : 2, # range 1 to m
#     "switch" : 1 # range 1 to n
# }
# print(f"migrate request data: {data}")
# response = requests.post(gc_base_url + 'migrate', json=data)
# if response.status_code != 200:
#     raise Exception("Failed to execute migration action with global controller.")


# # get the switches by controller
# try:
#     data = {
#     }
#     url = gc_base_url + "switches_by_controller"
#     response = requests.get(url, json=data)
#     if response.status_code == 200:
#         # get the state and return it
#         json_data = response.json()  # Convert response to a dictionary
#         print(json_data['data']) # this will be a list of lists, where the rows are the controllers (starting 0,1,2,3), and each holds the switches it controls (indexed starting at 1)
#     else:
#         raise Exception("Failed to retreive capacities from network.")
# except Exception as e:
#     print("there was an error in _get_switches_by_controller: ", e)


# # TEST RESET REQUEST

# response = requests.post(gc_base_url + 'reset', timeout=15)
# if response.status_code != 200:
#     error = response.json()
#     e = error["error"]
#     raise Exception(f"Failed to reset network, error: ({e})")

# # print("in reset: ")
# # print(type(response))

# json_data = response.json()  # Convert response to a dictionary
# observation = np.array(json_data["state"])  # Convert list back to NumPy array
# print("state: ", observation)
# # convert it to the observation space 
# observation = np.array(observation, dtype=np.float32)


# test the state request: 
data = {}
url = gc_base_url + "state"
response = requests.get(url, json=data)
if response.status_code == 200:
    # get the state and return it
    json_data = response.json()  # Convert response to a dictionary
    array = np.array(json_data["state"])  # Convert list back to NumPy array
    # convert it to the observation space 
    array = np.array(array, dtype=np.float32)
    print("array: ", array)
else:
    raise Exception("Failed to retreive state from network.")

# # get the new switch configuration

# try:
#     data = {
#     }
#     url = gc_base_url + "switches_by_controller"
#     response = requests.get(url, json=data)
#     if response.status_code == 200:
#         # get the state and return it
#         json_data = response.json()  # Convert response to a dictionary
#         print(json_data['data']) # this will be a list of lists, where the rows are the controllers (starting 0,1,2,3), and each holds the switches it controls (indexed starting at 1)
#     else:
#         raise Exception("Failed to retreive capacities from network.")
# except Exception as e:
#     print("there was an error in _get_switches_by_controller: ", e)