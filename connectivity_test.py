
import requests

global_controller_ip = "192.168.56.101"
global_controller_flask_port = "8000"
global_controller_url = f"http://{global_controller_ip}:{global_controller_flask_port}/register"

# define the payload
data = {
    "controller_ip": "127.0.0.1",  # The domain controller's IP
    # "controller_port": self.listen_port,
    "flask_port": "1234"
}

while True:
    try:
        # Send a POST request to the global controller to register
        response = requests.post(global_controller_url, json=data)
        if response.status_code == 200:
            print(f"Successfully registered with the global controller: {response.text}")
            break
        else:
            print(f"Failed to register with global controller, status code: {response.status_code}")
            raise Exception("Failed to register with the global controller")
    except requests.exceptions.RequestException as e:
        print(f"Error while trying to connect to the global controller: {e}")