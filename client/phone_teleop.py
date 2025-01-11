from scipy.spatial.transform import Rotation as R
import numpy as np
from loop_rate_limiters import RateLimiter
from mujoco_ar import MujocoARConnector
from FrankaClient import FrankaClient


# Connect to the server
interface = FrankaClient(
    server_ip='10.228.255.79',
)

# Start the AR connector
connector = MujocoARConnector(port=8888, debug=False) 
connector.start()

# Cartesian impedance controller gains
Kx = (np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0]) * 0.4)
Kxd = (np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0]) * 0.5)
rate = RateLimiter(frequency=200, warn=False)

# Start the cartesian impedance controller
interface.start_cartesian_impedance(Kx=Kx, Kxd=Kxd)

# Get the initial pose
start_rot = interface.get_ee_pose()[3:]
start_rot_matrix = R.from_rotvec(start_rot).as_matrix()
start_pos = interface.get_ee_pose()[:3]

# Wait for the AR connector to get the first data
while connector.get_latest_data()["position"] is None:
    pass

while True:
    
    new_pos = start_pos + connector.get_latest_data()["position"] * 0.7
    # add safe bounds

    transformation_matrix = connector.get_latest_data()["rotation"]
    transformed_matrix = transformation_matrix @ start_rot_matrix
    transformed_rot_vec = R.from_matrix(transformed_matrix).as_rotvec()
    updated_pose = np.concatenate([new_pos, transformed_rot_vec])

    interface.update_desired_ee_pose(updated_pose)

    interface.update_gripper(connector.get_latest_data()["toggle"])

    rate.sleep()
