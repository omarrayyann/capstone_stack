from scipy.spatial.transform import Rotation as R
import numpy as np
from loop_rate_limiters import RateLimiter
from mujoco_ar import MujocoARConnector
from FrankaClient import FrankaClient


# Connect to the server
interface = FrankaClient(
    server_ip='127.0.0.1',
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
    
    new_pos = start_pos + connector.get_latest_data()["position"] * 0.8
    # add safe bounds

    # new_pos[2] = max(0.155, new_pos[2])
    new_pos[2] = max(0.127, new_pos[2])
    transformation_matrix = connector.get_latest_data()["rotation"]
    transformed_matrix = transformation_matrix @ start_rot_matrix
    transformed_rot_vec = R.from_matrix(transformed_matrix).as_rotvec()
    updated_pose = np.concatenate([new_pos, transformed_rot_vec])

    interface.update_desired_ee_pose(updated_pose)

    if connector.get_latest_data()["toggle"] is True:
        interface.set_gripper_width(0.04)
    else:
        interface.set_gripper_width(0.085)

    

    rate.sleep()
