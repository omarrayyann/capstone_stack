from scipy.spatial.transform import Rotation as R
import zerorpc
import numpy as np
from loop_rate_limiters import RateLimiter
from mujoco_ar import MujocoARConnector

class FrankaClient:
    def __init__(self, server_ip):
        self.client = zerorpc.Client()
        self.client.connect(f"tcp://{server_ip}:4242")

    def get_ee_pose(self):
        return self.client.get_ee_pose()

    def get_joint_positions(self):
        return self.client.get_joint_positions()

    def move_to_joint_positions(self, positions, time_to_go):
        self.client.move_to_joint_positions(positions, time_to_go)

    def start_cartesian_impedance(self, Kx, Kxd):
        self.client.start_cartesian_impedance(Kx, Kxd)

    def update_desired_ee_pose(self, pose):
        self.client.update_desired_ee_pose(pose)

    def terminate_current_policy(self):
        self.client.terminate_current_policy()

    def get_gripper_width(self):
        return self.client.get_gripper_width()

    def update_gripper(self, flag):
        self.client.update_gripper(flag)


# Connect to the server
interface = FrankaClient(
    ip='localhost',
    port=4242
)

connector = MujocoARConnector(port=8888, debug=False) 
connector.start()

freq = 200
Kx_scale = 1.0
Kxd_scale = 1.0
Kx = (np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0]) * 0.4)
Kxd = (np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0]) * 0.5)

rate = RateLimiter(frequency=100, warn=False)

# Pass Kx and Kxd as lists
interface.start_cartesian_impedance(Kx=Kx, Kxd=Kxd)

start_rot = interface.get_ee_pose()[3:]
start_rot_matrix = R.from_rotvec(start_rot).as_matrix()

start_pos = interface.get_ee_pose()[:3]

while connector.get_latest_data()["position"] is None:
    pass

grasp = 0
while True:
    # new_pos = start_pos.copy()
    # new_pos[0] = np.clip(new_pos[0] + connector.get_latest_data()["position"][0] * 0.7, start_pos[0] - 1.5, start_pos[0] + 1.5)
    # new_pos[1] = np.clip(new_pos[1] + connector.get_latest_data()["position"][1] * 0.7, start_pos[1] - 1.5, start_pos[1] + 1.5)
    # new_pos[2] = np.clip(new_pos[2] + connector.get_latest_data()["position"][2] * 0.7, start_pos[2] - 1.5, start_pos[2] + 1.5)

    # transformation_matrix = connector.get_latest_data()["rotation"]
    # transformed_matrix = transformation_matrix @ start_rot_matrix
    # transformed_rot_vec = R.from_matrix(transformed_matrix).as_rotvec()

    # updated_pose = np.concatenate([new_pos, transformed_rot_vec])
    # interface.update_desired_ee_pose(updated_pose)
    # data = connector.get_latest_data()
    # if connector.get_latest_data()["toggle"] != grasp:
    #     grasp = connector.get_latest_data()["toggle"]
    #     interface.update_gripper(grasp)
    print(interface.get_ee_pose())
    rate.sleep()
