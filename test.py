from scipy.spatial.transform import Rotation as R
import zerorpc
import numpy as np
from loop_rate_limiters import RateLimiter
from mujoco_ar import MujocoARConnector

class FrankaInterface:
    def __init__(self, ip='localhost', port=4242):
        self.server = zerorpc.Client(heartbeat=20)
        self.server.connect(f"tcp://{ip}:{port}")

    def get_ee_pose(self):
        flange_pose = np.array(self.server.get_ee_pose())
        tip_pose = flange_pose
        return tip_pose
    
    def get_joint_positions(self):
        return np.array(self.server.get_joint_positions())
    
    def get_joint_velocities(self):
        return np.array(self.server.get_joint_velocities())

    def move_to_joint_positions(self, positions: np.ndarray, time_to_go: float):
        self.server.move_to_joint_positions(positions.tolist(), time_to_go)

    def start_cartesian_impedance(self, Kx: np.ndarray, Kxd: np.ndarray):
        self.server.start_cartesian_impedance(
            Kx.tolist(),
            Kxd.tolist()
        )
    
    def update_desired_ee_pose(self, pose: np.ndarray):
        self.server.update_desired_ee_pose(pose.tolist())

    def terminate_current_policy(self):
        self.server.terminate_current_policy()

    def close(self):
        self.server.close()

    def get_gripper_width(self):
        return self.server.get_gripper_width()
    
    def update_gripper(self, flag):
        self.server.update_gripper(flag)
    
# Connect to the server
interface = FrankaInterface(
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
    new_pos = start_pos.copy()
    new_pos[0] = np.clip(new_pos[0] + connector.get_latest_data()["position"][0] * 0.7, start_pos[0] - 1.5, start_pos[0] + 1.5)
    new_pos[1] = np.clip(new_pos[1] + connector.get_latest_data()["position"][1] * 0.7, start_pos[1] - 1.5, start_pos[1] + 1.5)
    new_pos[2] = np.clip(new_pos[2] + connector.get_latest_data()["position"][2] * 0.7, start_pos[2] - 1.5, start_pos[2] + 1.5)

    transformation_matrix = connector.get_latest_data()["rotation"]
    transformed_matrix = transformation_matrix @ start_rot_matrix
    transformed_rot_vec = R.from_matrix(transformed_matrix).as_rotvec()

    updated_pose = np.concatenate([new_pos, transformed_rot_vec])
    interface.update_desired_ee_pose(updated_pose)
    data = connector.get_latest_data()
    if connector.get_latest_data()["toggle"] != grasp:
        grasp = connector.get_latest_data()["toggle"]
        interface.update_gripper(grasp)
    rate.sleep()
