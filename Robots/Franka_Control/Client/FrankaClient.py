import zerorpc
import numpy as np

class FrankaClient:
    def __init__(self, server_ip='localhost', port=4242):
        self.server = zerorpc.Client(heartbeat=20)
        self.server.connect(f"tcp://{server_ip}:{port}")

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
    
    def set_gripper_width(self, width):
        self.server.set_gripper_width(width)
    