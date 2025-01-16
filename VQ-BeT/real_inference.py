import random
import time
from pathlib import Path
from collections import deque

import cv2
import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R

from RealSense import RealSense
from FrankaClient import FrankaClient
from vqvae.vqvae import *
from loop_rate_limiters import RateLimiter

def reset_position(client):
    """Reset the robot's position to a predefined goal."""
    speed = 0.5  # m/s
    freq = 10
    goal = np.array([0.40412879, 0.03975403, 0.31362468, -2.87557022, 1.19741026, -0.05542863])
    limiter = RateLimiter(frequency=freq, warn=True)  # Control frequency

    while np.linalg.norm(goal[:3] - client.get_ee_pose()[:3]) > 0.01:
        current = client.get_ee_pose()
        current_position = current[:3]
        delta = (goal[:3] - current_position) / np.linalg.norm(goal[:3] - current_position)
        current[:3] += delta * speed / freq
        client.update_desired_ee_pose(current)
        print(f"Position error: {np.linalg.norm(goal[:3] - current_position):.4f}")
        limiter.sleep()

@hydra.main(config_path="configs", config_name="inference", version_base="1.2")
def main(cfg):
    client = FrankaClient(server_ip='10.228.255.79')

    cbet_model = hydra.utils.instantiate(cfg.model).to(cfg.device)
    if cfg.load_path:
        cbet_model.load_model(Path(cfg.load_path))
    cbet_model.eval()

    rgb_queue = deque(maxlen=5)
    depth_queue = deque(maxlen=5)
    low_dim_queue = deque(maxlen=5)

    camera = RealSense()
    camera.start()

    Kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0]) * 1.0
    Kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0]) * 1.0
    client.start_cartesian_impedance(Kx=Kx, Kxd=Kxd)

    grasping = False
    last_time = None

    with torch.no_grad():
        while True:
            # Retrieve and preprocess frames
            rgb, depth = camera.get_latest_frames()
            rgb = cv2.resize(rgb, (320, 240))
            depth = cv2.resize(depth, (320, 240)).astype(np.float32) / 1000.0

            current_ee_pose = client.get_ee_pose()
            current_gripper_width = client.get_gripper_width()
            
            position = current_ee_pose[:3]
            rotation_quat = R.from_rotvec(current_ee_pose[3:]).as_quat()
            low_dim = np.concatenate([position, rotation_quat, [current_gripper_width]]).reshape(-1)

            rgb_queue.append(rgb)
            depth_queue.append(depth)
            low_dim_queue.append(low_dim)

            if len(rgb_queue) < 5:
                continue

            rgb_torch = torch.tensor(list(rgb_queue), dtype=torch.float32).unsqueeze(0).to(cfg.device)
            depth_torch = torch.tensor(list(depth_queue), dtype=torch.float32).unsqueeze(0).to(cfg.device)
            low_dim_torch = torch.tensor(list(low_dim_queue), dtype=torch.float32).unsqueeze(0).to(cfg.device)
            

            # Model inference
            predicted_act, _, _ = cbet_model(rgb_torch, depth_torch, low_dim_torch, None, None)

            # Update robot pose
            delta = predicted_act[-1, 0, :3].cpu().numpy()
            mag = np.linalg.norm(delta)
            # if mag > 0.01:
            #     delta /= mag
            #     delta *= 0.01
            new_ee_pose = np.array(current_ee_pose)
            new_ee_pose[:3] += delta

            delta_quat = predicted_act[-1, 0, 3:7].cpu().numpy()
            delta_quat /= np.linalg.norm(delta_quat)
            delta_rot = R.from_quat(delta_quat)
            current_rot = R.from_rotvec(current_ee_pose[3:])
            new_rot_vec = (delta_rot * current_rot).as_rotvec()

            new_ee_pose[3:] = new_rot_vec
            new_ee_pose[2] = max(0.13, new_ee_pose[2])
            client.update_desired_ee_pose(new_ee_pose)

            # Update gripper
            grip_width = predicted_act[-1, 0, -1].item()
            client.set_gripper_width(grip_width)

            # if grip_width < 0.075 and not grasping:
            #     client.set_gripper_width(0.055)
            #     grasping = True
            # elif grasping and grip_width > 0.056:
            #     client.set_gripper_width(0.085)
            #     grasping = False

            # Logging
            print(f"Delta: {delta}, Grip width: {grip_width:.3f}, Grasping: {grasping}, Magnitude: {mag:.4f}")

            if last_time:
                print(f"FPS: {1 / (time.time() - last_time):.2f}")
            last_time = time.time()

if __name__ == "__main__":
    main()
