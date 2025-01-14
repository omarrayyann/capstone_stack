import matplotlib.pyplot as plt
import random
import time
from pathlib import Path
from RealSense import *
from FrankaClient import *
from scipy.spatial.transform import Rotation as R

import hydra
import numpy as np
import torch
from vqvae.vqvae import *
from torch.utils.data import DataLoader

from collections import deque

@hydra.main(config_path="configs", config_name="inference", version_base="1.2")
def main(cfg):

    client = FrankaClient(
    server_ip='127.0.0.1',
)

    cbet_model = hydra.utils.instantiate(cfg.model).to(cfg.device)
    
    if cfg.load_path:
        cbet_model.load_model(Path(cfg.load_path))

    cbet_model.eval()

    rgb_queue = deque(maxlen=5)
    depth_queue = deque(maxlen=5)
    camera = RealSense()
    camera.start()
    

    Kx = (np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0]) * 0.4)
    Kxd = (np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0]) * 0.5)

    # Start the cartesian impedance controller
    client.start_cartesian_impedance(Kx=Kx, Kxd=Kxd)



    with torch.no_grad():
        last_time = None
        while True:

            

            rgb, depth = camera.get_latest_frames()

            rgb = cv2.resize(rgb, (320, 240))
            depth = cv2.resize(depth, (320, 240)).astype(np.float32) / 1000.0

            rgb_queue.append(rgb)
            depth_queue.append(depth)

            if len(rgb_queue) < 5:
                continue
            
            rgb_torch = torch.tensor(rgb_queue)
            depth_torch = torch.tensor(depth_queue)

            rgb = torch.stack(list(rgb_torch), dim=0).unsqueeze(0)
            depth = torch.stack(list(depth_torch), dim=0).unsqueeze(0)

            predicted_act, _, _ = cbet_model(rgb, depth, None, None)

            current_ee_pose = client.get_ee_pose()
            new_ee_pose = np.array(current_ee_pose)
            delta = np.array(predicted_act[-1, 0, 0:3].flatten())
            max_delta = np.max(np.abs(delta))
            if max_delta > 0.05:
                delta = delta / max_delta * 0.05
            print(delta)
            new_ee_pose[0:3] = current_ee_pose[0:3] + delta

            current_rot_vec = current_ee_pose[3:]
            current_rot = R.from_rotvec(current_rot_vec)

            delta_quat = np.array(predicted_act[-1, 0, 3:7].flatten())

            delta_rot = R.from_quat(delta_quat)

            new_rot = delta_rot * current_rot

            new_rot_vec = new_rot.as_rotvec()

            new_ee_pose[3:] = new_rot_vec
            

            new_ee_pose[2] = max(0.127, new_ee_pose[2])

            print("Before: ", current_ee_pose)
            client.update_desired_ee_pose(new_ee_pose)
            print("After: ", new_ee_pose)
            print("Action: ", predicted_act[-1, 0, -1])

            grip_width = predicted_act[-1, 0, -1].item()
            client.set_gripper_width(grip_width)


            if last_time is None:
                last_time = time.time()
            else:
                print(f"FPS: {1/(time.time() - last_time)}")
                last_time = time.time()

            # # Extract the xyz vector
            # print(predicted_act[-1,0,0:3])
            # xyz = predicted_act[-1, 0, 0:3]
            # xyz = np.array(xyz)*100.0

            # # Extract the grip value
            # grip = predicted_act[-1, 0, -1]

            # print(predicted_act[:, 0, -1])

            # # Prepare the RGB image
            # rgb = rgb[0][0].cpu().numpy().astype(np.uint8)

            # # Define the origin of the vector (center of the image)
            # origin_x, origin_y = rgb.shape[1] // 2, rgb.shape[0] // 2

            # # Scale the xyz vector for better visualization in the image
            # scale_factor = 50  # Adjust this factor to make the vector visible in the image
            # scaled_xyz = xyz * scale_factor

            # # Calculate the vector's end point
            # end_x = origin_x + scaled_xyz[0]
            # end_y = origin_y - scaled_xyz[1]  # Note: subtract y to match image coordinates

            # # # Plot the RGB image
            # plt.imshow(rgb)

            # # Plot the xyz vector
            # plt.quiver(
            #     origin_x, origin_y, 
            #     scaled_xyz[0], scaled_xyz[1],  # Negate y for image coordinates
            #     angles='xy', scale_units='xy', scale=1, color='red', width=0.01  # Slightly thicker arrow
            # )

            # # Display the grip value on the image
            # plt.text(
            #     10, 10,  # Position near the top-left corner
            #     f'Grip: {grip:.2f}', 
            #     color='yellow', 
            #     fontsize=12, 
            #     bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3')
            # )

            # # Show the plot
            # plt.axis('off')  # Turn off axis for better visualization
            # plt.show()


if __name__ == "__main__":
    main()
