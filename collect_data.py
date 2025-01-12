import numpy as np
import cv2
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from playsound import playsound
from Client.FrankaClient import FrankaClient
from Camera.RealSense import RealSense
import threading

def process_ee_pose(ee_pose):
    position = ee_pose[0:3]
    quat = R.from_rotvec(ee_pose[3:6]).as_quat()
    return position, quat

save_folder = Path("/home/franka/Desktop/franka_stack/data")
save_folder.mkdir(parents=True, exist_ok=True)

camera = RealSense()
franka = FrankaClient()

def save_episode(ep_rgb, ep_depth, ep_actions, ep_timestamps):
    npz_files = list(save_folder.glob("*.npz"))
    if npz_files:
        indices = []
        for f in npz_files:
            try:
                idx = int(f.stem)
                indices.append(idx)
            except ValueError:
                continue
        new_index = max(indices) + 1 if indices else 0
    else:
        new_index = 0
    save_path = save_folder / f"{new_index}.npz"
    np.savez_compressed(save_path,
                        rgb_frames=np.array(ep_rgb),
                        depth_frames=np.array(ep_depth),
                        actions=np.array(ep_actions),
                        timestamps=np.array(ep_timestamps))
    print(f"Episode data saved to {save_path}")

def wait_for_episode_end(stop_event):
    input("Press Enter to end the current episode...")
    stop_event.set()

print("Data collection starting. Type 'q' (followed by Enter) at the prompt to quit overall.")
camera.start()

try:
    while True:
        cmd = input("Press Enter to start a new episode, or type 'q' to quit: ").strip().lower()
        if cmd == 'q':
            break
        playsound("/home/franka/Desktop/franka_stack/countdown.mp3")
        print("Episode started.")
        ep_rgb, ep_depth, ep_actions, ep_timestamps = [], [], [], []
        ep_start = time.time()
        stop_event = threading.Event()
        stopper = threading.Thread(target=wait_for_episode_end, args=(stop_event,))
        stopper.start()
        while not stop_event.is_set():
            try:
                ee_pose = franka.get_ee_pose()
                gripper_width = franka.get_gripper_width()
                color_frame, depth_frame = camera.get_latest_frames()
            except RuntimeError as e:
                print("Warning:", e)
                break
            if color_frame is None or depth_frame is None:
                continue
            cur_time = time.time() - ep_start
            position, quat = process_ee_pose(ee_pose)
            action = np.concatenate([position, quat, np.array([gripper_width])], axis=0)
            ep_rgb.append(color_frame)
            ep_depth.append(depth_frame)
            ep_actions.append(action)
            ep_timestamps.append(cur_time)
        stopper.join()
        print("Episode done. Saving data...")
        save_episode(ep_rgb, ep_depth, ep_actions, ep_timestamps)
except KeyboardInterrupt:
    print("KeyboardInterrupt detected.")
finally:
    camera.stop()
    franka.close()

print("Data collection complete.")
