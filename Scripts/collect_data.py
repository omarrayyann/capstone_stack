import numpy as np
import cv2
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from playsound import playsound
from Client.FrankaClient import FrankaClient
from Camera.RealSense import RealSense
import threading
from flask import Flask, request, jsonify
from loop_rate_limiters import RateLimiter

def process_ee_pose(ee_pose):
    position = ee_pose[0:3]
    quat = R.from_rotvec(ee_pose[3:6]).as_quat()
    return position, quat

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




save_folder = Path("/home/franka/Desktop/capstone_stack/data")
save_folder.mkdir(parents=True, exist_ok=True)


episode_stop_event = threading.Event()


app = Flask(__name__)

@app.route('/trigger', methods=['POST'])
def trigger_episode():
    """
    When a POST request is sent to this endpoint, it will signal the current episode to stop.
    """
    if not episode_stop_event.is_set():
        episode_stop_event.set()
        print("Trigger received: stopping the current episode.")
        return jsonify({"status": "Triggered episode end"}), 200
    else:
          
        return jsonify({"status": "Already stopping"}), 200




def recording_loop():
    """
    This loop continuously records episodes. The start of an episode is implicit
    (immediately after finishing the prior one) and an HTTP POST to /trigger signals
    the current episode to stop and be saved.
    """
    
    camera = RealSense()
    franka = FrankaClient()
    camera.start()
    rate = RateLimiter(frequency=10, warn=False)

    try:
        episode_num = 0
        print("Data collection started. Trigger new episodes via HTTP POST requests to '/trigger'.")
        while True:
            
            if episode_num > 0:  
                playsound("/home/franka/Desktop/capstone_stack/5seconds.mp3")
            else:
                time.sleep(10)
                playsound("/home/franka/Desktop/capstone_stack/5seconds.mp3")
            print(f"Episode {episode_num} started.")

            
            ep_rgb, ep_depth, ep_actions, ep_timestamps = [], [], [], []
            ep_start = time.time()

            
            episode_stop_event.clear()

            
            while not episode_stop_event.is_set():
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

                color_frame_resized = cv2.resize(color_frame, (320, 240))
                depth_frame_resized = cv2.resize(depth_frame, (320, 240))
                
                ep_rgb.append(color_frame_resized)
                ep_depth.append(depth_frame_resized)
                ep_actions.append(action)
                ep_timestamps.append(cur_time)
                
                rate.sleep()

            
            if episode_num > 0 or (episode_num == 0 and len(ep_rgb) > 0):
                print(f"Episode {episode_num} complete. Saving data...")
                save_episode(ep_rgb, ep_depth, ep_actions, ep_timestamps)
            else:
                print("No data recorded for this episode.")

            episode_num += 1
            print("Starting a new episode...\n")
            
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Exiting...")
    finally:
        camera.stop()
        franka.close()
        print("Data collection complete.")




if __name__ == "__main__":
    
    flask_thread = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False))
    flask_thread.daemon = True
    flask_thread.start()

    
    recording_loop()
