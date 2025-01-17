import os
import time
import numpy as np
import cv2
from Camera.RealSense import RealSense

FOLDER_PATH = "/home/lambda1/Documents/sam-pipeline/data/realsense"

def get_next_index(prefix):
    """Find the next available index for the file prefix."""
    index = 0
    while os.path.exists(f"{FOLDER_PATH}/{prefix}_{index}.npy"):
        index += 1
    return index

if __name__ == "__main__":
    camera = RealSense()
    rgb_frames = []
    timestamps = []

    try:
        camera.start()

        start_time = time.time()
        while True:
            color_image, _, timestamp = camera.get_latest_frames_timestamp()
            if color_image is not None:
                
                rgb_frames.append(color_image)
                timestamps.append(timestamp)

                
                cv2.imshow('RGB Stream', cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))

            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted! Saving frames and timestamps...")
        next_index = get_next_index("rgbs")
        np.save(f"{FOLDER_PATH}/rgbs_{next_index}.npy", np.array(rgb_frames))
        np.save(f"{FOLDER_PATH}/timestamps_{next_index}.npy", np.array(timestamps))
        print(f"Frames saved as rgbs_{next_index}.npy and timestamps_{next_index}.npy.")

    finally:
        camera.stop()
        cv2.destroyAllWindows()
