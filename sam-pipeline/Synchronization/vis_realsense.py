import numpy as np
import cv2
import os
from datetime import datetime

def visualize_npy(rgb_file, timestamp_file):
    """Visualize RGB frames stored in a .npy file along with their global timestamps."""
    if not os.path.exists(rgb_file) or not os.path.exists(timestamp_file):
        print(f"Error: Files {rgb_file} and/or {timestamp_file} do not exist.")
        return

    
    rgb_frames = np.load(rgb_file)
    timestamps = np.load(timestamp_file)

    if len(rgb_frames) != len(timestamps):
        print("Error: Number of frames and timestamps do not match.")
        return

    print("Visualizing frames with global timestamps. Press 'q' to quit.")

    prev_timestamp = 0
    for frame, timestamp in zip(rgb_frames, timestamps):
        
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        
        global_timestamp = datetime.utcfromtimestamp(timestamp)
        formatted_timestamp = global_timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        
        overlay_frame = bgr_frame.copy()
        cv2.putText(overlay_frame, f"Timestamp: {formatted_timestamp}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Frame Visualization', overlay_frame)

        
        if prev_timestamp > 0:
            delay = int((timestamp - prev_timestamp) * 1000)
            if delay > 0:
                key = cv2.waitKey(delay)
            else:
                key = cv2.waitKey(1)
        else:
            key = cv2.waitKey(1)

        prev_timestamp = timestamp

        if key & 0xFF == ord('q'):  
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    index = 8
    rgb_file = f"/home/lambda1/Documents/GitHub/capstone_stack/sam-pipeline/Synchronization/data/realsense/rgbs_{index}.npy"
    timestamp_file = f"/home/lambda1/Documents/GitHub/capstone_stack/sam-pipeline/Synchronization/data/realsense/timestamps_{index}.npy"
    visualize_npy(rgb_file, timestamp_file)
