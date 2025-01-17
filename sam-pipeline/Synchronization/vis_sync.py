import numpy as np
import cv2
import os
from datetime import datetime, timedelta

def get_gopro_timestamps(video_file):
    """Extract global timestamps from GoPro video metadata."""
    from pymediainfo import MediaInfo

    def get_creation_datetime(video_file):
        media_info = MediaInfo.parse(video_file)
        for track in media_info.tracks:
            if track.track_type == "General":
                creation_time = track.tagged_date or track.encoded_date
                if creation_time:
                    if "UTC" in creation_time:
                        creation_time = creation_time.replace("UTC ", "")
                        return datetime.strptime(creation_time, "%Y-%m-%d %H:%M:%S")
        return None

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file}.")
        return None, None

    creation_datetime = get_creation_datetime(video_file)
    if not creation_datetime:
        print("Error: Could not extract creation time from GoPro metadata.")
        cap.release()
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_timestamps = []
    frames = []

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        global_timestamp = creation_datetime + timedelta(seconds=frame_count / fps)
        frame_timestamps.append(global_timestamp)
        frame_count += 1

    cap.release()
    return frames, frame_timestamps

def synchronize_frames(gopro_timestamps, rs_timestamps):
    """
    Synchronize GoPro timestamps with RealSense timestamps using binary search.
    Returns indices of matching frames in GoPro and RealSense.
    """
    from bisect import bisect_left

    matched_indices = []
    for gopro_idx, gopro_time in enumerate(gopro_timestamps):
        
        rs_idx = bisect_left(rs_timestamps, gopro_time)
        if rs_idx == 0:
            closest_rs_idx = rs_idx
        elif rs_idx == len(rs_timestamps):
            closest_rs_idx = rs_idx - 1
        else:
            before = rs_timestamps[rs_idx - 1]
            after = rs_timestamps[rs_idx]
            closest_rs_idx = rs_idx if abs(after - gopro_time) < abs(before - gopro_time) else rs_idx - 1

        
        time_diff = abs((gopro_time - rs_timestamps[closest_rs_idx]).total_seconds())
        if time_diff < 0.1:  
            matched_indices.append((gopro_idx, closest_rs_idx))

    return matched_indices

def synchronize_frames(gopro_timestamps, rs_timestamps):
    """
    Synchronize GoPro timestamps with RealSense timestamps using binary search.
    Returns indices of matching frames in GoPro and RealSense.
    """
    from bisect import bisect_left

    matched_indices = []
    for gopro_idx, gopro_time in enumerate(gopro_timestamps):
        
        rs_idx = bisect_left(rs_timestamps, gopro_time)
        if rs_idx == 0:
            closest_rs_idx = rs_idx
        elif rs_idx == len(rs_timestamps):
            closest_rs_idx = rs_idx - 1
        else:
            before = rs_timestamps[rs_idx - 1]
            after = rs_timestamps[rs_idx]
            closest_rs_idx = rs_idx if abs(after - gopro_time) < abs(before - gopro_time) else rs_idx - 1

        
        time_diff = abs((gopro_time - rs_timestamps[closest_rs_idx]).total_seconds())
        if time_diff < 0.1:  
            matched_indices.append((gopro_idx, closest_rs_idx))

    return matched_indices

def visualize_synced(gopro_video, realsense_rgb_file, realsense_timestamp_file):
    """Visualize GoPro video and RealSense frames synced by timestamps in real-time."""
    
    if not os.path.exists(realsense_rgb_file) or not os.path.exists(realsense_timestamp_file):
        print(f"Error: RealSense files {realsense_rgb_file} and/or {realsense_timestamp_file} do not exist.")
        return

    rgb_frames = np.load(realsense_rgb_file)
    rs_timestamps = np.load(realsense_timestamp_file) / 1000.0
    rs_timestamps = [datetime.utcfromtimestamp(ts) for ts in rs_timestamps]

    
    gopro_frames, gopro_timestamps = get_gopro_timestamps(gopro_video)
    if gopro_frames is None or gopro_timestamps is None:
        return

    
    matched_indices = synchronize_frames(gopro_timestamps, rs_timestamps)

    
    real_start_time = datetime.now()
    first_gopro_time = gopro_timestamps[matched_indices[0][0]]
    first_rs_time = rs_timestamps[matched_indices[0][1]]
    simulation_start_time = min(first_gopro_time, first_rs_time)

    for gopro_idx, rs_idx in matched_indices:
        
        gopro_time = gopro_timestamps[gopro_idx]
        rs_time = rs_timestamps[rs_idx]

        
        real_elapsed_time = (datetime.now() - real_start_time).total_seconds()

        
        simulation_elapsed_time = (gopro_time - simulation_start_time).total_seconds()

        
        delay = simulation_elapsed_time - real_elapsed_time
        if delay > 0:
            cv2.waitKey(int(delay * 1000))  

        
        gopro_frame = gopro_frames[gopro_idx]
        rs_frame = cv2.cvtColor(rgb_frames[rs_idx], cv2.COLOR_RGB2BGR)

        gopro_resized = cv2.resize(gopro_frame, (640, 360))
        rs_resized = cv2.resize(rs_frame, (640, 360))
        combined_frame = np.hstack((gopro_resized, rs_resized))

        
        gopro_time_str = gopro_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        rs_time_str = rs_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        cv2.putText(combined_frame, f"GoPro: {gopro_time_str}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined_frame, f"RealSense: {rs_time_str}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("GoPro and RealSense Synced", combined_frame)

        
        if cv2.getWindowProperty("GoPro and RealSense Synced", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()



if __name__ == "__main__":
    index = 5
    gopro_video = f"/home/lambda1/Documents/GitHub/capstone_stack/sam-pipeline/Synchronization/data/gopro/{index}.mp4"
    realsense_rgb_file = f"/home/lambda1/Documents/GitHub/capstone_stack/sam-pipeline/Synchronization/data/realsense/rgbs_{index}.npy"
    realsense_timestamp_file = f"/home/lambda1/Documents/GitHub/capstone_stack/sam-pipeline/Synchronization/data/realsense/timestamps_{index}.npy"
    visualize_synced(gopro_video, realsense_rgb_file, realsense_timestamp_file)
