import cv2
import numpy as np
import os
from datetime import datetime, timedelta
from bisect import bisect_left
from pymediainfo import MediaInfo


def get_creation_datetime(video_file):
    """
    Extract the creation datetime from the GoPro video metadata using pymediainfo.
    Returns a datetime object or None if unavailable.
    """
    media_info = MediaInfo.parse(video_file)
    for track in media_info.tracks:
        if track.track_type == "General":
            creation_time = track.tagged_date or track.encoded_date
            if creation_time:
                # Handle string format such as "UTC 2023-06-30 10:15:30"
                if "UTC" in creation_time:
                    creation_time = creation_time.replace("UTC ", "")
                try:
                    return datetime.strptime(creation_time, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    pass
    return None


def visualize_synced_optimized(gopro_video, realsense_rgb_file, realsense_timestamp_file, time_threshold=0.1):
    """
    Visualize GoPro and RealSense frames side by side in real-time.
    This version reads the GoPro video frame-by-frame (low-memory approach)
    and always calls cv2.waitKey(1) to ensure the window is updated.

    :param gopro_video: Path to the GoPro video file.
    :param realsense_rgb_file: Path to .npy with RealSense RGB frames (shape: [N, H, W, 3]).
    :param realsense_timestamp_file: Path to .npy with RealSense timestamps (in milliseconds).
    :param time_threshold: Maximum allowed time difference (in seconds) for a valid match.
    """

    # -----------------------------
    # 1. Check input file existence
    # -----------------------------
    if not os.path.exists(gopro_video):
        print(f"Error: GoPro video file {gopro_video} not found.")
        return
    if not os.path.exists(realsense_rgb_file):
        print(f"Error: RealSense RGB file {realsense_rgb_file} not found.")
        return
    if not os.path.exists(realsense_timestamp_file):
        print(f"Error: RealSense timestamps file {realsense_timestamp_file} not found.")
        return

    # -----------------------------
    # 2. Load RealSense data
    # -----------------------------
    print("Loading RealSense frames and timestamps...")
    rs_rgb_frames = np.load(realsense_rgb_file)  # Expected shape: [N, H, W, 3]
    rs_timestamps_ms = np.load(realsense_timestamp_file)  # Expected shape: [N]
    # Convert timestamps from milliseconds to datetime objects (UTC)
    rs_timestamps = [datetime.utcfromtimestamp(ts) for ts in rs_timestamps_ms]

    # -----------------------------
    # 3. Open the GoPro video
    # -----------------------------
    cap = cv2.VideoCapture(gopro_video)
    if not cap.isOpened():
        print(f"Error: Could not open video file {gopro_video}.")
        return

    # Extract creation datetime from video metadata
    creation_datetime = get_creation_datetime(gopro_video)
    if creation_datetime is None:
        print("Error: Could not extract creation time from GoPro metadata.")
        cap.release()
        return

    # Get FPS and total frame count for simulation and logging
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("Warning: Total frame count is unknown. Progress logging may be off.")

    print("Starting synchronized visualization...")

    # -----------------------------
    # 4. Prepare for real-time simulation
    # -----------------------------
    real_start_time = datetime.now()
    simulation_start_time = None  # Will be set once the first GoPro frame is read

    # We'll maintain a pointer for RealSense frames
    rs_idx = 0
    len_rs = len(rs_timestamps)

    last_progress_percent = 0  # Track progress percentage
    frame_idx = 0  # Current GoPro frame index

    while True:
        ret, gopro_frame = cap.read()
        if not ret:
            # No more frames
            print("Finished reading video.")
            break

        # Compute current global timestamp for this GoPro frame
        gopro_timestamp = creation_datetime + timedelta(seconds=frame_idx / fps)
        if simulation_start_time is None:
            simulation_start_time = gopro_timestamp

        # -----------------------------
        # 4a. Find the closest matching RealSense timestamp
        # -----------------------------
        # Keep rs_idx within valid range:
        rs_idx = min(rs_idx, len_rs - 1)
        # Advance rs_idx while next RealSense timestamp is closer than the current one
        while rs_idx < len_rs - 1 and rs_timestamps[rs_idx] < gopro_timestamp:
            next_diff = abs((rs_timestamps[rs_idx + 1] - gopro_timestamp).total_seconds())
            curr_diff = abs((rs_timestamps[rs_idx] - gopro_timestamp).total_seconds())
            if next_diff < curr_diff:
                rs_idx += 1
            else:
                break

        time_diff = abs((rs_timestamps[rs_idx] - gopro_timestamp).total_seconds())
        if time_diff <= time_threshold:
            rs_frame_bgr = cv2.cvtColor(rs_rgb_frames[rs_idx], cv2.COLOR_RGB2BGR)
        else:
            # If no good match, create a black frame of the same size as gopro_frame.
            rs_frame_bgr = np.zeros_like(gopro_frame)

        # -----------------------------
        # 4b. Simulate real-time playback synchronization
        # -----------------------------
        real_elapsed_time = (datetime.now() - real_start_time).total_seconds()
        simulation_elapsed_time = (gopro_timestamp - simulation_start_time).total_seconds()
        delay = simulation_elapsed_time - real_elapsed_time

        # Always call a minimal waitKey(1) so that the display updates.
        key = cv2.waitKey(1) & 0xFF  # minimal wait

        if delay > 0:
            # Additional delay if simulation is ahead of real time
            cv2.waitKey(int(delay * 1000))

        # -----------------------------
        # 4c. Combine and display frames
        # -----------------------------
        # Resize for side-by-side view
        target_width, target_height = 640, 360
        gopro_resized = cv2.resize(gopro_frame, (target_width, target_height))
        rs_resized = cv2.resize(rs_frame_bgr, (target_width, target_height))
        combined_frame = np.hstack((gopro_resized, rs_resized))

        # Overlay timestamps
        gopro_time_str = gopro_timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        rs_time_str = (rs_timestamps[rs_idx].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                       if time_diff <= time_threshold else "NO MATCH")
        cv2.putText(combined_frame, f"GoPro: {gopro_time_str}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined_frame, f"RealSense: {rs_time_str}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("GoPro and RealSense Synced", combined_frame)

        # Also check for ESC key or window close
        if key == 27 or cv2.getWindowProperty("GoPro and RealSense Synced", cv2.WND_PROP_VISIBLE) < 1:
            print("Exiting per user request.")
            break

        # -----------------------------
        # 4d. Log progress
        # -----------------------------
        frame_idx += 1
        if total_frames > 0:
            progress_percent = int((frame_idx / total_frames) * 100)
            if progress_percent != last_progress_percent:
                print(f"Progress: {progress_percent}%")
                last_progress_percent = progress_percent

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    index = 8
    gopro_video = f"/home/lambda1/Documents/GitHub/capstone_stack/sam-pipeline/Synchronization/data/gopro/{index}.mp4"
    realsense_rgb_file = f"/home/lambda1/Documents/GitHub/capstone_stack/sam-pipeline/Synchronization/data/realsense/rgbs_{index}.npy"
    realsense_timestamp_file = f"/home/lambda1/Documents/GitHub/capstone_stack/sam-pipeline/Synchronization/data/realsense/timestamps_{index}.npy"
    
    visualize_synced_optimized(
        gopro_video,
        realsense_rgb_file,
        realsense_timestamp_file,
        time_threshold=0.01  # Allowable time difference in seconds
    )
