import cv2
import os
from datetime import datetime, timedelta
from pymediainfo import MediaInfo

def get_creation_datetime(video_file):
    """Extract the creation datetime from video metadata."""
    media_info = MediaInfo.parse(video_file)
    for track in media_info.tracks:
        if track.track_type == "General":
            creation_time = track.tagged_date or track.encoded_date
            if creation_time:
                
                if "UTC" in creation_time:
                    creation_time = creation_time.replace("UTC ", "")
                    return datetime.strptime(creation_time, "%Y-%m-%d %H:%M:%S")
    return None

def extract_global_timestamps(video_file):
    """Extract global timestamps from video file metadata."""
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file}.")
        return

    creation_datetime = get_creation_datetime(video_file)
    if not creation_datetime:
        print("Error: Could not extract creation time from metadata.")
        cap.release()
        return

    print("Extracting frames and global timestamps from video...")

    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_timestamps = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        
        global_timestamp = creation_datetime + timedelta(seconds=frame_count / fps)
        frame_timestamps.append(global_timestamp)
        frame_count += 1

    cap.release()
    return frame_timestamps

def visualize_mp4(video_file):
    """Visualize video frames with global timestamps from .mp4 file."""
    if not os.path.exists(video_file):
        print(f"Error: File {video_file} does not exist.")
        return

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file}.")
        return

    creation_datetime = get_creation_datetime(video_file)
    if not creation_datetime:
        print("Error: Could not extract creation time from metadata.")
        cap.release()
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    print("Visualizing video frames with global timestamps. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        
        global_timestamp = creation_datetime + timedelta(seconds=frame_count / fps)

        
        display_frame = frame.copy()
        timestamp_str = global_timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        cv2.putText(display_frame, f"Timestamp: {timestamp_str}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('MP4 Visualization', display_frame)
        frame_count += 1

        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    index = 8
    video_file = f"/home/lambda1/Documents/GitHub/capstone_stack/sam-pipeline/Synchronization/data/gopro/{index}.mp4"
    visualize_mp4(video_file)
