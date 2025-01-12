import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable depth and color streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    ctx = rs.context()
    devices = ctx.query_devices()
    for dev in devices:
        dev.hardware_reset()

    
    try:
        # Start the pipeline
        pipeline.start(config)
        print("Streaming from RealSense D455... Press 'q' to quit.")
        
        while True:
            # Wait for a new set of frames from the camera
            frames = pipeline.wait_for_frames()
            
            # Get depth and color frames
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Apply colormap to depth image for better visualization
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            # Stack both images horizontally for display
            combined_images = np.hstack((color_image, depth_colormap))
            
            # Display the combined images
            cv2.imshow('RealSense RGB and Depth', combined_images)
            
            # Exit loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop the pipeline and close OpenCV windows
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
