import pyrealsense2 as rs
import numpy as np
import cv2

class RealSense:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable depth and color streams
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Internal variables to store the latest frames
        self.depth_image = None
        self.color_image = None
        self.running = False

    def start(self):
        """Start the RealSense camera pipeline."""
        self.pipeline.start(self.config)
        self.running = True

    def stop(self):
        """Stop the RealSense camera pipeline."""
        if self.running:
            self.pipeline.stop()
            self.running = False

    def get_latest_frames(self):
        """Retrieve the latest color and depth frames."""
        if not self.running:
            raise RuntimeError("Pipeline is not running. Please call start() before getting frames.")

        frames = self.pipeline.wait_for_frames()
        
        # Get depth and color frames
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None

        # Convert frames to numpy arrays.
        # IMPORTANT: Convert the color frame from BGR to RGB so that it is stored in RGB order.
        self.color_image = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)
        self.depth_image = np.asanyarray(depth_frame.get_data())
        
        return self.color_image, self.depth_image

    def get_combined_image(self):
        """Retrieve a combined color and depth visualization."""
        if self.color_image is None or self.depth_image is None:
            raise RuntimeError("No frames available. Please call get_latest_frames() first.")

        # For visualization, convert the raw depth image to an 8-bit image.
        # Typically, depth is 16-bit so we scale it down.
        depth_8bit = cv2.convertScaleAbs(self.depth_image, alpha=0.03)
        # Convert the depth image to a 3-channel image so it can be stacked with the RGB image.
        depth_gray = cv2.cvtColor(depth_8bit, cv2.COLOR_GRAY2RGB)

        # Stack the RGB image (which is already in RGB order) and the processed depth image horizontally.
        combined_images = np.hstack((self.color_image, depth_gray))
        return combined_images

if __name__ == "__main__":
    # Instantiate the RealSense camera.
    camera = RealSense()

    try:
        camera.start()
        print("Streaming from RealSense... Press 'q' to quit.")
        
        while True:
            # Get the latest frames.
            color_image, depth_image = camera.get_latest_frames()

            if color_image is not None and depth_image is not None:
                # Get the combined image for visualization.
                combined_image = camera.get_combined_image()

                # For display, convert the combined image from RGB back to BGR because OpenCV's imshow expects BGR.
                cv2.imshow('RealSense RGB and Depth', cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
                
            # Exit loop when 'q' is pressed.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()
