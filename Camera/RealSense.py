import pyrealsense2 as rs
import numpy as np
import cv2

class RealSense:
    def __init__(self):
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.depth_image = None
        self.color_image = None
        self.running = False

    def start(self):
        
        self.pipeline.start(self.config)
        self.running = True

    def stop(self):
        
        if self.running:
            self.pipeline.stop()
            self.running = False

    def get_latest_frames(self):
        
        if not self.running:
            raise RuntimeError("Pipeline is not running. Please call start() before getting frames.")
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        self.color_image = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)
        self.depth_image = np.asanyarray(depth_frame.get_data())
        return self.color_image, self.depth_image
        

    def get_combined_image(self):
        
        if self.color_image is None or self.depth_image is None:
            raise RuntimeError("No frames available. Please call get_latest_frames() first.")
        depth_8bit = cv2.convertScaleAbs(self.depth_image, alpha=0.03)
        depth_gray = cv2.cvtColor(depth_8bit, cv2.COLOR_GRAY2RGB)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_RGB2BGR)
        combined_images = np.hstack((self.color_image, depth_gray))
        return combined_images

if __name__ == "__main__":
    
    camera = RealSense()

    try:
        camera.start()
        print("Streaming from RealSense... Press 'q' to quit.")
        
        while True:
            
            color_image, depth_image = camera.get_latest_frames()
            if color_image is not None and depth_image is not None:
                combined_image = camera.get_combined_image()
                cv2.imshow('RealSense RGB and Depth', cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()
