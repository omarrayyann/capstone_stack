from RealSense import RealSense
import time 
import cv2

def main():
    # Initialize the RealSense camera
    camera = RealSense()

    try:
        # Start the camera pipeline
        camera.start()
        print("Streaming from RealSense D455... Press 'q' to quit.")
        last_time = None
        while True:
            
            # Get the latest frames
            color_image, depth_image = camera.get_latest_frames()

            if last_time is not None:
                print(1/(time.time()-last_time))
            
            last_time = time.time()

            if color_image is not None and depth_image is not None:
                # Get the combined image for visualization
                combined_image = camera.get_combined_image()

                # Display the combined images
                cv2.imshow('RealSense RGB and Depth', combined_image)

            # Exit loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop the camera pipeline and close OpenCV windows
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
