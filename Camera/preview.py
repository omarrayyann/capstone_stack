from Camera.RealSense import RealSense
import time 
import cv2

def main():

    camera = RealSense()

    try:
        camera.start()
        print("Streaming from RealSense D455... Press 'q' to quit.")

        while True:

            color_image, depth_image = camera.get_latest_frames()

            if color_image is not None and depth_image is not None:
                combined_image = camera.get_combined_image()

                cv2.imshow('RealSense RGB and Depth', combined_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
