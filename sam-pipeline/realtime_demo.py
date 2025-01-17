import time
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_camera_predictor
from Camera.RealSense import RealSense

checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
predictor = build_sam2_camera_predictor(model_cfg, checkpoint)

if_init = False

camera = RealSense()
camera.start()

selected_points = []
labels = []
selecting = True

# Callback function to capture mouse click
def select_point(event, x, y, flags, param):
    global selected_points, labels, selecting
    if selecting and event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append((x, y))
        labels.append(1)  # Label all selected points with 1 for segmentation
    elif selecting and event == cv2.EVENT_RBUTTONDOWN:
        selected_points.append((x, y))
        labels.append(0)  # Label point as 0 to exclude from segmentation

cv2.namedWindow("Select Points")
cv2.setMouseCallback("Select Points", select_point)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    
    while True:

        start_time = time.time()

        rgb, _ = camera.get_latest_frames()
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for consistency

        if selecting:
            temp_frame = rgb.copy()
            for point, label in zip(selected_points, labels):
                color = (0, 255, 0) if label == 1 else (0, 0, 255)  # Green for 1, Red for 0
                cv2.circle(temp_frame, point, 5, color, -1)

            cv2.imshow("Select Points", cv2.cvtColor(temp_frame, cv2.COLOR_RGB2BGR))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):  # Press 's' to stop selecting and start segmenting
                selecting = False
                cv2.destroyWindow("Select Points")

        else:
            if not if_init:
                predictor.load_first_frame(rgb)
                points = np.array(selected_points, dtype=np.float32)
                labels_np = np.array(labels, dtype=np.int32)
                _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(0, 1, points=points, labels=labels_np)
                if_init = True

            out_obj_ids, out_mask_logits = predictor.track(rgb)
            # Overlay mask logits on rgb for visualization
            mask = (out_mask_logits[0][0].cpu().numpy() > 0).astype(np.uint8) * 255
            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Convert mask to 3 channels
            mask_colored[:, :, 1:] = 0  # Retain only the red channel for clarity
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            visualized_frame = cv2.addWeighted(rgb, 0.9, mask_colored, 0.5, 0)

            cv2.imshow("Tracking", visualized_frame)

            print(f"FPS: {1 / (time.time() - start_time)}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
camera.stop()
