import matplotlib.pyplot as plt
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from vqvae.vqvae import *
from torch.utils.data import DataLoader

@hydra.main(config_path="configs", config_name="inference", version_base="1.2")
def main(cfg):


    if "visual_input" in cfg and cfg.visual_input:
        print("use visual environment")
        cfg.model.gpt_model.config.input_dim = 1536
    cbet_model = hydra.utils.instantiate(cfg.model).to(cfg.device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        cbet_model = DataParallel(cbet_model)
        multi_gpu = True
    else:
        print("Using a single GPU.")
    
    if cfg.load_path:
        cbet_model.load_model(Path(cfg.load_path))

    cbet_model.eval()

    dataset = hydra.utils.instantiate(cfg.dataset)
    train_dataset, val_dataset = hydra.utils.instantiate(cfg.split_dataset, dataset=dataset)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    with torch.no_grad():
        last_time = None
        while True:

            # get random batch from val loader
            batch = next(iter(train_loader))

            rgb = batch["rgb"][0].unsqueeze(0)
            depth = batch["depth"][0].unsqueeze(0)
            low_dim = batch["low_dim_obs"][0].unsqueeze(0)
            print(low_dim.shape)

            predicted_act, _, _ = cbet_model(rgb, depth, None, None)

            if last_time is None:
                last_time = time.time()
            else:
                print(f"FPS: {1/(time.time() - last_time)}")
                last_time = time.time()

            # Extract the xyz vector
            xyz = predicted_act[-1, 0, 0:3]
            xyz = np.array(xyz)*100.0

            # Extract the grip value
            grip = predicted_act[-1, 0, -1]

            print(predicted_act[:, 0, -1])

            # Prepare the RGB image
            rgb = rgb[0][0].cpu().numpy().astype(np.uint8)

            # Define the origin of the vector (center of the image)
            origin_x, origin_y = rgb.shape[1] // 2, rgb.shape[0] // 2

            # Scale the xyz vector for better visualization in the image
            scale_factor = 50  # Adjust this factor to make the vector visible in the image
            scaled_xyz = xyz * scale_factor

            # Calculate the vector's end point
            end_x = origin_x + scaled_xyz[0]
            end_y = origin_y - scaled_xyz[1]  # Note: subtract y to match image coordinates

            # Plot the RGB image
            plt.imshow(rgb)

            # Plot the xyz vector
            plt.quiver(
                origin_x, origin_y, 
                scaled_xyz[0], -scaled_xyz[1],  # Negate y for image coordinates
                angles='xy', scale_units='xy', scale=1, color='red', width=0.01  # Slightly thicker arrow
            )

            # Display the grip value on the image
            plt.text(
                10, 10,  # Position near the top-left corner
                f'Grip: {grip:.2f}', 
                color='yellow', 
                fontsize=12, 
                bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3')
            )

            # Show the plot
            plt.axis('off')  # Turn off axis for better visualization
            plt.show()


if __name__ == "__main__":
    main()
