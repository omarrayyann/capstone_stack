import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import FrankaDataset  # Assuming the dataset is defined in dataloader.py

# Parameters
data_folder = "/home/franka/Desktop/capstone_stack/sample_data"
sequence_length = 10
step_size = 2
batch_size = 8

# Initialize the dataset and dataloader
dataset = FrankaDataset(data_folder, sequence_length=sequence_length, step_size=step_size, frequency=5)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Fetch one batch
for batch in dataloader:
    low_dim_obs = batch['low_dim_obs']  # Shape: [batch_size, sequence_length, 8]
    rgbs = batch['rgb']  # Shape: [batch_size, sequence_length, H, W, C]
    depths = batch['depth']  # Shape: [batch_size, sequence_length, H, W]
    actions = batch['action']  # Shape: [batch_size, sequence_length, 8]
    break

# Select one item from the batch
item_index = 0
rgb_sequence = rgbs[item_index].numpy()  # Shape: [sequence_length, H, W, C]
depth_sequence = depths[item_index].numpy()  # Shape: [sequence_length, H, W]
action_sequence = actions[item_index].numpy()  # Shape: [sequence_length, 8]

# Ensure action_sequence has the correct shape
if action_sequence.ndim == 1:
    action_sequence = action_sequence[np.newaxis, :]  # Add a new axis if it's a single timestep

# Visualize RGB sequence
plt.figure(figsize=(15, 5))
for t, rgb_frame in enumerate(rgb_sequence):
    plt.subplot(1, sequence_length, t + 1)
    plt.imshow(rgb_frame.astype(np.uint8))
    plt.axis('off')
    plt.title(f"RGB t={t}")
plt.suptitle("RGB Sequence")
plt.show()

# Visualize depth sequence
plt.figure(figsize=(15, 5))
for t, depth_frame in enumerate(depth_sequence):
    plt.subplot(1, sequence_length, t + 1)
    plt.imshow(depth_frame, cmap='viridis')
    plt.axis('off')
    plt.title(f"Depth t={t}")
plt.suptitle("Depth Sequence")
plt.show()

# Visualize changes in the first 3 dimensions of the action
plt.figure(figsize=(10, 5))
for dim in range(3):
    plt.plot(range(len(action_sequence)), action_sequence[:, dim], label=f"Dim {dim}")
plt.xlabel("Time step")
plt.ylabel("Action Value")
plt.title("Change in First 3 Dimensions of Action")
plt.legend()
plt.grid()
plt.show()
