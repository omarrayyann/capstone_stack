import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------
# Load the Data
# ---------------------------

# Update this filename to the desired npz file (e.g., "0.npz")
npz_file = "/home/franka/Desktop/franka_stack/Data_Collection/data/0.npz"  # Change to your file as needed

data = np.load(npz_file, allow_pickle=True)
rgb_frames = data["rgb_frames"]         # Assumed shape: (N, H, W, C)
depth_frames = data["depth_frames"]     # Assumed shape: (N, H, W) or (N, H, W, 1)
actions = data["actions"]               # Assumed shape: (N, 8)
timestamps = data["timestamps"]         # Assumed shape: (N,)

num_frames = rgb_frames.shape[0]

# ---------------------------
# Setup the Figure and Subplots
# ---------------------------

plt.ion()  # Turn on interactive mode
fig = plt.figure(figsize=(12, 4))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])

# Subplot for the RGB image
ax_rgb = plt.subplot(gs[0])
ax_rgb.set_title("RGB Frame")
im_rgb = ax_rgb.imshow(rgb_frames[0])
ax_rgb.axis("off")

# Subplot for the Depth image
ax_depth = plt.subplot(gs[1])
ax_depth.set_title("Depth Frame")
# If depth is single channel, use cmap.
im_depth = ax_depth.imshow(depth_frames[0], cmap="gray")
ax_depth.axis("off")

# Subplot for the Action bar plot
ax_action = plt.subplot(gs[2])
ax_action.set_title("Action (8-d)")
# Draw a bar for each dimension; use indices 0..7.
bar_container = ax_action.bar(np.arange(8), actions[0], color="skyblue")
ax_action.set_ylim(np.min(actions) - 0.1, np.max(actions) + 0.1)
ax_action.set_xticks(np.arange(8))
ax_action.set_xticklabels([f"{i}" for i in range(8)])
ax_action.set_ylabel("Value")

plt.tight_layout()

# ---------------------------
# Function to Update the Plots
# ---------------------------
def update_frame(i):
    # Update RGB image
    im_rgb.set_data(rgb_frames[i])
    
    # Update Depth image
    im_depth.set_data(depth_frames[i])
    
    # Update Action bar plot
    for bar, new_val in zip(bar_container, actions[i]):
        bar.set_height(new_val)
    
    # Update the figure title with the timestamp (optional)
    fig.suptitle(f"Frame {i+1}/{num_frames} - Timestamp: {timestamps[i]:.2f}s", fontsize=14)
    plt.draw()
    plt.pause(0.001)

# ---------------------------
# Iterate Through the Frames
# ---------------------------
current_frame = 0
update_frame(current_frame)

print("Press the Right Arrow key in the figure window to advance one frame.")
print("Close the figure window to exit.")

def on_key(event):
    global current_frame
    if event.key == "right":
        current_frame = (current_frame + 1) % num_frames  # cycle through
        update_frame(current_frame)

fig.canvas.mpl_connect("key_press_event", on_key)

# Keep the plot open until closed by the user.
plt.ioff()
plt.show()
