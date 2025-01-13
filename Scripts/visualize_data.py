import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------
# Load the Data
# ---------------------------
npz_file = "/home/franka/Desktop/franka_stack/data/50.npz"  # Change to your file as needed

data = np.load(npz_file, allow_pickle=True)
rgb_frames = data["rgb_frames"]         # Assumed shape: (N, H, W, C)
depth_frames = data["depth_frames"]     # Assumed shape: (N, H, W) or (N, H, W, 1)
actions = data["actions"]               # Assumed shape: (N, 8)
timestamps = data["timestamps"]         # Assumed shape: (N,)

num_frames = rgb_frames.shape[0]
current_frame = 0  # Keep track of the current frame index

# ---------------------------
# Setup the Figure and Subplots
# ---------------------------
plt.ion()  # Turn on interactive mode
fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])

# Subplot for the RGB image
ax_rgb = plt.subplot(gs[0])
ax_rgb.set_title("RGB Frame")
im_rgb = ax_rgb.imshow(rgb_frames[0])
ax_rgb.axis("off")

# Subplot for the Depth image
ax_depth = plt.subplot(gs[1])
ax_depth.set_title("Depth Frame")
im_depth = ax_depth.imshow(depth_frames[0], cmap="gray")
ax_depth.axis("off")

# Subplot for the Action graph (last dimension)
ax_action = plt.subplot(gs[2])
ax_action.set_title("Action (Last Dimension)")
line, = ax_action.plot([], [], color="skyblue")
ax_action.set_ylim(np.min(actions) - 0.1, np.max(actions) + 0.1)
ax_action.set_xlim(0, num_frames)
ax_action.set_xlabel("Frame")
ax_action.set_ylabel("Action Value (last dimension)")
ax_action.grid(True)

plt.tight_layout()

# ---------------------------
# Function to Update the Plots
# ---------------------------
def update_frame(i):
    # Update RGB image
    im_rgb.set_data(rgb_frames[i])
    
    # Update Depth image
    im_depth.set_data(depth_frames[i])
    
    # Update Action graph (using 3rd column as "last dimension")
    line.set_data(np.arange(i + 1), actions[:i + 1, 2])  # Adjust index if needed
    
    # Update the figure title with the timestamp
    fig.suptitle(f"Frame {i+1}/{num_frames} - Timestamp: {timestamps[i]:.2f}s", fontsize=14)
    plt.draw()
    plt.pause(0.001)  # Short pause to force the GUI event loop to process the update

# ---------------------------
# Keyboard Event Callback
# ---------------------------
def on_key(event):
    global current_frame
    # Check if right arrow key is pressed
    if event.key == "right":
        if current_frame < num_frames:
            update_frame(current_frame)
            current_frame += 1
        else:
            print("Reached last frame.")
    if event.key == "left":
        if current_frame  > 0:
            update_frame(current_frame)
            current_frame -= 1
        else:
            print("Reached last frame.")

# Connect the key press event to the callback function
fig.canvas.mpl_connect("key_press_event", on_key)

# Show the plot window
plt.show(block=True)
