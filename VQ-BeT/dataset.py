from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torch.utils.data import random_split

class FrankaDataset(Dataset):
    def __init__(self, data_folder, sequence_length=3, step_size=2, frequency=5):
        self.data_folder = Path(data_folder)
        self.sequence_length = sequence_length
        self.step_size = step_size
        self.frequency = frequency  # Frequency for down-sampling (Hz)
        self.slices = []  # Store all slices from all files
        self._prepare_data()

    def _prepare_data(self):
        files = list(self.data_folder.glob("*.npz"))
        if not files:
            raise ValueError(f"No data files found in {self.data_folder}")

        for file_path in files:
            data = np.load(file_path)

            rgb_frames = data['rgb_frames']
            depth_frames = data['depth_frames']
            actions = data['actions']
            timestamps = data['timestamps']

            # Determine the step size for down-sampling based on frequency
            time_interval = 1.0 / self.frequency  # Target time interval
            mean_timestamp_diff = np.mean(np.diff(timestamps))
            actual_step = int(round(time_interval / mean_timestamp_diff))

            N = len(rgb_frames)

            for i in range(self.step_size, N - (self.sequence_length * actual_step) - self.step_size + 1, 1):
                rgb_slice = rgb_frames[i:i + (self.sequence_length * actual_step):actual_step]
                depth_slice = depth_frames[i:i + (self.sequence_length * actual_step):actual_step].astype(np.float32) / 1000.0  # Convert to meters

                # Compute low-dimensional observation (delta pose from past)
                low_dim_obs_sequence = []
                for j in range(self.sequence_length):
                    past_index = i - self.step_size + j * actual_step
                    current_index = i + j * actual_step

                    past_position = actions[past_index][:3]
                    current_position = actions[current_index][:3]
                    delta_past_position = current_position - past_position

                    past_quat = actions[past_index][3:7]
                    current_quat = actions[current_index][3:7]

                    past_rot = R.from_quat(past_quat)
                    current_rot = R.from_quat(current_quat)
                    delta_past_quat = (current_rot * past_rot.inv()).as_quat()

                    current_width = actions[current_index][-1]
                    low_dim_obs_current = np.concatenate([delta_past_position, delta_past_quat, [current_width]], axis=0)
                    low_dim_obs_sequence.append(low_dim_obs_current)

                low_dim_obs_sequence = np.stack(low_dim_obs_sequence)

                # Compute relative actions (delta pose to future)
                action_sequence = []
                for j in range(self.sequence_length):
                    current_index = i + j * actual_step
                    future_index = current_index + self.step_size

                    current_position = actions[current_index][:3]
                    future_position = actions[future_index][:3]
                    relative_position = future_position - current_position

                    current_quat = actions[current_index][3:7]
                    future_quat = actions[future_index][3:7]

                    current_rot = R.from_quat(current_quat)
                    future_rot = R.from_quat(future_quat)
                    relative_rot = future_rot * current_rot.inv()
                    relative_quat = relative_rot.as_quat()

                    future_width = actions[future_index][-1]
                    relative_action = np.concatenate([relative_position, relative_quat, [future_width]], axis=0)
                    action_sequence.append(relative_action)

                action_sequence = np.stack(action_sequence)

                # Store the slice
                self.slices.append({
                    'low_dim_obs': low_dim_obs_sequence,
                    'rgb': rgb_slice,
                    'depth': depth_slice,
                    'action': action_sequence
                })

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        slice_data = self.slices[index]
        return {
            'low_dim_obs': torch.tensor(slice_data['low_dim_obs'], dtype=torch.float32),
            'rgb': torch.tensor(slice_data['rgb'], dtype=torch.float32),
            'depth': torch.tensor(slice_data['depth'], dtype=torch.float32),
            'action': torch.tensor(slice_data['action'], dtype=torch.float32)
        }

# data_folder = "/home/franka/Desktop/capstone_stack/sample_data"
# sequence_length = 5
# step_size = 2
# frequency = 5  # Target frequency in Hz (must be <= 10 Hz, the original collection frequency)
# dataset = FrankaDataset(data_folder, sequence_length=sequence_length, step_size=step_size, frequency=frequency)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# for batch in dataloader:
#     low_dim_obs = batch['low_dim_obs']
#     rgbs = batch['rgb']
#     depths = batch['depth']
#     actions = batch['action']
#     print("Low-Dimensional Observations Shape:", low_dim_obs.shape)
#     print("RGBs Shape:", rgbs.shape)
#     print("Depths Shape:", depths.shape)
#     print("Actions Shape:", actions.shape)
#     break

def split_dataset(dataset, train_split=0.8):

    total_samples = len(dataset)
    train_size = int(train_split * total_samples)
    val_size = total_samples - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset