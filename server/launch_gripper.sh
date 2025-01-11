#!/bin/bash

# Activating Conda Environment
echo "Activating Conda Environment: polymetis-local"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate polymetis-local

# Running the Gripper Server in the background
echo "Starting the Gripper Server..."
launch_gripper.py gripper=franka_hand