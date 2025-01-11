#!/bin/bash

# Activating Conda Environment
echo "Activating Conda Environment: polymetis-local"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate polymetis-local

# Killing any Existing Servers
echo "Killing any existing servers"
sudo pkill -9 run_server

# Running the Franka Server in the background
echo "Starting the Franka Server..."
launch_robot.py robot_client=franka_hardware