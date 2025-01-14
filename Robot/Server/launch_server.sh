#!/bin/bash

# Activating Conda Environment
echo "Activating Conda Environment: polymetis-local"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate polymetis-local

# Kill port 4242
echo "Killing any existing servers"
sudo fuser -k 4242/tcp

# Running Server
echo "Starting the Server..."
python /home/franka/Desktop/capstone_stack/Robot/Server/server.py