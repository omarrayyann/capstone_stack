#!/bin/bash

# Activating Conda Environment
echo "Activating Conda Environment: polymetis-local"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate polymetis-local

# Starting Phone Teleop
python python /home/franka/Desktop/franka_stack/Clien/phone_teleop.py