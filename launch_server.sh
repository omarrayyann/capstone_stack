# Activating Conda Environment
conda activate polymetis-local

# Killing any Existing Servers
pkill -9 run_server

# Running the Franka Server
launch_robot.py robot_client=franka_hardware

# Running the Gripper Server
launch_gripper.py gripper=franka_hand

