#!/bin/bash

# Define the resource requirements
#SBATCH -c 10                # Request 10 CPUs
#SBATCH -t 6:00:00           # Maximum wall time
#SBATCH -p nvidia
#SBATCH --gres=gpu:1          
#SBATCH --mem=128GB               # Total of 4 GPUs (2 A100 + 2 V100)
 
# Load required modules
module purge
module load singularity

# Navigate to the working directory
cd /scratch/olr7742

# Run the Singularity container and execute the training script
singularity exec abc.sif bash -c "
    # Activate the Conda environment
    conda activate train_env

    # Check if Python is available
    if command -v python &> /dev/null; then
        echo 'Python is available.'
    else
        echo 'Python is not available in this environment.'
        exit 1
    fi

    # Check if CUDA is available and print GPU details
    python -c 'import torch; 
if torch.cuda.is_available(): 
    print(\"CUDA is available. GPU Name:\", torch.cuda.get_device_name(0)) 
else: 
    print(\"CUDA is not available.\")'
    cd /scratch/olr7742/franka
    python /scratch/olr7742/franka/capstone_stack/VQ-BeT/train_vae.py
"