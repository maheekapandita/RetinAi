#!/bin/bash
#SBATCH --job-name=idrid_train
#SBATCH --output=idrid_train_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100gb
#SBATCH --time=18:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=2

# Load required modules
module load conda
module load cuda

# Activate your environment
conda activate retinai_env

# Navigate to the project directory
cd /home/saririans/RetinAI/src/

# Run the training script
python train_idrid.py

