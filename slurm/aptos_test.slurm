#!/bin/bash
#SBATCH --job-name=aptos_test
#SBATCH --output=/home/saririans/RetinAI/slurm/aptos_test_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=150gb
#SBATCH --time=12:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=3

# Load required modules
module load conda
module load cuda

# Activate your environment
conda activate retinai_env

# Navigate to the project directory
cd /home/saririans/RetinAI/src/

# Run the training script
python /home/saririans/RetinAI/src/test_aptos.py

