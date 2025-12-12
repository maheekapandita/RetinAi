#!/bin/bash
#SBATCH --job-name=aptos_eval
#SBATCH --output=/home/saririans/RetinAI/slurm/aptos_eval_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100gb
#SBATCH --time=12:00:00
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
python /home/saririans/RetinAI/src/aptos_eval.py

