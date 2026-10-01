#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --job-name=drug_response_morphology
#SBATCH --output=./logs/drug_response_morphology_%j.out

module purge
eval "$(conda shell.bash hook)"

cd /gpfs/projects/b1042/GoyalLab/sohini/drug_response_morphology/
conda activate drug_response_morphology

mkdir -p logs
python main.py
