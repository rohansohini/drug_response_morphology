#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --job-name=process_imaging
#SBATCH --output=./logs/process_imaging_%j.out

module purge
eval "$(conda shell.bash hook)"

cd /projects/b1042/GoyalLab/sohini/drug_response_morphology/process_imaging_data/
conda activate drug_morpho

# unbuffer stdout so print() shows up in the log in real time, instead of
# sitting in a buffer until it fills up
export PYTHONUNBUFFERED=1

mkdir -p logs
python main.py
