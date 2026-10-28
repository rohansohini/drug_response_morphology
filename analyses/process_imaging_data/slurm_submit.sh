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

cd /projects/b1042/GoyalLab/sohini/drug_response_morphology/analyses/process_imaging_data/
conda activate drug_morpho

mkdir -p logs
python main.py
