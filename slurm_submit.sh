#!/bin/bash
#SBATCH --account=p32655
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --job-name=process_imaging
#SBATCH --output=./logs/process_imaging_%j.out

module purge
eval "$(conda shell.bash hook)"

cd /projects/p32655/sohini/process_imaging_data/
conda activate process_imaging

mkdir -p logs
python main.py
