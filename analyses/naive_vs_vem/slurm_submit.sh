#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=52
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --job-name=naive_vs_vem
#SBATCH --output=./logs/naive_vs_vem_%j.out

module purge
eval "$(conda shell.bash hook)"

cd /gpfs/projects/b1042/GoyalLab/sohini/drug_response_morphology/analyses/naive_vs_vem
conda activate drug_response_morphology

mkdir -p logs
python main.py
