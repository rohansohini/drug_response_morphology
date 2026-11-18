#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=52
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --job-name=02_vem_pretreatment
#SBATCH --output=./logs/02_vem_pretreatment_%j.out

module purge
eval "$(conda shell.bash hook)"

cd /gpfs/projects/b1042/GoyalLab/sohini/drug_response_morphology/analyses/02_vem_pretreatment
conda activate drug_morpho

# unbuffer stdout so print() shows up in the log in real time, instead of
# sitting in a buffer until it fills up (which also made CAJAL/UMAP/PHATE's
# progress interleave out of order with stderr warnings, which are unbuffered)
export PYTHONUNBUFFERED=1

mkdir -p logs
python main.py
