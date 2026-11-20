#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=52
#SBATCH --mem=192G
#SBATCH --time=36:00:00
#SBATCH --job-name=05_vem_144hr_lineage_relatedness
#SBATCH --output=./logs/05_vem_144hr_lineage_relatedness_%j.out

module purge
eval "$(conda shell.bash hook)"

cd /gpfs/projects/b1042/GoyalLab/sohini/drug_response_morphology/analyses/05_vem_144hr_lineage_relatedness
conda activate drug_morpho

# unbuffer stdout so print() shows up in the log in real time, instead of
# sitting in a buffer until it fills up (which also made CAJAL/UMAP/PHATE's
# progress interleave out of order with stderr warnings, which are unbuffered)
export PYTHONUNBUFFERED=1

mkdir -p logs
python main.py
