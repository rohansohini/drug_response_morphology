#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=52
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --job-name=vem_48hr_pretreatment
#SBATCH --output=./logs/vem_48hr_pretreatment_%j.out

module purge
eval "$(conda shell.bash hook)"

cd /gpfs/projects/b1042/GoyalLab/sohini/drug_response_morphology/analyses/1_vem_48hr_pretreatment
conda activate drug_morpho

mkdir -p logs
python main.py
