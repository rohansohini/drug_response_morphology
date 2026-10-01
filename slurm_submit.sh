#!/bin/bash
#SBATCH --account=p32655
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --job-name=drug_response_morphology
#SBATCH --output=./logs/drug_response_morphology_%j.out

# NOTE: --time and --mem are rough, UNVALIDATED estimates - there's no
# empirical runtime benchmark yet for CAJAL + UMAP at this repo's scale
# (up to 30,000 cells across two conditions and two modalities). Check the
# log after the first real run and adjust before resubmitting; if it times
# out, main.py's existing skip-if-exists checks (per mask type, per
# embedding) mean a re-submitted job resumes instead of restarting from
# scratch. --account/--partition are carried over from
# process_imaging_data's slurm_submit.sh (same lab/user, confirmed compute
# account) - the working directory below uses the confirmed
# /gpfs/projects/b1042/GoyalLab/sohini/ filesystem path, which is separate
# from the SLURM compute account.

module purge
eval "$(conda shell.bash hook)"

cd /gpfs/projects/b1042/GoyalLab/sohini/drug_response_morphology/
conda activate drug_response_morphology

mkdir -p logs
python main.py
