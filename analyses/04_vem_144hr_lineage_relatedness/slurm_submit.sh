#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=52
#SBATCH --mem=192G
# --time bumped 36h -> 54h (1.5x) for the deeper per-frame sample
# (SAMPLE_SIZE_PER_FRAME 1000 -> 1500 - see config.py): main.py now
# processes 3 CAJAL runs sequentially (before-drug, after-drug, naive)
# instead of 2, and every one of them got deeper, so TOTAL cells
# processed this job went from 15,000 (10,000 + 5,000) to 22,500
# (7,500 x 3) - a 1.5x increase, applied here as a straightforward
# scaling of the old budget (the mask-extraction/icdm steps scale
# linearly with cell count, so 1.5x is the safe multiplier to budget
# against, even though the quantized-GW step itself - quadratic in
# cell count per run - grows a bit less than that in total: splitting
# before-drug/after-drug into two smaller runs more than offsets their
# own deeper sample, so only naive's sample-size increase drives up its
# own GW cost).
# --mem is left unchanged: the largest SINGLE run's cell count actually
# went DOWN (10,000 in the old combined before+after run -> 7,500 in
# any one of the 3 new separate runs), and main.py processes conditions
# sequentially, so peak memory (dominated by one run's mask-extraction/
# icdm workers, not the small GW distance matrix itself) shouldn't
# increase.
#SBATCH --time=54:00:00
#SBATCH --job-name=04_vem_144hr_lineage_relatedness
#SBATCH --output=./logs/04_vem_144hr_lineage_relatedness_%j.out

module purge
eval "$(conda shell.bash hook)"

cd /gpfs/projects/b1042/GoyalLab/sohini/drug_response_morphology/analyses/04_vem_144hr_lineage_relatedness
conda activate drug_morpho

# unbuffer stdout so print() shows up in the log in real time, instead of
# sitting in a buffer until it fills up (which also made CAJAL/UMAP/PHATE's
# progress interleave out of order with stderr warnings, which are unbuffered)
export PYTHONUNBUFFERED=1

mkdir -p logs
python main.py
