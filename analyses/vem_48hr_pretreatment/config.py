import os

###### This dataset
#  vem_1uM_48hrsbefore in process_imaging_data: frames 1-2 are BEFORE drug
#  addition (pre-treatment baseline), frame 3 is day 0 (when 1uM
#  vemurafenib was added). Every available cell/nucleus across these 3
#  frames goes through CAJAL - no sampling budget, unlike naive_vs_vem.
DRUG   = 'vem_1uM_48hrsbefore'
FRAMES = [1, 2, 3]

###### Paths
#  process_imaging_data's output is read directly from its own
#  processed_data/ - no copy step. Point this at wherever that repo lives.
PROCESS_IMAGING_DATA_ROOT = '/gpfs/projects/b1042/GoyalLab/sohini/process_imaging_data'
PROCESSED_DATA_DIR        = os.path.join(PROCESS_IMAGING_DATA_ROOT, 'processed_data')
COMBINED_MASKS_DIR        = os.path.join(PROCESSED_DATA_DIR, 'combined_masks')

OUTPUT_DIR       = 'output'
MASKS_DIR        = os.path.join(OUTPUT_DIR, 'masks')
CAJAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'cajal_output')


def metadata_csv_path() -> str:
    return os.path.join(PROCESSED_DATA_DIR, f'{DRUG}_metadata.csv')


###### Mask extraction / CAJAL parameters
PAD_SIZE   = 200   # pixels of zero-padding added to each side of extracted masks
N_JOBS     = 52    # parallel jobs for mask extraction / CAJAL - must match slurm_submit.sh's --cpus-per-task
N_POINTS   = 75    # points sampled per cell for CAJAL's icdm
N_CLUSTERS = 20    # clusters for quantized GW

###### UMAP parameters
#  Fit directly on the CAJAL GW distance matrix (metric='precomputed').
UMAP_SEED = 42
