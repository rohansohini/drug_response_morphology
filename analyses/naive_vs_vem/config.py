import os

###### Drug conditions
#  Must exactly match process_imaging_data's own condition naming (it
#  determines both the {condition}_metadata.csv filenames and the
#  combined mask TIFF filenames) - confirmed against the real files:
#  A2_naive_metadata.csv, A1_1uM_vem_metadata.csv.
NAIVE_CONDITION = 'A2_naive'
VEM_CONDITION   = 'A1_1uM_vem'
CONDITIONS      = [NAIVE_CONDITION, VEM_CONDITION]

###### Sampling
#  Total cells selected for CAJAL, split across conditions by weight, then
#  water-filled within each condition across its own (frame, subclass)
#  buckets - see allocation.py and select_cells.py. The classification
#  scheme itself (m_label/k_label -> subclass) is universal across every
#  analysis in this repo, not specific to this one - see shared/classify.py.
TOTAL_CELL_BUDGET = 30_000
CONDITION_WEIGHTS = {NAIVE_CONDITION: 0.2, VEM_CONDITION: 0.8}   # must sum to 1.0

###### Paths
#  process_imaging_data's output is read directly from its own
#  processed_data/ - no copy step. Point this at wherever that repo lives.
PROCESS_IMAGING_DATA_ROOT = '/gpfs/projects/b1042/GoyalLab/sohini/process_imaging_data'
PROCESSED_DATA_DIR        = os.path.join(PROCESS_IMAGING_DATA_ROOT, 'processed_data')
COMBINED_MASKS_DIR        = os.path.join(PROCESSED_DATA_DIR, 'combined_masks')

OUTPUT_DIR       = 'output'
MASKS_DIR        = os.path.join(OUTPUT_DIR, 'masks')
CAJAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'cajal_output')


def metadata_csv_path(condition: str) -> str:
    return os.path.join(PROCESSED_DATA_DIR, f'{condition}_metadata.csv')


###### Mask extraction / CAJAL parameters
PAD_SIZE   = 200   # pixels of zero-padding added to each side of extracted masks
N_JOBS     = 52    # parallel jobs for mask extraction / CAJAL - must match slurm_submit.sh's --cpus-per-task
N_POINTS   = 75    # points sampled per cell for CAJAL's icdm
N_CLUSTERS = 20    # clusters for quantized GW

###### UMAP parameters
#  Fit directly on the CAJAL GW distance matrix (metric='precomputed').
UMAP_SEED = 42
