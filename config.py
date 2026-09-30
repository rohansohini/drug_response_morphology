import os

###### Drug conditions
#  Must exactly match the folder names process_imaging_data used under its
#  raw_data/ (and therefore the {condition}_metadata.csv / mask filenames
#  it produced).
CONDITIONS     = ['naive', 'A1_1uM_vem']
VEM_CONDITION  = 'A1_1uM_vem'

###### Subcluster classification
#  m_label: 1 = dying, 2 = persister, 3 = proliferator (see
#  process_imaging_data/README.md). Within m_label == 3, k_label splits
#  slow (1, 2) vs fast (3, 4, 5, 6) proliferators.
DYING_M_LABEL      = 1
PERSISTING_M_LABEL = 2
PROLIF_M_LABEL     = 3
SLOW_K_LABELS      = [1, 2]
FAST_K_LABELS      = [3, 4, 5, 6]

###### Sampling
#  Total cells selected for CAJAL, split across conditions by weight, then
#  water-filled within each condition across its own (frame, subclass)
#  buckets - see pipeline/allocation.py and pipeline/select_cells.py.
TOTAL_CELL_BUDGET = 30_000
CONDITION_WEIGHTS = {'naive': 0.2, 'A1_1uM_vem': 0.8}   # must sum to 1.0

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
N_JOBS     = 32    # parallel jobs for mask extraction / CAJAL
N_POINTS   = 100   # points sampled per cell for CAJAL's icdm
N_CLUSTERS = 25    # clusters for quantized GW

###### UMAP parameters
#  Fit directly on the CAJAL GW distance matrix (metric='precomputed') -
#  deliberately no Isomap step.
UMAP_SEED = 42
