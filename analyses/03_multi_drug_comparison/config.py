import os

###### Drug conditions
#  Must exactly match process_imaging_data's own condition naming (it
#  determines both the {condition}_metadata.csv filenames and the
#  combined mask TIFF filenames) - see process_imaging_data/config.py's
#  own DRUG_DIRS, which already has all four of these registered.
CONDITIONS = ['naive', 'vem_1uM', 'gem_2.5nM', 'tram_6nM']

###### Frames
#  Compares conditions at two timepoints that share ONE embedding space
#  (one CAJAL run per modality, pooling both frames' cells together - see
#  main.py) rather than one independent run per frame, so frame 1 and
#  frame 13 positions are directly comparable, not just visually similar.
#  process_imaging_data itself no longer caps frames (see its own
#  config.py) - {drug}_metadata.csv can now contain far more of them -
#  so frame 13 here is this analysis's own deliberate choice (frames
#  beyond that are known to get unreliable, heavy overcrowding from
#  repeated division), explicitly capped rather than silently absorbing
#  whatever else becomes available upstream.
FRAMES = [1, 13]

###### Sampling
#  Cells sampled per (frame, condition) bucket - see select_cells.py. A
#  flat per-bucket size, not a total budget split by weight like
#  naive_vs_vem - no water-filling needed since every condition is
#  sampled independently and equally.
SAMPLE_SIZE_PER_CONDITION = 1000
SAMPLE_SEED = 42

###### Paths
#  process_imaging_data's output is read directly from its own
#  processed_data/ - no copy step. process_imaging_data/ is part of this
#  same repo, at a fixed location relative to this file (not wherever the
#  repo happens to be checked out), so its root is derived from config.py's
#  own location rather than a hardcoded absolute path.
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
PROCESS_IMAGING_DATA_ROOT = os.path.join(_REPO_ROOT, 'process_imaging_data')
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

###### Embedding parameters
#  Both fit directly on the CAJAL GW distance matrix (precomputed), which
#  pools every condition AND both frames together - one shared space, not
#  a separate fit per frame.
UMAP_SEED  = 42
PHATE_SEED = 42
