import os

###### Drug conditions
#  Must exactly match process_imaging_data's own condition naming (it
#  determines both the {condition}_metadata.csv filenames and the
#  combined mask TIFF filenames) - see process_imaging_data/config.py's
#  own DRUG_DIRS, which already has all four of these registered.
CONDITIONS = ['naive', 'vem_1uM', 'gem_2.5nM', 'tram_6nM']

###### Frames
#  Compares conditions at two INDEPENDENT timepoints, each with its own
#  CAJAL run (not one shared embedding across frames like naive_vs_vem) -
#  frame 1's embedding space has no relationship to frame 13's, since
#  the question here is how conditions differ WITHIN a frame, never how
#  a frame differs from another frame. Frame 13 is the LAST frame any
#  condition's metadata.csv can contain: process_imaging_data's own
#  main.py already excludes anything beyond its MAX_FRAME (see
#  process_imaging_data/config.py - frames beyond that are unreliable,
#  heavy overcrowding) before this analysis ever sees the data, so
#  there's no separate "end frame" cutoff to maintain here too.
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


def frame_masks_dir(frame: int) -> str:
    '''
    Each frame's own masks/ subfolder - frame 1 and frame 13's masks
    never share a folder, since CAJAL treats everything in one folder as
    one shape space and these two frames must stay independent (see main.py).
    '''
    return os.path.join(MASKS_DIR, f'frame_{frame:02d}')


def frame_cajal_dir(frame: int) -> str:
    '''Each frame's own cajal_output/ subfolder - see frame_masks_dir.'''
    return os.path.join(CAJAL_OUTPUT_DIR, f'frame_{frame:02d}')


###### Mask extraction / CAJAL parameters
PAD_SIZE   = 200   # pixels of zero-padding added to each side of extracted masks
N_JOBS     = 52    # parallel jobs for mask extraction / CAJAL - must match slurm_submit.sh's --cpus-per-task
N_POINTS   = 75    # points sampled per cell for CAJAL's icdm
N_CLUSTERS = 20    # clusters for quantized GW

###### Embedding parameters
#  Both fit directly on each frame's own CAJAL GW distance matrix.
UMAP_SEED  = 42
PHATE_SEED = 42
