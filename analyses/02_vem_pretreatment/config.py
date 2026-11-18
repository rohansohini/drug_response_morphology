import os

###### These datasets
#  Same question, asked at two different pre-treatment durations:
#  vem_1uM_48hrsbefore (frame 1 is day -2, frame 3 is day 0) and
#  vem_1uM_144hrsbefore (frame 1 is day -6, frame 7 is day 0). Neither
#  dataset's frame-1 cells get a classification from process_imaging_data
#  (classification starts at day 0), so each frame-1 cell instead gets a
#  fate-mixture percentage (see shared/lineage_fate.py) from its day-0
#  descendants. Every available frame-1 cell/nucleus goes through CAJAL -
#  no sampling budget, unlike naive_vs_vem.
FRAMES = [1]

#  day0_frame picks out which lineage_frame{N}_to_frame1.mat file resolves
#  each frame-1 lineage's eventual day-0 descendants (see
#  shared/lineage_fate.py) - everything condition-specific below is
#  derived from drug + day0_frame alone.
CONDITIONS = {
    '48hr': {
        'drug':       'vem_1uM_48hrsbefore',
        'day0_frame': 3,
    },
    '144hr': {
        'drug':       'vem_1uM_144hrsbefore',
        'day0_frame': 7,
    },
}

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

#  raw_data/ is gitignored and may live outside the repo checkout
#  entirely, so (unlike PROCESS_IMAGING_DATA_ROOT above) this stays an
#  explicit path - same root process_imaging_data's own config.DRUG_DIRS
#  values live under.
_RAW_DATA_ROOT = '/projects/b1042/GoyalLab/sohini/drug_response_morphology/raw_data'


def raw_drug_dir(condition: str) -> str:
    return os.path.join(_RAW_DATA_ROOT, CONDITIONS[condition]['drug'])


#  lineage_table: file_idx, new_clone_id_global, old_clone_id_frame1 - see
#  shared/lineage_fate.py for exactly how these relate a condition's day-0
#  cells back to its frame-1 lineage.
def lineage_mat_path(condition: str) -> str:
    day0_frame = CONDITIONS[condition]['day0_frame']
    return os.path.join(raw_drug_dir(condition), f'lineage_frame{day0_frame}_to_frame1.mat')


#  cloneClassification - same file process_imaging_data's append_labels.py
#  reads; new_clone_id_global above is the 1-indexed row number into it.
def clone_kmeans_mat_path(condition: str) -> str:
    return os.path.join(raw_drug_dir(condition), 'clone_kmeans_id.mat')


def metadata_csv_path(condition: str) -> str:
    return os.path.join(PROCESSED_DATA_DIR, f"{CONDITIONS[condition]['drug']}_metadata.csv")


###### Classification scheme (from process_imaging_data's cloneClassification)
#  m_label: 1 = dying, 2 = persister, 3 = proliferator. Within m_label == 3,
#  k_label splits slow (1, 2) vs fast (3, 4, 5, 6) proliferators. This
#  mapping's biological basis (what the upstream k-means clusters actually
#  represent) isn't derivable from this repo - confirm with whoever set up
#  that clustering if you need to verify it. See shared/classify.py. Same
#  scheme for both conditions.
#  Used by shared/lineage_fate.py to classify each day-0 cell before
#  rolling its subclass up into each frame-1 lineage's fate-mixture
#  percentages.
DYING_M_LABEL      = 1
PERSISTING_M_LABEL = 2
PROLIF_M_LABEL     = 3
SLOW_K_LABELS      = [1, 2]
FAST_K_LABELS      = [3, 4, 5, 6]

#  each condition gets its own output subtree (masks/CAJAL distances can't
#  be mixed across datasets) - output/48hr/..., output/144hr/...
OUTPUT_DIR = 'output'


def output_dir(condition: str) -> str:
    return os.path.join(OUTPUT_DIR, condition)


def masks_dir(condition: str) -> str:
    return os.path.join(output_dir(condition), 'masks')


def cajal_output_dir(condition: str) -> str:
    return os.path.join(output_dir(condition), 'cajal_output')


###### Mask extraction / CAJAL parameters (same for both conditions)
PAD_SIZE   = 200   # pixels of zero-padding added to each side of extracted masks
N_JOBS     = 52    # parallel jobs for mask extraction / CAJAL - must match slurm_submit.sh's --cpus-per-task
N_POINTS   = 75    # points sampled per cell for CAJAL's icdm
N_CLUSTERS = 20    # clusters for quantized GW
