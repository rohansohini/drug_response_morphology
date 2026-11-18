import os

###### This dataset
#  vem_1uM_144hrsbefore in process_imaging_data: frame 1 is day -6 (6
#  days BEFORE drug addition), frame 7 is day 0 (when 1uM vemurafenib
#  was added). This analysis only ever looks at frame 1 - its cells
#  don't get a classification from process_imaging_data (classification
#  starts at day 0), so each frame-1 cell instead gets a fate-mixture
#  percentage (see shared/lineage_fate.py) from its day-0 descendants.
#  Every available frame-1 cell/nucleus goes through CAJAL - no sampling
#  budget, unlike naive_vs_vem. Same analysis as
#  02_vem_48hr_pretreatment, just further back before treatment.
DRUG   = 'vem_1uM_144hrsbefore'
FRAMES = [1]

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

#  this condition's RAW data root - same value as process_imaging_data's
#  own config.DRUG_DIRS['vem_1uM_144hrsbefore'] (not the processed_data output
#  above). Needed to read raw track_table_*.mat (to resolve
#  lineage_frame7_to_frame1.mat's track-local IDs into global lin_ids -
#  see shared/lineage_fate.py) and the two .mat files below, which live
#  alongside the raw track tables, not in processed_data/. Unlike
#  PROCESS_IMAGING_DATA_ROOT above, raw_data/ is gitignored and may live
#  outside the repo checkout entirely, so this stays an explicit path.
RAW_DRUG_DIR = '/projects/b1042/GoyalLab/sohini/drug_response_morphology/raw_data/vem_1uM_144hrsbefore'

#  lineage_table: file_idx, new_clone_id_global, old_clone_id_frame1 - see
#  shared/lineage_fate.py for exactly how these relate day-0 (frame 7)
#  cells back to their frame-1 lineage. Same internal field names as
#  vem_1uM_48hrsbefore's own lineage_frame3_to_frame1.mat - only the filename
#  (and which frame it points back from) differs between datasets.
LINEAGE_MAT_PATH      = os.path.join(RAW_DRUG_DIR, 'lineage_frame7_to_frame1.mat')
#  cloneClassification - same file process_imaging_data's append_labels.py
#  reads; new_clone_id_global above is the 1-indexed row number into it.
CLONE_KMEANS_MAT_PATH = os.path.join(RAW_DRUG_DIR, 'clone_kmeans_id.mat')

###### Classification scheme (from process_imaging_data's cloneClassification)
#  m_label: 1 = dying, 2 = persister, 3 = proliferator. Within m_label == 3,
#  k_label splits slow (1, 2) vs fast (3, 4, 5, 6) proliferators. This
#  mapping's biological basis (what the upstream k-means clusters actually
#  represent) isn't derivable from this repo - confirm with whoever set up
#  that clustering if you need to verify it. See shared/classify.py.
#  Used by shared/lineage_fate.py to classify each day-0 cell before
#  rolling its subclass up into each frame-1 lineage's fate-mixture
#  percentages.
DYING_M_LABEL      = 1
PERSISTING_M_LABEL = 2
PROLIF_M_LABEL     = 3
SLOW_K_LABELS      = [1, 2]
FAST_K_LABELS      = [3, 4, 5, 6]

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
