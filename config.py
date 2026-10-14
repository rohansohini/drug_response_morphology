import os

##############################################################
# DRUG_DIRS — edit this to point at wherever each drug
# condition's raw data actually lives on disk. Each directory
# must contain:
#   Cropped_XX/track_table_*.mat
#   Cropped_XX/segmentation_info_*.mat
#   Output_Segmentation/Transmission_region_XX/Transmission_YY_masks.tiff
#   clone_kmeans_id.mat
##############################################################

DRUG_DIRS = {
    'A2_naive':   '/path/to/A2_naive/drug_dir',
    'A1_1uM_vem': '/path/to/A1_1uM_vem/drug_dir',
}

##############################################################
# DAY0_FRAMES — the frame number (1-indexed) each condition's
# drug was actually added on. Most conditions image starting
# exactly at day 0, so DAY0_FRAMES is 1 for them - but some
# start imaging before the drug is added (e.g. a
# 48-hours-before-treatment baseline), in which case
# clone_kmeans_id.mat's lineage classification only begins at
# that later frame, not frame 1 (see pipeline/append_labels.py
# for exactly how this controls pairing labels to lineages).
# EVERY key in DRUG_DIRS must have a matching entry here -
# main.py asserts this and fails loudly if one's missing, since
# getting day0_frame wrong silently mis-labels cells rather
# than erroring.
##############################################################

DAY0_FRAMES = {
    'A2_naive':   1,
    'A1_1uM_vem': 1,
}

##############################################################

PROCESSED_DATA_DIR = 'processed_data/'
COMBINED_MASKS_DIR = os.path.join(PROCESSED_DATA_DIR, 'combined_masks')

N_JOBS    = 16
BUFFER    = 2
IMG_SIZE  = 2501
MAX_FRAME = 13   # frames beyond this are unreliable (heavy overcrowding) and excluded entirely
