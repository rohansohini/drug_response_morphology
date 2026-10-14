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

PROCESSED_DATA_DIR = 'processed_data/'
COMBINED_MASKS_DIR = os.path.join(PROCESSED_DATA_DIR, 'combined_masks')

N_JOBS    = 16
BUFFER    = 2
IMG_SIZE  = 2501
MAX_FRAME = 13   # frames beyond this are unreliable (heavy overcrowding) and excluded entirely
