import os

###### This dataset
#  vem_1uM_144hrsbefore, the same dataset 05_vem_144hr_pretreatment uses:
#  frame 7 is day 0 (drug addition). Frames 1-CONTROL_END_FRAME are
#  BEFORE treatment (~naive growth, used here as a control); frames
#  TREATMENT_START_FRAME-TREATMENT_END_FRAME are AFTER treatment. Unlike
#  05 (which only ever looks at frame 1), this analysis pools every
#  frame in [1, TREATMENT_END_FRAME] into ONE CAJAL run, and splits into
#  control vs. treatment only when building sister/cousin pairs in the
#  notebook - not at the sampling/CAJAL stage.
DRUG = 'vem_1uM_144hrsbefore'

TREATMENT_START_FRAME = 7    # day 0 - fixed by the data, not tunable
CONTROL_END_FRAME      = 5    # control range: frames 1-CONTROL_END_FRAME
TREATMENT_END_FRAME    = 11   # treatment range: frames TREATMENT_START_FRAME-TREATMENT_END_FRAME
#  frame 6 (day -1, the frame immediately before day 0) is missing
#  entirely from this dataset's raw data as of writing - it falls
#  outside BOTH ranges above (frames 1-5 and 7-11), so it's simply never
#  selected at all, deliberately. select_cells.py and the notebook also
#  both skip any OTHER empty frame gracefully, rather than assuming
#  every configured frame has cells - useful if a frame somewhere in
#  7-11 also turns out to be missing.

###### Sampling
#  No subclass stratification here - this analysis never classifies
#  cells by subclass at all (see root README's "Classification scheme",
#  not used here), so a flat per-frame sample size is enough; no budget
#  split/water-filling needed (c.f. naive_vs_vem/vem_full_timecourse).
#  A simple, lineage-UNaware subsample - which real sister/cousin pairs
#  survive (both members independently selected) is left to chance, not
#  biased towards keeping more of them.
SAMPLE_SIZE_PER_FRAME = 1000
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


def metadata_csv_path() -> str:
    return os.path.join(PROCESSED_DATA_DIR, f'{DRUG}_metadata.csv')


###### Mask extraction / CAJAL parameters
PAD_SIZE   = 200   # pixels of zero-padding added to each side of extracted masks
N_JOBS     = 52    # parallel jobs for mask extraction / CAJAL - must match slurm_submit.sh's --cpus-per-task
N_POINTS   = 75    # points sampled per cell for CAJAL's icdm
N_CLUSTERS = 20    # clusters for quantized GW

###### Pair-similarity permutation test parameters (shared/pair_similarity.py)
#  No UMAP/PHATE here, same as vem_48hr/144hr_pretreatment: the question
#  below needs the real CAJAL distances directly, not a 2D projection.
N_PERMUTATIONS = 10_000
TEST_SEED      = 42
