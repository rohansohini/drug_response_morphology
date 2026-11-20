import os

###### These conditions
#  vem_1uM_144hrsbefore (the same dataset 02_vem_pretreatment's '144hr'
#  condition uses): frame 7 is day 0 (drug addition). Frames
#  1-BEFORE_DRUG_END_FRAME are BEFORE the drug was added; frames
#  AFTER_DRUG_START_FRAME-AFTER_DRUG_END_FRAME are AFTER. This analysis
#  pools every frame in [1, AFTER_DRUG_END_FRAME] into ONE CAJAL run, and
#  splits into before-drug vs. after-drug only when building sister/
#  cousin pairs in the notebook - not at the sampling/CAJAL stage.
#
#  naive (a second, entirely separate condition - never treated with any
#  drug at all): used as a sanity-check control, expected to look like
#  the vem before-drug window since neither has seen a drug yet. Sampled
#  and CAJAL'd on its own (own output subtree - it is NOT part of the vem
#  CAJAL run above), over frames 1-BEFORE_DRUG_END_FRAME - the SAME
#  length as the vem before-drug window, for a fair comparison.
DRUG = 'vem_1uM_144hrsbefore'

AFTER_DRUG_START_FRAME = 7    # day 0 - fixed by the data, not tunable
BEFORE_DRUG_END_FRAME   = 5    # before-drug range: frames 1-BEFORE_DRUG_END_FRAME (also naive's own comparison window length)
AFTER_DRUG_END_FRAME    = 11   # after-drug range: frames AFTER_DRUG_START_FRAME-AFTER_DRUG_END_FRAME
#  frame 6 (day -1, the frame immediately before day 0) is missing
#  entirely from the vem dataset's raw data as of writing - it falls
#  outside BOTH ranges above (frames 1-5 and 7-11), so it's simply never
#  selected at all, deliberately. select_cells.py and the notebook also
#  both skip any OTHER empty frame gracefully, rather than assuming
#  every configured frame has cells - useful if a frame somewhere in
#  7-11 also turns out to be missing.

#  each condition's own drug name (for metadata_csv_path()/extract_masks)
#  and the single highest frame it needs sampled/CAJAL'd - vem needs
#  every frame up to AFTER_DRUG_END_FRAME (both its before- and
#  after-drug ranges are carved out of this ONE run later); naive only
#  ever needs its own single comparison window.
CONDITIONS = {
    'vem':   {'drug': DRUG,    'max_frame': AFTER_DRUG_END_FRAME},
    'naive': {'drug': 'naive', 'max_frame': BEFORE_DRUG_END_FRAME},
}

###### Sampling
#  No subclass stratification here - this analysis never classifies
#  cells by subclass at all (see root README's "Classification scheme",
#  not used here), so a flat per-frame sample size is enough; no budget
#  split/water-filling needed (c.f. naive_vs_vem/vem_full_timecourse).
#  A simple, lineage-UNaware subsample - which real sister/cousin pairs
#  survive (both members independently selected) is left to chance, not
#  biased towards keeping more of them. Same sample size for both
#  conditions.
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


def metadata_csv_path(condition: str) -> str:
    return os.path.join(PROCESSED_DATA_DIR, f"{CONDITIONS[condition]['drug']}_metadata.csv")


#  each condition gets its own output subtree (masks/CAJAL distances
#  can't be mixed across datasets, and naive was never part of the vem
#  CAJAL run) - output/vem/..., output/naive/...
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

###### Pair-similarity test parameters (shared/pair_similarity.py)
#  No UMAP/PHATE here, same as vem_48hr/144hr_pretreatment: the question
#  below needs the real CAJAL distances directly, not a 2D projection.
N_NULL_SAMPLES = 10_000   # individual random pairs making up each test's null distribution
TEST_SEED      = 42
