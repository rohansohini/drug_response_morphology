import os

###### Drug condition
#  Must exactly match process_imaging_data's own condition naming. Only
#  vem_1uM, deliberately - running CAJAL on naive across its own full
#  timecourse too would mean far more sampled cells (naive divides a lot
#  over time) and CAJAL's pairwise distance computation is too slow at
#  that scale. Restricting to one condition keeps the cell count - and
#  so the CAJAL run - tractable even across every available frame.
DRUG = 'vem_1uM'

###### Sampling
#  Cells sampled per (frame, subclass) bucket - see select_cells.py and
#  shared/allocation.py. Same water-filling mechanism as naive_vs_vem,
#  just without a cross-condition split, since every cell here is
#  already vem_1uM - the whole budget goes to this one condition.
TOTAL_CELL_BUDGET = 30_000

#  No frame cap, deliberately: process_imaging_data no longer caps
#  frames itself (see its own config.py), and the whole point of this
#  analysis is to look across vem_1uM's ENTIRE available timecourse
#  (more frames than naive_vs_vem's analyses use), on the hypothesis
#  that vem's subclasses only separate clearly later than frame 13.

###### Classification scheme (from process_imaging_data's cloneClassification)
#  m_label: 1 = dying, 2 = persister, 3 = proliferator. Within m_label == 3,
#  k_label splits slow (1, 2) vs fast (3, 4, 5, 6) proliferators. This
#  mapping's biological basis (what the upstream k-means clusters actually
#  represent) isn't derivable from this repo - confirm with whoever set up
#  that clustering if you need to verify it. See shared/classify.py.
DYING_M_LABEL      = 1
PERSISTING_M_LABEL = 2
PROLIF_M_LABEL     = 3
SLOW_K_LABELS      = [1, 2]
FAST_K_LABELS      = [3, 4, 5, 6]

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
N_JOBS     = 64    # parallel jobs for mask extraction / CAJAL - must match slurm_submit.sh's --cpus-per-task
N_POINTS   = 75    # points sampled per cell for CAJAL's icdm
N_CLUSTERS = 20    # clusters for quantized GW

###### Embedding parameters
#  Both fit directly on the CAJAL GW distance matrix (precomputed). This
#  is its OWN embedding space - not comparable to naive_vs_vem's, since
#  that one also includes naive and is capped at frame 13.
UMAP_SEED  = 42
PHATE_SEED = 42
