import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# repo root (where shared/ lives) is two levels up from this analysis's
# own directory - add it to sys.path so `shared.*` imports resolve when
# this is run directly as `python main.py`, regardless of invocation cwd
# (Python always puts the executed script's own directory on sys.path,
# so imports already resolve correctly either way - but config.OUTPUT_DIR
# et al are cwd-relative, not script-relative, so they do NOT without the
# re-anchoring below)
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.abspath(os.path.join(ANALYSIS_DIR, '..', '..'))
sys.path.insert(0, REPO_ROOT)

import config
from select_cells import select_cells
from shared.cajal.extract_masks import extract_masks
from shared.cajal.run_cajal import run_cajal, needs_recompute
from shared.cajal.embed import get_umap_embedding, get_phate_embedding

# config.py's OUTPUT_DIR/MASKS_DIR/CAJAL_OUTPUT_DIR are relative paths,
# which Python resolves against the current working directory at runtime,
# not this script's own location - if main.py is ever invoked as
# `python analyses/03_multi_drug_comparison/main.py` from somewhere else
# (e.g. the repo root) rather than `cd`-ing into this directory first,
# those paths would silently resolve under the WRONG directory, including
# possibly colliding with another analysis's output. Re-anchor them to
# this script's own directory, same as every notebook already does.
config.OUTPUT_DIR       = os.path.join(ANALYSIS_DIR, config.OUTPUT_DIR)
config.MASKS_DIR        = os.path.join(ANALYSIS_DIR, config.MASKS_DIR)
config.CAJAL_OUTPUT_DIR = os.path.join(ANALYSIS_DIR, config.CAJAL_OUTPUT_DIR)


def process_frame(frame, metadata_by_condition, rng):
    '''
    Select cells, extract masks, run CAJAL, and fit embeddings for ONE
    frame - entirely independently of every other frame. Each frame gets
    its own self-contained masks/frame_NN/ and cajal_output/frame_NN/
    (config.frame_masks_dir/frame_cajal_dir), not a shared one, since
    this analysis compares conditions WITHIN a frame and never compares
    frame to frame - there's no reason their shape spaces should share
    coordinates.

    Inputs:
        frame:                  frame number to process
        metadata_by_condition:  {condition: full metadata DataFrame}
        rng:                    shared np.random.Generator - see select_cells
    '''
    frame_masks_dir = config.frame_masks_dir(frame)
    frame_cajal_dir = config.frame_cajal_dir(frame)
    os.makedirs(frame_masks_dir, exist_ok=True)
    os.makedirs(frame_cajal_dir, exist_ok=True)

    # --- select cells across every condition, this frame only ---
    print("  Selecting cells...")
    selected = select_cells(metadata_by_condition, frame, rng)
    selected['sampled'] = True   # extract_masks expects this column name

    print(f"  Selected {len(selected)} cells total")
    for condition in config.CONDITIONS:
        n = (selected['condition'] == condition).sum()
        print(f"    {condition}: {n} cells")

    selected_path = os.path.join(frame_cajal_dir, 'selected_metadata.csv')
    selected.to_csv(selected_path, index=False)
    print(f"  Saved selection to {selected_path}")

    # --- extract masks, per condition ---
    print("  Extracting masks...")
    for condition in config.CONDITIONS:
        condition_df = selected[selected['condition'] == condition]
        extract_masks(
            condition_df, condition, config.COMBINED_MASKS_DIR, frame_masks_dir,
            n_jobs=config.N_JOBS, pad_size=config.PAD_SIZE,
        )

    # --- run CAJAL, once per modality, this frame only ---
    print("  Running CAJAL...")
    cell_dir = os.path.join(frame_masks_dir, 'cell')
    nuc_dir  = os.path.join(frame_masks_dir, 'nuc')

    cell_dist_csv = os.path.join(frame_cajal_dir, f"cell_qgw_{config.N_POINTS:03d}P_dists.csv")
    nuc_dist_csv  = os.path.join(frame_cajal_dir, f"nuc_qgw_{config.N_POINTS:03d}P_dists.csv")

    # needs_recompute, not a bare exists check, so a dist CSV left over
    # from before extract_masks re-extracted a now-stale selection gets
    # correctly redone instead of silently reused.
    needs_cajal = {
        'cell': needs_recompute(cell_dist_csv, cell_dir),
        'nuc':  needs_recompute(nuc_dist_csv, nuc_dir),
    }
    for modality, needed in needs_cajal.items():
        if not needed:
            dist_csv = cell_dist_csv if modality == 'cell' else nuc_dist_csv
            print(f"    Skipping {modality} CAJAL run: {dist_csv} already exists and is up to date")

    n_tasks = sum(needs_cajal.values())
    if n_tasks > 0:
        per_task_jobs = max(1, config.N_JOBS // n_tasks)
        print(f"    Processing {n_tasks} mask type(s) in parallel ({per_task_jobs} jobs each)...")

        tasks = []
        if needs_cajal['cell']:
            tasks.append(delayed(run_cajal)(cell_dir, frame_cajal_dir, 'cell', config.N_POINTS, per_task_jobs, config.N_CLUSTERS))
        if needs_cajal['nuc']:
            tasks.append(delayed(run_cajal)(nuc_dir, frame_cajal_dir, 'nuc', config.N_POINTS, per_task_jobs, config.N_CLUSTERS))

        Parallel(n_jobs=n_tasks)(tasks)

    # --- fit UMAP and PHATE directly on this frame's CAJAL distances ---
    print("  Fitting embeddings...")
    get_umap_embedding(cell_dist_csv, os.path.join(frame_cajal_dir, 'cell_umap.npz'), seed=config.UMAP_SEED)
    get_umap_embedding(nuc_dist_csv, os.path.join(frame_cajal_dir, 'nuc_umap.npz'), seed=config.UMAP_SEED)
    get_phate_embedding(cell_dist_csv, os.path.join(frame_cajal_dir, 'cell_phate.npz'), seed=config.PHATE_SEED)
    get_phate_embedding(nuc_dist_csv, os.path.join(frame_cajal_dir, 'nuc_phate.npz'), seed=config.PHATE_SEED)


def main():
    os.makedirs(config.MASKS_DIR, exist_ok=True)
    os.makedirs(config.CAJAL_OUTPUT_DIR, exist_ok=True)

    print("Loading metadata...")
    metadata_by_condition = {c: pd.read_csv(config.metadata_csv_path(c)) for c in config.CONDITIONS}

    rng = np.random.default_rng(config.SAMPLE_SEED)
    for i, frame in enumerate(config.FRAMES, 1):
        print(f"\n=== Frame {frame} ({i}/{len(config.FRAMES)}) ===")
        process_frame(frame, metadata_by_condition, rng)

    print("\nDone.")


if __name__ == '__main__':
    main()
