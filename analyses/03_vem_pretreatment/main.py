import os
import sys

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
from shared.load_all_cells import load_all_cells
from shared.cajal.extract_masks import extract_masks
from shared.cajal.run_cajal import run_cajal, needs_recompute

# config.py's OUTPUT_DIR is a relative path, which Python resolves against
# the current working directory at runtime, not this script's own
# location - if main.py is ever invoked as
# `python analyses/03_vem_pretreatment/main.py` from somewhere else (e.g.
# the repo root) rather than `cd`-ing into this directory first, that path
# would silently resolve under the WRONG directory, including possibly
# colliding with another analysis's output. Re-anchor it to this script's
# own directory, same as every notebook already does. config.output_dir()/
# masks_dir()/cajal_output_dir() all derive from this one variable, so
# re-anchoring it here is enough for every condition's own subtree too.
config.OUTPUT_DIR = os.path.join(ANALYSIS_DIR, config.OUTPUT_DIR)


def process_condition(condition: str, drug: str) -> None:
    masks_dir        = config.masks_dir(condition)
    cajal_output_dir = config.cajal_output_dir(condition)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(cajal_output_dir, exist_ok=True)

    # --- step 1/3: load metadata, restrict to config.FRAMES (frame 1 only), use every available cell ---
    #  unlike naive_vs_vem, there's no sampling budget here - every
    #  has_mask=True frame-1 cell goes through CAJAL. No subclass here -
    #  process_imaging_data doesn't classify frame-1 cells (classification
    #  starts at day 0) - see shared/lineage_fate.py for each cell's fate mixture.
    print(f"[1/3] ({condition}) Loading metadata...")
    metadata = load_all_cells(config.metadata_csv_path(condition), config.FRAMES, drug)

    print(f"  {len(metadata)} cells across frames {sorted(metadata['frame'].unique())}")

    selected_path = os.path.join(cajal_output_dir, 'selected_metadata.csv')
    metadata.to_csv(selected_path, index=False)
    print(f"  Saved to {selected_path}")

    # --- step 2/3: extract masks for every cell ---
    print(f"\n[2/3] ({condition}) Extracting masks...")
    extract_masks(
        metadata, drug, config.COMBINED_MASKS_DIR, masks_dir,
        n_jobs=config.N_JOBS, pad_size=config.PAD_SIZE,
    )

    # --- step 3/3: run CAJAL once per modality, on frame-1 cells only ---
    #  the GW distance matrix this produces IS the morphology
    #  representation used downstream (see notebooks/01_morphology_predicts_fate.ipynb) -
    #  no UMAP/PHATE embedding is fit here, since the prediction test
    #  needs the real distances, not a 2D projection that distorts them.
    print(f"\n[3/3] ({condition}) Running CAJAL...")
    cell_dir = os.path.join(masks_dir, 'cell')
    nuc_dir  = os.path.join(masks_dir, 'nuc')

    cell_dist_csv = os.path.join(cajal_output_dir, f"cell_qgw_{config.N_POINTS:03d}P_dists.csv")
    nuc_dist_csv  = os.path.join(cajal_output_dir, f"nuc_qgw_{config.N_POINTS:03d}P_dists.csv")

    # figure out how many of {cell, nuc} actually need to run BEFORE
    # deciding how many cores to give each one - splitting config.N_JOBS in
    # half unconditionally would waste half the allocation whenever only
    # one of the two is still needed (e.g. resuming after a partial run
    # that already finished one modality). needs_recompute, not a bare
    # exists check, so a dist CSV left over from BEFORE extract_masks
    # re-extracted a now-stale frame (selection changed upstream) gets
    # correctly redone instead of silently reused.
    needs_cajal = {
        'cell': needs_recompute(cell_dist_csv, cell_dir),
        'nuc':  needs_recompute(nuc_dist_csv, nuc_dir),
    }
    for modality, needed in needs_cajal.items():
        if not needed:
            dist_csv = cell_dist_csv if modality == 'cell' else nuc_dist_csv
            print(f"  Skipping {modality} CAJAL run: {dist_csv} already exists and is up to date")

    n_tasks = sum(needs_cajal.values())
    if n_tasks > 0:
        per_task_jobs = max(1, config.N_JOBS // n_tasks)
        print(f"  Processing {n_tasks} mask type(s) in parallel ({per_task_jobs} jobs each)...")

        tasks = []
        if needs_cajal['cell']:
            tasks.append(delayed(run_cajal)(cell_dir, cajal_output_dir, 'cell', config.N_POINTS, per_task_jobs, config.N_CLUSTERS))
        if needs_cajal['nuc']:
            tasks.append(delayed(run_cajal)(nuc_dir, cajal_output_dir, 'nuc', config.N_POINTS, per_task_jobs, config.N_CLUSTERS))

        Parallel(n_jobs=n_tasks)(tasks)


def main():
    for condition, info in config.CONDITIONS.items():
        print(f"=== {condition} ({info['drug']}) ===")
        process_condition(condition, info['drug'])
        print()

    print("Done.")


if __name__ == '__main__':
    main()
