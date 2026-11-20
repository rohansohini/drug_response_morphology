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
from select_cells import select_cells
from shared.cajal.extract_masks import extract_masks
from shared.cajal.run_cajal import run_cajal, needs_recompute
from shared.cajal.embed import get_umap_embedding, get_phate_embedding

# config.py's OUTPUT_DIR/MASKS_DIR/CAJAL_OUTPUT_DIR are relative paths,
# which Python resolves against the current working directory at runtime,
# not this script's own location - if main.py is ever invoked as
# `python analyses/05_vem_full_timecourse/main.py` from somewhere else
# (e.g. the repo root) rather than `cd`-ing into this directory first,
# those paths would silently resolve under the WRONG directory, including
# possibly colliding with another analysis's output. Re-anchor them to
# this script's own directory, same as every notebook already does.
config.OUTPUT_DIR       = os.path.join(ANALYSIS_DIR, config.OUTPUT_DIR)
config.MASKS_DIR        = os.path.join(ANALYSIS_DIR, config.MASKS_DIR)
config.CAJAL_OUTPUT_DIR = os.path.join(ANALYSIS_DIR, config.CAJAL_OUTPUT_DIR)


def main():
    os.makedirs(config.MASKS_DIR, exist_ok=True)
    os.makedirs(config.CAJAL_OUTPUT_DIR, exist_ok=True)

    # --- step 1/4: select cells, every available frame ---
    print("[1/4] Loading metadata and selecting cells...")
    metadata = pd.read_csv(config.metadata_csv_path())
    selected = select_cells(metadata)
    selected['sampled'] = True   # extract_masks expects this column name

    print(f"  Selected {len(selected)} cells total across {selected['frame'].nunique()} frames "
          f"(frames {selected['frame'].min()}-{selected['frame'].max()})")
    print("  By subclass:")
    for subclass, n in selected['subclass'].value_counts().items():
        print(f"    {subclass}: {n}")

    selected_path = os.path.join(config.CAJAL_OUTPUT_DIR, 'selected_metadata.csv')
    selected.to_csv(selected_path, index=False)
    print(f"  Saved selection to {selected_path}")

    # --- step 2/4: extract masks for every selected cell ---
    print("\n[2/4] Extracting masks...")
    extract_masks(
        selected, config.DRUG, config.COMBINED_MASKS_DIR, config.MASKS_DIR,
        n_jobs=config.N_JOBS, pad_size=config.PAD_SIZE,
    )

    # --- step 3/4: run CAJAL once per modality ---
    print("\n[3/4] Running CAJAL...")
    cell_dir = os.path.join(config.MASKS_DIR, 'cell')
    nuc_dir  = os.path.join(config.MASKS_DIR, 'nuc')

    cell_dist_csv = os.path.join(config.CAJAL_OUTPUT_DIR, f"cell_qgw_{config.N_POINTS:03d}P_dists.csv")
    nuc_dist_csv  = os.path.join(config.CAJAL_OUTPUT_DIR, f"nuc_qgw_{config.N_POINTS:03d}P_dists.csv")

    # figure out how many of {cell, nuc} actually need to run BEFORE
    # deciding how many cores to give each one - splitting config.N_JOBS in
    # half unconditionally would waste half the allocation whenever only
    # one of the two is still needed (e.g. resuming after a partial run
    # that already finished one modality). needs_recompute, not a bare
    # exists check, so a dist CSV left over from BEFORE extract_masks
    # re-extracted a now-stale selection gets correctly redone instead of
    # silently reused.
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
            tasks.append(delayed(run_cajal)(cell_dir, config.CAJAL_OUTPUT_DIR, 'cell', config.N_POINTS, per_task_jobs, config.N_CLUSTERS))
        if needs_cajal['nuc']:
            tasks.append(delayed(run_cajal)(nuc_dir, config.CAJAL_OUTPUT_DIR, 'nuc', config.N_POINTS, per_task_jobs, config.N_CLUSTERS))

        Parallel(n_jobs=n_tasks)(tasks)

    # --- step 4/4: fit UMAP and PHATE directly on the CAJAL distances ---
    print("\n[4/4] Fitting embeddings...")
    cell_umap_path  = os.path.join(config.CAJAL_OUTPUT_DIR, 'cell_umap.npz')
    nuc_umap_path   = os.path.join(config.CAJAL_OUTPUT_DIR, 'nuc_umap.npz')
    cell_phate_path = os.path.join(config.CAJAL_OUTPUT_DIR, 'cell_phate.npz')
    nuc_phate_path  = os.path.join(config.CAJAL_OUTPUT_DIR, 'nuc_phate.npz')
    get_umap_embedding(cell_dist_csv, cell_umap_path, seed=config.UMAP_SEED)
    get_umap_embedding(nuc_dist_csv, nuc_umap_path, seed=config.UMAP_SEED)
    get_phate_embedding(cell_dist_csv, cell_phate_path, seed=config.PHATE_SEED)
    get_phate_embedding(nuc_dist_csv, nuc_phate_path, seed=config.PHATE_SEED)

    print("\nDone.")


if __name__ == '__main__':
    main()
