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

# config.py's OUTPUT_DIR is a relative path, which Python resolves
# against the current working directory at runtime, not this script's
# own location - if main.py is ever invoked as
# `python analyses/05_vem_144hr_lineage_relatedness/main.py` from
# somewhere else (e.g. the repo root) rather than `cd`-ing into this
# directory first, that path would silently resolve under the WRONG
# directory, including possibly colliding with another analysis's
# output. Re-anchor it to this script's own directory, same as every
# notebook already does. config.output_dir()/masks_dir()/
# cajal_output_dir() all derive from this one variable, so re-anchoring
# it here is enough for every condition's own subtree too.
config.OUTPUT_DIR = os.path.join(ANALYSIS_DIR, config.OUTPUT_DIR)


def process_condition(condition: str, drug: str, max_frame: int) -> None:
    masks_dir        = config.masks_dir(condition)
    cajal_output_dir = config.cajal_output_dir(condition)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(cajal_output_dir, exist_ok=True)

    # --- step 1/3: select cells across every frame 1-max_frame ---
    #  this is a flat per-frame subsample of the FULL metadata - the
    #  notebook separately re-reads that same full metadata (not just
    #  this selection) to reconstruct the true lineage tree, since a
    #  sister/cousin relationship must be resolved from every cell ever
    #  tracked, not just the ones that happened to survive this
    #  subsample (see shared/lineage_tree.py and the notebook's own
    #  "Method" section).
    print(f"[1/3] ({condition}) Loading metadata and selecting cells...")
    metadata = pd.read_csv(config.metadata_csv_path(condition))

    rng = np.random.default_rng(config.SAMPLE_SEED)
    selected = select_cells(metadata, drug, max_frame, rng)
    selected['sampled'] = True   # extract_masks expects this column name

    print(f"  Selected {len(selected)} cells total across {selected['frame'].nunique()} frames")
    for frame in sorted(selected['frame'].unique()):
        n = (selected['frame'] == frame).sum()
        print(f"    frame {frame}: {n} cells")

    selected_path = os.path.join(cajal_output_dir, 'selected_metadata.csv')
    selected.to_csv(selected_path, index=False)
    print(f"  Saved selection to {selected_path}")

    # --- step 2/3: extract masks for every selected cell ---
    print(f"\n[2/3] ({condition}) Extracting masks...")
    extract_masks(
        selected, drug, config.COMBINED_MASKS_DIR, masks_dir,
        n_jobs=config.N_JOBS, pad_size=config.PAD_SIZE,
    )

    # --- step 3/3: run CAJAL once per modality ---
    #  the GW distance matrix this produces IS the morphology
    #  representation used downstream (see notebooks/01_relatedness.ipynb) -
    #  no UMAP/PHATE embedding is fit here, since the test needs the real
    #  distances, not a 2D projection that distorts them.
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
            tasks.append(delayed(run_cajal)(cell_dir, cajal_output_dir, 'cell', config.N_POINTS, per_task_jobs, config.N_CLUSTERS))
        if needs_cajal['nuc']:
            tasks.append(delayed(run_cajal)(nuc_dir, cajal_output_dir, 'nuc', config.N_POINTS, per_task_jobs, config.N_CLUSTERS))

        Parallel(n_jobs=n_tasks)(tasks)


def main():
    for condition, info in config.CONDITIONS.items():
        print(f"=== {condition} ({info['drug']}) ===")
        process_condition(condition, info['drug'], info['max_frame'])
        print()

    print("Done.")


if __name__ == '__main__':
    main()
