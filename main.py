import os
import pandas as pd
from joblib import Parallel, delayed

import config
from pipeline.select_cells import select_cells
from cajal_pipeline.extract_masks import extract_masks
from cajal_pipeline.run_cajal import run_cajal
from cajal_pipeline.embed import get_umap_embedding


def main():
    os.makedirs(config.MASKS_DIR, exist_ok=True)
    os.makedirs(config.CAJAL_OUTPUT_DIR, exist_ok=True)

    # --- step 1: select cells across both conditions ---
    print("[1/4] Loading metadata and selecting cells...")
    metadata_by_condition = {c: pd.read_csv(config.metadata_csv_path(c)) for c in config.CONDITIONS}
    selected = select_cells(metadata_by_condition)
    selected['sampled'] = True   # extract_masks expects this column name

    print(f"  Selected {len(selected)} cells total")
    for condition in config.CONDITIONS:
        n = (selected['condition'] == condition).sum()
        print(f"    {condition}: {n} cells")
    print("  By subclass:")
    for subclass, n in selected['subclass'].value_counts().items():
        print(f"    {subclass}: {n}")

    selected_path = os.path.join(config.CAJAL_OUTPUT_DIR, 'selected_metadata.csv')
    selected.to_csv(selected_path, index=False)
    print(f"  Saved selection to {selected_path}")

    # --- step 2: extract masks for every selected cell, per condition ---
    print("\n[2/4] Extracting masks...")
    for condition in config.CONDITIONS:
        condition_df = selected[selected['condition'] == condition]
        extract_masks(
            condition_df, condition, config.COMBINED_MASKS_DIR, config.MASKS_DIR,
            n_jobs=config.N_JOBS, pad_size=config.PAD_SIZE,
        )

    # --- step 3: run CAJAL once per modality, across BOTH conditions together ---
    #  (masks from every condition land in the same cell/ and nuc/ folders,
    #  so CAJAL computes one shared distance matrix covering both - distances
    #  are only comparable if they come from the same CAJAL run)
    print("\n[3/4] Running CAJAL...")
    cell_dir = os.path.join(config.MASKS_DIR, 'cell')
    nuc_dir  = os.path.join(config.MASKS_DIR, 'nuc')

    cell_dist_csv = os.path.join(config.CAJAL_OUTPUT_DIR, f"cell_qgw_{config.N_POINTS:03d}P_dists.csv")
    nuc_dist_csv  = os.path.join(config.CAJAL_OUTPUT_DIR, f"nuc_qgw_{config.N_POINTS:03d}P_dists.csv")

    half_jobs = max(1, config.N_JOBS // 2)
    tasks = []
    if os.path.exists(cell_dist_csv):
        print(f"  Skipping cell CAJAL run: {cell_dist_csv} already exists")
    else:
        tasks.append(delayed(run_cajal)(cell_dir, config.CAJAL_OUTPUT_DIR, 'cell', config.N_POINTS, half_jobs, config.N_CLUSTERS))

    if os.path.exists(nuc_dist_csv):
        print(f"  Skipping nuclear CAJAL run: {nuc_dist_csv} already exists")
    else:
        tasks.append(delayed(run_cajal)(nuc_dir, config.CAJAL_OUTPUT_DIR, 'nuc', config.N_POINTS, half_jobs, config.N_CLUSTERS))

    if tasks:
        print(f"  Processing {len(tasks)} mask type(s) in parallel ({half_jobs} jobs each)...")
        Parallel(n_jobs=len(tasks))(tasks)

    # --- step 4: fit UMAP directly on the CAJAL distances ---
    print("\n[4/4] Fitting UMAP embeddings...")
    cell_umap_path = os.path.join(config.CAJAL_OUTPUT_DIR, 'cell_umap.npz')
    nuc_umap_path  = os.path.join(config.CAJAL_OUTPUT_DIR, 'nuc_umap.npz')
    get_umap_embedding(cell_dist_csv, cell_umap_path, seed=config.UMAP_SEED)
    get_umap_embedding(nuc_dist_csv, nuc_umap_path, seed=config.UMAP_SEED)

    print("\nDone.")


if __name__ == '__main__':
    main()
