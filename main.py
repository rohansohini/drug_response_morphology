import os
import time
import numpy as np
import pandas as pd
from natsort import natsorted

from pipeline.track_table        import load_track_table
from pipeline.append_klabels     import append_klabels
from pipeline.load_nuclear_data  import load_nuclear_data
from pipeline.pixel_ids          import add_pixel_ids
from pipeline.join_data          import join_data
from pipeline.build_masks        import build_masks
from pipeline.combine_tracks     import combine_tracks

##############################################################
# CONSTANTS — edit these to match your directory structure
##############################################################

RAW_DATA_DIR       = 'raw_data/'
PROCESSED_DATA_DIR = 'processed_data/'
COMBINED_MASKS_DIR = os.path.join(PROCESSED_DATA_DIR, 'combined_masks')

N_JOBS   = 16
BUFFER   = 5
IMG_SIZE = 2501

##############################################################


def process_drug(drug: str, raw_dir: str) -> None:
    '''
    Run the full pipeline for a single drug condition.

    Inputs:
        drug:    drug condition name (e.g. 'naive', 'vemurafenib')
        raw_dir: path to raw_data/{drug}/ directory
    '''
    print(f"\n{'='*50}")
    print(f"Processing: {drug}")
    print(f"{'='*50}")
    t_start = time.time()

    # --- paths ---
    track_table_dir  = os.path.join(raw_dir, 'track_tables/')
    nuclear_data_dir = os.path.join(raw_dir, 'nuclear_segmentation_info/')
    seg_mask_dir     = raw_dir
    klabels_path     = os.path.join(raw_dir, 'sorted_idx.mat')

    # --- step 1: load cell track tables ---
    print("\n[1/7] Loading track tables...")
    t = time.time()
    track_files = natsorted([
        os.path.join(track_table_dir, f)
        for f in os.listdir(track_table_dir) if f.endswith('.mat')
    ])
    df = load_track_table(track_files)
    print(f"  Loaded {len(df)} rows across {df['track'].nunique()} tracks ({time.time()-t:.1f}s)")

    # --- step 2: append k-means labels ---
    print("\n[2/7] Appending k-means labels...")
    t = time.time()
    df = append_klabels(df, klabels_path)
    print(f"  Done ({time.time()-t:.1f}s)")

    # --- step 3: load nuclear data ---
    print("\n[3/7] Loading nuclear data...")
    t = time.time()
    nuc_df = load_nuclear_data(nuclear_data_dir, img_size=IMG_SIZE)
    print(f"  Loaded {len(nuc_df)} nuclei ({time.time()-t:.1f}s)")

    # --- step 4: assign pixel IDs ---
    print("\n[4/7] Assigning pixel IDs...")
    t = time.time()
    df = add_pixel_ids(df, seg_mask_dir, buffer=BUFFER, n_jobs=N_JOBS)
    print(f"  Done ({time.time()-t:.1f}s)")

    # --- step 5: join cell and nuclear data ---
    print("\n[5/7] Joining cell and nuclear data...")
    t = time.time()
    df = join_data(df, nuc_df)
    print(f"  Done ({time.time()-t:.1f}s)")

    # --- step 6: build masks ---
    print("\n[6/7] Building masks...")
    t = time.time()
    df, masks, dtype = build_masks(df, seg_mask_dir, n_jobs=N_JOBS)
    print(f"  Done ({time.time()-t:.1f}s)")

    # --- step 7: combine tracks into per-frame TIFFs ---
    print("\n[7/7] Combining tracks...")
    t = time.time()
    combine_tracks(df, masks, drug, COMBINED_MASKS_DIR, dtype, img_size=IMG_SIZE)
    print(f"  Done ({time.time()-t:.1f}s)")

    # --- export metadata ---
    print("\nExporting metadata...")
    meta_cols = ['frame', 'node_id', 'pred_node_id', 'lin_id', 'pixel_id', 'has_mask', 'pixel_dup', 'mask_fail_reason', 'k_label']
    out_path  = os.path.join(PROCESSED_DATA_DIR, f"{drug}_metadata.csv")
    df[meta_cols].to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")

    # --- summary ---
    t_total  = time.time() - t_start
    n_total  = len(df)
    n_valid  = df['has_mask'].sum()
    n_failed = n_total - n_valid
    reasons  = df['mask_fail_reason'].value_counts()

    print(f"\n{'='*50}")
    print(f"Summary: {drug}")
    print(f"  Total cells:          {n_total}")
    print(f"  has_mask=True:        {n_valid} ({100*n_valid/n_total:.1f}%)")
    print(f"  has_mask=False:       {n_failed} ({100*n_failed/n_total:.1f}%)")
    print(f"    no_nuc_match:       {reasons.get('no_nuc_match', 0)} ({100*reasons.get('no_nuc_match', 0)/n_total:.1f}%)")
    print(f"    no_cell_seg:        {reasons.get('no_cell_seg', 0)} ({100*reasons.get('no_cell_seg', 0)/n_total:.1f}%)")
    print(f"    pixel_dup:          {reasons.get('pixel_dup', 0)} ({100*reasons.get('pixel_dup', 0)/n_total:.1f}%)")
    print(f"    multi_nuc:          {reasons.get('multi_nuc', 0)} ({100*reasons.get('multi_nuc', 0)/n_total:.1f}%)")
    print(f"    nuc_outside_cell:   {reasons.get('nuc_outside_cell', 0)} ({100*reasons.get('nuc_outside_cell', 0)/n_total:.1f}%)")
    print(f"    multi_cell:         {reasons.get('multi_cell', 0)} ({100*reasons.get('multi_cell', 0)/n_total:.1f}%)")
    print(f"  Total time:           {t_total:.1f}s")
    print(f"{'='*50}")


def main():
    os.makedirs(COMBINED_MASKS_DIR, exist_ok=True)

    # infer drug conditions from folder names under raw_data/
    drugs = natsorted([
        d for d in os.listdir(RAW_DATA_DIR)
        if os.path.isdir(os.path.join(RAW_DATA_DIR, d))
    ])

    print(f"Found conditions: {drugs}")

    for drug in drugs:
        raw_dir = os.path.join(RAW_DATA_DIR, drug)
        process_drug(drug, raw_dir)


if __name__ == '__main__':
    main()
