import os
import time

import config
from pipeline.discover_raw_data  import discover_tracks
from pipeline.track_table        import load_track_table
from pipeline.append_labels      import append_labels
from pipeline.load_nuclear_data  import load_nuclear_data
from pipeline.pixel_ids          import add_pixel_ids
from pipeline.join_data          import join_data
from pipeline.build_masks        import build_masks
from pipeline.combine_tracks     import combine_tracks
from pipeline.utils.image_utils  import save_failure_examples


def process_drug(drug: str, drug_dir: str) -> None:
    '''
    Run the full pipeline for a single drug condition.

    Inputs:
        drug:     drug condition name (e.g. 'naive', 'vemurafenib')
        drug_dir: path to this drug's raw data root (contains Cropped_XX/,
                  Output_Segmentation/, sorted_idx.mat - see config.DRUG_DIRS)
    '''
    print(f"\n{'='*50}")
    print(f"Processing: {drug}")
    print(f"{'='*50}")
    t_start = time.time()

    # --- paths ---
    seg_mask_dir = drug_dir
    labels_path  = os.path.join(drug_dir, 'sorted_idx.mat')

    # --- step 1: discover and load cell track tables ---
    print("\n[1/7] Discovering and loading track tables...")
    t = time.time()
    track_table_files, segmentation_info_files = discover_tracks(drug_dir)
    df = load_track_table(track_table_files)
    print(f"  Loaded {len(df)} rows across {df['track'].nunique()} tracks ({time.time()-t:.1f}s)")

    # restrict to reliable frames only - dropped here, before any of the
    # expensive per-frame mask work below, so frames beyond MAX_FRAME cost
    # nothing rather than just getting filtered out of the final export
    n_before = len(df)
    df = df[df['frame'] <= config.MAX_FRAME].reset_index(drop=True)
    print(f"  Restricted to frames <= {config.MAX_FRAME}: {len(df)}/{n_before} rows kept")

    # --- step 2: append lineage classification labels ---
    print("\n[2/7] Appending classification labels...")
    t = time.time()
    df = append_labels(df, labels_path)
    print(f"  Done ({time.time()-t:.1f}s)")

    # --- step 3: load nuclear data ---
    print("\n[3/7] Loading nuclear data...")
    t = time.time()
    nuc_df = load_nuclear_data(segmentation_info_files, img_size=config.IMG_SIZE)
    print(f"  Loaded {len(nuc_df)} nuclei ({time.time()-t:.1f}s)")

    n_before = len(nuc_df)
    nuc_df = nuc_df[nuc_df['frame'] <= config.MAX_FRAME].reset_index(drop=True)
    print(f"  Restricted to frames <= {config.MAX_FRAME}: {len(nuc_df)}/{n_before} nuclei kept")

    # --- step 4: assign pixel IDs ---
    print("\n[4/7] Assigning pixel IDs...")
    t = time.time()
    df = add_pixel_ids(df, seg_mask_dir, buffer=config.BUFFER, n_jobs=config.N_JOBS)
    print(f"  Done ({time.time()-t:.1f}s)")

    # --- step 5: join cell and nuclear data ---
    print("\n[5/7] Joining cell and nuclear data...")
    t = time.time()
    df = join_data(df, nuc_df)
    print(f"  Done ({time.time()-t:.1f}s)")

    # --- step 6: build masks ---
    print("\n[6/7] Building masks...")
    t = time.time()
    df, masks, dtype = build_masks(df, seg_mask_dir, n_jobs=config.N_JOBS)
    print(f"  Done ({time.time()-t:.1f}s)")

    # --- step 7: combine tracks into per-frame TIFFs ---
    print("\n[7/7] Combining tracks...")
    t = time.time()
    combine_tracks(df, masks, drug, config.COMBINED_MASKS_DIR, dtype, img_size=config.IMG_SIZE)
    print(f"  Done ({time.time()-t:.1f}s)")

    # --- save a few failure example crops for visual inspection ---
    failure_examples_dir = os.path.join(config.PROCESSED_DATA_DIR, 'failure_examples', drug)
    save_failure_examples(df, seg_mask_dir, drug, failure_examples_dir)

    # --- export metadata ---
    print("\nExporting metadata...")
    meta_cols = ['frame', 'node_id', 'pred_node_id', 'lin_id', 'pixel_id', 'has_mask', 'pixel_dup', 'mask_fail_reason', 'm_label', 'k_label']
    out_path  = os.path.join(config.PROCESSED_DATA_DIR, f"{drug}_metadata.csv")
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

    # --- per-frame retention breakdown ---
    # lets you see whether loss is concentrated in particular frames (e.g.
    # later frames getting more crowded) rather than spread evenly, and
    # which specific failure mode drives it in each frame
    print(f"\nPer-frame retention: {drug}")
    header = (
        f"  {'frame':>5} {'total':>7} {'valid':>7} {'valid%':>7} "
        f"{'no_nuc':>7} {'no_seg':>7} {'dup':>6} {'m_nuc':>6} {'nuc_out':>8} {'m_cell':>7}"
    )
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for frame in sorted(df['frame'].unique()):
        frame_df    = df[df['frame'] == frame]
        frame_total = len(frame_df)
        frame_valid = frame_df['has_mask'].sum()
        frame_pct   = 100 * frame_valid / frame_total if frame_total > 0 else 0.0
        frame_reasons = frame_df['mask_fail_reason'].value_counts()
        print(
            f"  {frame:>5} {frame_total:>7} {frame_valid:>7} {frame_pct:>6.1f}% "
            f"{frame_reasons.get('no_nuc_match', 0):>7} {frame_reasons.get('no_cell_seg', 0):>7} "
            f"{frame_reasons.get('pixel_dup', 0):>6} {frame_reasons.get('multi_nuc', 0):>6} "
            f"{frame_reasons.get('nuc_outside_cell', 0):>8} {frame_reasons.get('multi_cell', 0):>7}"
        )


def main():
    os.makedirs(config.COMBINED_MASKS_DIR, exist_ok=True)

    print(f"Found conditions: {list(config.DRUG_DIRS.keys())}")

    for drug, drug_dir in config.DRUG_DIRS.items():
        # {drug}_metadata.csv is the last file process_drug() writes, so its
        # presence means that drug's run already finished - skip it rather
        # than redoing the full (slow) pipeline. Delete the CSV to force a
        # drug to be reprocessed (e.g. after changing config.py or raw data).
        out_path = os.path.join(config.PROCESSED_DATA_DIR, f"{drug}_metadata.csv")
        if os.path.exists(out_path):
            print(f"\nSkipping {drug}: {out_path} already exists")
            continue

        process_drug(drug, drug_dir)


if __name__ == '__main__':
    main()
