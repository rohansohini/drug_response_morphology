'''
Run directly: python3 tests/shared/test_extract_masks.py
'''
import os
import sys
import shutil
import tempfile

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.cajal.extract_masks import extract_masks, extracted_mask_paths


def write_combined_tiff(path, cell_mask, nuc_mask):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tifffile.imwrite(path, np.stack([cell_mask, nuc_mask]).astype(np.uint16))


def make_selected_df(drug, frame, pixel_ids):
    return pd.DataFrame({
        'frame':    [frame] * len(pixel_ids),
        'pixel_id': pixel_ids,
        'sampled':  [True] * len(pixel_ids),
    })


def test_extract_masks_filters_to_sampled_pixels_and_pads():
    tmp = tempfile.mkdtemp()
    try:
        combined_dir = os.path.join(tmp, 'combined')
        out_dir      = os.path.join(tmp, 'out')

        cell_mask = np.array([[1, 2], [0, 3]])
        nuc_mask  = np.array([[1, 2], [0, 3]])
        write_combined_tiff(os.path.join(combined_dir, 'naive_frame_01_mask.tiff'), cell_mask, nuc_mask)

        df = make_selected_df('naive', 1, [1, 3])   # pixel_id 2 not selected
        extract_masks(df, 'naive', combined_dir, out_dir, n_jobs=1, pad_size=1)

        cell_path, nuc_path = extracted_mask_paths('naive', 1, out_dir)
        out_cell = tifffile.imread(cell_path)
        out_nuc  = tifffile.imread(nuc_path)

        # padded by 1 on every side: (2,2) -> (4,4)
        assert out_cell.shape == (4, 4)
        # pixel_id 2 zeroed out, 1 and 3 kept
        assert set(np.unique(out_cell)) == {0, 1, 3}
        assert set(np.unique(out_nuc)) == {0, 1, 3}

        print("PASS: extract_masks keeps only sampled pixel_ids and pads as configured")
    finally:
        shutil.rmtree(tmp)


def test_extract_masks_skips_frames_whose_output_already_exists():
    tmp = tempfile.mkdtemp()
    try:
        combined_dir = os.path.join(tmp, 'combined')
        out_dir      = os.path.join(tmp, 'out')

        cell_mask = np.array([[1, 2], [0, 3]])
        nuc_mask  = np.array([[1, 2], [0, 3]])
        write_combined_tiff(os.path.join(combined_dir, 'naive_frame_01_mask.tiff'), cell_mask, nuc_mask)

        df = make_selected_df('naive', 1, [1, 2, 3])
        extract_masks(df, 'naive', combined_dir, out_dir, n_jobs=1, pad_size=0)

        cell_path, nuc_path = extracted_mask_paths('naive', 1, out_dir)
        first_mtime = os.path.getmtime(cell_path)

        # delete the source combined TIFF - if extract_masks tries to
        # re-extract this frame, it has nothing to read and would raise
        os.remove(os.path.join(combined_dir, 'naive_frame_01_mask.tiff'))

        extract_masks(df, 'naive', combined_dir, out_dir, n_jobs=1, pad_size=0)

        assert os.path.getmtime(cell_path) == first_mtime, (
            "expected the already-extracted frame to be left untouched, not re-extracted"
        )
        print("PASS: extract_masks skips a frame whose extracted masks already exist")
    finally:
        shutil.rmtree(tmp)


def test_extract_masks_re_extracts_a_frame_whose_cached_selection_is_stale():
    # simulates the real failure mode: masks were extracted once for an
    # OLDER selected set (e.g. before an upstream metadata fix), and the
    # cache was never invalidated because the files still exist on disk -
    # extract_masks must notice the cached file's own baked-in pixel_ids
    # no longer match the CURRENT df and re-extract, not silently reuse
    # a selection that's gone stale
    tmp = tempfile.mkdtemp()
    try:
        combined_dir = os.path.join(tmp, 'combined')
        out_dir      = os.path.join(tmp, 'out')

        cell_mask = np.array([[1, 2], [0, 3]])
        nuc_mask  = np.array([[1, 2], [0, 3]])
        write_combined_tiff(os.path.join(combined_dir, 'naive_frame_01_mask.tiff'), cell_mask, nuc_mask)

        # first run: only pixel_id 1 selected
        stale_df = make_selected_df('naive', 1, [1])
        extract_masks(stale_df, 'naive', combined_dir, out_dir, n_jobs=1, pad_size=0)

        cell_path, nuc_path = extracted_mask_paths('naive', 1, out_dir)
        assert set(np.unique(tifffile.imread(cell_path))) == {0, 1}

        # second run: selection changed upstream to include 1 AND 3 - the
        # cached file from the first run is now stale
        fresh_df = make_selected_df('naive', 1, [1, 3])
        extract_masks(fresh_df, 'naive', combined_dir, out_dir, n_jobs=1, pad_size=0)

        out_cell = tifffile.imread(cell_path)
        assert set(np.unique(out_cell)) == {0, 1, 3}, (
            f"expected the stale cache to be re-extracted with the current selection {{0, 1, 3}}, "
            f"got {set(np.unique(out_cell))}"
        )
        print("PASS: extract_masks re-extracts a frame whose cached selection no longer matches df")
    finally:
        shutil.rmtree(tmp)


if __name__ == '__main__':
    test_extract_masks_filters_to_sampled_pixels_and_pads()
    test_extract_masks_skips_frames_whose_output_already_exists()
    test_extract_masks_re_extracts_a_frame_whose_cached_selection_is_stale()
    print("\nAll tests passed.")
