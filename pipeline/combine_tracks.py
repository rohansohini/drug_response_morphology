import os
import numpy as np
import pandas as pd
import tifffile

from tqdm import tqdm

###### Output:
#  - one 2-layer uint32 TIFF per frame: {drug_name}_frame_YY_mask.tiff
#  - saved to out_dir/


def get_grid_dims(n_tracks: int) -> tuple[int, int]:
    '''
    Infer grid dimensions from number of tracks.
    Asserts that tracks form a perfect square grid.

    Inputs:
        n_tracks: total number of expected tracks (including missing)
    Outputs:
        n_rows, n_cols: grid dimensions
    '''
    n = int(np.sqrt(n_tracks))
    assert n * n == n_tracks, f"Tracks do not form a perfect square grid: {n_tracks} tracks"
    return n, n


def combine_frame(
    frame:           int,
    actual_tracks:   list[int],
    expected_tracks: list[int],
    masks:           dict,
    img_size:        int,
    n_rows:          int,
    n_cols:          int,
    dtype:           np.dtype
) -> np.ndarray:
    '''
    Stitch all tracks for a single frame into one large 2-layer mask.
    Missing tracks are left as empty (zero) regions.

    Inputs:
        frame:           frame number
        actual_tracks:   sorted list of actual track numbers
        expected_tracks: sorted list of expected track numbers (including missing)
        masks:           dict of {(track, frame): 2-layer mask array}
        img_size:        side length of each track image
        n_rows:          number of rows in the grid
        n_cols:          number of cols in the grid
        dtype:           np.uint16 or np.uint32 depending on the number of cells
    Outputs:
        combined: 2-layer uint16/32 array of shape [2, n_rows*img_size, n_cols*img_size]
    '''
    combined = np.zeros((2, n_rows * img_size, n_cols * img_size), dtype=dtype)

    for i, track in enumerate(expected_tracks):
        row = i // n_cols
        col = i  % n_cols

        r0, r1 = row * img_size, (row + 1) * img_size
        c0, c1 = col * img_size, (col + 1) * img_size

        # skip missing tracks, leave region as zeros
        if track not in actual_tracks:
            continue

        combined[:, r0:r1, c0:c1] = masks[(track, frame)]

    return combined


def combine_tracks(
    df:       pd.DataFrame,
    masks:    dict,
    drug:     str,
    out_dir:  str,
    dtype:    np.dtype,
    img_size: int = 2501,
) -> None:
    '''
    Stitch per-track masks into one combined TIFF per frame and save.

    Inputs:
        df:       updated cell DataFrame (must have: track, frame)
        masks:    dict of {(track, frame): 2-layer uint32 mask array}
        drug:     drug condition name (used for output filename)
        out_dir:  output directory
        dtype:    np.uint16 or np.uint32 depending on the number of cells
        img_size: side length of each track image (default 2501)
    '''
    os.makedirs(out_dir, exist_ok=True)

    actual_tracks   = sorted(df['track'].unique())
    min_track       = min(actual_tracks)
    max_track       = max(actual_tracks)
    expected_tracks = list(range(min_track, max_track + 1))
    n_expected      = len(expected_tracks)

    n_rows, n_cols = get_grid_dims(n_expected)
    print(f"  Grid: {n_rows}x{n_cols} ({n_expected} expected tracks)")

    missing = set(expected_tracks) - set(actual_tracks)
    if missing:
        print(f"  Missing tracks: {sorted(missing)} — regions will be empty")
    else:
        print(f"  No missing tracks")

    frames = sorted(df['frame'].unique())
    print(f"  Writing {len(frames)} frames...")

    for frame in tqdm(frames, desc='  Combining frames'):
        combined = combine_frame(
            frame, actual_tracks, expected_tracks,
            masks, img_size, n_rows, n_cols, dtype
        )
        out_path = os.path.join(out_dir, f"{drug}_frame_{str(frame).zfill(2)}_mask.tiff")
        tifffile.imwrite(out_path, combined)

    print(f"  Saved {len(frames)} TIFFs to {out_dir}")