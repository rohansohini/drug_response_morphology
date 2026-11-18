import numpy as np
import pandas as pd
from scipy.io import loadmat

from pipeline.utils.image_utils import ind2sub_square

###### Nuclear Segmentation Info Column Reference (raw .mat file):
#  - clusters[f][0]['Centroid']:     nucleus centroid coordinates (1-indexed)
#  - clusters[f][0]['PixelIdxList']: linear indices of nucleus pixels (1-indexed, MATLAB-style)

###### Output Columns:
#  - frame:          frame number (1-indexed)
#  - x_coord:        nucleus centroid x-coordinate (0-indexed)
#  - y_coord:        nucleus centroid y-coordinate (0-indexed)
#  - track:          track number (1-indexed)
#  - pixel_idx_list: (N x 2) array of 0-indexed [row, col] pixel indices for each nucleus


def load_nuclear_data(segmentation_info_files: dict[int, str], img_size: int = 2501) -> pd.DataFrame:
    '''
    Load nuclear segmentation data from each track's segmentation_info .mat file.
    Returns one row per nucleus per frame across all tracks.

    Inputs:
        segmentation_info_files: {track_number: path to that track's
                                  segmentation_info_*.mat} (see
                                  discover_raw_data.discover_tracks) - track
                                  number comes from this mapping, not the
                                  filename, since the number embedded in
                                  segmentation_info_*.mat filenames is not
                                  reliably correct
        img_size: side length of the square image for pixel index conversion (default 2501)
    Outputs:
        df: nuclear DataFrame with columns [frame, x_coord, y_coord, track, pixel_idx_list]
    '''
    rows = []

    for track, file_path in segmentation_info_files.items():
        data     = loadmat(file_path)
        n_frames = data['clusters'].shape[0]

        for f in range(n_frames):
            cluster = data['clusters'][f][0]

            # a frame where this track had NO nuclei detected at all is
            # saved as an empty matrix ([] in MATLAB), not a struct with
            # empty Centroid/PixelIdxList fields - cluster['PixelIdxList']
            # only works on an actual struct (string field access raises
            # IndexError on a plain ndarray), so check for this case
            # first. A struct that DOES exist but has an empty
            # PixelIdxList (n_nuclei == 0) is the same situation
            # serialized differently - treat both the same way: no
            # nuclei this frame, contribute no rows. join_data.py
            # already expects this (a cell with no nuclear match at all
            # gets mask_fail_reason='no_nuc_match' via its left join),
            # so skipping is the correct behavior, not a workaround.
            if cluster.dtype.names is None:
                continue
            n_nuclei = cluster['PixelIdxList'].shape[1]
            if n_nuclei == 0:
                continue

            # centroid coordinates, converted from 1-indexed (MATLAB) to 0-indexed (Python)
            centroid = np.array([i.flatten() for i in cluster['Centroid'][0]]) - 1

            # convert linear pixel indices to (row, col) subscripts for each nucleus
            pixel_idx_list = [
                ind2sub_square(img_size, cluster['PixelIdxList'][0][i].flatten())
                for i in range(n_nuclei)
            ]

            rows.append(pd.DataFrame({
                'frame':          f + 1,          # 1-indexed
                'x_coord':        centroid[:, 0],
                'y_coord':        centroid[:, 1],
                'track':          track,          # 1-indexed
                'pixel_idx_list': pixel_idx_list,
            }))

    assert rows, (
        f"every frame across every one of {len(segmentation_info_files)} track(s) had zero nuclei - "
        f"nothing to return. This is almost certainly a sign something's wrong upstream (e.g. the "
        f"wrong segmentation_info_*.mat files), not a real all-empty dataset."
    )
    return pd.concat(rows, ignore_index=True)