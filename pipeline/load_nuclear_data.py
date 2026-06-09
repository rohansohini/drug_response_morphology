import os
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


def load_nuclear_data(direc: str, img_size: int = 2501) -> pd.DataFrame:
    '''
    Load nuclear segmentation data from a directory of .mat files.
    Returns one row per nucleus per frame across all tracks.

    Inputs:
        direc:    path to nuclear_segmentation_info/ directory
        img_size: side length of the square image for pixel index conversion (default 2501)
    Outputs:
        df: nuclear DataFrame with columns [frame, x_coord, y_coord, track, pixel_idx_list]
    '''
    rows      = []
    mat_files = np.sort([f for f in os.listdir(direc) if f.endswith('.mat')])

    for file in mat_files:
        data     = loadmat(os.path.join(direc, file))
        n_frames = data['clusters'].shape[0]
        track = int(file.split('segmentation_info_')[1].split('.mat')[0])

        for f in range(n_frames):
            cluster  = data['clusters'][f][0]
            n_nuclei = cluster['PixelIdxList'].shape[1]

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

    return pd.concat(rows, ignore_index=True)