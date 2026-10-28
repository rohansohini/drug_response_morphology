import numpy as np
import pandas as pd

from tqdm import tqdm
from joblib import Parallel, delayed
from scipy import stats

from pipeline.utils.image_utils import load_mask

###### Output Columns:
#  - pixel_id:  segmentation label of the cell in its mask image (mode of nucleus region)
#  - pixel_dup: True if another cell in the same (track, frame) shares the same pixel_id


def get_pixel_id(row: pd.Series, mask_img: np.ndarray, buffer: int = 0) -> np.float64:
    '''
    Return the segmentation label at a cell's nucleus coordinates.

    Takes the mode of non-background pixels in a buffered region around
    the nucleus. Returns 0 if no foreground pixels are found.

    Inputs:
        row:      track DataFrame row (must have: x_coord, y_coord)
        mask_img: segmentation mask as a NumPy array
        buffer:   pixel buffer around nucleus coordinate (default 0)
    Outputs:
        pixel_id: segmentation label (mode of region), or 0 if empty
    '''
    x = round(row['x_coord'])
    y = round(row['y_coord'])

    n_rows, n_cols = mask_img.shape

    # define buffered region, clamped to image bounds
    x_min, x_max = max(0, x - buffer), min(n_cols, x + buffer + 1)
    y_min, y_max = max(0, y - buffer), min(n_rows, y + buffer + 1)

    # extract region, handling edge cases where one axis collapses
    if x_min == x_max and y_min == y_max:
        region = np.array([mask_img[y, x]])
    elif x_min == x_max:
        region = mask_img[y_min:y_max, x].flatten()
    elif y_min == y_max:
        region = mask_img[y, x_min:x_max].flatten()
    else:
        region = mask_img[y_min:y_max, x_min:x_max].flatten()

    # exclude background (0) pixels before taking mode
    region = region[region != 0]

    if len(region) == 0:
        return 0

    return stats.mode(region, keepdims=False).mode


def process_group(group: pd.DataFrame, direc: str, buffer: int = 0) -> pd.DataFrame:
    '''
    Assign pixel_id to all rows in a (track, frame) group using one mask load.

    Inputs:
        group:  DataFrame rows sharing the same (track, frame)
        direc:  directory containing mask images
        buffer: pixel buffer around nucleus coordinate (default 0)
    Outputs:
        group: same DataFrame with pixel_id column added
    '''
    mask_img          = load_mask(group.iloc[0]['track'], group.iloc[0]['frame'], direc)
    group['pixel_id'] = group.apply(lambda row: get_pixel_id(row, mask_img, buffer), axis=1)
    return group


def add_pixel_ids(df: pd.DataFrame, direc: str, buffer: int = 0, n_jobs: int = 8) -> pd.DataFrame:
    '''
    Add pixel_id and pixel_dup columns to a single-drug track DataFrame.

    For each (track, frame) group, loads the mask image once and assigns
    each cell its segmentation label. Flags duplicate labels within a group.

    Inputs:
        df:     single-drug track DataFrame (must have: track, frame, x_coord, y_coord)
        direc:  directory containing mask images for this drug
        buffer: pixel buffer around nucleus coordinate (default 0)
        n_jobs: number of parallel jobs (default 8)
    Outputs:
        df: input DataFrame with added pixel_id and pixel_dup columns
    '''
    groups = [group for _, group in df.groupby(['track', 'frame'])]

    # process each (track, frame) group in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_group)(group, direc, buffer) for group in tqdm(groups, desc='  Assigning pixel IDs')
    )

    df = pd.concat(results).sort_index().reset_index(drop=True)

    # flag rows where two cells in the same (track, frame) share a pixel_id (excluding those that are 0)
    df['pixel_dup'] = (
        df.groupby(['track', 'frame'])['pixel_id']
          .transform(lambda x: (x.duplicated(keep=False)) & (x != 0))
    )

    df['pixel_id'] = df['pixel_id'].astype(np.uint32)

    n_missing = (df['pixel_id'] == 0).sum()
    n_dup     = df['pixel_dup'].sum()
    print(f"  pixel_id assigned: {n_missing} missing (no foreground pixels), {n_dup} duplicates flagged")

    return df