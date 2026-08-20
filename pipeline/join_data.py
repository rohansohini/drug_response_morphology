import numpy as np
import pandas as pd

###### Output Columns (additions):
#  - pixel_idx_list:    (N x 2) array of 0-indexed [row, col] pixel indices for each nucleus
#  - has_mask:          True if cell was matched to a nucleus and has a valid mask
#  - mask_fail_reason:  reason for has_mask=False, set here to one of:
#                         'none'         — cell is valid so far
#                         'no_nuc_match' — no nucleus found (no join match, NaN, or empty array)
#                         'no_cell_seg'  — no cell segmentation label at the nucleus
#                         'pixel_dup'    — pixel_id shared with another cell in the same (track, frame)
#                       checks are applied in this order and the first one to match wins;
#                       build_masks.py may later overwrite this with 'multi_nuc',
#                       'nuc_outside_cell', or 'multi_cell'

def is_missing_nucleus(x) -> bool:
    '''Return True if pixel_idx_list has no valid nucleus data.'''
    if x is None:                              return True
    if isinstance(x, float) and np.isnan(x):  return True
    if isinstance(x, np.ndarray) and x.shape[0] == 0: return True
    return False

def join_data(cell_df: pd.DataFrame, nuc_df: pd.DataFrame, round_precision: int = 6) -> pd.DataFrame:
    '''
    Join cell and nuclear DataFrames on (frame, track, x_coord, y_coord).
    Unmatched cells get has_mask=False and a null pixel_idx_list.

    Inputs:
        cell_df:         cell track DataFrame (must have: frame, track, x_coord, y_coord)
        nuc_df:          nuclear DataFrame (must have: frame, track, x_coord, y_coord, pixel_idx_list)
        round_precision: decimal places for coordinate rounding to handle float differences (default 6)
    Outputs:
        df: joined DataFrame with all cell columns plus pixel_idx_list, has_mask, and mask_fail_reason
    '''
    cell_df = cell_df.copy()
    nuc_df  = nuc_df.copy()

    # round coordinates to handle slight floating point differences between datasets
    for df in [cell_df, nuc_df]:
        df['x_coord'] = df['x_coord'].round(round_precision)
        df['y_coord'] = df['y_coord'].round(round_precision)

    # left join so unmatched cells are kept with null pixel_idx_list
    merged = cell_df.merge(
        nuc_df[['frame', 'track', 'x_coord', 'y_coord', 'pixel_idx_list']],
        on=['frame', 'track', 'x_coord', 'y_coord'],
        how='left'
    )

    # initialize all cells as valid
    merged['has_mask']         = True
    merged['mask_fail_reason'] = 'none'

    # checks below are applied in priority order (highest first); a cell already
    # marked invalid by an earlier check keeps that reason even if a later
    # check would also apply to it

    # no nucleus match (no entry or empty pixel_idx_list)
    no_nuc = merged['pixel_idx_list'].apply(is_missing_nucleus)
    merged.loc[no_nuc, 'has_mask']         = False
    merged.loc[no_nuc, 'mask_fail_reason'] = 'no_nuc_match'

    # no cell segmentation label
    no_seg = (merged['pixel_id'] == 0) & (merged['mask_fail_reason'] == 'none')
    merged.loc[no_seg, 'has_mask']         = False
    merged.loc[no_seg, 'mask_fail_reason'] = 'no_cell_seg'

    # duplicate pixel ID
    dup = merged['pixel_dup'] & (merged['mask_fail_reason'] == 'none')
    merged.loc[dup, 'has_mask']            = False
    merged.loc[dup, 'mask_fail_reason']    = 'pixel_dup'

    n_total   = len(merged)
    n_valid   = merged['has_mask'].sum()
    n_no_nuc  = no_nuc.sum()
    n_no_seg  = no_seg.sum()
    n_dup     = dup.sum()
    print(f"  Joined {n_total} cells: {n_valid} valid, {n_no_nuc} no nucleus match, {n_no_seg} no cell seg, {n_dup} pixel dup")

    return merged.reset_index(drop=True)