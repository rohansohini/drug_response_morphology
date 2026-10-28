import numpy as np
import pandas as pd

from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.ndimage import label as cc_label

from pipeline.utils.image_utils import load_mask

###### Output:
#  - updated DataFrame with has_mask, pixel_id, and pixel_dup columns updated
#  - masks dict: {(track, frame): 2-layer uint32 mask array}
#  - dtype: returns np.uint16 or np.unit32 depending the number of cells

def get_bounding_box(binary: np.ndarray) -> tuple:
    '''
    Get bounding box of True pixels in a binary mask.

    Inputs:
        binary: boolean mask
    Outputs:
        r_min, r_max, c_min, c_max: bounding box indices
    '''
    rows  = np.any(binary, axis=1)
    cols  = np.any(binary, axis=0)
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    return r_min, r_max + 1, c_min, c_max + 1


def resolve_nucleus_blobs(nuc_mask: np.ndarray, pixel_id: int) -> bool:
    '''
    Check if a nucleus has multiple disconnected blobs.
    Returns True if valid (single blob), False if invalid (multiple blobs).

    Inputs:
        nuc_mask: nuclear mask layer
        pixel_id: pixel ID of the nucleus to check
    Outputs:
        valid: True if single blob, False if multiple
    '''
    s = np.array([[0,1,0],
                  [1,1,1],
                  [0,1,0]])
    binary = (nuc_mask == pixel_id)
    
    # no nucleus pixels found. shouldn't happen for has_mask=True cells
    if not binary.any():
        return True

    # crop to bounding box before cc_label for speed
    r_min, r_max, c_min, c_max = get_bounding_box(binary)
    cropped    = binary[r_min:r_max, c_min:c_max]
    _, n_blobs = cc_label(cropped, structure=s)

    return n_blobs <= 1

def resolve_cell_blobs(cell_mask: np.ndarray, nuc_mask: np.ndarray, pixel_id: int) -> bool:
    '''
    If a cell has multiple blobs, keep the smallest one that fully encloses
    the nucleus. Modifies cell_mask in place.
    Returns True if a valid blob was found, False otherwise.

    Inputs:
        cell_mask: cell mask layer (modified in place)
        nuc_mask:  nuclear mask layer
        pixel_id:  pixel ID of the cell/nucleus to resolve
    Outputs:
        valid: True if a valid enclosing blob was found, False otherwise
    '''
    s = np.array([[0,1,0],
                  [1,1,1],
                  [0,1,0]])
    binary           = (cell_mask == pixel_id)
    labeled, n_blobs = cc_label(binary, structure=s)

    # nothing to resolve
    if n_blobs <= 1:
        return True

    nuc_pixels = (nuc_mask == pixel_id)

    # early return if nucleus is already zeroed
    if nuc_pixels.sum() == 0:
        return False

    # find blobs that fully enclose the nucleus
    enclosing = [
        k for k in range(1, n_blobs + 1)
        if np.all((labeled == k)[nuc_pixels])
    ]

    if len(enclosing) == 0:
        return False

    # keep the smallest enclosing blob
    keep = min(enclosing, key=lambda k: (labeled == k).sum())

    # zero out all other blobs
    cell_mask[(binary) & (labeled != keep)] = 0

    return True


def zero_out(cell_mask: np.ndarray, nuc_mask: np.ndarray, pixel_id: int) -> None:
    '''
    Zero out a cell and its nucleus in both mask layers.

    Inputs:
        cell_mask: cell mask layer (modified in place)
        nuc_mask:  nuclear mask layer (modified in place)
        pixel_id:  pixel ID to zero out
    '''
    cell_mask[cell_mask == pixel_id] = 0
    nuc_mask[nuc_mask   == pixel_id] = 0


def process_group(group: pd.DataFrame, direc: str) -> tuple[pd.DataFrame, np.ndarray]:
    '''
    Load mask, resolve blobs, and return updated group and 2-layer mask.

    Inputs:
        group: DataFrame rows for a single (track, frame)
        direc: directory containing mask images
    Outputs:
        group: updated DataFrame with has_mask, pixel_id, and mask_fail_reason changes
        mask:  2-layer uint32 mask array [cell_mask, nuc_mask]
    '''
    group     = group.copy()
    track     = group.iloc[0]['track']
    frame     = group.iloc[0]['frame']

    cell_mask = load_mask(track, frame, direc).astype(np.uint32)
    nuc_mask  = np.zeros_like(cell_mask)

    # build nuclear mask from pixel_idx_list for all valid cells
    matched = group[group['has_mask']]
    for _, row in matched.iterrows():
        coords = row['pixel_idx_list']
        if isinstance(coords, np.ndarray):
            nuc_mask[coords[:, 0], coords[:, 1]] = row['pixel_id']

    # zero out any cell pixels not belonging to a valid cell
    valid_ids = set(group[group['has_mask']]['pixel_id'].values)
    cell_mask[~np.isin(cell_mask, list(valid_ids))] = 0

    # resolve each valid cell in order
    for idx, row in group[group['has_mask']].iterrows():
        pixel_id = int(row['pixel_id'])

        # step 1: if nucleus has multiple blobs, cell is invalid
        if not resolve_nucleus_blobs(nuc_mask, pixel_id):
            zero_out(cell_mask, nuc_mask, pixel_id)
            group.loc[idx, 'has_mask']         = False
            group.loc[idx, 'mask_fail_reason'] = 'multi_nuc'
            group.loc[idx, 'pixel_id']         = 0
            continue

        # step 2: if nucleus is not fully contained within cell, cell is invalid
        nuc_pixels  = (nuc_mask  == pixel_id)
        cell_pixels = (cell_mask == pixel_id)
        if not np.all(cell_pixels[nuc_pixels]):
            zero_out(cell_mask, nuc_mask, pixel_id)
            group.loc[idx, 'has_mask']         = False
            group.loc[idx, 'mask_fail_reason'] = 'nuc_outside_cell'
            group.loc[idx, 'pixel_id']         = 0
            continue

        # step 3: if cell has multiple blobs, keep smallest one enclosing nucleus
        if not resolve_cell_blobs(cell_mask, nuc_mask, pixel_id):
            zero_out(cell_mask, nuc_mask, pixel_id)
            group.loc[idx, 'has_mask']         = False
            group.loc[idx, 'mask_fail_reason'] = 'multi_cell'
            group.loc[idx, 'pixel_id']         = 0

    return group, np.stack([cell_mask, nuc_mask], axis=0)


def reassign_frame_ids(df: pd.DataFrame, masks: dict, frame: int) -> tuple[pd.DataFrame, int]:
    '''
    Reassign pixel IDs for a single frame to a contiguous range starting at 1.
    Updates both the masks dict and the DataFrame.

    IDs are scoped to one frame rather than threaded globally across the
    whole drug condition. Each frame's combined mask is its own
    self-contained TIFF file, and every downstream consumer always pairs
    pixel_id with frame, so there is no need for IDs to be unique or
    meaningful across different frames - raw pixel_id itself is only ever
    meaningful within a single frame's independent segmentation to begin
    with. Scoping per frame also keeps the max pixel ID as small as
    possible (bounded by how many cells appear in the single busiest
    frame, not accumulated across every track and frame in the drug
    condition), which keeps mask storage at uint16 instead of spilling
    into uint32.

    A new ID is further scoped per (track, raw pixel_id) pair within the
    frame, not pooled across tracks: two different tracks coincidentally
    reusing the same raw label number in the same frame must not collapse
    onto one new ID either.

    Inputs:
        df:    full DataFrame
        masks: dict of {(track, frame): 2-layer mask array}
        frame: frame number to reassign
    Outputs:
        df:     updated DataFrame
        max_id: highest new pixel ID assigned in this frame (0 if none)
    '''
    frame_mask = df['frame'] == frame
    next_id    = 1

    for track in sorted(df.loc[frame_mask, 'track'].unique()):
        track_frame_mask = frame_mask & (df['track'] == track)

        # old ids for just this (track, frame) - never pooled across tracks or frames
        old_ids = sorted(df.loc[track_frame_mask & (df['pixel_id'] != 0), 'pixel_id'].unique())
        id_map  = {old: new for new, old in enumerate(old_ids, start=next_id)}
        next_id += len(old_ids)

        # update DataFrame rows for this (track, frame) (unmapped stays 0)
        df.loc[track_frame_mask, 'pixel_id'] = df.loc[track_frame_mask, 'pixel_id'].map(
            lambda x: id_map.get(x, 0)
        )

        if not id_map:
            continue

        # build lookup table for vectorized mask remapping
        # lut[old_id] = new_id, index is old pixel ID, value is new pixel ID
        max_old_id = max(id_map.keys())
        lut        = np.zeros(max_old_id + 1, dtype=np.uint32)
        for old, new in id_map.items():
            lut[old] = new

        old_mask = masks[(track, frame)]
        new_mask = np.zeros_like(old_mask)

        # remap all valid pixels in one vectorized operation
        valid           = (old_mask > 0) & (old_mask <= max_old_id)
        new_mask[valid] = lut[old_mask[valid]]
        masks[(track, frame)] = new_mask

    max_id = next_id - 1
    return df, max_id


def build_masks(df: pd.DataFrame, direc: str, n_jobs: int = 8) -> tuple[pd.DataFrame, dict, np.dtype]:
    '''
    Resolve blobs, reassign pixel IDs, and return updated DataFrame and masks.

    Inputs:
        df:     joined cell+nuclear DataFrame (must have: track, frame, pixel_id, has_mask, pixel_dup, pixel_idx_list)
        direc:  directory containing mask images
        n_jobs: number of parallel jobs (default 8)
    Outputs:
        df:    updated DataFrame with resolved has_mask, pixel_id, and pixel_dup columns
        masks: dict of {(track, frame): 2-layer uint32 mask array}
        dtype: either np.uint16 or np.uint32 depending on the number of cells
    '''
    groups = [group for _, group in df.groupby(['track', 'frame'])]

    # process each (track, frame) group in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_group)(group, direc) for group in tqdm(groups, desc='  Resolving blobs')
    )

    # unpack results
    updated_groups, mask_list = zip(*results)
    df    = pd.concat(updated_groups).sort_index().reset_index(drop=True)
    masks = {
        (int(g.iloc[0]['track']), int(g.iloc[0]['frame'])): m
        for g, m in zip(updated_groups, mask_list)
    }

    n_invalid = (~df['has_mask']).sum()
    print(f"  Blob resolution complete: {n_invalid} cells marked has_mask=False")

    # reassign pixel IDs per frame - IDs only need to be unique within a
    # single frame's combined mask (see reassign_frame_ids), so this is no
    # longer threaded sequentially across frames the way per-track
    # threading was; each frame's reassignment is independent
    print("  Reassigning pixel IDs...")
    max_id = 0
    for frame in tqdm(sorted(df['frame'].unique()), desc='  Reassigning frames'):
        df, frame_max_id = reassign_frame_ids(df, masks, frame)
        max_id = max(max_id, frame_max_id)
    print(f"  Pixel IDs reassigned: max pixel_id per frame = {max_id}")

    # choose smallest dtype that fits the max pixel ID
    dtype = np.uint16 if max_id <= 65535 else np.uint32
    print(f"  Using dtype: {dtype}")

    # convert all masks to the chosen dtype
    for key in masks:
        masks[key] = masks[key].astype(dtype)

    return df, masks, dtype