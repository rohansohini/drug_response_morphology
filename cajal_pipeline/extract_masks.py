import os
import numpy as np
import pandas as pd
import tifffile

from tqdm import tqdm
from joblib import Parallel, delayed

###### Ported unchanged from vem_fast_prolif_cajal/extract_masks.py
#  (itself copied unchanged from cancer_morphology_analysis/sampling_pipeline/extract_masks.py)
#  Selection logic lives in pipeline/select_cells.py instead of a random
#  draw - this function itself doesn't care how the selection was made, it
#  just needs a DataFrame with a 'sampled' column.

###### Output:
#  - two single-layer TIFFs per frame (cell, nuc), zeroed to only the
#    pixel_ids where df['sampled'] is True
#  - saved to out_dir/cell/ and out_dir/nuc/


def extract_frame_masks(
    drug: str,
    frame: int,
    sampled_ids: set,
    combined_masks_dir: str,
    out_dir: str,
    pad_size: int,
) -> None:
    '''
    Load a combined 2-layer TIFF, zero out non-selected pixels, pad with zeros,
    and save separate cell and nuclear single-layer TIFFs.

    Inputs:
        drug:               drug condition name
        frame:              frame number
        sampled_ids:        set of pixel_ids to keep
        combined_masks_dir: directory containing combined {drug}_frame_YY_mask.tiff files
        out_dir:            output directory for extracted masks
        pad_size:           number of zero-padding pixels added to each side
    '''
    in_path  = os.path.join(combined_masks_dir, f"{drug}_frame_{str(frame).zfill(2)}_mask.tiff")
    combined = tifffile.imread(in_path)

    cell_mask = combined[0]
    nuc_mask  = combined[1]

    # zero out any pixels not in the selected set
    sampled_arr = np.array(list(sampled_ids))
    cell_mask   = np.where(np.isin(cell_mask, sampled_arr), cell_mask, 0)
    nuc_mask    = np.where(np.isin(nuc_mask,  sampled_arr), nuc_mask,  0)

    # preserve original dtype
    cell_mask = cell_mask.astype(combined.dtype)
    nuc_mask  = nuc_mask.astype(combined.dtype)

    # pad with zeros on all sides
    cell_mask = np.pad(cell_mask, pad_width=pad_size, mode='constant', constant_values=0)
    nuc_mask  = np.pad(nuc_mask,  pad_width=pad_size, mode='constant', constant_values=0)

    cell_dir = os.path.join(out_dir, 'cell')
    nuc_dir  = os.path.join(out_dir, 'nuc')

    cell_path = os.path.join(cell_dir, f"{drug}_frame_{str(frame).zfill(2)}_cell_mask_sampled.tiff")
    nuc_path  = os.path.join(nuc_dir,  f"{drug}_frame_{str(frame).zfill(2)}_nuc_mask_sampled.tiff")

    tifffile.imwrite(cell_path, cell_mask)
    tifffile.imwrite(nuc_path, nuc_mask)


def extract_masks(
    df: pd.DataFrame,
    drug: str,
    combined_masks_dir: str,
    out_dir: str,
    n_jobs: int = 8,
    pad_size: int = 0,
) -> None:
    '''
    For each frame, extract selected cell and nuclear masks from the combined TIFFs.

    Inputs:
        df:                 metadata DataFrame (must have: frame, pixel_id, sampled)
        drug:               drug condition name
        combined_masks_dir: directory containing combined {drug}_frame_YY_mask.tiff files
        out_dir:            output directory for extracted masks
        n_jobs:             number of parallel jobs (default 8)
        pad_size:           number of zero-padding pixels added to each side (default 0)
    '''
    os.makedirs(os.path.join(out_dir, 'cell'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'nuc'), exist_ok=True)

    frames = sorted(df['frame'].unique())

    Parallel(n_jobs=n_jobs)(
        delayed(extract_frame_masks)(
            drug, frame,
            set(df[(df['frame'] == frame) & (df['sampled'])]['pixel_id'].values),
            combined_masks_dir, out_dir, pad_size
        )
        for frame in tqdm(frames, desc=f'  Extracting masks ({drug})')
    )
