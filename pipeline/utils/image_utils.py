import io
import numpy as np
import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_mask(track: int, frame: int, direc: str) -> np.ndarray:
    '''
    Load and return the segmentation mask for a given track and frame.

    Inputs:
        track: track number
        frame: frame number
        direc: directory containing the mask image folders
    Outputs:
        mask_img: segmentation mask as a NumPy array
    '''
    folder    = 'transmission_region_' + str(int(track)).zfill(2) + '/'
    file_name = 'transmission_' + str(int(frame)).zfill(2) + '_masks.tiff'
    full_path = os.path.join(direc, folder, file_name)

    with open(full_path, 'rb') as f:
        mask_img = np.array(Image.open(io.BytesIO(f.read())))

    return mask_img

def save_failure_examples(
    df,
    seg_mask_dir: str,
    drug: str,
    out_dir: str,
    n_per_reason: int = 5,
    crop_size: int = 100,
) -> None:
    '''
    Save a few example crops for visual inspection of the overcrowding-
    related failure reasons. For each example cell, re-loads the raw
    (unmodified) cell mask from disk, crops around the cell's nucleus
    centroid, and highlights its nucleus pixels in red - so overlapping/
    crowded cells are visually obvious at a glance.

    Inputs:
        df:            processed DataFrame, read before CSV export (must
                        have: track, frame, x_coord, y_coord,
                        pixel_idx_list, mask_fail_reason)
        seg_mask_dir:  directory containing raw per-track mask images
        drug:          drug condition name, used in output filenames
        out_dir:       output directory for saved crops
        n_per_reason:  number of example crops to save per failure reason (default 5)
        crop_size:     half-width of the cropped region in pixels (default 100)
    '''
    os.makedirs(out_dir, exist_ok=True)

    for reason in ['multi_nuc', 'nuc_outside_cell', 'pixel_dup']:
        examples = df[df['mask_fail_reason'] == reason].head(n_per_reason)

        for i, (_, row) in enumerate(examples.iterrows()):
            cell_mask = load_mask(row['track'], row['frame'], seg_mask_dir)

            x, y = int(round(row['x_coord'])), int(round(row['y_coord']))
            r0, r1 = max(0, y - crop_size), y + crop_size
            c0, c1 = max(0, x - crop_size), x + crop_size
            crop = cell_mask[r0:r1, c0:c1]

            # shade by blob id so distinct cells are visually distinguishable
            shaded = np.zeros_like(crop, dtype=np.uint8)
            if crop.max() > 0:
                shaded = (50 + (crop.astype(np.int64) % 41) * 5).astype(np.uint8)
                shaded[crop == 0] = 0
            img = np.stack([shaded, shaded, shaded], axis=-1)

            # highlight this cell's nucleus pixels in red
            coords = row['pixel_idx_list']
            if isinstance(coords, np.ndarray) and len(coords) > 0:
                in_crop = coords[(coords[:, 0] >= r0) & (coords[:, 0] < r1) & (coords[:, 1] >= c0) & (coords[:, 1] < c1)]
                img[in_crop[:, 0] - r0, in_crop[:, 1] - c0] = [255, 0, 0]

            out_path = os.path.join(
                out_dir, f"{drug}_{reason}_example{i+1}_track{int(row['track'])}_frame{int(row['frame'])}.png"
            )
            Image.fromarray(img).save(out_path)

    print(f"  Saved failure example crops to {out_dir}")


def ind2sub_square(sz: int, ind: np.ndarray) -> np.ndarray:
    '''
    Convert MATLAB-style linear indices to (row, col) subscripts for a square matrix.
    Equivalent to MATLAB's ind2sub for square matrices.

    Inputs:
        sz:  side length of the square matrix
        ind: 1-based linear indices (MATLAB-style)
    Outputs:
        idxs: (N x 2) array of 0-based [row, col] indices
    '''
    # MATLAB stores PixelIdxList per-nucleus in the smallest integer dtype
    # that fits that nucleus's own values (e.g. uint16 for a nucleus near
    # the image's low-index edge), which can be narrower than sz - widen
    # to int64 before doing arithmetic so this doesn't overflow.
    ind0 = ind.astype(np.int64) - 1   # convert from 1-based (MATLAB) to 0-based (Python)
    row  = ind0 % sz                  # row = remainder when dividing by matrix height
    col  = ind0 // sz                 # col = quotient when dividing by matrix height
    return np.column_stack((row.astype(int), col.astype(int)))