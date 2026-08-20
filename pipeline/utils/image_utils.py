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