# process_imaging_data

Pipeline for processing melanoma cell and nuclear segmentation data into combined masks and metadata.

By Rohan Sohini with help from Claude Sonnet 4.6 Medium

---

## Directory Structure

```
process_imaging_data/
├── main.py
├── config.py
├── slurm_submit.sh
├── environment.yml
├── processed_data/
│   ├── {drug_name}_metadata.csv
│   ├── combined_masks/
│   │   └── {drug_name}_frame_YY_mask.tiff
│   └── failure_examples/
│       └── {drug_name}/
│           └── {drug_name}_{reason}_exampleN_trackXX_frameYY.png
├── logs/
│   └── process_imaging_{job_id}.out
└── pipeline/
    ├── discover_raw_data.py
    ├── track_table.py
    ├── load_nuclear_data.py
    ├── pixel_ids.py
    ├── join_data.py
    ├── build_masks.py
    ├── combine_tracks.py
    ├── append_labels.py
    └── utils/
        └── image_utils.py
```

Raw data is **not** stored inside this repo - each drug condition's data lives wherever it actually sits on disk, and `config.DRUG_DIRS` points at it (see Configuration below). Each drug directory must look like:

```
{drug_dir}/
├── sorted_idx.mat
├── Cropped_XX/
│   ├── track_table_*.mat
│   └── segmentation_info_*.mat
└── Output_Segmentation/
    └── Transmission_region_XX/
        └── Transmission_YY_masks.tiff
```

Where:
- `{drug_dir}` — one directory per condition, as given in `config.DRUG_DIRS` (e.g. `A2_naive`, `A1_1uM_vem`)
- `Cropped_XX/` — one folder per track (`XX` ranges 1-25); **only the folder's own number is trusted as the track number** - the numbers embedded in `track_table_*.mat` and `segmentation_info_*.mat` filenames inside it are not reliably correct, so each is found by globbing the folder for exactly one match rather than by constructing an exact filename (see `pipeline/discover_raw_data.py`)
- `track_table_*.mat` — cell track data for that folder's track
- `segmentation_info_*.mat` — nuclear segmentation data for that folder's track
- `sorted_idx.mat` — lineage classification labels (`cloneClassification`), one row per lineage: column 0 is the broad class (`m_label`), column 1 is the sub-class (`k_label`)
- `Output_Segmentation/Transmission_region_XX/` — cell segmentation masks, one folder per track
- `Transmission_YY_masks.tiff` — segmentation mask, one per frame

---

## Pipeline Overview

The pipeline runs end-to-end for each drug condition in `config.DRUG_DIRS`. Steps run in the following order:

**1. `discover_raw_data.py`** — Discover each track's `track_table_*.mat` and `segmentation_info_*.mat` files from the `Cropped_XX/` folders, keyed by the folder's own track number (never the filename's).

**2. `track_table.py`** — Load cell track tables from `.mat` files. Assigns globally unique `node_id`, `pred_node_id`, and `lin_id` across all tracks. Rows beyond `config.MAX_FRAME` are dropped here, before any of the expensive per-frame mask work below runs.

**3. `append_labels.py`** — Append lineage classification labels (`m_label`, `k_label`) to the cell DataFrame. One label pair per lineage, broadcast to all rows sharing that `lin_id`.

**4. `load_nuclear_data.py`** — Load nuclear segmentation data from `.mat` files. Converts MATLAB linear pixel indices to `(row, col)` subscripts. Also restricted to `config.MAX_FRAME`.

**5. `pixel_ids.py`** — Assign each cell a `pixel_id` by looking up its nucleus coordinates in the segmentation mask. Flags duplicate pixel IDs within a `(track, frame)` group.

**6. `join_data.py`** — Join cell and nuclear DataFrames on `(frame, track, x_coord, y_coord)`. Sets `has_mask=False` for unmatched cells, cells with no cell segmentation label, and duplicate pixel IDs. Assigns `mask_fail_reason` for each failure.

**7. `build_masks.py`** — For each `(track, frame)` group, resolves multi-blob segmentation conflicts and builds a 2-layer mask. Reassigns pixel IDs to be contiguous within each frame, across that frame's tracks (IDs are not meaningful across different frames — each frame's mask is its own independent segmentation). Updates `has_mask` and `pixel_id` in the DataFrame accordingly.

**8. `combine_tracks.py`** — Stitches per-track masks into a single 2-layer TIFF per frame. Handles missing tracks by leaving their regions empty.

**9. Export** — Saves `{drug_name}_metadata.csv` with final metadata columns.

---

## Outputs

### `{drug_name}_metadata.csv`
One row per cell per frame with the following columns:

| Column | Description |
|---|---|
| `frame` | Frame number (1-indexed) |
| `node_id` | Globally unique node ID |
| `pred_node_id` | Globally unique predecessor node ID (0 for root nodes) |
| `lin_id` | Globally unique lineage ID |
| `pixel_id` | Segmentation label in the combined mask (0 if `has_mask=False`) |
| `has_mask` | True if cell was matched to a nucleus and passed blob resolution |
| `pixel_dup` | True if cell shared a pixel ID with another cell in the same `(track, frame)` |
| `mask_fail_reason` | Reason for `has_mask=False`: `none`, `no_nuc_match` (no nucleus found, NaN, or empty array), `no_cell_seg`, `pixel_dup`, `multi_nuc`, `nuc_outside_cell`, `multi_cell` |
| `m_label` | Broad lineage class: `1` = dying, `2` = persister, `3` = proliferator |
| `k_label` | Sub-class (1-6) within `m_label`, for a total of 18 combined labels |

### `{drug_name}_frame_YY_mask.tiff`
2-layer TIFF where:
- Layer 0: cell segmentation mask
- Layer 1: nuclear segmentation mask

Both layers share the same pixel ID labels, which are unique and contiguous across all tracks *within this frame* — they reset per frame and are not meaningful across different frames. dtype is `uint16` if max pixel ID ≤ 65535, otherwise `uint32`.

### `failure_examples/{drug_name}/`
A handful of example crops (up to 5 per reason, 100px half-width around the nucleus centroid) for visual inspection of cells with `mask_fail_reason` in `multi_nuc`, `nuc_outside_cell`, or `pixel_dup` — the failure modes expected under cell overcrowding. Each crop re-loads the raw, unmodified mask from disk, shades distinct segmentation blobs in gray, and highlights the example cell's nucleus pixels in red.

---

## Blob Resolution Logic

During mask building, the following rules are applied in order to resolve segmentation conflicts:

1. **Multi-blob nucleus** — if a nucleus has more than one disconnected blob, zero out both the cell and nuclear mask for that pixel ID and set `has_mask=False`
2. **Nucleus outside cell** — if any nuclear pixels fall outside the cell boundary, zero out both masks and set `has_mask=False`
3. **Multi-blob cell** — if a cell has multiple blobs, keep the smallest blob that fully encloses all nuclear pixels. If no blob satisfies this, zero out both masks and set `has_mask=False`
4. **Duplicate pixel IDs** (`pixel_dup=True`) — `has_mask=False` is set in `join_data.py`. During mask building, their cell pixels are excluded via `valid_ids` filtering.

Connected components are computed using 4-connectivity (no diagonal connections), meaning two pixels are considered part of the same blob only if they share an edge, not just a corner.

---

## Configuration

Edit `config.py`:

```python
DRUG_DIRS = {
    'A2_naive':   '/path/to/A2_naive/drug_dir',    # wherever this condition's raw data actually lives
    'A1_1uM_vem': '/path/to/A1_1uM_vem/drug_dir',
}

PROCESSED_DATA_DIR = 'processed_data/' # path to output directory
N_JOBS              = 16                # number of parallel jobs (set to match allocated cores)
BUFFER              = 2                 # pixel buffer for nucleus coordinate lookup
IMG_SIZE            = 2501              # side length of each track image (pixels)
MAX_FRAME           = 13                # frames beyond this are excluded entirely (unreliable, heavy overcrowding)
```

Raw data is read directly from each path in `DRUG_DIRS` - nothing is copied into this repo. Each path must follow the layout described in Directory Structure above.

---

## Setup

All dependencies are defined in `environment.yml`. Create the environment once before running:

```bash
mamba env create -f environment.yml
conda activate process_imaging
```

---

## Running

### Locally

```bash
conda activate process_imaging
cd process_imaging_data
python main.py
```

### On Quest (HPC)

Make sure the environment is created first (one-time setup):

```bash
mamba env create -f environment.yml
```

Then submit the job:

```bash
cd /{YOUR_PATH}/process_imaging_data
sbatch slurm_submit.sh
```

Logs are written to `logs/process_imaging_{job_id}.out`.

Drug conditions come from `config.DRUG_DIRS`. Each condition is processed independently and sequentially.

A drug is skipped if its `processed_data/{drug_name}_metadata.csv` already exists, rather than being reprocessed from scratch. Delete that file (there's no need to delete `combined_masks/` or `failure_examples/`, those get overwritten on a fresh run) to force that drug to be reprocessed - e.g. after changing `config.py` or the raw data itself.