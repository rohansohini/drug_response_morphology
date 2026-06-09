# process_imaging_data

Pipeline for processing melanoma cell and nuclear segmentation data into combined masks and metadata.

By Rohan Sohini with help from Claude Sonnet 4.6 Medium

---

## Directory Structure

```
process_imaging_data/
├── main.py
├── slurm_submit.sh
├── environment.yml
├── raw_data/
│   └── {drug_name}/
│       ├── track_tables/
│       │   └── track_table_XX.mat
│       ├── nuclear_segmentation_info/
│       │   └── segmentation_info_XX.mat
│       ├── sorted_idx.mat
│       └── transmission_region_XX/
│           └── transmission_YY_masks.tiff
├── processed_data/
│   ├── {drug_name}_metadata.csv
│   └── combined_masks/
│       └── {drug_name}_frame_YY_mask.tiff
├── logs/
│   └── process_imaging_{job_id}.out
└── pipeline/
    ├── track_table.py
    ├── load_nuclear_data.py
    ├── pixel_ids.py
    ├── join_data.py
    ├── build_masks.py
    ├── combine_tracks.py
    ├── append_klabels.py
    └── utils/
        └── image_utils.py
```

Where:
- `{drug_name}` — one folder per condition (e.g. `naive`, `vemurafenib`)
- `track_table_XX.mat` — cell track data, one per track
- `segmentation_info_XX.mat` — nuclear segmentation data, one per track
- `sorted_idx.mat` — k-means cluster labels, one per lineage
- `transmission_region_XX/` — cell segmentation masks, one folder per track
- `transmission_YY_masks.tiff` — segmentation mask, one per frame

---

## Pipeline Overview

The pipeline runs end-to-end for each drug condition found under `raw_data/`. Steps run in the following order:

**1. `track_table.py`** — Load cell track tables from `.mat` files. Assigns globally unique `node_id`, `pred_node_id`, and `lin_id` across all tracks.

**2. `append_klabels.py`** — Append k-means cluster labels to the cell DataFrame. One label per lineage, broadcast to all rows sharing that `lin_id`.

**3. `load_nuclear_data.py`** — Load nuclear segmentation data from `.mat` files. Converts MATLAB linear pixel indices to `(row, col)` subscripts.

**4. `pixel_ids.py`** — Assign each cell a `pixel_id` by looking up its nucleus coordinates in the segmentation mask. Flags duplicate pixel IDs within a `(track, frame)` group.

**5. `join_data.py`** — Join cell and nuclear DataFrames on `(frame, track, x_coord, y_coord)`. Sets `has_mask=False` for unmatched cells, cells with no cell segmentation label, and duplicate pixel IDs. Assigns `mask_fail_reason` for each failure.

**6. `build_masks.py`** — For each `(track, frame)` group, resolves multi-blob segmentation conflicts and builds a 2-layer mask. Reassigns pixel IDs to be globally contiguous across all tracks. Updates `has_mask` and `pixel_id` in the DataFrame accordingly.

**7. `combine_tracks.py`** — Stitches per-track masks into a single 2-layer TIFF per frame. Handles missing tracks by leaving their regions empty.

**8. Export** — Saves `{drug_name}_metadata.csv` with final metadata columns.

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
| `k_label` | K-means cluster label for this cell's lineage |

### `{drug_name}_frame_YY_mask.tiff`
2-layer TIFF where:
- Layer 0: cell segmentation mask
- Layer 1: nuclear segmentation mask

Both layers share the same pixel ID labels, which are globally unique and contiguous across all tracks. dtype is `uint16` if max pixel ID ≤ 65535, otherwise `uint32`.

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

Edit the constants at the top of `main.py`:

```python
RAW_DATA_DIR       = 'raw_data/'       # path to raw data
PROCESSED_DATA_DIR = 'processed_data/' # path to output directory
N_JOBS             = 16                # number of parallel jobs (set to match allocated cores)
BUFFER             = 5                 # pixel buffer for nucleus coordinate lookup
IMG_SIZE           = 2501              # side length of each track image (pixels)
```

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

Drug conditions are inferred automatically from folder names under `raw_data/`. Each condition is processed independently and sequentially.