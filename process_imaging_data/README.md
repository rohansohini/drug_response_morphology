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

Raw data is **not** stored inside this repo. Each drug condition's data lives wherever it actually sits on disk, and `config.DRUG_DIRS` points at it (see Configuration below). Each drug directory must look like:

```
{drug_dir}/
├── clone_kmeans_id.mat
├── Cropped_XX/
│   ├── track_table_*.mat
│   └── segmentation_info_*.mat
└── Output_Segmentation/
    └── Transmission_region_XX/
        └── Transmission_YY_masks.tiff
```

Where:
- `{drug_dir}`: one directory per condition, as given in `config.DRUG_DIRS` (e.g. `naive`, `vem_1uM`)
- `Cropped_XX/`: one folder per track (`XX` ranges 1 to 25). **Only the folder's own number is trusted as the track number.** The numbers embedded in `track_table_*.mat` and `segmentation_info_*.mat` filenames inside it are not reliably correct, so each is found by globbing the folder for exactly one match rather than by constructing an exact filename (see `pipeline/discover_raw_data.py`).
- `track_table_*.mat`: cell track data for that folder's track
- `segmentation_info_*.mat`: nuclear segmentation data for that folder's track
- `clone_kmeans_id.mat`: lineage classification labels (`cloneClassification`), one row per lineage. Column 0 is the broad class (`m_label`), column 1 is the subclass (`k_label`).
- `Output_Segmentation/Transmission_region_XX/`: cell segmentation masks, one folder per track
- `Transmission_YY_masks.tiff`: segmentation mask, one per frame

---

## Pipeline Overview

The pipeline runs end-to-end for each drug condition in `config.DRUG_DIRS`. Steps run in the following order:

**1. `discover_raw_data.py`**: Discover each track's `track_table_*.mat` and `segmentation_info_*.mat` files from the `Cropped_XX/` folders, keyed by the folder's own track number, never the filename's.

**2. `track_table.py`**: Load cell track tables from `.mat` files. Assigns globally unique `node_id`, `pred_node_id`, and `lin_id` across all tracks.

**3. `append_labels.py`**: Append classification labels (`m_label`, `k_label`) to the cell DataFrame. One label pair per *cell* present at `config.DAY0_FRAMES[drug]`, day 0, the frame the drug was actually added on, propagated forward to each day-0 cell's own descendants via the `node_id`/`pred_node_id` chain. Cells before day 0, or from a lineage that dies out before reaching it, get `NaN` labels. See "Day 0 and Cell Classification" below.

**4. `load_nuclear_data.py`**: Load nuclear segmentation data from `.mat` files. Converts MATLAB linear pixel indices to `(row, col)` subscripts.

**5. `pixel_ids.py`**: Assign each cell a `pixel_id` by looking up its nucleus coordinates in the segmentation mask. Flags duplicate pixel IDs within a `(track, frame)` group.

**6. `join_data.py`**: Join cell and nuclear DataFrames on `(frame, track, x_coord, y_coord)`. Sets `has_mask=False` for unmatched cells, cells with no cell segmentation label, and duplicate pixel IDs. Assigns `mask_fail_reason` for each failure.

**7. `build_masks.py`**: For each `(track, frame)` group, resolves multi-blob segmentation conflicts and builds a 2-layer mask. Reassigns pixel IDs to be contiguous within each frame, across that frame's tracks (IDs are not meaningful across different frames; each frame's mask is its own independent segmentation). Updates `has_mask` and `pixel_id` in the DataFrame accordingly.

**8. `combine_tracks.py`**: Stitches per-track masks into a single 2-layer TIFF per frame. Handles missing tracks by leaving their regions empty.

**9. Export**: Saves `{drug_name}_metadata.csv` with final metadata columns.

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
| `m_label` | Broad class: `1` = dying, `2` = persister, `3` = proliferator. `NaN` for a cell before day 0, or whose lineage died before reaching it (see below); no classification exists for it |
| `k_label` | Subclass (1-6) within `m_label`, for a total of 18 combined labels. `NaN` under the same condition as `m_label` |

### `{drug_name}_frame_YY_mask.tiff`
2-layer TIFF where:
- Layer 0: cell segmentation mask
- Layer 1: nuclear segmentation mask

Both layers share the same pixel ID labels, which are unique and contiguous across all tracks *within this frame*. They reset per frame and are not meaningful across different frames. dtype is `uint16` if max pixel ID ≤ 65535, otherwise `uint32`.

### `failure_examples/{drug_name}/`
A handful of example crops (up to 5 per reason, 100px half-width around the nucleus centroid) for visual inspection of cells with `mask_fail_reason` in `multi_nuc`, `nuc_outside_cell`, or `pixel_dup`, the failure modes expected under cell overcrowding. Each crop re-loads the raw, unmodified mask from disk, shades distinct segmentation blobs in gray, and highlights the example cell's nucleus pixels in red.

---

## Day 0 and Cell Classification

Most drug conditions image starting exactly at day 0 (the frame the drug is added on is frame 1). Some conditions instead image for a while *before* the drug is added, e.g. a 48-hours-before-treatment baseline, where imaging starts 2 days early and the drug isn't added until the 3rd frame. `config.DAY0_FRAMES[drug]` tells `append_labels.py` which frame day 0 actually falls on for that condition.

Classification (`clone_kmeans_id.mat`'s `cloneClassification`) is assigned per **cell present at day 0**, not per lineage. A lineage can already have divided multiple times during any pre-treatment imaging: e.g. 3 cells at frame 1 can expand to 100 cells by day 0 at frame 3, and each of those 100 day-0 cells gets its *own* distinct label, independent of its siblings, even though all 100 share the same original lineage (`lin_id`). This matters: `cloneClassification` is **not** indexed by `lin_id` at all.

- `cloneClassification` has exactly one row per cell present at `frame == day0_frame`. Row `i` (0-indexed) is the day-0 cell with the `i`-th smallest `node_id`, in flat ascending order, not grouped by lineage.
- Each day-0 cell's label is propagated forward to every one of its own descendants, via the `node_id`/`pred_node_id` parent-child chain. A descendant of a *different* day-0 sibling, even one sharing the same `lin_id`, gets that sibling's own (different) label instead.
- A cell strictly before day 0 gets `NaN` for both `m_label` and `k_label`. Its fate isn't classified until day 0, and a shared pre-day0 ancestor can fan out into several differently-labeled day-0 descendants, so no single label could correctly apply to it anyway.
- A lineage that dies out before ever reaching day0_frame gets `NaN` throughout: no day-0 cell, no classification.

For any condition where `DAY0_FRAMES[drug] == 1` (the drug is added on the very first imaged frame), frame 1 is the very start of imaging, so every lineage has exactly one cell there (no divisions have happened yet). Every cell is either a day-0 cell itself or a descendant of one, so no row ever ends up `NaN`.

**Every key in `DRUG_DIRS` must have a matching entry in `DAY0_FRAMES`.** `main.py` asserts this up front and fails loudly if one's missing, since getting `day0_frame` wrong for a condition silently mis-pairs labels with cells rather than erroring in an obvious way.

---

## Blob Resolution Logic

During mask building, the following rules are applied in order to resolve segmentation conflicts:

1. **Multi-blob nucleus**: if a nucleus has more than one disconnected blob, zero out both the cell and nuclear mask for that pixel ID and set `has_mask=False`
2. **Nucleus outside cell**: if any nuclear pixels fall outside the cell boundary, zero out both masks and set `has_mask=False`
3. **Multi-blob cell**: if a cell has multiple blobs, keep the smallest blob that fully encloses all nuclear pixels. If no blob satisfies this, zero out both masks and set `has_mask=False`
4. **Duplicate pixel IDs** (`pixel_dup=True`): `has_mask=False` is set in `join_data.py`. During mask building, their cell pixels are excluded via `valid_ids` filtering.

Connected components are computed using 4-connectivity (no diagonal connections), meaning two pixels are considered part of the same blob only if they share an edge, not just a corner.

---

## Configuration

Edit `config.py`:

```python
DRUG_DIRS = {
    'naive':   '/path/to/naive/drug_dir',    # wherever this condition's raw data actually lives
    'vem_1uM': '/path/to/vem_1uM/drug_dir',
}

DAY0_FRAMES = {
    'naive':   1,   # drug added on frame 1, the common case
    'vem_1uM': 1,
    # a condition with a pre-treatment baseline (e.g. imaging starts 2 days
    # before the drug is added) would instead have its drug-addition frame
    # here, e.g. 3; see "Day 0 and Lineage Classification" above
}

PROCESSED_DATA_DIR = 'processed_data/' # path to output directory
N_JOBS              = 16                # number of parallel jobs (set to match allocated cores)
BUFFER              = 2                 # pixel buffer for nucleus coordinate lookup
IMG_SIZE            = 2501              # side length of each track image (pixels)
```

No frame cutoff here: every available raw frame is processed for every condition (frames beyond ~13 are known to get unreliable, heavy overcrowding from repeated division, but which frames are trustworthy enough to actually use is left to each downstream analysis's own config, not enforced here).

Raw data is read directly from each path in `DRUG_DIRS`; nothing is copied into this repo. Each path must follow the layout described in Directory Structure above. Every key in `DRUG_DIRS` must have a matching key in `DAY0_FRAMES`.

---

## Setup

Dependencies are defined in the repo's shared `environment.yml`. See the root README's "Setup" section to create and activate it before running anything here.

---

## Running

### Locally

```bash
conda activate drug_morpho
cd process_imaging_data
python main.py
```

### On Quest (HPC)

```bash
cd /{YOUR_PATH}/process_imaging_data
sbatch slurm_submit.sh
```

Logs are written to `logs/process_imaging_{job_id}.out`.

Drug conditions come from `config.DRUG_DIRS`. Each condition is processed independently and sequentially.

A drug is skipped if its `processed_data/{drug_name}_metadata.csv` already exists, rather than being reprocessed from scratch. Delete that file (there's no need to delete `combined_masks/` or `failure_examples/`, those get overwritten on a fresh run) to force that drug to be reprocessed, e.g. after changing `config.py` or the raw data itself.