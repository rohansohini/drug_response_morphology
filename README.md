# drug_response_morphology

Morphology analyses built on [CAJAL](https://github.com/CamaraLab/CAJAL) Gromov-Wasserstein shape distances and UMAP, reading cell/nuclear segmentation metadata produced by `process_imaging_data`.

This repo holds multiple, independent analyses, each self-contained under `analyses/`, sharing common building blocks (CAJAL wrappers, classification, data merging, plotting) from `shared/`.

---

## Analyses

### `naive_vs_vem`

Compares cell and nuclear morphology between `naive` and `A1_1uM_vem` conditions over time.

1. **Selects cells** for CAJAL from both conditions, every subclass (dying, persisting, slow proliferator, fast proliferator), restricted to frames present in *both* conditions and sampled with an 80/20 vem/naive budget split (`select_cells.py` / `allocation.py`).
2. **Runs CAJAL** once per modality (cell, nucleus) on the combined selection.
3. **Fits UMAP directly on the CAJAL distance matrix** (`metric='precomputed'`).
4. **Three notebooks** read that one shared embedding and plot different comparisons:
   - `01_umap_by_condition_over_time.ipynb` - every cell, every subclass, KDE per condition
   - `02_proliferators_vem_vs_naive.ipynb` - proliferators only (`m_label == 3`), KDE per condition
   - `03_vem_subclasses_over_time.ipynb` - vem only, KDE per subclass

### `vem_48hr_pretreatment`

For the `vem_1uM_48hrsbefore` dataset: frame 1 is day -2 (2 days *before* drug addition), frame 3 is day 0 (when 1µM vemurafenib was added). This analysis only ever looks at **frame 1** (`config.FRAMES = [1]`) - `process_imaging_data` doesn't classify a cell until day 0, so a frame-1 cell has no single subclass to compare across frames; the question here is whether a cell's shape *before* treatment relates to what it eventually becomes.

1. **Uses every available frame-1 cell/nucleus** - no sampling budget, unlike `naive_vs_vem` (`load_all_cells.py`).
2. **Runs CAJAL** once per modality on frame-1 cells, then fits UMAP directly on that distance matrix.
3. **Resolves each frame-1 lineage's eventual fate mixture** (`lineage_fate.py`): a frame-1 lineage can already have divided by day 0 (e.g. 3 cells at frame 1 expanding to 100 by day 0), and each of those day-0 descendants can land in a *different* subclass - so instead of one hard label, each frame-1 cell gets `%dying`/`%persisting`/`%slow_prolif`/`%fast_prolif`, resolved from `lineage_frame3_to_frame1.mat` and `clone_kmeans_id.mat` (raw, track-local lineage IDs resolved to this analysis's own `lin_id` by reusing `process_imaging_data`'s `load_track`/`make_master_ids` directly).
4. **`01_frame1_fate_mixture.ipynb`** plots that mixture as a weighted KDE per fate category over the frame-1 UMAP - every cell contributes to every category's contour in proportion to its own percentage in that category, rather than being forced into one dominant-fate bucket (`shared/density_plots.py`'s `plot_umap_with_weighted_kde`).

---

## Directory structure

```
drug_response_morphology/
├── process_imaging_data/              # nested standalone pipeline (own repo history, own environment.yml/tests) -
│                                       #   turns raw track tables into {drug}_metadata.csv + combined_masks/;
│                                       #   see process_imaging_data/README.md for its own structure
├── shared/                            # reusable across every analysis
│   ├── cajal/
│   │   ├── extract_masks.py           # crops selected cells' masks
│   │   ├── run_cajal.py               # CAJAL icdm + quantized GW distances
│   │   └── embed.py                   # loads the distance matrix; UMAP on it
│   ├── classify.py                    # m_label/k_label -> subclass (scheme passed in by each analysis's own config.py)
│   ├── load_data.py                   # merges UMAP coordinates / distance-matrix labels with metadata
│   └── density_plots.py               # shared KDE-over-UMAP plotting (hard-label and weighted), PDF export
├── analyses/
│   ├── naive_vs_vem/
│   │   ├── main.py                    # runs the pipeline end to end
│   │   ├── config.py                  # conditions, sampling budget, paths
│   │   ├── allocation.py              # water-filling budget allocation
│   │   ├── select_cells.py            # cell selection/sampling
│   │   ├── slurm_submit.sh
│   │   ├── notebooks/                 # the 3 notebooks (see above)
│   │   └── output/                    # generated (gitignored)
│   └── vem_48hr_pretreatment/
│       ├── main.py
│       ├── config.py                  # dataset name, frames, paths
│       ├── load_all_cells.py          # loads every available frame-1 cell (no classification - see lineage_fate.py)
│       ├── lineage_fate.py            # frame-1 lineage -> day-0 fate-mixture percentages
│       ├── slurm_submit.sh
│       ├── notebooks/
│       └── output/                    # generated (gitignored)
├── tests/
│   ├── shared/                        # tests for shared/ - no real data needed
│   └── analyses/
│       ├── naive_vs_vem/
│       └── vem_48hr_pretreatment/
└── environment.yml                    # shared across every analysis
```

Each analysis's `output/` follows the same shape:
```
output/
├── masks/, cajal_output/              # extracted masks, CAJAL distances, UMAP embeddings
└── plots/
    └── {notebook_name}/
        ├── pdfs/                      # one combined, multi-page PDF per modality, e.g. cell.pdf
        └── svgs/                      # one SVG per frame, e.g. cell_frame_01.svg
```

---

## Adding a new analysis

1. `mkdir -p analyses/<name>/notebooks`
2. Write `analyses/<name>/config.py` - at minimum the paths into `process_imaging_data`'s output (see either existing `config.py` as a template) and whatever's specific to this analysis (conditions/frames/sampling).
3. Write `analyses/<name>/main.py` - reuse `shared.cajal.extract_masks/run_cajal/embed` directly; only the cell-selection step (how you decide which cells go into CAJAL) is usually analysis-specific logic, worth its own small module next to `main.py` (see `naive_vs_vem/select_cells.py` for budget-based sampling, or `vem_48hr_pretreatment/load_all_cells.py` for "use everything").
4. Write notebook(s) under `analyses/<name>/notebooks/`, using `shared.load_data.load_full_dataset` and `shared.density_plots.export_kde_by_frame` (or `plot_umap_with_kde` directly for something custom) - copy an existing notebook's cell 1 path-setup verbatim, it's analysis-agnostic.
5. Add a `slurm_submit.sh` that `cd`s into `analyses/<name>/` before running `python main.py`.

Anything genuinely reusable across analyses (not just this one) belongs in `shared/`, not copied into the new analysis folder.

---

## Prerequisites

- `process_imaging_data/` (nested in this repo - see below) has already been run for whatever drug condition(s) this analysis needs, producing `{drug}_metadata.csv` and `combined_masks/` under its own `processed_data/`.
- Each analysis's `config.py`'s `PROCESS_IMAGING_DATA_ROOT` points at that directory's location (`/projects/b1042/GoyalLab/sohini/drug_response_morphology/process_imaging_data`).

---

## Setup

```bash
conda env create -f environment.yml
conda activate drug_response_morphology
```

If `mamba` is available (`mamba --version`), use it instead - it's a drop-in replacement that reads this same `environment.yml` but resolves dependencies much faster:

```bash
mamba env create -f environment.yml
conda activate drug_response_morphology
```

One environment is shared across every analysis (`naive_vs_vem` and `vem_48hr_pretreatment`).

`process_imaging_data/` is a separate, self-contained sub-project with its own `environment.yml` (MATLAB `.mat` parsing, segmentation mask building - a different dependency set than CAJAL/UMAP) - set it up separately per `process_imaging_data/README.md` before running anything there.

---

## Running an analysis

**Step 1 - the heavy part (cell selection/loading, mask extraction, CAJAL, UMAP):**

```bash
cd analyses/<name>
sbatch slurm_submit.sh
```

This runs that analysis's `main.py`, which is safe to re-submit if it times out or fails partway - CAJAL and the UMAP fit each skip recomputation if their output file already exists, so a re-run resumes rather than restarting from scratch.

To run it directly instead (e.g. for a quick local check):
```bash
cd analyses/<name>
python main.py
```

**Step 2 - the analysis notebooks:**

Once `main.py` has completed (so `output/cajal_output/selected_metadata.csv`, `cell_umap.npz`, and `nuc_umap.npz` all exist under that analysis's own `output/`), open the notebooks in `analyses/<name>/notebooks/` with Jupyter and run them top to bottom - no SLURM job needed, they just load the saved embedding and plot.

---

## Classification scheme

`shared/classify.py`'s `classify_subclass()` maps `m_label`/`k_label` (from `process_imaging_data`'s `cloneClassification`) to a subclass - but the actual numeric scheme is declared explicitly in each analysis's own `config.py` (`DYING_M_LABEL`, `PERSISTING_M_LABEL`, `PROLIF_M_LABEL`, `SLOW_K_LABELS`, `FAST_K_LABELS`), not hardcoded in `shared/`. Every analysis currently uses the same values:

| Subclass | Rule |
|---|---|
| `dying` | `m_label == 1` |
| `persisting` | `m_label == 2` |
| `slow_prolif` | `m_label == 3` and `k_label` in {1, 2} |
| `fast_prolif` | `m_label == 3` and `k_label` in {3, 4, 5, 6} |

What the upstream k-means clusters (`k_label` 1-6) actually represent, and why 1-2 means "slow" vs. 3-6 "fast", isn't derivable from either this repo or `process_imaging_data` - that's domain knowledge from whoever set up the clustering.

---

## Testing

The test suite needs no real data - everything is synthetic:

```bash
for f in $(find tests -name "test_*.py"); do python3 "$f"; done
```
