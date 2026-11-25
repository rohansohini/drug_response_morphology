# drug_response_morphology

Morphology analyses built on [CAJAL](https://github.com/CamaraLab/CAJAL) Gromov-Wasserstein shape distances and UMAP, reading cell/nuclear segmentation metadata produced by `process_imaging_data`.

This repo holds multiple, independent analyses, each self-contained under `analyses/`, sharing common building blocks (CAJAL wrappers, classification, data merging, plotting) from `shared/`. `process_imaging_data/` is the pipeline every analysis depends on for its input data: it isn't itself one of the analyses, so like `shared/` it sits outside `analyses/`.

---

## [`process_imaging_data`](process_imaging_data/README.md)

Turns raw track/segmentation `.mat` data (under `raw_data/`) into the `{drug}_metadata.csv` and `combined_masks/` that every analysis below reads as its input.

---

## Analyses

Each analysis is self-contained under `analyses/<name>/`, with its own README covering what it does and how its pieces fit together.

### [`naive_vs_vem`](analyses/01_naive_vs_vem/README.md)

Compares cell and nuclear morphology between the `naive` and `vem_1uM` conditions over time, via a shared CAJAL embedding sampled across both conditions.

### [`multi_drug_comparison`](analyses/02_multi_drug_comparison/README.md)

Compares cell and nuclear morphology across four conditions (`naive`, `vem_1uM`, `gem_2.5nM`, `tram_6nM`) over every frame shared by all of them, sharing one CAJAL embedding per modality across every frame.

### [`vem_pretreatment`](analyses/03_vem_pretreatment/README.md)

Asks whether a cell's shape *before* vemurafenib treatment predicts what it eventually becomes, via a k-nearest-neighbor prediction test directly on CAJAL distances. Runs the same question for two pre-treatment durations (`vem_1uM_48hrsbefore`, `vem_1uM_144hrsbefore`).

### [`vem_144hr_lineage_relatedness`](analyses/04_vem_144hr_lineage_relatedness/README.md)

Asks whether sister cells, and separately cousin cells, are morphologically more alike than a random pair - compared before-drug vs. after-drug vemurafenib treatment, plus a wholly separate never-treated `naive` condition as a sanity-check control, using each condition's own full lineage tree (not just `vem`'s frame-1-to-day-0 mapping).

### [`vem_full_timecourse`](analyses/05_vem_full_timecourse/README.md)

Runs CAJAL on `vem_1uM` alone, across its entire available timecourse (not just the first 13 frames), to test whether its subclasses separate more clearly later than the frame range `naive_vs_vem` uses.

---

## Directory structure

```
drug_response_morphology/
├── raw_data/                          # gitignored: each drug condition's raw .mat/tiff data; see process_imaging_data/config.py's DRUG_DIRS
├── process_imaging_data/              # pipeline analyses/ depends on for its input data
│   ├── README.md                      # raw data format, pipeline steps, output columns, classification logic
│   ├── main.py
│   ├── config.py                      # drug conditions, raw data paths, day-0 frames
│   ├── pipeline/
│   ├── slurm_submit.sh
│   └── processed_data/                # generated (gitignored)
├── shared/                            # reusable across every analysis
│   ├── cajal/
│   │   ├── extract_masks.py           # crops selected cells' masks
│   │   ├── run_cajal.py               # CAJAL icdm + quantized GW distances
│   │   └── embed.py                   # loads the distance matrix; UMAP/PHATE on it
│   ├── classify.py                    # m_label/k_label -> subclass (scheme passed in by each analysis's own config.py)
│   ├── load_data.py                   # merges embedding coordinates / distance-matrix labels with metadata; exports a notebook's combined plot_data.csv
│   ├── density_plots.py               # shared KDE-over-embedding plotting (hard-label and weighted), PNG/SVG/GIF export
│   ├── representative_shapes.py       # k-means regions over an embedding + each region's real representative cell shape
│   ├── morphology_predicts_fate.py    # KNN prediction of a per-cell mixture from real (GW) morphological distance + permutation test
│   ├── lineage_fate.py                # frame-1 lineage -> day-0 fate-mixture percentages (from a lineage_frame{N}_to_frame1.mat mapping)
│   ├── load_all_cells.py              # loads every available cell within given frames (no classification, no sampling budget)
│   ├── lineage_tree.py                # reconstructs division trees from node_id/pred_node_id/lin_id; finds sister/cousin pairs
│   ├── pair_similarity.py             # shared-null permutation test + plot for "are these pairs more alike than random cells"
│   └── allocation.py                  # water-filling budget allocation
├── analyses/
│   ├── 01_naive_vs_vem/
│   │   ├── README.md
│   │   ├── main.py                    # runs the pipeline end to end
│   │   ├── config.py                  # conditions, sampling budget, paths
│   │   ├── select_cells.py            # cell selection/sampling
│   │   ├── slurm_submit.sh
│   │   ├── notebooks/                 # the 6 notebooks (see analyses/01_naive_vs_vem/README.md)
│   │   └── output/                    # generated (gitignored)
│   ├── 02_multi_drug_comparison/
│   │   ├── README.md
│   │   ├── main.py                    # one CAJAL run per modality, pooling every shared frame together
│   │   ├── config.py                  # conditions, frames, sample size per condition, paths
│   │   ├── select_cells.py            # flat per-(frame, condition) sampling, no budget-splitting
│   │   ├── slurm_submit.sh
│   │   ├── notebooks/
│   │   └── output/                    # generated (gitignored)
│   ├── 03_vem_pretreatment/
│   │   ├── README.md
│   │   ├── main.py                    # loops over both conditions in one job
│   │   ├── config.py                  # CONDITIONS (per-duration drug name, day0_frame), frames, paths
│   │   ├── slurm_submit.sh
│   │   ├── notebooks/
│   │   └── output/                    # generated (gitignored): output/{condition}/...
│   ├── 04_vem_144hr_lineage_relatedness/
│   │   ├── README.md
│   │   ├── main.py                    # two-condition (vem, naive) CAJAL run, no embedding
│   │   ├── config.py                  # CONDITIONS (vem/naive), before/after-drug frame ranges, sampling, paths
│   │   ├── select_cells.py            # flat per-frame sampling, no budget-splitting, no subclass
│   │   ├── slurm_submit.sh
│   │   ├── notebooks/
│   │   └── output/                    # generated (gitignored): output/{condition}/...
│   └── 05_vem_full_timecourse/
│       ├── README.md
│       ├── main.py                    # single-condition CAJAL run, every available frame
│       ├── config.py                  # drug (fixed to vem_1uM), sampling budget, paths
│       ├── select_cells.py            # single-condition water-filled sampling, no frame cap
│       ├── slurm_submit.sh
│       ├── notebooks/
│       └── output/                    # generated (gitignored)
├── tests/
│   ├── process_imaging_data/          # tests for process_imaging_data/
│   ├── shared/                        # tests for shared/, no real data needed
│   └── analyses/
│       ├── 01_naive_vs_vem/
│       ├── 02_multi_drug_comparison/
│       ├── 04_vem_144hr_lineage_relatedness/
│       └── 05_vem_full_timecourse/
└── environment.yml                    # shared across every analysis and process_imaging_data
```

Each analysis's `output/` follows the same shape:
```
output/
├── masks/, cajal_output/              # extracted masks, CAJAL distances, UMAP/PHATE embeddings
└── plots/
    └── {notebook_name}/
        ├── {modality}_{embedding}.png   # a single static plot - omitted for a per-frame (time-series) plot, which has no single combined image
        ├── {modality}_{embedding}.gif   # only for a per-frame (time-series) plot: every frame, animated at 2fps
        ├── svgs/                        # one SVG per frame (time-series), or one SVG per static plot, e.g. cell_umap_frame_01.svg / cell_umap.svg
        └── plot_data.csv                # the same data each plot above is drawn from
```

Every notebook also exports its own `plot_data.csv` (`shared.load_data.export_plot_data`), so someone without this repo can recreate or extend a plot from raw numbers alone. For a scatter/KDE notebook (including the representative-shapes ones, minus the actual shape crops - those don't fit a flat CSV), it's one row per cell: `modality`, `embedding_method`, `dim_0`/`dim_1`, plus whatever metadata and grouping column that notebook used (`condition`, `subclass`, `frame`, `region`, ...). The distribution-vs-null notebooks instead export one row per individual value: `vem_pretreatment` has `modality`, `group` (`'real'`/`'null'`), `error`, `p_value` (repeated on every row of its own modality+group, one `plot_data.csv` per condition); `vem_144hr_lineage_relatedness`'s `01_relatedness.ipynb` has `modality`, `group` (before-drug/after-drug/naive), `series` (`'sisters'`/`'cousins'`/`'null'`), `distance`, `p_value`, `effect_size`.

---

## Adding a new analysis

1. `mkdir -p analyses/<name>/notebooks`
2. Write `analyses/<name>/config.py`: at minimum the paths into `process_imaging_data`'s output (see either existing `config.py` as a template) and whatever's specific to this analysis (conditions, frames, sampling).
3. Write `analyses/<name>/main.py`, reusing `shared.cajal.extract_masks/run_cajal/embed` directly. Only the cell-selection step (how you decide which cells go into CAJAL) is usually analysis-specific logic, worth its own small module next to `main.py` (see `01_naive_vs_vem/select_cells.py` for budget-based sampling, or `shared/load_all_cells.py` for "use everything within some frames").
4. Write notebook(s) under `analyses/<name>/notebooks/`, using `shared.load_data.load_full_dataset` and `shared.density_plots.export_kde_by_frame` (or `plot_embedding_with_kde` directly for something custom). Copy an existing notebook's cell 1 path-setup verbatim, it's analysis-agnostic. Every existing notebook loops over both `'umap'` and `'phate'` and plots each with the embedding method as a filename suffix (see any existing notebook's cell 3): follow that same pattern rather than picking just one. End with `shared.load_data.export_plot_data` on the `{embedding_method: df}` dict(s) `process_modality()` already built, saved to `NOTEBOOK_PLOTS_DIR/plot_data.csv` - see any existing notebook's last cell.
5. Add a `slurm_submit.sh` that `cd`s into `analyses/<name>/` before running `python main.py`.
6. Add `analyses/<name>/README.md` describing what the analysis does and how its pieces fit together. See any existing analysis's README for the expected level of detail.

Anything genuinely reusable across analyses (not just this one) belongs in `shared/`, not copied into the new analysis folder.

---

## Prerequisites

- `process_imaging_data` has already been run for whatever drug condition(s) this analysis needs, producing `{drug}_metadata.csv` and `combined_masks/` under its own `processed_data/` (see [`process_imaging_data/README.md`](process_imaging_data/README.md)).
- Each analysis's `config.py`'s `PROCESS_IMAGING_DATA_ROOT` points at that directory's location (`/projects/b1042/GoyalLab/sohini/drug_response_morphology/process_imaging_data`).

---

## Setup

```bash
conda env create -f environment.yml
conda activate drug_morpho
```

If `mamba` is available (`mamba --version`), use it instead: a drop-in replacement that reads this same `environment.yml` but resolves dependencies much faster.

```bash
mamba env create -f environment.yml
conda activate drug_morpho
```

One environment (`drug_morpho`) is shared across every analysis and `process_imaging_data`.

### Using this environment in a Jupyter notebook on Quest

Creating the environment above doesn't make it selectable in Quest's OnDemand Jupyter Notebook/JupyterLab interactive sessions. It has to be registered as a kernel once, with the environment activated:

```bash
conda activate drug_morpho
python -m ipykernel install --user --name drug_morpho --display-name "Python (drug_morpho)"
```

After that, `Python (drug_morpho)` appears as a selectable kernel in any new Quest OnDemand Jupyter session. No need to re-register it unless the environment is recreated from scratch.

---

## Running an analysis

**Step 1: the heavy part (cell selection/loading, mask extraction, CAJAL, UMAP, PHATE)**

```bash
cd analyses/<name>
sbatch slurm_submit.sh
```

This runs that analysis's `main.py`, which is safe to re-submit if it times out or fails partway: mask extraction, CAJAL, and the UMAP/PHATE fits each skip recomputation of what's already on disk, so a re-run resumes rather than restarting from scratch.

To run it directly instead (e.g. for a quick local check):
```bash
cd analyses/<name>
python main.py
```

**Step 2: the analysis notebooks**

Once `main.py` has completed (so `output/cajal_output/selected_metadata.csv` and each modality's `_umap.npz`/`_phate.npz` embeddings all exist under that analysis's own `output/`), open the notebooks in `analyses/<name>/notebooks/` with Jupyter and run them top to bottom. No SLURM job needed, they just load the saved embedding and plot.

---

## Classification scheme

`shared/classify.py`'s `classify_subclass()` maps `m_label`/`k_label` (from `process_imaging_data`'s `cloneClassification`) to a subclass. The actual numeric scheme is declared explicitly in each analysis's own `config.py` (`DYING_M_LABEL`, `PERSISTING_M_LABEL`, `PROLIF_M_LABEL`, `SLOW_K_LABELS`, `FAST_K_LABELS`) rather than hardcoded in `shared/`. Every analysis currently uses the same values:

| Subclass | Rule |
|---|---|
| `dying` | `m_label == 1` |
| `persisting` | `m_label == 2` |
| `slow_prolif` | `m_label == 3` and `k_label` in {1, 2} |
| `fast_prolif` | `m_label == 3` and `k_label` in {3, 4, 5, 6} |

What the upstream k-means clusters (`k_label` 1-6) actually represent, and why 1-2 means "slow" vs. 3-6 "fast", isn't derivable from either this repo or `process_imaging_data`. That's domain knowledge from whoever set up the clustering.

---

## Testing

The test suite needs no real data: everything is synthetic.

```bash
for f in $(find tests -name "test_*.py"); do python3 "$f"; done
```
