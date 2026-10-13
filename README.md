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

For the `vem_1uM_48hrsbefore` dataset: frames 1-2 are *before* drug addition (pre-treatment baseline), frame 3 is day 0 (when 1µM vemurafenib was added).

1. **Uses every available cell/nucleus** across frames 1-3 - no sampling budget, unlike `naive_vs_vem` (`load_all_cells.py`).
2. **Runs CAJAL** once per modality on all 3 frames together, so distances are comparable across them.
3. **Fits one shared UMAP** directly on that distance matrix.
4. **`01_subclass_by_frame.ipynb`** plots each frame on that shared embedding, colored by subclass via KDE contours - one combined PDF per modality (3 pages, one per frame), so 2 files / 6 plots total.
5. **`02_subclass_separation_stats.ipynb`** tests, per frame per modality, whether the 4 subclasses are statistically distinguishable - PERMANOVA + PERMDISP directly on CAJAL's GW distance matrix (never the UMAP projection, which distorts distances), plus pairwise subclass comparisons with Benjamini-Hochberg correction. See "Statistical testing" below.

---

## Directory structure

```
drug_response_morphology/
├── shared/                            # reusable across every analysis
│   ├── cajal/
│   │   ├── extract_masks.py           # crops selected cells' masks
│   │   ├── run_cajal.py               # CAJAL icdm + quantized GW distances
│   │   └── embed.py                   # loads the distance matrix; UMAP on it
│   ├── classify.py                    # m_label/k_label -> subclass (fixed scheme)
│   ├── load_data.py                   # merges UMAP coordinates / distance-matrix labels with metadata
│   ├── density_plots.py               # shared KDE-over-UMAP plotting, PDF export
│   └── permanova.py                   # PERMANOVA/PERMDISP on the distance matrix, BH correction
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
│       ├── load_all_cells.py          # loads + classifies every available cell
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

- `process_imaging_data` has already been run for whatever drug condition(s) this analysis needs, producing `{drug}_metadata.csv` and `combined_masks/` under its own `processed_data/`.
- Each analysis's `config.py`'s `PROCESS_IMAGING_DATA_ROOT` points at that repo's location (`/gpfs/projects/b1042/GoyalLab/sohini/process_imaging_data`).

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

One environment is shared across every analysis.

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

Fixed and universal across every analysis (`shared/classify.py`), from `process_imaging_data`'s `cloneClassification`:

| Subclass | Rule |
|---|---|
| `dying` | `m_label == 1` |
| `persisting` | `m_label == 2` |
| `slow_prolif` | `m_label == 3` and `k_label` in {1, 2} |
| `fast_prolif` | `m_label == 3` and `k_label` in {3, 4, 5, 6} |

---

## Statistical testing

`shared/permanova.py` (used by `vem_48hr_pretreatment/notebooks/02_subclass_separation_stats.ipynb`) tests whether a categorical grouping (e.g. subclass) is statistically distinguishable in morphology - operating directly on CAJAL's Gromov-Wasserstein distance matrix, **never** on a UMAP projection of it. UMAP optimizes a 2D layout for visual cluster separation, not for preserving actual distances, so it is the wrong representation for any quantitative claim, including a statistical test.

Two complementary, permutation-based tests are run together, because a significant result from one alone is ambiguous:

| Test | Question | Sensitive to |
|---|---|---|
| `permanova()` (Anderson 2001) | Do groups occupy different *locations* in distance space? | Location, but also dispersion |
| `permdisp()` (Anderson 2006) | Do groups differ in *dispersion* (spread)? | Dispersion only |

A PERMANOVA hit with no PERMDISP hit is a clean location effect. A hit on both is ambiguous - part or all of the PERMANOVA signal may be a dispersion difference rather than a location difference, so the notebook reports both outcomes side by side rather than reading PERMANOVA alone as evidence of separation.

`permanova_pairwise()` runs PERMANOVA on every pair of groups and applies `benjamini_hochberg()` FDR correction across the pairs (e.g. the 6 pairwise subclass comparisons within one frame) - use the resulting `q_value`, not the raw `p_value`, to judge pairwise significance.

All three report an effect size (`R2` for PERMANOVA, `mean_dispersion` per group for PERMDISP) alongside the permutation p-value, since p-values alone are close to meaningless at the cell counts these analyses work with (thousands of cells per frame).

---

## Testing

The test suite needs no real data - everything is synthetic:

```bash
for f in $(find tests -name "test_*.py"); do python3 "$f"; done
```
