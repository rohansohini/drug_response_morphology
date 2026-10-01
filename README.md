# drug_response_morphology

Compares cell and nuclear morphology between `naive` and `A1_1uM_vem` conditions over time, using [CAJAL](https://github.com/CamaraLab/CAJAL) Gromov-Wasserstein shape distances and UMAP (no Isomap step).

---

## What this does

1. **Selects cells** for CAJAL from `process_imaging_data`'s metadata, across both conditions and every subclass (dying, persisting, slow proliferator, fast proliferator), restricted to frames present in *both* conditions and sampled with an 80/20 vem/naive budget split.
2. **Runs CAJAL** once per modality (cell, nucleus) on the combined selection, so every selected cell's shape distance is comparable to every other's.
3. **Fits UMAP directly on the CAJAL distance matrix** (`metric='precomputed'`) - deliberately no Isomap step.
4. **Three notebooks** read that one shared embedding and plot different comparisons:
   - `01_umap_by_condition_over_time.ipynb` - every cell, every subclass, KDE per condition
   - `02_proliferators_vem_vs_naive.ipynb` - proliferators only (`m_label == 3`), KDE per condition
   - `03_vem_subclasses_over_time.ipynb` - vem only, KDE per subclass

---

## Directory structure

```
drug_response_morphology/
├── main.py                      # runs steps 1-3 above, end to end
├── slurm_submit.sh              # SLURM job for main.py
├── environment.yml
├── config.py                    # conditions, classification scheme, sampling budget, paths
├── pipeline/
│   ├── allocation.py            # water-filling budget allocation
│   └── select_cells.py          # cell selection (step 1)
├── cajal_pipeline/
│   ├── extract_masks.py         # crops selected cells' masks
│   ├── run_cajal.py             # CAJAL icdm + quantized GW distances (step 2)
│   └── embed.py                 # UMAP on precomputed distances (step 3)
├── helpers/
│   ├── classify.py              # m_label/k_label -> subclass
│   ├── load_data.py             # merges UMAP coordinates with metadata
│   └── density_plots.py         # shared KDE-over-UMAP plotting
├── analysis/                    # the three notebooks (step 4)
├── tests/                       # unit tests - no real data needed, run anywhere
├── output/                      # generated - masks, CAJAL distances, embeddings, plots (gitignored)
└── logs/                        # generated - SLURM job logs (gitignored)
```

---

## Prerequisites

- `process_imaging_data` has already been run for both the `naive` and `A1_1uM_vem` conditions, producing `{condition}_metadata.csv` and `combined_masks/` under its own `processed_data/`.
- `config.py`'s `PROCESS_IMAGING_DATA_ROOT` points at that repo's location (`/gpfs/projects/b1042/GoyalLab/sohini/process_imaging_data`).

---

## Setup

```bash
conda env create -f environment.yml
conda activate drug_response_morphology
```

---

## Running the pipeline

**Step 1 - the heavy part (cell selection, mask extraction, CAJAL, UMAP):**

```bash
sbatch slurm_submit.sh
```

This runs `main.py`, which is safe to re-submit if it times out or fails partway - CAJAL and the UMAP fit each skip recomputation if their output file already exists, so a re-run resumes rather than restarting from scratch. The `--time`/`--mem` values in `slurm_submit.sh` are rough estimates with no empirical benchmark behind them yet at this repo's scale (up to 30,000 cells, two conditions, two modalities) - check the job log after the first run and adjust before relying on them.

To run it directly instead (e.g. for a quick local check on a small budget):
```bash
python main.py
```

**Step 2 - the analysis notebooks:**

Once `main.py` has completed (so `output/cajal_output/selected_metadata.csv`, `cell_umap.npz`, and `nuc_umap.npz` all exist), open the notebooks in `analysis/` with Jupyter and run them top to bottom - no SLURM job needed, they just load the saved embedding and plot. Each one writes its PDF output to `output/plots/`.

---

## Configuration

Everything you're likely to want to change lives in `config.py`:

| Setting | Meaning |
|---|---|
| `CONDITIONS`, `VEM_CONDITION` | drug condition names - must match `process_imaging_data`'s folder/metadata naming exactly |
| `TOTAL_CELL_BUDGET` | total cells sampled for CAJAL across both conditions (default 30,000) |
| `CONDITION_WEIGHTS` | how the budget splits across conditions (default 80% vem / 20% naive) |
| `DYING_M_LABEL` / `PERSISTING_M_LABEL` / `PROLIF_M_LABEL`, `SLOW_K_LABELS` / `FAST_K_LABELS` | the classification scheme (see below) |
| `N_JOBS`, `N_POINTS`, `N_CLUSTERS`, `PAD_SIZE` | mask extraction / CAJAL parameters |

**Classification scheme:**

| Subclass | Rule |
|---|---|
| `dying` | `m_label == 1` |
| `persisting` | `m_label == 2` |
| `slow_prolif` | `m_label == 3` and `k_label` in {1, 2} |
| `fast_prolif` | `m_label == 3` and `k_label` in {3, 4, 5, 6} |

**Sampling:** `select_cells.py` first splits `TOTAL_CELL_BUDGET` across conditions by `CONDITION_WEIGHTS`, then water-fills each condition's share evenly across its own `(frame, subclass)` buckets - a bucket with fewer real cells than an equal share simply contributes everything it has, and the leftover redistributes to the rest, rather than demanding cells that don't exist. See `pipeline/allocation.py`.

---

## Testing

The test suite needs no real data - everything is synthetic:

```bash
for f in tests/test_*.py; do python3 "$f"; done
```
