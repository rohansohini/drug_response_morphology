# multi_drug_comparison

Compares cell and nuclear morphology across four conditions (`naive`, `vem_1uM`, `gem_2.5nM`, `tram_6nM`) at two timepoints: frame 1 and frame 13 (this analysis's own deliberate cutoff - `process_imaging_data` itself no longer caps frames, so `config.FRAMES` explicitly stops at 13, since frames beyond that are known to get unreliable, heavy overcrowding from repeated division). Both frames go through the SAME CAJAL run and the SAME UMAP/PHATE fit per modality - every condition and both frames are pooled together before embedding - so frame 1 and frame 13 positions are directly comparable, not two independently-fit spaces that just happen to look similar. Frame is simply a page/filter within that one shared space, the same way `naive_vs_vem` pages through frames within its own shared embedding.

1. **Samples up to `config.SAMPLE_SIZE_PER_CONDITION`** (default 1000) `has_mask=True` cells per (frame, condition) - 2 frames × 4 conditions × 1000 = up to 8000 cells per modality. A flat per-bucket size, not a budget split by weight like `naive_vs_vem`, so no water-filling is needed (`select_cells.py`).
2. **Runs CAJAL once per modality** (cell, nucleus) across every condition and both frames pooled together, then fits one UMAP and one PHATE embedding per modality on that single distance matrix - same structure as `naive_vs_vem`'s own `main.py`.
3. **`00_representative_shapes.ipynb`** k-means clusters the FULL shared embedding (every condition, both frames pooled together) into a few regions and shows each region's actual representative cell shape - a region's representative cell can come from either frame. Same pattern as `01_naive_vs_vem`'s and `02_vem_48hr_pretreatment`'s own `00_representative_shapes.ipynb`, run once per (modality, embedding method) - 4 plots total.
4. **`01_kde_by_condition.ipynb`** produces one KDE-contour-per-condition plot per frame - one combined PDF per (modality, embedding method) with a page per frame and fixed shared axis limits across both pages (`shared/density_plots.py`'s `export_kde_by_frame`, same as `naive_vs_vem`) - 2 modalities × 2 embedding methods × 2 frames = 8 plots total.

---

## Configuration

`config.py` sets `CONDITIONS`, `FRAMES`, `SAMPLE_SIZE_PER_CONDITION`, and the paths into `process_imaging_data`'s output (see the root README's "Classification scheme" section - not used here, since this analysis never classifies cells by subclass, only compares by condition).

---

## Running

Follows the same pattern as every other analysis. See the root README's "Running an analysis" section:

```bash
cd analyses/03_multi_drug_comparison
sbatch slurm_submit.sh
```
