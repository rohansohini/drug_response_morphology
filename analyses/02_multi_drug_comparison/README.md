# multi_drug_comparison

Compares cell and nuclear morphology across four conditions (`naive`, `vem_1uM`, `gem_2.5nM`, `tram_6nM`) across *every frame shared by all of them*, up to `config.MAX_FRAME` (13 - this analysis's own deliberate cutoff, since `process_imaging_data` itself no longer caps frames and frames beyond that are known to get unreliable, heavy overcrowding from repeated division). The four conditions don't necessarily have the same number of frames, so "shared" is computed as the true intersection, not assumed (`select_cells.py`). Every selected cell, whatever frame it's from, goes through the SAME CAJAL run and the SAME UMAP/PHATE fit per modality - every condition and every shared frame are pooled together before embedding - so every frame's positions are directly comparable, not independently-fit spaces that just happen to look similar. Frame is simply a page/filter within that one shared space, the same way `naive_vs_vem` pages through frames within its own shared embedding.

1. **Samples up to `config.SAMPLE_SIZE_PER_CONDITION`** (default 250) `has_mask=True` cells per (frame, condition) - e.g. 13 shared frames × 4 conditions × 250 = up to 13,000 cells per modality. A flat per-bucket size, not a budget split by weight like `naive_vs_vem`, so no water-filling is needed (`select_cells.py`).
2. **Runs CAJAL once per modality** (cell, nucleus) across every condition and every shared frame pooled together, then fits one UMAP and one PHATE embedding per modality on that single distance matrix - same structure as `naive_vs_vem`'s own `main.py`.
3. **`00_representative_shapes.ipynb`** k-means clusters the FULL shared embedding (every condition, every shared frame pooled together) into a few regions and shows each region's actual representative cell shape - a region's representative cell can come from any of the shared frames. Same pattern as `01_naive_vs_vem`'s and `03_vem_pretreatment`'s own `00_representative_shapes.ipynb`, run once per (modality, embedding method) - 8 plots total (an embedding-by-region scatter and a representative-shapes row, exported separately, for each of the 4 combinations).
4. **`01_kde_by_condition.ipynb`** produces one KDE-contour-per-condition plot per frame - one combined PDF per (modality, embedding method) with a page per shared frame and fixed shared axis limits across every page, plus an animated GIF stepping through those same pages at 2fps (`shared/density_plots.py`'s `export_kde_by_frame`, same as `naive_vs_vem`) - 2 modalities × 2 embedding methods × however many frames are actually shared.

---

## Configuration

`config.py` sets `CONDITIONS`, `MAX_FRAME`, `SAMPLE_SIZE_PER_CONDITION`, and the paths into `process_imaging_data`'s output (see the root README's "Classification scheme" section - not used here, since this analysis never classifies cells by subclass, only compares by condition).

---

## Running

Follows the same pattern as every other analysis. See the root README's "Running an analysis" section:

```bash
cd analyses/02_multi_drug_comparison
sbatch slurm_submit.sh
```
