# multi_drug_comparison

Compares cell and nuclear morphology across four conditions (`naive`, `vem_1uM`, `gem_2.5nM`, `tram_6nM`) at two independent timepoints: frame 1 and `config.END_FRAME` (default 13 - `process_imaging_data`'s own `MAX_FRAME` cutoff, since frames beyond that are unreliable). Each frame gets its own, completely separate CAJAL run: frame 1's embedding space has no relationship to `config.END_FRAME`'s, since this analysis only ever asks how the four conditions differ from each other *within* a frame, never how a frame differs from another frame.

1. **Samples up to `config.SAMPLE_SIZE_PER_CONDITION`** (default 1000) `has_mask=True` cells per (frame, condition) - a flat per-bucket size, not a budget split by weight like `naive_vs_vem`, so no water-filling is needed (`select_cells.py`).
2. **Runs CAJAL 4 times total**: once per (frame, modality) - frame 1 cell, frame 1 nucleus, `END_FRAME` cell, `END_FRAME` nucleus - each in its own `output/masks/frame_NN/` and `output/cajal_output/frame_NN/` subfolder (`config.frame_masks_dir`/`frame_cajal_dir`), then fits UMAP and PHATE on each.
3. **`01_kde_by_condition.ipynb`** produces one KDE-contour-per-condition plot for every (frame, modality, embedding method) combination - 2 frames × 2 modalities × 2 embedding methods = 8 plots total, each a standalone figure (no paging across frames, since there's no shared embedding to page through).

---

## Configuration

`config.py` sets `CONDITIONS`, `START_FRAME`/`END_FRAME`, `SAMPLE_SIZE_PER_CONDITION`, and the paths into `process_imaging_data`'s output (see the root README's "Classification scheme" section - not used here, since this analysis never classifies cells by subclass, only compares by condition).

---

## Running

Follows the same pattern as every other analysis. See the root README's "Running an analysis" section:

```bash
cd analyses/03_multi_drug_comparison
sbatch slurm_submit.sh
```
