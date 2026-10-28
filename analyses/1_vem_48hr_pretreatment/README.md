# vem_48hr_pretreatment

For the `vem_1uM_48hrsbefore` dataset: frame 1 is day -2 (2 days *before* drug addition), frame 3 is day 0 (when 1µM vemurafenib was added). This analysis only ever looks at **frame 1** (`config.FRAMES = [1]`) - `process_imaging_data` doesn't classify a cell until day 0, so a frame-1 cell has no single subclass to compare across frames; the question here is whether a cell's shape *before* treatment relates to what it eventually becomes.

1. **Uses every available frame-1 cell/nucleus** - no sampling budget, unlike `naive_vs_vem` (`load_all_cells.py`).
2. **Runs CAJAL** once per modality on frame-1 cells, then fits UMAP directly on that distance matrix.
3. **Resolves each frame-1 lineage's eventual fate mixture** (`lineage_fate.py`): a frame-1 lineage can already have divided by day 0 (e.g. 3 cells at frame 1 expanding to 100 by day 0), and each of those day-0 descendants can land in a *different* subclass - so instead of one hard label, each frame-1 cell gets `%dying`/`%persisting`/`%slow_prolif`/`%fast_prolif`, resolved from `lineage_frame3_to_frame1.mat` and `clone_kmeans_id.mat` (raw, track-local lineage IDs resolved to this analysis's own `lin_id` by reusing `process_imaging_data`'s `load_track`/`make_master_ids` directly).
4. **`01_frame1_fate_mixture.ipynb`** plots that mixture as a weighted KDE per fate category over the frame-1 embedding (both UMAP and PHATE) - every cell contributes to every category's contour in proportion to its own percentage in that category, rather than being forced into one dominant-fate bucket (`shared/density_plots.py`'s `plot_embedding_with_weighted_kde`).

---

## Configuration

`config.py` sets the dataset name and frame list, `PROCESS_IMAGING_DATA_ROOT` (where `process_imaging_data`'s processed output lives), `RAW_DRUG_DIR` and the two raw `.mat` paths `lineage_fate.py` needs (`LINEAGE_MAT_PATH`, `CLONE_KMEANS_MAT_PATH`), and the classification scheme (see the root README's "Classification scheme" section).

---

## Running

Follows the same pattern as every other analysis - see the root README's "Running an analysis" section:

```bash
cd analyses/1_vem_48hr_pretreatment
sbatch slurm_submit.sh
```
