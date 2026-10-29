# vem_48hr_pretreatment

For the `vem_1uM_48hrsbefore` dataset: frame 1 is day -2 (2 days *before* drug addition), frame 3 is day 0 (when 1µM vemurafenib was added). This analysis only ever looks at **frame 1** (`config.FRAMES = [1]`). `process_imaging_data` doesn't classify a cell until day 0, so a frame-1 cell has no single subclass to compare across frames; the question here is whether a cell's shape *before* treatment **predicts** what it eventually becomes.

1. **Uses every available frame-1 cell/nucleus**, no sampling budget, unlike `naive_vs_vem` (`load_all_cells.py`).
2. **Runs CAJAL** once per modality on frame-1 cells. That's it for the pipeline: no UMAP/PHATE embedding is fit, since the question below is answered directly from the real distance matrix, not a 2D projection that distorts it.
3. **Resolves each frame-1 lineage's eventual fate mixture** (`lineage_fate.py`): a frame-1 lineage can already have divided by day 0 (e.g. 3 cells at frame 1 expanding to 100 by day 0), and each of those day-0 descendants can land in a *different* subclass. So instead of one hard label, each frame-1 cell gets `%dying`/`%persisting`/`%slow_prolif`/`%fast_prolif`, resolved from `lineage_frame3_to_frame1.mat` and `clone_kmeans_id.mat` (raw, track-local lineage IDs resolved to this analysis's own `lin_id` by reusing `process_imaging_data`'s `load_track`/`make_master_ids` directly).
4. **`01_morphology_predicts_fate.ipynb`** asks: does pre-treatment shape predict that mixture? For each cell, it predicts the cell's own fate mixture from the mixtures of its *k* nearest neighbors in real morphological (CAJAL GW) distance, then tests whether that prediction beats chance by comparing its error against many random reassignments of mixture to cell (`shared/morphology_predicts_fate.py`'s `knn_prediction_permutation_test`). One figure per modality: predicted vs. actual mixture fractions next to the permutation null distribution.

---

## Configuration

`config.py` sets the dataset name and frame list, `PROCESS_IMAGING_DATA_ROOT` (where `process_imaging_data`'s processed output lives), `RAW_DRUG_DIR` and the two raw `.mat` paths `lineage_fate.py` needs (`LINEAGE_MAT_PATH`, `CLONE_KMEANS_MAT_PATH`), and the classification scheme (see the root README's "Classification scheme" section).

---

## Running

Follows the same pattern as every other analysis. See the root README's "Running an analysis" section:

```bash
cd analyses/02_vem_48hr_pretreatment
sbatch slurm_submit.sh
```
