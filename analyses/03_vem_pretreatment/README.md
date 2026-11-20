# vem_pretreatment

Same question, asked for two pre-treatment durations (`config.CONDITIONS`): `48hr` (`vem_1uM_48hrsbefore` - frame 1 is day -2, 2 days *before* drug addition; frame 3 is day 0) and `144hr` (`vem_1uM_144hrsbefore` - frame 1 is day -6; frame 7 is day 0). Both only ever look at **frame 1** (`config.FRAMES = [1]`). `process_imaging_data` doesn't classify a cell until day 0, so a frame-1 cell has no single subclass to compare across frames; the question here is whether a cell's shape *before* treatment **predicts** what it eventually becomes.

1. **Uses every available frame-1 cell/nucleus**, no sampling budget, unlike `naive_vs_vem` (`shared/load_all_cells.py`).
2. **Runs CAJAL** once per modality, once per condition, on frame-1 cells. That's it for the pipeline: no UMAP/PHATE embedding is fit, since the question below is answered directly from the real distance matrix, not a 2D projection that distorts it. `main.py` runs both conditions back to back in one job, each into its own `output/{condition}/` subtree - the two datasets are never pooled into a shared CAJAL run.
3. **Resolves each frame-1 lineage's eventual fate mixture** (`shared/lineage_fate.py`): a frame-1 lineage can already have divided by day 0 (e.g. 3 cells at frame 1 expanding to 100 by day 0), and each of those day-0 descendants can land in a *different* subclass. So instead of one hard label, each frame-1 cell gets `%dying`/`%persisting`/`%slow_prolif`/`%fast_prolif`, resolved from that condition's own `lineage_frame{N}_to_frame1.mat` and `clone_kmeans_id.mat` (raw, track-local lineage IDs resolved to this analysis's own `lin_id` by reusing `process_imaging_data`'s `load_track`/`make_master_ids` directly).
4. **`01_morphology_predicts_fate.ipynb`** asks: does pre-treatment shape predict that mixture? For each condition and each cell, it predicts the cell's own fate mixture from the mixtures of its *k* nearest neighbors in real morphological (CAJAL GW) distance, then tests whether that prediction beats chance by comparing its error against many random reassignments of mixture to cell (`shared/morphology_predicts_fate.py`'s `knn_prediction_permutation_test`). One figure per (condition, modality) - 4 total: the permutation null distribution with the real (observed) prediction error marked.

This analysis used to be two separate ones (`vem_48hr_pretreatment`, `vem_144hr_pretreatment`) - same code and question in both, differing only in `config.py`'s dataset name/paths - now merged into one, parametrized by `config.CONDITIONS`.

---

## Configuration

`config.py`'s `CONDITIONS` dict holds each condition's drug name and `day0_frame` (which frame is day 0 for that dataset); `raw_drug_dir()`/`lineage_mat_path()`/`clone_kmeans_mat_path()`/`metadata_csv_path()`/`output_dir()`/`masks_dir()`/`cajal_output_dir()` all derive a per-condition path from it. `PROCESS_IMAGING_DATA_ROOT` (where `process_imaging_data`'s processed output lives) and the classification scheme (see the root README's "Classification scheme" section) are shared across both conditions.

---

## Running

Follows the same pattern as every other analysis, except `main.py` itself loops over both conditions in sequence (so one `sbatch` submission runs both). See the root README's "Running an analysis" section:

```bash
cd analyses/03_vem_pretreatment
sbatch slurm_submit.sh
```
