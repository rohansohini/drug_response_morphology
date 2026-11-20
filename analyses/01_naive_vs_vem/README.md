# naive_vs_vem

Compares cell and nuclear morphology between the `naive` and `vem_1uM` conditions over time.

1. **Selects cells** for CAJAL from both conditions, every subclass (dying, persisting, slow proliferator, fast proliferator), restricted to frames present in *both* conditions and <= `config.MAX_FRAME` (13 - `process_imaging_data` itself no longer caps frames, so this analysis caps its own range explicitly rather than silently absorbing whatever else becomes available upstream), sampled with an 80/20 vem/naive budget split (`select_cells.py` / `shared/allocation.py`).
2. **Runs CAJAL** once per modality (cell, nucleus) on the combined selection.
3. **Fits UMAP and PHATE directly on the CAJAL distance matrix** (`metric='precomputed'`/`knn_dist='precomputed_distance'`).
4. **Six notebooks** read those shared embeddings:
   - `00_representative_shapes.ipynb`: k-means clusters the full embedding into a few regions and shows each region's actual representative cell shape
   - `01_umap_by_condition_over_time.ipynb`: every cell, every subclass, KDE per condition
   - `02_proliferators_vem_vs_naive.ipynb`: proliferators only (`m_label == 3`), KDE per condition
   - `03_vem_subclasses_over_time.ipynb`: vem only, KDE per subclass
   - `04_prolif_vs_persisters.ipynb`: three-way comparison (naive proliferators / vem proliferators / vem persisters), KDE per group, paged by frame like the others
   - `05_naive_early_vs_vem_late.ipynb`: naive pooled across an early frame window vs. vem pooled across a late frame window (both windows are notebook-level constants), one single plot - not paged by frame, since each group already spans several frames

---

## Configuration

`config.py` sets the two condition names (`NAIVE_CONDITION`, `VEM_CONDITION`, which must exactly match `process_imaging_data`'s own condition naming), the total CAJAL sampling budget and its per-condition weights, `MAX_FRAME` (this analysis's own frame cutoff), the classification scheme (see the root README's "Classification scheme" section), and `PROCESS_IMAGING_DATA_ROOT`.

---

## Running

Follows the same pattern as every other analysis. See the root README's "Running an analysis" section:

```bash
cd analyses/01_naive_vs_vem
sbatch slurm_submit.sh
```
