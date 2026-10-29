# naive_vs_vem

Compares cell and nuclear morphology between the `naive` and `A1_1uM_vem` conditions over time.

1. **Selects cells** for CAJAL from both conditions, every subclass (dying, persisting, slow proliferator, fast proliferator), restricted to frames present in *both* conditions and sampled with an 80/20 vem/naive budget split (`select_cells.py` / `allocation.py`).
2. **Runs CAJAL** once per modality (cell, nucleus) on the combined selection.
3. **Fits UMAP and PHATE directly on the CAJAL distance matrix** (`metric='precomputed'`/`knn_dist='precomputed_distance'`).
4. **Three notebooks** read those shared embeddings and plot different comparisons:
   - `01_umap_by_condition_over_time.ipynb` - every cell, every subclass, KDE per condition
   - `02_proliferators_vem_vs_naive.ipynb` - proliferators only (`m_label == 3`), KDE per condition
   - `03_vem_subclasses_over_time.ipynb` - vem only, KDE per subclass

---

## Configuration

`config.py` sets the two condition names (`NAIVE_CONDITION`, `VEM_CONDITION` - must exactly match `process_imaging_data`'s own condition naming), the total CAJAL sampling budget and its per-condition weights, the classification scheme (see the root README's "Classification scheme" section), and `PROCESS_IMAGING_DATA_ROOT`.

---

## Running

Follows the same pattern as every other analysis - see the root README's "Running an analysis" section:

```bash
cd analyses/01_naive_vs_vem
sbatch slurm_submit.sh
```
