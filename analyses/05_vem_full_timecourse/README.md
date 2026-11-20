# vem_full_timecourse

Runs CAJAL on `vem_1uM` alone, across its *entire* available timecourse - not just the first 13 frames `naive_vs_vem`'s own analyses use. The hypothesis: `naive_vs_vem`'s `03_vem_subclasses_over_time.ipynb` doesn't show clear separation between vem's subclasses (dying/persisting/slow proliferator/fast proliferator) because it's only looking at the earliest frames - maybe the separation shows up later.

Restricted to `vem_1uM` only, deliberately: running CAJAL on `naive` across its own full timecourse too would mean far more sampled cells (naive keeps dividing), and CAJAL's pairwise distance computation gets too slow at that scale. One condition keeps the cell count - and so the CAJAL run - tractable even across every available frame.

This is its own, separate embedding space from `naive_vs_vem`'s - not comparable to it, since that one also includes `naive` and is capped at frame 13.

1. **Samples cells** for CAJAL from `vem_1uM`, every subclass, **every available frame** (no frame cap - `process_imaging_data` itself no longer caps frames either, see its own README), water-filled across `(frame, subclass)` buckets (`select_cells.py` / `shared/allocation.py`). Simpler than `naive_vs_vem`'s own selection: a single condition, so there's no cross-condition budget split.
2. **Runs CAJAL** once per modality (cell, nucleus) on the selection.
3. **Fits UMAP and PHATE directly on the CAJAL distance matrix.**
4. **Three notebooks** read those shared embeddings:
   - `00_representative_shapes.ipynb`: k-means clusters the full embedding into a few regions and shows each region's actual representative cell shape
   - `01_clonal_dynamics.ipynb`: for persisting, slow proliferator, and fast proliferator cells (dying excluded), finds the 2 clones (lineages, by `lin_id`) whose own cells are most TIGHTLY CLUSTERED morphologically - lowest mean pairwise CAJAL GW distance among themselves, on the real distance matrix, not the UMAP/PHATE projection - among clones with enough cells in the embedded sample to make that mean meaningful, and paints each one's own cells onto the embedding in its own color (`shared/clonal_dynamics.py`, same layout as Figure 1g of Shaffer et al., "Diverse clonal fates emerge upon drug treatment of homogeneous cancer cells"). Separately crops each clone's own MEDOID cell's actual shape - the cell with the lowest mean real distance to the rest of its own clone, picked the same real-distance way as the clone itself, not via the 2D projection - into its own exported row, same "embedding and shapes as separate files" pattern `00_representative_shapes.ipynb` uses. This is the right analysis for this question specifically because of point 1 above: `vem_1uM`'s own dedicated, heavier-sampled CAJAL run (not diluted across a shared naive+vem budget like `naive_vs_vem`) gives any given clone a much better chance of having enough of its own cells actually embedded. Clone selection (the expensive part - it scans every candidate lineage's own distance submatrix) runs once per modality and is reused for both UMAP and PHATE, since it never depends on which embedding method is being plotted.
   - `02_subclasses_over_time.ipynb`: every cell, every subclass, KDE per subclass at each frame - the same thing as `naive_vs_vem`'s `03_vem_subclasses_over_time.ipynb`, just across the full timecourse instead of frames 1-13

---

## Configuration

`config.py` sets `DRUG` (fixed to `'vem_1uM'`), the CAJAL sampling budget, the classification scheme (see the root README's "Classification scheme" section), and `PROCESS_IMAGING_DATA_ROOT`. No frame-range constant - every available frame is used.

---

## Running

Follows the same pattern as every other analysis. See the root README's "Running an analysis" section:

```bash
cd analyses/05_vem_full_timecourse
sbatch slurm_submit.sh
```
