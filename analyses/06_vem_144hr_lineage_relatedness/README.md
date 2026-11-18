# vem_144hr_lineage_relatedness

Same `vem_1uM_144hrsbefore` dataset as [`vem_144hr_pretreatment`](../05_vem_144hr_pretreatment/README.md), but a different question: is a cell's morphology more similar to its **sister**'s (born from the same division) than to a random cell's? What about its **cousins**' (its parent and its parent's sister's children - sharing a grandparent, not a parent)?

Frame 7 is day 0 (drug addition). Frames 1-`config.CONTROL_END_FRAME` (5, for now) are pre-treatment - effectively naive growth, used here as a **control**. Frames `config.TREATMENT_START_FRAME` (7, fixed by the data) through `config.TREATMENT_END_FRAME` (25, for now - pushed out to use as much of the available data as possible) are post-treatment, the **treatment** range. Frame 6 (day -1, the frame immediately before day 0) is currently missing from the raw data entirely, and falls outside both named ranges above; `select_cells.py`'s own sampling loop still visits it (it pools every frame up to `TREATMENT_END_FRAME`, not just the two named ranges) and skips it gracefully, same as it would any other empty frame. Both questions (sisters vs. random, cousins vs. random) are asked separately for each range, so a change under treatment can be compared against how things looked before it.

1. **Selects cells** with a flat per-frame subsample across frames 1-`config.TREATMENT_END_FRAME`, pooled into ONE CAJAL run - no subclass stratification (this analysis never classifies cells), and deliberately lineage-unaware: which true sister/cousin pairs end up with both members sampled is left to chance, not biased towards keeping more of them (`select_cells.py`).
2. **Runs CAJAL** once per modality (cell, nucleus) across every selected cell. No UMAP/PHATE embedding, same as the `_pretreatment` analyses: the question needs the real distances, not a 2D projection that distorts them.
3. **Reconstructs every lineage tree** directly from the FULL `{drug}_metadata.csv` (not just the CAJAL-sampled cells - see below) using its `node_id`/`pred_node_id`/`lin_id` columns, and finds every sister pair and cousin pair in it (`shared/lineage_tree.py`). This is intentionally **not** based on `lineage_frame7_to_frame1.mat` (that file only resolves each day-0 lineage's single frame-1 ancestor, collapsing every division in between - the generation-by-generation structure this analysis needs isn't in it).
4. **`01_relatedness.ipynb`**: for each modality and each range, restricts each sister/cousin pair to whichever of their own cells actually survived CAJAL subsampling (every surviving frame combination on each side counts as its own pair - sisters and cousins alike are not required to share a frame, since a cousin pair's two parents can divide at different times), builds ONE shared null distribution from random pairs drawn from every CAJAL-embedded cell in that range (any frame), and marks both the real sister-pair and real cousin-pair mean distance against it (`shared/pair_similarity.py`'s `pair_distance_permutation_test` + `plot_pair_similarity_test`). One figure per (modality, range) - up to 4 total.

---

## Why the full metadata, not just the sampled cells

`main.py` only sends a subsample of cells through CAJAL (mask extraction and the GW distance computation don't scale to every cell). But a sister/cousin *relationship* is a fact about the real lineage tree, independent of which cells happened to get sampled - resolving it from the subsample alone would silently miss (or misidentify) relationships that pass through unsampled intermediate rows. The notebook therefore reads `config.metadata_csv_path()` (the full, unsampled metadata) to build the tree, and only afterwards checks which of each relationship's own cells are also in `selected_metadata.csv`/the CAJAL output.

---

## Configuration

`config.py` sets `TREATMENT_START_FRAME` (fixed - frame 7, when the drug was actually added), `CONTROL_END_FRAME`/`TREATMENT_END_FRAME` (this analysis's own tunable range limits), `SAMPLE_SIZE_PER_FRAME`, and the pair-similarity permutation test's `N_PERMUTATIONS`/`TEST_SEED`.

---

## Running

Follows the same pattern as every other analysis. See the root README's "Running an analysis" section:

```bash
cd analyses/06_vem_144hr_lineage_relatedness
sbatch slurm_submit.sh
```
