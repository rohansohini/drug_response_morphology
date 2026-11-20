# vem_144hr_lineage_relatedness

Same `vem_1uM_144hrsbefore` dataset as the `144hr` condition of [`vem_pretreatment`](../03_vem_pretreatment/README.md), but a different question: is a cell's morphology more similar to its **sister**'s (born from the same division) than to a random cell's? What about its **cousins**' (its parent and its parent's sister's children - sharing a grandparent, not a parent)?

Frame 7 is day 0 (drug addition). Frames 1-`config.BEFORE_DRUG_END_FRAME` (5, for now) are **before-drug** - effectively naive growth. Frames `config.AFTER_DRUG_START_FRAME` (7, fixed by the data) through `config.AFTER_DRUG_END_FRAME` (11, for now) are **after-drug**. Frame 6 (day -1, the frame immediately before day 0) is currently missing from the raw data entirely, and falls outside both named ranges above; `select_cells.py`'s own sampling loop still visits it (it pools every frame up to `AFTER_DRUG_END_FRAME`, not just the two named ranges) and skips it gracefully, same as it would any other empty frame. A third, wholly separate **naive** condition (`config.CONDITIONS['naive']`) is included as a sanity-check control, sampled and CAJAL'd entirely on its own over the same number of frames as the before-drug window (1-`config.BEFORE_DRUG_END_FRAME`) - it should look like before-drug, since neither has seen a drug yet. All three questions (sisters vs. random, cousins vs. random) are asked separately for each of the three groups, so a change after-drug can be compared against both before-drug and a cell line that was never treated at all.

1. **Selects cells** with a flat per-frame subsample, pooled into ONE CAJAL run per condition (`vem` across frames 1-`config.AFTER_DRUG_END_FRAME`; `naive` across frames 1-`config.BEFORE_DRUG_END_FRAME`, its own separate run) - no subclass stratification (this analysis never classifies cells), and deliberately lineage-unaware: which true sister/cousin pairs end up with both members sampled is left to chance, not biased towards keeping more of them (`select_cells.py`).
2. **Runs CAJAL** once per modality (cell, nucleus) per condition. No UMAP/PHATE embedding, same as the `_pretreatment` analyses: the question needs the real distances, not a 2D projection that distorts them.
3. **Reconstructs each condition's own lineage tree** directly from its FULL `{drug}_metadata.csv` (not just the CAJAL-sampled cells - see below) using its `node_id`/`pred_node_id`/`lin_id` columns, and finds every sister pair and cousin pair in it (`shared/lineage_tree.py`). This is intentionally **not** based on `lineage_frame7_to_frame1.mat` (that file only resolves each day-0 lineage's single frame-1 ancestor, collapsing every division in between - the generation-by-generation structure this analysis needs isn't in it; `naive` doesn't even have such a file, having no day-0 drug-addition event to anchor one).
4. **`01_relatedness.ipynb`**: for each modality and each group (before-drug, after-drug, naive), restricts each sister/cousin pair to whichever of their own cells actually survived CAJAL subsampling (every surviving frame combination on each side counts as its own pair - sisters and cousins alike are not required to share a frame, since a cousin pair's two parents can divide at different times), builds ONE shared null distribution of *individual* random-pair distances drawn from every CAJAL-embedded cell in that group (any frame), and compares the real sister-pair and real cousin-pair distance *distributions* against it via a one-sided Mann-Whitney U test - not a single collapsed mean against a null of means, which would hide whether the effect holds for most real pairs or is driven by just a few (`shared/pair_similarity.py`'s `pair_distance_permutation_test` + `plot_pair_similarity_test`). One figure per (modality, group) - up to 6 total.
5. **`02_relatedness_decay.ipynb`** (prototype): `01` found both sisters (degree 1: common parent) and cousins (degree 2: common grandparent) significantly more similar than random, in both modalities and all three groups - this asks how much further that holds. `shared/lineage_tree.py`'s `find_relative_pairs(segments, degree)` generalizes `find_sister_pairs`/`find_cousin_pairs` to any degree (3 = second cousins/common great-grandparent, and so on); this notebook sweeps `degree = 1, 2, 3, ...` (stopping a (modality, group) line as soon as it runs out of surviving pairs, and the whole sweep once neither condition's tree has divided that many times yet) and plots `-log10(p-value)` vs. `degree` - one figure per modality, 3 lines each (before-drug, after-drug, naive) - with a `p = 0.05` reference line, so where each line crosses it is visible directly. The printed `n_pairs` per point matters as much as the plot itself: a line that stops because it ran out of data reads very differently from one that keeps going but crosses above `p = 0.05`.

---

## Why the full metadata, not just the sampled cells

`main.py` only sends a subsample of cells through CAJAL (mask extraction and the GW distance computation don't scale to every cell). But a sister/cousin *relationship* is a fact about the real lineage tree, independent of which cells happened to get sampled - resolving it from the subsample alone would silently miss (or misidentify) relationships that pass through unsampled intermediate rows. Each notebook therefore reads `config.metadata_csv_path(condition)` (that condition's full, unsampled metadata) to build its tree, and only afterwards checks which of each relationship's own cells are also in that condition's own `selected_metadata.csv`/CAJAL output.

---

## Configuration

`config.py`'s `CONDITIONS` dict holds each condition's drug name and the single highest frame it needs sampled/CAJAL'd (`vem`'s own run covers both its before- and after-drug ranges at once; `naive` only ever needs its own comparison window). `AFTER_DRUG_START_FRAME` (fixed - frame 7, when the drug was actually added), `BEFORE_DRUG_END_FRAME`/`AFTER_DRUG_END_FRAME` (this analysis's own tunable range limits - `BEFORE_DRUG_END_FRAME` also sets naive's comparison window length), `SAMPLE_SIZE_PER_FRAME`, and the pair-similarity test's `N_NULL_SAMPLES`/`TEST_SEED`. `output_dir()`/`masks_dir()`/`cajal_output_dir()`/`metadata_csv_path()` all take a `condition` argument, since `vem` and `naive` are processed and stored entirely separately.

---

## Running

Follows the same pattern as every other analysis. See the root README's "Running an analysis" section:

```bash
cd analyses/04_vem_144hr_lineage_relatedness
sbatch slurm_submit.sh
```
