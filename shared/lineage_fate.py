import os
import sys

import numpy as np
import pandas as pd
from scipy.io import loadmat

# process_imaging_data/ is part of this same repo, at a fixed location
# relative to this file (shared/ and process_imaging_data/ are both
# direct children of the repo root) - derived from __file__ rather than
# depending on whichever analysis's own config.py happens to already be
# on sys.path, since this module is shared across analyses, not owned by
# any one of them.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'process_imaging_data'))
from pipeline.discover_raw_data import discover_tracks
from pipeline.track_table import load_track, make_master_ids

from shared.classify import classify_subclass

SUBCLASSES = ['dying', 'persisting', 'slow_prolif', 'fast_prolif']


def build_raw_to_global_lin_id(drug_dir: str) -> dict[tuple[int, int], int]:
    '''
    Map (track, raw per-track lin_id) -> global lin_id.

    lineage_frame3_to_frame1.mat's old_clone_id_frame1 is a raw,
    track-local lineage ID (scoped by file_idx, i.e. the Cropped_XX track
    number) - the same raw numbering track_table_*.mat's column 9 holds,
    before process_imaging_data's make_master_ids() applies its per-track
    offset to make it globally unique. {drug}_metadata.csv only has the
    GLOBAL lin_id, so resolving old_clone_id_frame1 against it requires
    this raw -> global mapping.

    Reuses process_imaging_data's own load_track/make_master_ids directly
    (not a reimplementation) so this is guaranteed to produce the exact
    same global lin_id numbering already baked into {drug}_metadata.csv,
    rather than risking a second, subtly different implementation of the
    same offsetting logic.

    Inputs:
        drug_dir: this condition's raw data root - same value as
                 process_imaging_data's config.DRUG_DIRS['vem_48hrsbefore']
                 (see config.RAW_DRUG_DIR)
    Outputs:
        {(track, raw_lin_id): global_lin_id}
    '''
    track_files, _ = discover_tracks(drug_dir)

    raw_df = pd.concat(
        [load_track(fp, track) for track, fp in track_files.items()],
        ignore_index=True,
    )
    global_df = make_master_ids(raw_df.copy())

    # raw_df and global_df are row-for-row aligned - make_master_ids only
    # reassigns ID values in place, it never reorders or drops rows - so
    # zipping them positionally gives the raw -> global correspondence
    # directly. Multiple rows share the same (track, raw lin_id) (every
    # frame of that lineage), which is fine: they all map to the same
    # global lin_id, so the dict collapses them consistently.
    return dict(zip(zip(raw_df['track'], raw_df['lin_id']), global_df['lin_id']))


def compute_frame1_fate_percentages(lineage_mat_path: str, clone_kmeans_mat_path: str,
                                     raw_to_global: dict[tuple[int, int], int],
                                     dying_m_label: int, persisting_m_label: int, prolif_m_label: int,
                                     slow_k_labels: list[int], fast_k_labels: list[int]) -> pd.DataFrame:
    '''
    For each frame-1 lineage (identified by its global lin_id), compute
    the fraction of its day-0 descendants in each of the 4 subclasses:
    %dying, %persisting, %slow_prolif, %fast_prolif. A lineage with only
    one day-0 descendant gets a "percentage" of 100% in that one
    descendant's subclass and 0% in the other three - percentages are
    not restricted to lineages that actually split.

    Inputs:
        lineage_mat_path:      path to this dataset's own
                               lineage_frame{N}_to_frame1.mat
                               (var.lineage_table: file_idx,
                               new_clone_id_global, old_clone_id_frame1) -
                               any row with a NaN in it is dropped before
                               use (see below)
        clone_kmeans_mat_path: path to clone_kmeans_id.mat
                               (cloneClassification - new_clone_id_global
                               is the 1-indexed row number into it)
        raw_to_global:         from build_raw_to_global_lin_id - resolves
                               (file_idx, old_clone_id_frame1) to a global lin_id
        dying_m_label, persisting_m_label, prolif_m_label,
        slow_k_labels, fast_k_labels: this dataset's own classification
                               scheme - see shared.classify.classify_subclass
    Outputs:
        df: one row per frame-1 lineage (global lin_id) that has at least
            one day-0 descendant, columns [lin_id, n_day0_descendants,
            pct_dying, pct_persisting, pct_slow_prolif, pct_fast_prolif]
    '''
    lineage_table = loadmat(lineage_mat_path)['lineage_table']

    # some raw data now has lineage_table rows with no valid mapping at
    # all (NaN) - e.g. a cloneClassification entry that never resolved
    # to a frame-1 ancestor. Drop those before anything else: left in,
    # .astype(int) below would either raise or silently produce garbage
    # from a NaN value.
    n_total = len(lineage_table)
    valid = ~np.isnan(lineage_table).any(axis=1)
    lineage_table = lineage_table[valid]
    n_dropped = n_total - len(lineage_table)
    if n_dropped > 0:
        print(f"  Dropping {n_dropped}/{n_total} lineage_table row(s) with no valid (NaN) mapping")

    lineage_df = pd.DataFrame(
        lineage_table, columns=['file_idx', 'new_clone_id_global', 'old_clone_id_frame1']
    ).astype(int)

    class_data = pd.DataFrame(
        loadmat(clone_kmeans_mat_path)['cloneClassification'], columns=['m_label', 'k_label']
    )
    class_data['subclass'] = classify_subclass(
        class_data,
        dying_m_label=dying_m_label, persisting_m_label=persisting_m_label,
        prolif_m_label=prolif_m_label,
        slow_k_labels=slow_k_labels, fast_k_labels=fast_k_labels,
    )

    assert lineage_df['new_clone_id_global'].between(1, len(class_data)).all(), (
        f"lineage_table has new_clone_id_global value(s) outside [1, {len(class_data)}] "
        f"(cloneClassification's row range) - check that both files came from the same run"
    )
    lineage_df['subclass'] = lineage_df['new_clone_id_global'].map(
        lambda i: class_data.iloc[i - 1]['subclass']
    )

    lineage_df['lin_id'] = lineage_df.apply(
        lambda row: raw_to_global.get((row['file_idx'], row['old_clone_id_frame1'])), axis=1
    )
    n_unresolved = lineage_df['lin_id'].isna().sum()
    assert n_unresolved == 0, (
        f"{n_unresolved}/{len(lineage_df)} lineage_frame3_to_frame1.mat row(s) had no matching "
        f"(track, raw lin_id) in the raw track tables - check that drug_dir points at the same "
        f"raw data lineage_frame3_to_frame1.mat was computed from"
    )
    lineage_df['lin_id'] = lineage_df['lin_id'].astype(int)

    pct = (
        lineage_df.groupby('lin_id')['subclass']
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .reindex(columns=SUBCLASSES, fill_value=0.0)
    )
    pct.columns = [f'pct_{c}' for c in pct.columns]

    counts = lineage_df.groupby('lin_id').size().rename('n_day0_descendants')

    return pd.concat([counts, pct], axis=1).reset_index()
