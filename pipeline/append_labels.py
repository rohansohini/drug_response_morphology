import numpy as np
import pandas as pd
from scipy.io import loadmat


def append_labels(df: pd.DataFrame, labels_path: str, day0_frame: int) -> pd.DataFrame:
    '''
    Append lineage classification labels to the cell DataFrame.
    Each label pair corresponds to a lineage and is broadcast to all rows sharing that lin_id.

    The classification has two levels:
        m_label: broad class (1 = dying, 2 = persister, 3 = proliferator)
        k_label: sub-class within m_label (1-6), 18 labels total (3 x 6)

    cloneClassification is computed starting from day 0 (day0_frame), not
    from frame 1 - some drug conditions image for a while before actually
    adding the drug (e.g. a 48-hours-before-treatment baseline). It has
    exactly one row per lineage that's still alive at or after day0_frame -
    NOT one row per lin_id this drug's track tables contain. A lineage
    that dies out before day0_frame has no row at all in
    cloneClassification (the dead lineage isn't a placeholder row to skip,
    it's absent entirely), so row i (0-indexed) is the i-th SMALLEST
    lin_id among lineages alive at or after day0_frame - not lin_id - 1
    directly, since dead-before-day0 lineages are compacted out rather
    than leaving gaps in the row numbering.

    For every drug condition where day0_frame is 1 (the drug is added on
    the very first imaged frame - true of every condition except the
    48-hours-before one), every lineage is by definition "alive at or
    after frame 1", so this reduces exactly to the dense lin_id -> row
    mapping this function used before day0_frame existed as a concept.

    Inputs:
        df:          cell track DataFrame (must have: lin_id, frame)
        labels_path: path to clone_kmeans_id.mat file
        day0_frame:  frame number (1-indexed) the drug was added on for
                     this condition - see config.DAY0_FRAMES
    Outputs:
        df: input DataFrame with m_label and k_label columns added (NaN
            for any lineage that died out before day0_frame, since no
            classification exists for it)
    '''
    df         = df.copy()
    class_data = pd.DataFrame(loadmat(labels_path)['cloneClassification'])

    # lineages alive at or after day0_frame, in ascending lin_id order -
    # this is the same order cloneClassification's rows were computed in.
    # A lineage that never reaches day0_frame is excluded here entirely,
    # not assigned a (missing) placeholder row.
    day0_lin_ids = sorted(df.loc[df['frame'] >= day0_frame, 'lin_id'].unique())

    # an exact match, not just an upper bound - cloneClassification is
    # expected to have exactly one row per lineage alive at day 0.
    # cloneClassification having MORE rows than needed would previously
    # pass silently - that's exactly the kind of symptom you'd see if
    # cloneClassification were computed over a different frame range (a
    # wrong day0_frame, or a different track table) than this run's.
    assert len(day0_lin_ids) == len(class_data), (
        f"{len(day0_lin_ids)} lineage(s) alive at or after frame {day0_frame} (day 0) does not "
        f"exactly match cloneClassification length ({len(class_data)}) - expected exactly one row "
        f"per lineage alive at day 0. A mismatch means day0_frame is probably wrong for this drug "
        f"(see config.DAY0_FRAMES), or cloneClassification was computed over a different frame "
        f"range than these track tables."
    )

    lin_id_to_row = {lin_id: row for row, lin_id in enumerate(day0_lin_ids)}

    # lineages that die before day0_frame have no row to look up - leave
    # them as NaN rather than erroring, since "no label yet" is the
    # expected, normal state for a lineage that never reaches day 0
    row_idx = df['lin_id'].map(lin_id_to_row)
    df['m_label'] = row_idx.map(lambda r: class_data.iloc[int(r), 0] if pd.notna(r) else np.nan)
    df['k_label'] = row_idx.map(lambda r: class_data.iloc[int(r), 1] if pd.notna(r) else np.nan)

    n_unlabeled = int(row_idx.isna().sum())
    if n_unlabeled > 0:
        print(f"  {n_unlabeled} row(s) from lineage(s) that died before day 0 (frame {day0_frame}) have no label")

    n_m_labels = df['m_label'].nunique()
    n_k_labels = df['k_label'].nunique()
    print(f"  Appended labels: {len(class_data)} lineages, {n_m_labels} unique m_labels, {n_k_labels} unique k_labels")

    return df
