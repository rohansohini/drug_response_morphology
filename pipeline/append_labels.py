import numpy as np
import pandas as pd
from scipy.io import loadmat


def append_labels(df: pd.DataFrame, labels_path: str) -> pd.DataFrame:
    '''
    Append lineage classification labels to the cell DataFrame.
    Each label pair corresponds to a lineage and is broadcast to all rows sharing that lin_id.

    The classification has two levels:
        m_label: broad class (1 = dying, 2 = persister, 3 = proliferator)
        k_label: sub-class within m_label (1-6), 18 labels total (3 x 6)

    Inputs:
        df:          cell track DataFrame (must have: lin_id)
        labels_path: path to sorted_idx.mat file
    Outputs:
        df: input DataFrame with m_label and k_label columns added
    '''
    df         = df.copy()
    class_data = pd.DataFrame(loadmat(labels_path)['cloneClassification'])

    # an exact match, not just an upper bound - cloneClassification is
    # expected to have exactly one row per lineage, since each lin_id is
    # used directly as a 1-indexed row position into it (see the lookup
    # below). cloneClassification having MORE rows than lin_id needs would
    # previously pass silently - that's exactly the kind of symptom you'd
    # see if cloneClassification were computed over a different lineage
    # set than this run's track tables produced, which would mean every
    # row index still "works" but may not refer to the lineage we think
    # it does.
    assert df['lin_id'].max() == len(class_data), (
        f"lin_id max ({df['lin_id'].max()}) does not exactly match cloneClassification length "
        f"({len(class_data)}) - expected exactly one row per lineage. A mismatch in either "
        f"direction means lin_id numbering and cloneClassification's row order may not "
        f"correspond to the same lineage set."
    )

    # map each lin_id to its label pair (lin_id is 1-indexed)
    df['m_label'] = df['lin_id'].map(lambda i: class_data.iloc[i - 1, 0])
    df['k_label'] = df['lin_id'].map(lambda i: class_data.iloc[i - 1, 1])

    n_m_labels = df['m_label'].nunique()
    n_k_labels = df['k_label'].nunique()
    print(f"  Appended labels: {len(class_data)} lineages, {n_m_labels} unique m_labels, {n_k_labels} unique k_labels")

    return df
