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
    class_data = pd.DataFrame(loadmat(labels_path)['CloneClassification'])

    assert df['lin_id'].max() <= len(class_data), (
        f"lin_id max ({df['lin_id'].max()}) exceeds CloneClassification length ({len(class_data)})"
    )

    # map each lin_id to its label pair (lin_id is 1-indexed)
    df['m_label'] = df['lin_id'].map(lambda i: class_data.iloc[i - 1, 0])
    df['k_label'] = df['lin_id'].map(lambda i: class_data.iloc[i - 1, 1])

    n_m_labels = df['m_label'].nunique()
    n_k_labels = df['k_label'].nunique()
    print(f"  Appended labels: {len(class_data)} lineages, {n_m_labels} unique m_labels, {n_k_labels} unique k_labels")

    return df
