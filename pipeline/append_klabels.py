import numpy as np
import pandas as pd
from scipy.io import loadmat


def append_klabels(df: pd.DataFrame, klabels_path: str) -> pd.DataFrame:
    '''
    Append k-means cluster labels to the cell DataFrame.
    Each label corresponds to a lineage and is broadcast to all rows sharing that lin_id.

    Inputs:
        df:           cell track DataFrame (must have: lin_id)
        klabels_path: path to sorted_idx.mat file
    Outputs:
        df: input DataFrame with k_label column added
    '''
    df       = df.copy()
    k_labels = loadmat(klabels_path)['sortedIdx'].flatten()

    assert df['lin_id'].max() <= len(k_labels), (
        f"lin_id max ({df['lin_id'].max()}) exceeds k_labels length ({len(k_labels)})"
    )

    # map each lin_id to its label (lin_id is 1-indexed)
    df['k_label'] = df['lin_id'].map(lambda i: k_labels[i - 1])

    n_labels = df['k_label'].nunique()
    print(f"  Appended k_labels: {len(k_labels)} total labels, {n_labels} unique clusters")

    return df