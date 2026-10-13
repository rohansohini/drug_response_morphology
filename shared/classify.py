import numpy as np
import pandas as pd

###### Classification scheme from process_imaging_data's cloneClassification
#  (see process_imaging_data/README.md). This is a fixed, universal
#  biological scheme shared by every analysis in this repo - not
#  per-analysis configuration - so it lives here rather than in any
#  analysis's own config.py.
#  m_label: 1 = dying, 2 = persister, 3 = proliferator. Within m_label == 3,
#  k_label splits slow (1, 2) vs fast (3, 4, 5, 6) proliferators.
DYING_M_LABEL      = 1
PERSISTING_M_LABEL = 2
PROLIF_M_LABEL     = 3
SLOW_K_LABELS      = [1, 2]
FAST_K_LABELS      = [3, 4, 5, 6]


def classify_subclass(df: pd.DataFrame) -> pd.Series:
    '''
    Assign each row's subclass: 'dying', 'persisting', 'slow_prolif', or
    'fast_prolif', from its m_label/k_label.

    Inputs:
        df: must have m_label, k_label columns
    Outputs:
        subclass: one label per row, same index as df
    '''
    conditions = [
        df['m_label'] == DYING_M_LABEL,
        df['m_label'] == PERSISTING_M_LABEL,
        (df['m_label'] == PROLIF_M_LABEL) & df['k_label'].isin(SLOW_K_LABELS),
        (df['m_label'] == PROLIF_M_LABEL) & df['k_label'].isin(FAST_K_LABELS),
    ]
    choices = ['dying', 'persisting', 'slow_prolif', 'fast_prolif']

    subclass = pd.Series(np.select(conditions, choices, default=None), index=df.index)

    unmatched = subclass.isna()
    if unmatched.any():
        bad = df.loc[unmatched, ['m_label', 'k_label']].drop_duplicates()
        raise ValueError(
            f"{unmatched.sum()} row(s) have an (m_label, k_label) combination that doesn't map to "
            f"any subclass: {bad.to_dict('records')} - check shared/classify.py's classification constants"
        )

    return subclass
