import numpy as np
import pandas as pd

import config


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
        df['m_label'] == config.DYING_M_LABEL,
        df['m_label'] == config.PERSISTING_M_LABEL,
        (df['m_label'] == config.PROLIF_M_LABEL) & df['k_label'].isin(config.SLOW_K_LABELS),
        (df['m_label'] == config.PROLIF_M_LABEL) & df['k_label'].isin(config.FAST_K_LABELS),
    ]
    choices = ['dying', 'persisting', 'slow_prolif', 'fast_prolif']

    subclass = pd.Series(np.select(conditions, choices, default=None), index=df.index)

    unmatched = subclass.isna()
    if unmatched.any():
        bad = df.loc[unmatched, ['m_label', 'k_label']].drop_duplicates()
        raise ValueError(
            f"{unmatched.sum()} row(s) have an (m_label, k_label) combination that doesn't map to "
            f"any subclass: {bad.to_dict('records')} - check config.py's classification constants"
        )

    return subclass
