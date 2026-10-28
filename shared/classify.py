import numpy as np
import pandas as pd


def classify_subclass(df: pd.DataFrame, dying_m_label: int, persisting_m_label: int, prolif_m_label: int,
                       slow_k_labels: list[int], fast_k_labels: list[int]) -> pd.Series:
    '''
    Assign each row's subclass: 'dying', 'persisting', 'slow_prolif', or
    'fast_prolif', from its m_label/k_label.

    The m_label/k_label -> subclass mapping comes from process_imaging_data's
    cloneClassification (see process_imaging_data/README.md), but the
    specific numeric scheme (which m_label means 'dying', which k_labels
    are 'slow' vs 'fast') is domain knowledge about the upstream k-means
    clustering that isn't derivable from this repo - each analysis
    declares it explicitly in its own config.py rather than this function
    assuming a fixed default, so the scheme is visible right next to
    everything else about how that analysis was run.

    Inputs:
        df:                 must have m_label, k_label columns
        dying_m_label:      m_label value meaning 'dying'
        persisting_m_label: m_label value meaning 'persisting'
        prolif_m_label:     m_label value meaning 'proliferator' (split
                            into slow_prolif/fast_prolif by k_label)
        slow_k_labels:      k_label values meaning 'slow' proliferator
        fast_k_labels:      k_label values meaning 'fast' proliferator
    Outputs:
        subclass: one label per row, same index as df
    '''
    conditions = [
        df['m_label'] == dying_m_label,
        df['m_label'] == persisting_m_label,
        (df['m_label'] == prolif_m_label) & df['k_label'].isin(slow_k_labels),
        (df['m_label'] == prolif_m_label) & df['k_label'].isin(fast_k_labels),
    ]
    choices = ['dying', 'persisting', 'slow_prolif', 'fast_prolif']

    subclass = pd.Series(np.select(conditions, choices, default=None), index=df.index)

    unmatched = subclass.isna()
    if unmatched.any():
        bad = df.loc[unmatched, ['m_label', 'k_label']].drop_duplicates()
        raise ValueError(
            f"{unmatched.sum()} row(s) have an (m_label, k_label) combination that doesn't map to "
            f"any subclass: {bad.to_dict('records')} - check this analysis's config.py classification constants"
        )

    return subclass
