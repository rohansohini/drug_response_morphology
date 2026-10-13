import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from shared.classify import classify_subclass


def load_all_cells(metadata_csv_path: str, frames: list[int], drug: str) -> pd.DataFrame:
    '''
    Load this drug's metadata CSV, restrict to has_mask=True cells within
    the given frames, and add 'condition' (the drug name itself - this
    analysis is a single dataset, not a cross-condition comparison),
    'subclass', and 'sampled' columns.

    Unlike naive_vs_vem, there is no sampling budget here - every
    available cell in these frames goes through CAJAL, so 'sampled' is
    True for every row.

    Inputs:
        metadata_csv_path: path to {drug}_metadata.csv
        frames:            frame numbers to include
        drug:              drug/dataset name, used as the 'condition' value
    Outputs:
        metadata: filtered DataFrame with condition, subclass, sampled columns added
    '''
    metadata = pd.read_csv(metadata_csv_path)
    metadata = metadata[metadata['has_mask'] & metadata['frame'].isin(frames)].copy()
    metadata['condition'] = drug
    metadata['subclass']  = classify_subclass(metadata)
    metadata['sampled']   = True
    return metadata
