import pandas as pd


def load_all_cells(metadata_csv_path: str, frames: list[int], drug: str) -> pd.DataFrame:
    '''
    Load this drug's metadata CSV, restrict to has_mask=True cells within
    the given frames, and add 'condition' (the drug name itself - this
    analysis is a single dataset, not a cross-condition comparison) and
    'sampled' columns.

    Unlike naive_vs_vem, there is no sampling budget here - every
    available cell in these frames goes through CAJAL, so 'sampled' is
    True for every row.

    No 'subclass' column here, unlike the other analyses: this analysis
    only ever covers frame 1, and process_imaging_data's classification
    doesn't begin until day 0 (frame 3) - m_label/k_label are NaN for
    every frame-1 row, so classify_subclass() would raise on all of them.
    Each frame-1 cell's eventual fate is instead a MIXTURE across its
    day-0 descendants, computed separately by lineage_fate.py, not a
    single subclass here.

    Inputs:
        metadata_csv_path: path to {drug}_metadata.csv
        frames:            frame numbers to include
        drug:              drug/dataset name, used as the 'condition' value
    Outputs:
        metadata: filtered DataFrame with condition, sampled columns added
    '''
    metadata = pd.read_csv(metadata_csv_path)
    metadata = metadata[metadata['has_mask'] & metadata['frame'].isin(frames)].copy()
    metadata['condition'] = drug
    metadata['sampled']   = True
    return metadata
