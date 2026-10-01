import os
import re

import numpy as np
import pandas as pd

import config


def parse_label(label: str) -> tuple[str, int, int]:
    '''
    Parse a node label into its components.

    Inputs:
        label: '{condition}_frame_{frame}_pixel_{pixel_id}'
    Outputs:
        condition, frame, pixel_id
    '''
    match = re.match(r'(.+)_frame_(\d+)_pixel_(\d+)', label)
    condition, frame, pixel_id = match.groups()
    return condition, int(frame), int(pixel_id)


def build_dataset(labels: list[str], embedding: np.ndarray, selected: pd.DataFrame) -> pd.DataFrame:
    '''
    Merge UMAP coordinates with the selected-cell metadata.

    Inputs:
        labels:    from cajal_pipeline.embed.get_umap_embedding
        embedding: (n, 2) UMAP coordinates, same row order as labels
        selected:  from pipeline.select_cells.select_cells (must have:
                   condition, frame, pixel_id, subclass, m_label, k_label)
    Outputs:
        df: one row per embedded cell, columns
            [label, condition, frame, pixel_id, umap_0, umap_1, subclass, m_label, k_label]
    '''
    parsed = [parse_label(l) for l in labels]
    df = pd.DataFrame(parsed, columns=['condition', 'frame', 'pixel_id'])
    df.insert(0, 'label', labels)
    df['umap_0'] = embedding[:, 0]
    df['umap_1'] = embedding[:, 1]

    meta = selected[['condition', 'frame', 'pixel_id', 'subclass', 'm_label', 'k_label']]

    # a duplicate (condition, frame, pixel_id) key in meta would silently
    # fan out matching rows in the merge below (confirmed directly: a
    # left-merge against a non-unique key duplicates the embedded cell's
    # row rather than raising, and introduces no NaN - so the n_unmatched
    # check further down would NOT catch it) - check uniqueness explicitly
    # instead of relying on that check to catch this class of bug too
    n_dup_keys = meta.duplicated(['condition', 'frame', 'pixel_id']).sum()
    assert n_dup_keys == 0, (
        f"{n_dup_keys} duplicate (condition, frame, pixel_id) key(s) in `selected` metadata - "
        f"the merge below would silently duplicate rows for those cells rather than erroring"
    )

    df = df.merge(meta, on=['condition', 'frame', 'pixel_id'], how='left')

    n_unmatched = df['subclass'].isna().sum()
    if n_unmatched > 0:
        raise ValueError(
            f"{n_unmatched}/{len(df)} embedded cells had no matching row in `selected` metadata "
            f"when looked up by (condition, frame, pixel_id) - check that `selected` covers every "
            f"cell CAJAL was run on."
        )

    return df


def load_full_dataset(modality: str) -> pd.DataFrame:
    '''
    Load the selected-cell metadata and the UMAP embedding for one
    modality from disk (config.CAJAL_OUTPUT_DIR), merged into one
    DataFrame. Thin I/O wrapper around build_dataset for notebook use.

    Inputs:
        modality: 'cell' or 'nuc'
    Outputs:
        df: from build_dataset - one row per embedded cell
    '''
    selected_path = os.path.join(config.CAJAL_OUTPUT_DIR, 'selected_metadata.csv')
    selected = pd.read_csv(selected_path)

    umap_path = os.path.join(config.CAJAL_OUTPUT_DIR, f'{modality}_umap.npz')
    data = np.load(umap_path, allow_pickle=True)

    return build_dataset(list(data['labels']), data['embedding'], selected)
