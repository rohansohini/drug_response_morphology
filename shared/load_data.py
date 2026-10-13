import os
import re

import numpy as np
import pandas as pd


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
        labels:    from shared.cajal.embed.get_umap_embedding
        embedding: (n, 2) UMAP coordinates, same row order as labels
        selected:  an analysis's own selected-cell DataFrame (e.g. from
                   naive_vs_vem's select_cells.select_cells, or
                   vem_48hr_pretreatment's load_all_cells.load_all_cells)
                   (must have: condition, frame, pixel_id, subclass,
                   m_label, k_label)
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


def load_full_dataset(cajal_output_dir: str, modality: str, metadata_filename: str = 'selected_metadata.csv') -> pd.DataFrame:
    '''
    Load the selected-cell metadata and the UMAP embedding for one
    modality from disk, merged into one DataFrame. Thin I/O wrapper
    around build_dataset for notebook use.

    Inputs:
        cajal_output_dir:  directory containing {metadata_filename} and
                           {modality}_umap.npz - an analysis's own
                           config.CAJAL_OUTPUT_DIR
        modality:          'cell' or 'nuc'
        metadata_filename: filename of the selected-cell metadata CSV
                           within cajal_output_dir
    Outputs:
        df: from build_dataset - one row per embedded cell
    '''
    selected_path = os.path.join(cajal_output_dir, metadata_filename)
    selected = pd.read_csv(selected_path)

    umap_path = os.path.join(cajal_output_dir, f'{modality}_umap.npz')
    data = np.load(umap_path, allow_pickle=True)

    return build_dataset(list(data['labels']), data['embedding'], selected)
