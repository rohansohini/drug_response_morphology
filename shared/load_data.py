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


def match_labels_to_metadata(labels: list[str], selected: pd.DataFrame) -> pd.DataFrame:
    '''
    Parse CAJAL point labels and look up each one's full metadata row -
    the label<->metadata merge, without requiring embedding coordinates
    (unlike build_dataset, which additionally attaches a UMAP embedding
    for notebook plotting). Used wherever the 2D UMAP projection itself
    isn't needed - e.g. statistical tests that operate directly on
    CAJAL's distance matrix instead (UMAP distorts distances, so it's
    the wrong representation for anything quantitative).

    Inputs:
        labels:   from shared.cajal.embed.get_umap_embedding,
                  get_phate_embedding, or load_distance_matrix
        selected: an analysis's own selected-cell DataFrame (e.g. from
                  naive_vs_vem's select_cells.select_cells, or
                  vem_48hr_pretreatment's load_all_cells.load_all_cells)
                  (must have: condition, frame, pixel_id; every other
                  column - subclass, m_label, k_label, lin_id, node_id,
                  etc. - is carried through too, whatever `selected`
                  happens to have, and none of them are required)
    Outputs:
        df: one row per label, in the SAME row order as `labels` (so row
            i still corresponds to a distance matrix's row/col i), with
            columns [label, condition, frame, pixel_id, <every other
            column `selected` has>]
    '''
    parsed = [parse_label(l) for l in labels]
    df = pd.DataFrame(parsed, columns=['condition', 'frame', 'pixel_id'])
    df.insert(0, 'label', labels)

    meta = selected

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

    # indicator=True, rather than checking some particular column for NaN,
    # detects an unmatched row correctly regardless of which columns
    # `selected` happens to have (some analyses, e.g. vem_48hr_pretreatment's
    # frame-1-only metadata, have no 'subclass' column at all to check)
    df = df.merge(meta, on=['condition', 'frame', 'pixel_id'], how='left', indicator=True)

    n_unmatched = (df['_merge'] == 'left_only').sum()
    if n_unmatched > 0:
        raise ValueError(
            f"{n_unmatched}/{len(df)} embedded cells had no matching row in `selected` metadata "
            f"when looked up by (condition, frame, pixel_id) - check that `selected` covers every "
            f"cell CAJAL was run on."
        )
    df = df.drop(columns='_merge')

    return df


def build_dataset(labels: list[str], embedding: np.ndarray, selected: pd.DataFrame) -> pd.DataFrame:
    '''
    Merge 2D embedding coordinates with the selected-cell metadata. Thin
    wrapper around match_labels_to_metadata() that additionally attaches
    the embedding coordinates, for callers (notebook plotting) that need
    them. Works identically regardless of which method (UMAP, PHATE, ...)
    produced `embedding` - the columns are always named dim_0/dim_1,
    never e.g. umap_0/umap_1, so notebook/plotting code never needs to
    know or care which method was used.

    Inputs:
        labels:    from shared.cajal.embed.get_umap_embedding or
                   get_phate_embedding
        embedding: (n, 2) coordinates, same row order as labels
        selected:  see match_labels_to_metadata
    Outputs:
        df: from match_labels_to_metadata, plus dim_0/dim_1 columns
    '''
    df = match_labels_to_metadata(labels, selected)
    df['dim_0'] = embedding[:, 0]
    df['dim_1'] = embedding[:, 1]
    return df


def load_full_dataset(cajal_output_dir: str, modality: str, embedding_method: str,
                       metadata_filename: str = 'selected_metadata.csv') -> pd.DataFrame:
    '''
    Load the selected-cell metadata and one embedding method's 2D
    coordinates for one modality from disk, merged into one DataFrame.
    Thin I/O wrapper around build_dataset for notebook use.

    Inputs:
        cajal_output_dir:  directory containing {metadata_filename} and
                           {modality}_{embedding_method}.npz - an
                           analysis's own config.CAJAL_OUTPUT_DIR
                           (main.py's Step 4 saves both 'umap' and
                           'phate' for every modality)
        modality:          'cell' or 'nuc'
        embedding_method:  'umap' or 'phate' - which saved embedding to load
        metadata_filename: filename of the selected-cell metadata CSV
                           within cajal_output_dir
    Outputs:
        df: from build_dataset - one row per embedded cell
    '''
    selected_path = os.path.join(cajal_output_dir, metadata_filename)
    selected = pd.read_csv(selected_path)

    embedding_path = os.path.join(cajal_output_dir, f'{modality}_{embedding_method}.npz')
    data = np.load(embedding_path, allow_pickle=True)

    return build_dataset(list(data['labels']), data['embedding'], selected)


def export_plot_data(dfs: dict, output_path: str) -> pd.DataFrame:
    '''
    Combine every (modality, embedding_method) DataFrame a notebook's own
    process_modality()-style loop already built into one flat CSV, so
    someone without this repo can recreate (or extend) its plots from raw
    per-cell coordinates and metadata alone, no code required.

    Inputs:
        dfs:         {(modality, embedding_method): df} - each df must
                      already have everything a plot needs (dim_0, dim_1
                      from load_full_dataset, plus whatever grouping/
                      metadata columns this notebook added - condition,
                      subclass, frame, region, ...)
        output_path: where to save the combined CSV
    Outputs:
        combined: the saved DataFrame, for notebook inspection
    '''
    parts = []
    for (modality, embedding_method), df in dfs.items():
        part = df.copy()
        part.insert(0, 'embedding_method', embedding_method)
        part.insert(0, 'modality', modality)
        parts.append(part)
    combined = pd.concat(parts, ignore_index=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"  Saved plot data ({len(combined)} rows) to {output_path}")
    return combined
