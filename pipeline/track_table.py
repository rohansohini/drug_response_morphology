import numpy as np
import pandas as pd
from scipy.io import loadmat

###### Track Table Column Reference (raw .mat file):
#  - Column 0:  Node ID (unique within a track)
#  - Column 1:  Frame number (1-indexed)
#  - Column 3:  Predecessor node ID (within-track)
#  - Column 4:  Nucleus x-coordinate (1-indexed)
#  - Column 5:  Nucleus y-coordinate (1-indexed)
#  - Column 9:  Lineage ID (unique within a track)

###### Output Columns:
#  - frame:        frame number (1-indexed)
#  - node_id:      globally unique node ID across all tracks
#  - pred_node_id: globally unique predecessor node ID (0 for root nodes)
#  - lin_id:       globally unique lineage ID across all tracks
#  - x_coord:      nucleus x-coordinate (0-indexed)
#  - y_coord:      nucleus y-coordinate (0-indexed)
#  - track:        track number (1-indexed)


def load_track(file_path: str) -> pd.DataFrame:
    '''
    Load a single .mat track table and return a tidy DataFrame.

    Inputs:
        file_path: path to the .mat file
    Outputs:
        df: cleaned DataFrame with columns [frame, node_id, pred_node_id, lin_id, x_coord, y_coord, track]
    '''
    raw = loadmat(file_path)
    df  = pd.DataFrame(raw['track_table'])

    df = df.dropna()

    # rename only the columns we need; drop the rest
    df = df.rename(columns={
        0: 'node_id',
        1: 'frame',
        3: 'pred_node_id',
        4: 'x_coord',
        5: 'y_coord',
        9: 'lin_id',
    })[['frame', 'node_id', 'pred_node_id', 'lin_id', 'x_coord', 'y_coord']]

    df = df.astype({
        'frame':        'int32',
        'node_id':      'int32',
        'pred_node_id': 'int32',
        'lin_id':       'int32',
        'x_coord':      'float64',
        'y_coord':      'float64',
    })

    # convert coordinates from 1-indexed (MATLAB) to 0-indexed (Python)
    df['x_coord'] = df['x_coord'] - 1
    df['y_coord'] = df['y_coord'] - 1

    # parse track number from filename (e.g. "track_table_03.mat" -> 3)
    df['track'] = int(file_path.split('track_table_')[1].split('.mat')[0])

    return df


def make_master_ids(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Replace within-track IDs with IDs that are unique across the entire dataset.

    node_id is reassigned globally by row position. lin_id is shifted by a
    running per-track offset (a constant shift preserves uniqueness
    regardless of any dropped rows). pred_node_id is resolved via an
    explicit old-node_id -> new-node_id lookup per track, NOT a fixed
    offset: load_track()'s dropna() can remove rows and leave gaps in a
    track's raw node_id numbering, and a fixed offset assumes no gaps -
    it silently mis-resolves any pred_node_id referencing a row after one.

    Inputs:
        df: concatenated track DataFrame (must have: node_id, pred_node_id, lin_id, track)
    Outputs:
        df: same DataFrame with globally unique node_id, pred_node_id, lin_id
    '''
    df = df.copy()

    # raw, track-local node_id - node_id is about to be overwritten below,
    # so pred_node_id (which references this raw numbering) must be
    # resolved against it before that happens
    local_node_id = df['node_id'].copy()

    # assign globally unique node_ids using row index (1-indexed)
    df['node_id'] = df.index + 1

    lin_offset = 0

    for track in np.unique(df['track']):
        mask = df['track'] == track

        local_to_global = dict(zip(local_node_id[mask], df.loc[mask, 'node_id']))

        raw_pred = df.loc[mask, 'pred_node_id']
        non_root = raw_pred[raw_pred != 0]
        missing  = non_root[~non_root.isin(local_to_global)]
        assert missing.empty, (
            f"track {track}: pred_node_id value(s) {sorted(missing.unique().tolist())} "
            f"reference a node_id not present among this track's own (post-dropna) rows - "
            f"the true parent row was dropped, so this child's lineage can't be resolved"
        )

        df.loc[mask, 'pred_node_id'] = raw_pred.map(
            lambda pid: 0 if pid == 0 else local_to_global[pid]
        )

        # offset lin_id
        df.loc[mask, 'lin_id'] = df.loc[mask, 'lin_id'] + lin_offset
        lin_offset = df.loc[mask, 'lin_id'].max()

    return df


def load_track_table(file_paths: list[str]) -> pd.DataFrame:
    '''
    Load, concatenate, and assign master IDs across all track tables.

    Inputs:
        file_paths: list of paths to .mat track table files
    Outputs:
        df: master DataFrame with columns [frame, node_id, pred_node_id, lin_id, x_coord, y_coord, track]
    '''
    # load and concatenate all tracks
    df = pd.concat(
        [load_track(fp) for fp in file_paths],
        ignore_index=True
    )

    # replace within-track IDs with dataset-wide unique IDs
    df = make_master_ids(df)

    return df