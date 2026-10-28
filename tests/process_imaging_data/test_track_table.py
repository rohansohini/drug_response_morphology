'''
Regression test for the make_master_ids() pred_node_id-resolution bug.

Real data showed: a lineage's cells referencing a pred_node_id that isn't
among that lineage's own rows. Root cause, confirmed against real track
tables on Quest: load_track()'s dropna() removes some rows (for reasons
unrelated to lineage structure - e.g. a missing coordinate), which leaves
gaps in a track's raw, file-local node_id numbering. The old make_master_ids
resolved pred_node_id with a fixed per-track offset, which implicitly
assumes no gaps; once a row is dropped, every pred_node_id referencing a
row after the gap resolves to the wrong node.

This test builds a small synthetic track with an injected gap (mirroring
the real scenario) and checks that every pred_node_id resolves to the
correct global node_id after make_master_ids - i.e., to the row that is
actually that cell's parent, not just some row that used to sit near it.

Run directly: python3 tests/process_imaging_data/test_track_table.py
'''
import os
import sys

import numpy as np
import pandas as pd

ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'process_imaging_data')
sys.path.insert(0, ANALYSIS_DIR)
from pipeline.track_table import make_master_ids


def build_track_with_gap() -> pd.DataFrame:
    '''
    One track, one lineage, mimicking a division with a dropped row.

    Raw (local) structure, before the gap:
        node 1 (root, frame 1)
         -> node 2 (frame 2)
             -> node 3 (frame 3, division)
             -> node 4 (frame 3, division)
                 -> node 5 (frame 4)

    Row for local node_id=2 gets dropped (as if dropna() removed it for an
    unrelated reason, e.g. missing coordinate) - node 3, 4, and 5 still
    carry their original pred_node_id values (2 and 3 respectively), which
    is exactly the gap scenario confirmed on real data.
    '''
    rows = [
        # node_id, pred_node_id, lin_id, frame, track
        (1, 0, 1, 1, 1),
        # (2, 1, 1, 2, 1),  <- dropped by dropna(), as on real data
        (3, 2, 1, 3, 1),   # division child A - parent is dropped node 2
        (4, 2, 1, 3, 1),   # division child B - parent is dropped node 2
        (5, 3, 1, 4, 1),   # child of node 3
    ]
    return pd.DataFrame(rows, columns=['node_id', 'pred_node_id', 'lin_id', 'frame', 'track'])


def test_gap_with_missing_parent_raises():
    # node 2 is the TRUE parent of nodes 3 and 4, but it was dropped -
    # there is no correct resolution, so this must fail loudly, not silently
    # mis-resolve to an unrelated row (the old bug's failure mode).
    df = build_track_with_gap()
    try:
        make_master_ids(df)
    except AssertionError as e:
        assert 'pred_node_id' in str(e)
        assert '2' in str(e)
        print("PASS: missing-parent case raises a clear assertion, as expected")
        return
    raise AssertionError("expected make_master_ids to raise when a referenced parent was dropped")


def test_gap_after_parent_survives_resolves_correctly():
    '''
    Same gap shape, but this time the dropped row is NOT anyone's parent
    (matching what was actually confirmed on real data: dropped rows are
    never referenced by a surviving pred_node_id). Every remaining
    pred_node_id must resolve to the correct row.
    '''
    rows = [
        (1, 0, 1, 1, 1),
        (2, 1, 1, 2, 1),
        # (3, 2, 1, 3, 1),  <- dropped: a childless dead-end, nobody's parent
        (4, 2, 1, 3, 1),    # true parent is node 2 (still present)
        (5, 4, 1, 4, 1),    # true parent is node 4
    ]
    df = pd.DataFrame(rows, columns=['node_id', 'pred_node_id', 'lin_id', 'frame', 'track'])

    result = make_master_ids(df.copy())

    # build the expected old_local_id -> new_global_id mapping directly
    expected_global = dict(zip(df['node_id'], np.arange(1, len(df) + 1)))

    # row order in `result` matches `df` (positional reassignment), so
    # result.iloc[i] corresponds to df.iloc[i]
    assert result.iloc[0]['pred_node_id'] == 0                                    # node 1: root
    assert result.iloc[1]['pred_node_id'] == expected_global[1]                    # node 2 -> parent node 1
    assert result.iloc[2]['pred_node_id'] == expected_global[2]                    # node 4 -> parent node 2
    assert result.iloc[3]['pred_node_id'] == expected_global[4]                    # node 5 -> parent node 4

    print("PASS: pred_node_id resolves correctly across a gap that isn't anyone's parent")


def test_multi_track_lin_id_offset_still_unique():
    '''
    Sanity check that the lin_id constant-shift logic (unchanged by this
    fix) still produces globally unique lin_ids across multiple tracks,
    with gaps present.
    '''
    track1 = pd.DataFrame([
        (1, 0, 1, 1, 1),
        (2, 1, 1, 2, 1),
    ], columns=['node_id', 'pred_node_id', 'lin_id', 'frame', 'track'])

    track2 = pd.DataFrame([
        (1, 0, 1, 1, 2),
        (2, 1, 1, 2, 2),
    ], columns=['node_id', 'pred_node_id', 'lin_id', 'frame', 'track'])

    df = pd.concat([track1, track2], ignore_index=True)
    result = make_master_ids(df)

    assert result['lin_id'].nunique() == 2
    assert result.loc[result['track'] == 1, 'lin_id'].nunique() == 1
    assert result.loc[result['track'] == 2, 'lin_id'].nunique() == 1
    assert result.loc[result['track'] == 1, 'lin_id'].iloc[0] != result.loc[result['track'] == 2, 'lin_id'].iloc[0]

    print("PASS: lin_id remains globally unique across tracks")


if __name__ == '__main__':
    test_gap_with_missing_parent_raises()
    test_gap_after_parent_survives_resolves_correctly()
    test_multi_track_lin_id_offset_still_unique()
    print("\nAll tests passed.")
