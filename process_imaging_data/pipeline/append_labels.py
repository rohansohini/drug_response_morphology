import numpy as np
import pandas as pd
from scipy.io import loadmat


def append_labels(df: pd.DataFrame, labels_path: str, day0_frame: int,
                   lineage_mat_path: str = None) -> pd.DataFrame:
    '''
    Append lineage classification labels to the cell DataFrame.

    Classification happens per CELL present at day 0 (day0_frame), not per
    lineage - an original lineage can already have divided multiple times
    during any pre-treatment imaging (e.g. 3 cells at frame 1 expanding to
    100 by day 0 at frame 3), and each of those day-0 cells gets its own
    distinct label, independent of its siblings or its lineage's other
    day-0 descendants. cloneClassification therefore has exactly one row
    per cell present at frame==day0_frame, NOT one row per lin_id.

    Row i (0-indexed) is the day-0 cell with the i-th smallest node_id
    among cells present at frame==day0_frame (flat ascending order, not
    grouped by lineage). That cell's label is then propagated FORWARD to
    every one of its own descendants, via the node_id/pred_node_id
    parent-child chain (walked regardless of any frame-number gaps,
    matching how make_master_ids already tolerates gaps in that chain) -
    descendants of a DIFFERENT day-0 sibling, even one sharing the same
    lin_id, get that sibling's own (different) label instead.

    Any cell that is neither itself present at day0_frame nor a
    descendant of one gets NaN labels, for one of two distinct reasons
    (reported as separate counts - see the printed summary below):
      - every cell strictly before day0_frame - its fate isn't classified
        until day 0 (a shared pre-day0 ancestor can fan out into several
        differently-labeled day-0 descendants, so no single label could
        correctly apply to it anyway). A lineage that dies out before
        ever reaching day0_frame only ever has rows in this category -
        it contributes no rows at or after day0_frame to count separately.
      - a cell at or after day0_frame whose own lineage nonetheless has
        no recorded row at EXACTLY frame==day0_frame - e.g. a tracking/
        segmentation gap right at that one frame, with tracking resuming
        later and referencing an earlier (pre-day0) node_id as
        predecessor. The forward walk below only ever starts from real
        day0_frame rows, so this lineage (and everything downstream of
        it) is never reached, even though rows after day 0 exist for it.

    For any drug where day0_frame is 1 (every condition except a
    pre-treatment-baseline one), frame 1 is the very start of imaging, so
    every lineage has exactly one cell there (no divisions have happened
    yet) and no row ever needs NaN - every cell is a day-0 cell or one of
    its descendants.

    Some raw data (for a pre-treatment-baseline condition, day0_frame
    != 1) now has MORE cloneClassification rows than there are real
    day-0 track-table cells - the matching lineage_frame{day0_frame}_
    to_frame1.mat (same row count, same row order as cloneClassification)
    marks which rows are real with a non-NaN row and which aren't (NaN).
    Pass lineage_mat_path to filter cloneClassification down to only the
    real rows before anything else below - the row-by-row correspondence
    this function otherwise assumes (row i == the i-th smallest day0_frame
    node_id) would silently misalign cells to the wrong label without it.

    The classification has two levels:
        m_label: broad class (1 = dying, 2 = persister, 3 = proliferator)
        k_label: sub-class within m_label (1-6), 18 labels total (3 x 6)

    Inputs:
        df:               cell track DataFrame (must have: node_id,
                          pred_node_id, frame)
        labels_path:      path to clone_kmeans_id.mat file
        day0_frame:       frame number (1-indexed) the drug was added on
                          for this condition - see config.DAY0_FRAMES
        lineage_mat_path: path to this drug's own lineage_frame
                          {day0_frame}_to_frame1.mat, or None for a
                          condition with day0_frame == 1 (no pre-
                          treatment baseline, so no such file exists).
                          See above for what this is used for.
    Outputs:
        df: input DataFrame with m_label and k_label columns added (NaN
            for any cell before day0_frame, or unreachable from a
            day0_frame cell due to a tracking gap at that frame - see
            above)
    '''
    df         = df.copy()
    class_data = loadmat(labels_path)['cloneClassification']

    if lineage_mat_path is not None:
        lineage_table = loadmat(lineage_mat_path)['lineage_table']
        assert len(lineage_table) == len(class_data), (
            f"lineage_table has {len(lineage_table)} row(s) but cloneClassification has "
            f"{len(class_data)} - expected them to be the same length, in the same row order, so "
            f"lineage_table's NaN rows can mark which cloneClassification rows aren't real day-0 cells"
        )
        valid = ~np.isnan(lineage_table).any(axis=1)
        n_dropped = int((~valid).sum())
        if n_dropped > 0:
            print(f"  Dropping {n_dropped}/{len(class_data)} cloneClassification row(s) with no "
                  f"matching (non-NaN) lineage_table row")
        class_data = class_data[valid]

    class_data = pd.DataFrame(class_data)

    # cells present at day0_frame, in ascending node_id order - this is
    # the same order cloneClassification's rows were computed in. Each is
    # its own independent label group, even if it shares a lin_id with
    # another day0_frame cell (both can descend from a division that
    # happened before day 0).
    day0_node_ids = sorted(df.loc[df['frame'] == day0_frame, 'node_id'].unique())

    # an exact match, not just an upper bound - cloneClassification is
    # expected to have exactly one row per cell present at day 0.
    assert len(day0_node_ids) == len(class_data), (
        f"{len(day0_node_ids)} cell(s) present at frame {day0_frame} (day 0) does not exactly match "
        f"(possibly already NaN-filtered) cloneClassification length ({len(class_data)}) - expected "
        f"exactly one row per cell at day 0. A mismatch means day0_frame is probably wrong for this "
        f"drug (see config.DAY0_FRAMES), cloneClassification was computed over a different frame "
        f"than these track tables, or (if lineage_mat_path was given) its NaN filtering didn't fully "
        f"resolve the count - check that lineage_table's row order truly corresponds to "
        f"cloneClassification's."
    )

    node_to_row = {node_id: row for row, node_id in enumerate(day0_node_ids)}

    # forward parent -> children adjacency, across every row regardless of
    # frame - used to walk each day0 cell's descendants forward in time,
    # following the actual pred_node_id chain rather than assuming
    # descendants are always exactly one frame apart (gaps are possible -
    # see tests/test_track_table.py)
    children_of = {}
    for node_id, pred in zip(df['node_id'], df['pred_node_id']):
        if pred != 0:
            children_of.setdefault(pred, []).append(node_id)

    # propagate each day0 cell's row assignment forward to its own
    # descendants only - a cell not reachable from any day0 cell (every
    # cell strictly before day0_frame, or a lineage that dies out before
    # reaching day0_frame at all) is left unassigned, and gets NaN labels
    # below instead of a lookup
    frontier = list(day0_node_ids)
    while frontier:
        next_frontier = []
        for node_id in frontier:
            row = node_to_row[node_id]
            for child in children_of.get(node_id, []):
                if child not in node_to_row:
                    node_to_row[child] = row
                    next_frontier.append(child)
        frontier = next_frontier

    row_idx = df['node_id'].map(node_to_row)
    df['m_label'] = row_idx.map(lambda r: class_data.iloc[int(r), 0] if pd.notna(r) else np.nan)
    df['k_label'] = row_idx.map(lambda r: class_data.iloc[int(r), 1] if pd.notna(r) else np.nan)

    # two distinct causes, reported separately rather than lumped into one
    # count: a row strictly before day0_frame is ALWAYS unlabeled (fate
    # isn't determined yet - node_to_row only ever grows from day0_frame
    # nodes forward, never backward, so this split doesn't need its own
    # isna() check to be exact, but one is used anyway for robustness).
    # A row at or after day0_frame can still end up unlabeled if its own
    # lineage has no recorded row at exactly frame==day0_frame - e.g. a
    # tracking/segmentation gap right at that one frame, with tracking
    # resuming later referencing an earlier (pre-day0) node_id as
    # predecessor - the forward walk below only ever starts from REAL
    # day0_frame rows, so such a lineage (and everything downstream of
    # it) is never reached, even though it has rows after day 0.
    unlabeled = row_idx.isna()
    n_before_day0 = int((unlabeled & (df['frame'] < day0_frame)).sum())
    n_gap_at_day0 = int((unlabeled & (df['frame'] >= day0_frame)).sum())
    n_unlabeled   = n_before_day0 + n_gap_at_day0

    if n_unlabeled > 0:
        print(f"  {n_unlabeled} row(s) have no label:")
        print(f"    {n_before_day0} before day 0 (frame {day0_frame}) - not yet classifiable, "
              f"fate isn't determined until day 0")
        print(f"    {n_gap_at_day0} at or after day 0, but unreachable from any day-0 cell - "
              f"this lineage has no recorded row at exactly frame {day0_frame} itself (a tracking/"
              f"segmentation gap at that frame, with tracking resuming later)")

    n_m_labels = df['m_label'].nunique()
    n_k_labels = df['k_label'].nunique()
    print(f"  Appended labels: {len(class_data)} day-0 cells, {n_m_labels} unique m_labels, {n_k_labels} unique k_labels")

    return df
