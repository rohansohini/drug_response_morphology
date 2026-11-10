import pandas as pd


class Segment:
    '''
    One uninterrupted stretch of one physical cell's life: from its birth
    (at its parent's division, or this lin_id's first frame) to its own
    next division (or wherever tracking of it ends) - this is what
    "sister"/"cousin" below mean by "cell". A segment typically has
    observations (rows) at several consecutive frames, not just one.

    parent/left_child/right_child link segments into one tree per
    lin_id. A segment with both children set divided; one still being
    None means it hasn't (yet, within the frame range `df` covers).
    '''

    def __init__(self, lin_id: int):
        self.lin_id = lin_id
        self.parent = None
        self.left_child = None
        self.right_child = None
        self.observations: list[tuple[int, int]] = []   # (frame, pixel_id)


def build_segments(df: pd.DataFrame) -> list[Segment]:
    '''
    Reconstruct every division tree in `df` as Segment objects, one tree
    per lin_id (sister/cousin relationships never cross between
    different lin_ids, so each is built independently).

    Mirrors how daughters are told apart from simple frame-to-frame
    continuation: node_id/pred_node_id chain ONE ROW PER (cell, frame) -
    a row's pred_node_id is the node_id of the SAME physical cell's row
    at the previous frame, except at division, where the mother's final
    node_id becomes the pred_node_id of BOTH daughters' first rows.  So
    a node_id referenced as pred_node_id by exactly one later row is a
    plain continuation (same segment); referenced by exactly two is a
    division (two new segments, linked as left_child/right_child).

    pred_node_id == 0 means "no known predecessor row" - either this is
    lin_id's very first frame, or (per this repo's own reference
    implementation, dataFunctions.py's sixth_step) a tracking gap
    further into the movie. Either way every such row is folded into
    the SAME root segment, rather than starting a new, disconnected one
    - a deliberate simplification inherited from that reference
    implementation, not a modeling choice made here.

    Inputs:
        df: must have columns [lin_id, frame, node_id, pred_node_id,
            pixel_id] - e.g. a drug condition's {drug}_metadata.csv
            (restrict to has_mask=True first if that's what the caller
            wants; this function doesn't filter on it itself)
    Outputs:
        segments: every Segment across every lin_id in `df`, flattened
                  into one list (each still tags its own lin_id)
    '''
    segments = []
    for lin_id, lin_df in df.groupby('lin_id'):
        segments.extend(_build_one_tree(lin_id, lin_df))
    return segments


def _build_one_tree(lin_id, lin_df: pd.DataFrame) -> list[Segment]:
    # how many later rows reference each node_id as their own
    # pred_node_id - 1 means a plain continuation, 2 means a division;
    # pred_node_id == 0 rows are excluded here (handled as "root" below,
    # never as someone's division target)
    child_counts = lin_df.loc[lin_df['pred_node_id'] != 0, 'pred_node_id'].value_counts()

    too_many = child_counts[child_counts > 2]
    assert too_many.empty, (
        f"lin_id {lin_id}: node_id(s) {too_many.index.tolist()} are the pred_node_id of more "
        f"than 2 rows - a division producing more than 2 daughters isn't supported"
    )

    node_to_segment: dict[int, Segment] = {}
    root = None
    segments = []

    for row in lin_df.sort_values('frame').itertuples():
        if row.pred_node_id == 0:
            if root is None:
                root = Segment(lin_id)
                segments.append(root)
            root.observations.append((row.frame, row.pixel_id))
            node_to_segment[row.node_id] = root
            continue

        parent_segment = node_to_segment.get(row.pred_node_id)
        assert parent_segment is not None, (
            f"lin_id {lin_id}: row at frame {row.frame} (node_id {row.node_id}) references "
            f"pred_node_id {row.pred_node_id}, which no earlier-frame row provided - rows must "
            f"be processable in frame order, with a predecessor appearing before its successor"
        )

        if child_counts.get(row.pred_node_id, 0) == 1:
            parent_segment.observations.append((row.frame, row.pixel_id))
            node_to_segment[row.node_id] = parent_segment
        else:
            new_segment = Segment(lin_id)
            new_segment.parent = parent_segment
            new_segment.observations.append((row.frame, row.pixel_id))
            if parent_segment.left_child is None:
                parent_segment.left_child = new_segment
            else:
                parent_segment.right_child = new_segment
            node_to_segment[row.node_id] = new_segment
            segments.append(new_segment)

    return segments


def find_sister_pairs(segments: list[Segment]) -> list[tuple[Segment, Segment]]:
    '''
    Every pair of segments born from the same division - i.e. every
    segment with both children present contributes exactly one pair.

    Inputs:
        segments: from build_segments (or any subset/pool of trees -
                 sisters are found purely from each segment's own
                 left_child/right_child, so trees never interfere)
    Outputs:
        pairs: [(left_child, right_child), ...]
    '''
    return [
        (s.left_child, s.right_child)
        for s in segments
        if s.left_child is not None and s.right_child is not None
    ]


def find_cousin_pairs(segments: list[Segment]) -> list[tuple[Segment, Segment]]:
    '''
    Every pair of segments whose PARENTS were sisters (same
    grandparent, different parents) - i.e. both of a division's two
    children must themselves have gone on to divide; every cross
    between one side's children and the other's counts (up to 4 pairs
    per grandparent). A grandparent whose two children divided at
    different frames still contributes pairs - cousins are not
    required to share a frame (their parents' divisions can land at
    different times).

    Inputs:
        segments: from build_segments
    Outputs:
        pairs: [(cousin_a, cousin_b), ...]
    '''
    pairs = []
    for s in segments:
        if s.left_child is None or s.right_child is None:
            continue
        left, right = s.left_child, s.right_child
        left_children  = [c for c in (left.left_child, left.right_child) if c is not None]
        right_children = [c for c in (right.left_child, right.right_child) if c is not None]
        for lc in left_children:
            for rc in right_children:
                pairs.append((lc, rc))
    return pairs
