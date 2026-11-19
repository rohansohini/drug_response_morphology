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


def _descendants_at_depth(segment: Segment, depth: int) -> list[Segment]:
    '''
    Every segment reached by exactly `depth` divisions below `segment`
    (depth=0 is `segment` itself). Stops wherever a branch hasn't
    divided that many more times (yet, within the frame range the tree
    was built from) and contributes nothing from that branch, rather
    than erroring - "hasn't divided enough times yet" is the expected
    case for a branch observed near the edge of a frame window, not a
    bug.
    '''
    if depth == 0:
        return [segment]
    descendants = []
    for child in (segment.left_child, segment.right_child):
        if child is not None:
            descendants.extend(_descendants_at_depth(child, depth - 1))
    return descendants


def find_relative_pairs(segments: list[Segment], degree: int) -> list[tuple[Segment, Segment]]:
    '''
    Every pair of segments that share a common ancestor exactly `degree`
    generations back, reached via the ancestor's TWO DIFFERENT children
    (not nested within one side - that would make them parent/child or
    aunt/niece, not same-generation relatives). Generalizes
    find_sister_pairs (degree=1: common parent) and find_cousin_pairs
    (degree=2: common grandparent) to arbitrarily distant relatives
    (degree=3: common great-grandparent, i.e. "second cousins"; etc).

    A grandparent (or further ancestor) whose two sides divided at
    different frames, or different numbers of times so far, still
    contributes whatever pairs both sides actually have at this depth -
    relatives are never required to share a frame or a generation count
    in wall-clock time, only in tree topology.

    Inputs:
        segments: from build_segments (or any subset/pool of trees -
                 relatives are found purely from each segment's own
                 left_child/right_child, so trees never interfere)
        degree:   generations back to the shared ancestor (>= 1; 1 is
                 sisters, the shallowest possible relationship)
    Outputs:
        pairs: [(relative_a, relative_b), ...] - every cross-combination
               of the ancestor's left-side degree-(degree-1) descendants
               with its right-side ones (up to 4 at degree=2, more at
               higher degrees if more branches survived that deep)
    '''
    assert degree >= 1, f"degree must be >= 1 (got {degree}) - degree 1 (sisters) is the shallowest relationship"

    pairs = []
    for s in segments:
        if s.left_child is None or s.right_child is None:
            continue
        left_descendants  = _descendants_at_depth(s.left_child, degree - 1)
        right_descendants = _descendants_at_depth(s.right_child, degree - 1)
        for a in left_descendants:
            for b in right_descendants:
                pairs.append((a, b))
    return pairs


def find_sister_pairs(segments: list[Segment]) -> list[tuple[Segment, Segment]]:
    '''
    Every pair of segments born from the same division - i.e. every
    segment with both children present contributes exactly one pair.
    See find_relative_pairs (degree=1) for the general case.

    Inputs:
        segments: from build_segments (or any subset/pool of trees -
                 sisters are found purely from each segment's own
                 left_child/right_child, so trees never interfere)
    Outputs:
        pairs: [(left_child, right_child), ...]
    '''
    return find_relative_pairs(segments, degree=1)


def find_cousin_pairs(segments: list[Segment]) -> list[tuple[Segment, Segment]]:
    '''
    Every pair of segments whose PARENTS were sisters (same
    grandparent, different parents) - i.e. both of a division's two
    children must themselves have gone on to divide; every cross
    between one side's children and the other's counts (up to 4 pairs
    per grandparent). See find_relative_pairs (degree=2) for the
    general case.

    Inputs:
        segments: from build_segments
    Outputs:
        pairs: [(cousin_a, cousin_b), ...]
    '''
    return find_relative_pairs(segments, degree=2)


def observations_in_range(segment: Segment, frame_range: tuple[int, int], label_to_index: dict) -> list[int]:
    '''
    A segment's own (frame, pixel_id) observations, restricted to a
    frame range AND to whichever ones actually got a CAJAL embedding -
    returned as indices into the distance matrix label_to_index was
    built from.

    Inputs:
        segment:        a Segment (from build_segments, or one of
                        find_relative_pairs' returned pair members)
        frame_range:    (lo, hi) frame numbers, inclusive
        label_to_index: {(frame, pixel_id): row/col index into a CAJAL
                        distance matrix} - e.g. built from
                        shared.cajal.embed.load_distance_matrix's
                        labels via shared.load_data.parse_label
    Outputs:
        indices: distance-matrix indices for this segment's own
                 observations within frame_range that survived CAJAL
                 subsampling
    '''
    lo, hi = frame_range
    return [
        label_to_index[(frame, pixel_id)]
        for frame, pixel_id in segment.observations
        if lo <= frame <= hi and (frame, pixel_id) in label_to_index
    ]


def real_index_pairs(pairs: list[tuple[Segment, Segment]], frame_range: tuple[int, int],
                      label_to_index: dict) -> list[tuple[int, int]]:
    '''
    Every (index_a, index_b) cross-combination of a surviving
    observation from one side of a relative pair (e.g. from
    find_relative_pairs) with a surviving observation from the other -
    every combination available, not just one per pair, since relatives
    are not required to share a frame (one side's own track can run
    longer than the other's; a cousin pair's parents, or a deeper
    ancestor's two sides, can divide at different times).

    Inputs:
        pairs:          [(segment_a, segment_b), ...] - e.g. from
                        find_relative_pairs
        frame_range:    (lo, hi) frame numbers, inclusive
        label_to_index: see observations_in_range
    Outputs:
        index_pairs: [(i, j), ...] indices into the distance matrix
                     label_to_index was built from
    '''
    index_pairs = []
    for segment_a, segment_b in pairs:
        idx_a = observations_in_range(segment_a, frame_range, label_to_index)
        idx_b = observations_in_range(segment_b, frame_range, label_to_index)
        index_pairs.extend((i, j) for i in idx_a for j in idx_b)
    return index_pairs


def pool_in_range(label_to_index: dict, frame_range: tuple[int, int]) -> list[int]:
    '''Every CAJAL-embedded cell (any lineage relationship) within a frame range - a null distribution is drawn from this.'''
    lo, hi = frame_range
    return [idx for (frame, pixel_id), idx in label_to_index.items() if lo <= frame <= hi]
