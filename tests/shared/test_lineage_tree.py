'''
Run directly: python3 tests/shared/test_lineage_tree.py
'''
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.lineage_tree import build_segments, find_sister_pairs, find_cousin_pairs


def make_df(rows):
    '''rows: list of (lin_id, frame, node_id, pred_node_id, pixel_id)'''
    return pd.DataFrame(rows, columns=['lin_id', 'frame', 'node_id', 'pred_node_id', 'pixel_id'])


def test_simple_chain_no_division_is_one_segment_with_no_pairs():
    # one lineage, 3 frames, never divides - node_id chains frame to frame
    df = make_df([
        (1, 1, 100, 0,   1),
        (1, 2, 101, 100, 1),
        (1, 3, 102, 101, 1),
    ])
    segments = build_segments(df)

    assert len(segments) == 1, f"expected exactly 1 segment (no division), got {len(segments)}"
    assert segments[0].observations == [(1, 1), (2, 1), (3, 1)]
    assert find_sister_pairs(segments) == []
    assert find_cousin_pairs(segments) == []
    print("PASS: a non-dividing chain is one segment with its full per-frame observation history, no pairs")


def test_one_division_produces_exactly_one_sister_pair():
    # mother (frames 1-2, node 100->101) divides at frame 3 into node 102 and 103
    df = make_df([
        (1, 1, 100, 0,   1),
        (1, 2, 101, 100, 1),
        (1, 3, 102, 101, 10),   # daughter A
        (1, 3, 103, 101, 11),   # daughter B
    ])
    segments = build_segments(df)

    assert len(segments) == 3, f"expected 3 segments (mother + 2 daughters), got {len(segments)}"
    sister_pairs = find_sister_pairs(segments)
    assert len(sister_pairs) == 1
    a, b = sister_pairs[0]
    assert {a.observations[0], b.observations[0]} == {(3, 10), (3, 11)}
    assert find_cousin_pairs(segments) == [], "neither daughter has divided again - no cousins yet"
    print("PASS: a single division produces exactly one sister pair and no cousin pairs")


def test_two_generations_produce_three_sister_pairs_and_four_cousin_pairs():
    # root divides at frame 2 into L and R; L divides at frame 4 into L1/L2;
    # R divides at frame 5 (a DIFFERENT frame than L's division) into R1/R2
    df = make_df([
        (1, 1, 1, 0, 1),                               # root, frame 1
        (1, 2, 2, 1, 1),                                # L, frame 2 (born at root's division)
        (1, 2, 3, 1, 2),                                # R, frame 2 (born at root's division)
        (1, 3, 4, 2, 1),                                # L continues, frame 3
        (1, 4, 5, 4, 10),                               # L1, frame 4 (L divides)
        (1, 4, 6, 4, 11),                               # L2, frame 4
        (1, 3, 7, 3, 2),                                # R continues, frame 3
        (1, 4, 8, 7, 2),                                # R continues, frame 4 (R divides LATER than L)
        (1, 5, 9, 8, 20),                                # R1, frame 5
        (1, 5, 10, 8, 21),                               # R2, frame 5
    ])
    segments = build_segments(df)

    # segments: root, L, R, L1, L2, R(continuation collapses into same R segment), R1, R2 = 7 total
    assert len(segments) == 7, f"expected 7 segments, got {len(segments)}"

    sister_pairs = find_sister_pairs(segments)
    assert len(sister_pairs) == 3, f"expected 3 sister pairs (L/R, L1/L2, R1/R2), got {len(sister_pairs)}"

    cousin_pairs = find_cousin_pairs(segments)
    assert len(cousin_pairs) == 4, f"expected 4 cousin pairs (L1,L2 x R1,R2), got {len(cousin_pairs)}"
    cousin_obs_pairs = {
        frozenset([a.observations[0], b.observations[0]]) for a, b in cousin_pairs
    }
    expected = {
        frozenset([(4, 10), (5, 20)]), frozenset([(4, 10), (5, 21)]),
        frozenset([(4, 11), (5, 20)]), frozenset([(4, 11), (5, 21)]),
    }
    assert cousin_obs_pairs == expected
    print("PASS: two generations produce 3 sister pairs and all 4 cross cousin pairs, even with parents dividing at different frames")


def test_cousins_require_both_parents_to_have_divided():
    # root divides into L/R; only L divides again - R never does, so there
    # are no cousins at all (cousins need BOTH parents' own children)
    df = make_df([
        (1, 1, 1, 0, 1),
        (1, 2, 2, 1, 1),   # L
        (1, 2, 3, 1, 2),   # R (never divides again)
        (1, 3, 4, 2, 10),  # L1
        (1, 3, 5, 2, 11),  # L2
    ])
    segments = build_segments(df)

    assert len(find_sister_pairs(segments)) == 2  # (L,R) and (L1,L2)
    assert find_cousin_pairs(segments) == [], "R never divided again - there should be no cousin pairs"
    print("PASS: no cousin pairs are produced unless BOTH parents went on to divide")


def test_pred_node_id_zero_mid_tree_folds_into_the_same_root_segment():
    # a tracking gap: the row at frame 3 has pred_node_id=0 (not really a
    # new root, just an unresolved link) - per this module's reference
    # implementation, it's folded into the SAME root segment rather than
    # treated as a disconnected second root
    df = make_df([
        (1, 1, 1, 0, 1),
        (1, 2, 2, 1, 1),
        (1, 3, 3, 0, 1),   # gap: pred_node_id reset to 0, not a real new root
    ])
    segments = build_segments(df)

    assert len(segments) == 1, f"expected the gap row folded into the single root segment, got {len(segments)} segments"
    assert segments[0].observations == [(1, 1), (2, 1), (3, 1)]
    print("PASS: a mid-tree pred_node_id=0 row is folded into the existing root segment, not a new one")


def test_independent_lin_ids_never_produce_cross_tree_pairs():
    df = make_df([
        (1, 1, 1, 0, 1), (1, 2, 2, 1, 1), (1, 2, 3, 1, 2),   # lin_id 1: one division
        (2, 1, 4, 0, 1), (2, 2, 5, 4, 1), (2, 2, 6, 4, 2),   # lin_id 2: one division
    ])
    segments = build_segments(df)

    assert len(segments) == 6
    sister_pairs = find_sister_pairs(segments)
    assert len(sister_pairs) == 2
    for a, b in sister_pairs:
        assert a.lin_id == b.lin_id, "a sister pair must never cross between two different lin_ids"
    print("PASS: segments from independent lin_ids never pair with each other")


def test_division_with_more_than_two_daughters_raises_clearly():
    df = make_df([
        (1, 1, 1, 0, 1),
        (1, 2, 2, 1, 1), (1, 2, 3, 1, 2), (1, 2, 4, 1, 3),   # 3 rows reference node 1 - not supported
    ])
    try:
        build_segments(df)
    except AssertionError as e:
        assert 'more than 2' in str(e)
        print("PASS: a division producing more than 2 daughters raises clearly")
        return
    raise AssertionError("expected build_segments to raise on a >2-daughter division")


def test_predecessor_appearing_after_successor_raises_clearly():
    # node 2's row references pred_node_id 1, but node 1's own row is
    # MISSING entirely from df - this must raise, not silently drop the cell
    df = make_df([
        (1, 2, 2, 1, 1),
    ])
    try:
        build_segments(df)
    except AssertionError as e:
        assert 'no earlier-frame row provided' in str(e)
        print("PASS: a row whose predecessor was never seen raises clearly")
        return
    raise AssertionError("expected build_segments to raise on an unresolvable pred_node_id")


if __name__ == '__main__':
    test_simple_chain_no_division_is_one_segment_with_no_pairs()
    test_one_division_produces_exactly_one_sister_pair()
    test_two_generations_produce_three_sister_pairs_and_four_cousin_pairs()
    test_cousins_require_both_parents_to_have_divided()
    test_pred_node_id_zero_mid_tree_folds_into_the_same_root_segment()
    test_independent_lin_ids_never_produce_cross_tree_pairs()
    test_division_with_more_than_two_daughters_raises_clearly()
    test_predecessor_appearing_after_successor_raises_clearly()
    print("\nAll tests passed.")
