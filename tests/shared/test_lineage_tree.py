'''
Run directly: python3 tests/shared/test_lineage_tree.py
'''
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.lineage_tree import (
    build_segments, find_sister_pairs, find_cousin_pairs, find_relative_pairs,
    observations_in_range, real_index_pairs, pool_in_range,
)


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


def test_find_relative_pairs_degree_1_and_2_match_find_sister_and_cousin_pairs():
    df = make_df([
        (1, 1, 1, 0, 1), (1, 2, 2, 1, 1), (1, 2, 3, 1, 2),
        (1, 3, 4, 2, 10), (1, 3, 5, 2, 11), (1, 4, 6, 3, 20), (1, 4, 7, 3, 21),
    ])
    segments = build_segments(df)

    assert find_relative_pairs(segments, degree=1) == find_sister_pairs(segments)
    assert find_relative_pairs(segments, degree=2) == find_cousin_pairs(segments)
    print("PASS: find_relative_pairs(degree=1/2) match find_sister_pairs/find_cousin_pairs exactly")


def test_find_relative_pairs_degree_3_finds_second_cousins():
    # root -> L/R; L -> L1/L2, R -> R1/R2 (cousins at degree 2); only L1
    # and R1 go on to divide again, into L1a/L1b and R1a/R1b - these are
    # "second cousins" (degree 3, common great-grandparent = root)
    df = make_df([
        (1, 1, 1, 0, 1),
        (1, 2, 2, 1, 1), (1, 2, 3, 1, 2),          # L, R
        (1, 3, 4, 2, 10), (1, 3, 5, 2, 11),        # L1, L2
        (1, 3, 6, 3, 20), (1, 3, 7, 3, 21),        # R1, R2
        (1, 4, 8, 4, 100), (1, 4, 9, 4, 101),      # L1a, L1b (L2 never divides again)
        (1, 4, 10, 6, 200), (1, 4, 11, 6, 201),    # R1a, R1b (R2 never divides again)
    ])
    segments = build_segments(df)

    second_cousin_pairs = find_relative_pairs(segments, degree=3)
    assert len(second_cousin_pairs) == 4, (
        f"expected 4 second-cousin pairs (L1a/L1b x R1a/R1b), got {len(second_cousin_pairs)}"
    )
    obs_pairs = {frozenset([a.observations[0], b.observations[0]]) for a, b in second_cousin_pairs}
    expected = {
        frozenset([(4, 100), (4, 200)]), frozenset([(4, 100), (4, 201)]),
        frozenset([(4, 101), (4, 200)]), frozenset([(4, 101), (4, 201)]),
    }
    assert obs_pairs == expected
    print("PASS: find_relative_pairs(degree=3) finds second cousins, skipping branches that haven't divided that deep yet")


def test_find_relative_pairs_rejects_degree_below_1():
    try:
        find_relative_pairs([], degree=0)
    except AssertionError as e:
        assert 'degree' in str(e)
        print("PASS: find_relative_pairs rejects degree < 1")
        return
    raise AssertionError("expected find_relative_pairs to raise on degree=0")


def test_observations_in_range_filters_by_frame_and_cajal_survival():
    segment = build_segments(make_df([(1, 1, 1, 0, 1), (1, 2, 2, 1, 1), (1, 3, 3, 2, 1)]))[0]
    # frame 3's observation never made it into label_to_index (e.g. it
    # wasn't sampled for CAJAL) - must be silently skipped, not KeyError
    label_to_index = {(1, 1): 0, (2, 1): 1}

    assert observations_in_range(segment, (1, 2), label_to_index) == [0, 1]
    assert observations_in_range(segment, (2, 3), label_to_index) == [1]
    assert observations_in_range(segment, (1, 3), label_to_index) == [0, 1], (
        "frame 3's observation has no entry in label_to_index and must be dropped, not raise"
    )
    print("PASS: observations_in_range restricts to a frame range and drops observations that never got a CAJAL embedding")


def test_real_index_pairs_cross_combines_surviving_observations_on_both_sides():
    df = make_df([
        (1, 1, 1, 0, 1),
        (1, 2, 2, 1, 1), (1, 2, 3, 1, 2),            # A, B (the one division)
        (1, 3, 4, 2, 10), (1, 3, 5, 3, 20),          # A, B both keep tracking into frame 3
        (1, 4, 6, 4, 10), (1, 4, 7, 5, 20),          # ... and frame 4
    ])
    segments = build_segments(df)
    sister_pairs = find_sister_pairs(segments)
    assert len(sister_pairs) == 1, "expected exactly one division (A, B), each continuing as a chain afterward"

    # frame 2's observations are deliberately left out, so restricting to
    # frames 3-4 below exercises both the frame-range filter AND the
    # "this frame wasn't sampled" filter at once
    label_to_index = {(3, 10): 0, (4, 10): 1, (3, 20): 2, (4, 20): 3}

    index_pairs = real_index_pairs(sister_pairs, (3, 4), label_to_index)
    assert set(index_pairs) == {(0, 2), (0, 3), (1, 2), (1, 3)}, (
        "expected every combination of A's 2 surviving observations with B's 2, not just one pair"
    )
    print("PASS: real_index_pairs cross-combines every surviving observation on each side, not just one pair per relationship")


def test_pool_in_range_filters_by_frame_only():
    label_to_index = {(1, 1): 0, (2, 1): 1, (3, 1): 2}
    assert sorted(pool_in_range(label_to_index, (1, 2))) == [0, 1]
    assert sorted(pool_in_range(label_to_index, (1, 3))) == [0, 1, 2]
    print("PASS: pool_in_range keeps every embedded cell within a frame range, regardless of lineage")


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
    test_find_relative_pairs_degree_1_and_2_match_find_sister_and_cousin_pairs()
    test_find_relative_pairs_degree_3_finds_second_cousins()
    test_find_relative_pairs_rejects_degree_below_1()
    test_observations_in_range_filters_by_frame_and_cajal_survival()
    test_real_index_pairs_cross_combines_surviving_observations_on_both_sides()
    test_pool_in_range_filters_by_frame_only()
    test_division_with_more_than_two_daughters_raises_clearly()
    test_predecessor_appearing_after_successor_raises_clearly()
    print("\nAll tests passed.")
