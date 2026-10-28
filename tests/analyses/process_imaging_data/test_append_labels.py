'''
Run directly: python3 tests/analyses/process_imaging_data/test_append_labels.py
'''
import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'analyses', 'process_imaging_data')
sys.path.insert(0, ANALYSIS_DIR)
from pipeline.append_labels import append_labels


def fake_loadmat(class_data):
    def _loadmat(path):
        return {'cloneClassification': class_data}
    return _loadmat


def test_day0_frame_1_single_cell_per_lineage_labels_every_descendant():
    # day0_frame=1: frame 1 is the very start, so each lineage has exactly
    # one (root) cell there - 3 lineages, one divides once after frame 1
    df = pd.DataFrame({
        'node_id':      [1, 2, 3, 4, 5],
        'pred_node_id': [0, 0, 0, 1, 1],   # nodes 4, 5 are node 1's two children (division)
        'frame':        [1, 1, 1, 2, 2],
    })
    class_data = np.array([[3, 1], [1, 2], [3, 5]])  # row0->node1, row1->node2, row2->node3

    with patch('pipeline.append_labels.loadmat', side_effect=fake_loadmat(class_data)):
        result = append_labels(df, '/fake/path.mat', day0_frame=1)

    assert not result['m_label'].isna().any(), "every cell should be labeled when day0_frame=1"
    # node 1's label (row0) propagates to both of its children (4, 5)
    assert list(result.loc[result['node_id'].isin([1, 4, 5]), 'm_label']) == [3, 3, 3]
    assert list(result.loc[result['node_id'] == 2, 'm_label']) == [1]
    assert list(result.loc[result['node_id'] == 3, 'm_label']) == [3]
    print("PASS: day0_frame=1 labels every cell, with a root's label propagating to its children")


def test_raises_when_class_data_too_short():
    df = pd.DataFrame({'node_id': [1, 2, 3], 'pred_node_id': [0, 0, 0], 'frame': [1, 1, 1]})
    class_data = np.array([[3, 1], [1, 2]])  # only 2 rows, but 3 cells at frame 1

    with patch('pipeline.append_labels.loadmat', side_effect=fake_loadmat(class_data)):
        try:
            append_labels(df, '/fake/path.mat', day0_frame=1)
        except AssertionError as e:
            assert 'does not exactly match' in str(e)
            print("PASS: too-short cloneClassification raises clearly")
            return
    raise AssertionError("expected append_labels to raise when cloneClassification is shorter than day0 cells need")


def test_raises_when_class_data_too_long():
    df = pd.DataFrame({'node_id': [1, 2, 3], 'pred_node_id': [0, 0, 0], 'frame': [1, 1, 1]})
    class_data = np.array([[3, 1], [1, 2], [3, 5], [2, 4]])  # 4 rows, but only 3 cells at frame 1

    with patch('pipeline.append_labels.loadmat', side_effect=fake_loadmat(class_data)):
        try:
            append_labels(df, '/fake/path.mat', day0_frame=1)
        except AssertionError as e:
            assert 'does not exactly match' in str(e)
            print("PASS: too-long cloneClassification raises clearly too")
            return
    raise AssertionError("expected append_labels to raise when cloneClassification is longer than day0 cells need")


def test_siblings_from_a_pre_day0_division_get_independent_labels():
    # THE central scenario this whole feature exists for: one lineage (lin
    # 1) divides TWICE before day 0, becoming 3 separate cells by frame 3
    # (day 0) - each of those 3 day-0 cells must get its OWN label, and
    # each one's own descendants must inherit THAT cell's label, never a
    # sibling's, despite all three sharing the same original lineage.
    #
    #   node 1 (frame 1, root)
    #    -> node 2 (frame 2)
    #        -> node 4 (frame 3, day0)  == DAY-0 CELL A
    #        -> node 5 (frame 3, day0)  == DAY-0 CELL B
    #            -> node 7 (frame 4, B's child)
    #    -> node 3 (frame 2)
    #        -> node 6 (frame 3, day0)  == DAY-0 CELL C
    #            -> node 8 (frame 4, C's child)
    #                -> node 9 (frame 5, C's grandchild)
    df = pd.DataFrame({
        'node_id':      [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'pred_node_id': [0, 1, 1, 2, 2, 3, 5, 6, 8],
        'frame':        [1, 2, 2, 3, 3, 3, 4, 4, 5],
    })
    # day0 node_ids present at frame 3, ascending: 4, 5, 6 -> rows 0, 1, 2
    class_data = np.array([[1, 1], [2, 2], [3, 3]])  # A=dying, B=persisting, C=proliferator

    with patch('pipeline.append_labels.loadmat', side_effect=fake_loadmat(class_data)):
        result = append_labels(df, '/fake/path.mat', day0_frame=3)

    def label_of(node_id):
        row = result.loc[result['node_id'] == node_id].iloc[0]
        return int(row['m_label'])

    assert label_of(4) == 1,  "day-0 cell A should get row 0's label"
    assert label_of(5) == 2,  "day-0 cell B should get row 1's label, independent of sibling A"
    assert label_of(6) == 3,  "day-0 cell C should get row 2's label, independent of A and B"
    assert label_of(7) == 2,  "B's child must inherit B's label, not A's or C's"
    assert label_of(8) == 3,  "C's child must inherit C's label"
    assert label_of(9) == 3,  "C's grandchild must also inherit C's label (multi-generation propagation)"

    # pre-day0 cells (the shared ancestor and its first division) get NaN -
    # no single label could correctly apply to a cell with 3 differently
    # labeled day-0 descendants
    assert result.loc[result['node_id'] == 1, 'm_label'].isna().all()
    assert result.loc[result['node_id'] == 2, 'm_label'].isna().all()
    assert result.loc[result['node_id'] == 3, 'm_label'].isna().all()

    print("PASS: siblings from a pre-day0 division each get their own independent label and propagate it correctly")


def test_lineage_that_dies_before_day0_gets_nan_throughout():
    # lineage rooted at node 1 never reaches frame 3 (dies after frame 2) -
    # it contributes no day-0 cell and gets NaN on every one of its rows.
    # lineage rooted at node 10 does reach day0 (frame 3) normally.
    df = pd.DataFrame({
        'node_id':      [1, 2, 10, 11],
        'pred_node_id': [0, 1, 0, 10],
        'frame':        [1, 2, 1, 3],
    })
    class_data = np.array([[5, 5]])  # exactly one day0 cell: node 11

    with patch('pipeline.append_labels.loadmat', side_effect=fake_loadmat(class_data)):
        result = append_labels(df, '/fake/path.mat', day0_frame=3)

    assert result.loc[result['node_id'].isin([1, 2]), 'm_label'].isna().all()
    assert list(result.loc[result['node_id'] == 11, 'm_label']) == [5]
    print("PASS: a lineage that dies before day 0 gets NaN throughout, without affecting surviving lineages")


def test_day0_frame_mismatch_raises_clearly():
    df = pd.DataFrame({
        'node_id':      [1, 2, 3],
        'pred_node_id': [0, 1, 0],
        'frame':        [1, 2, 3],   # only node 3 is present at frame 3
    })
    class_data = np.array([[1, 1], [2, 2], [3, 3]])  # 3 rows, but only 1 cell at frame 3

    with patch('pipeline.append_labels.loadmat', side_effect=fake_loadmat(class_data)):
        try:
            append_labels(df, '/fake/path.mat', day0_frame=3)
        except AssertionError as e:
            assert 'does not exactly match' in str(e)
            assert 'day0_frame' in str(e)
            print("PASS: a wrong day0_frame raises clearly instead of silently mis-indexing labels")
            return
    raise AssertionError("expected append_labels to raise when day0_frame doesn't match the data")


if __name__ == '__main__':
    test_day0_frame_1_single_cell_per_lineage_labels_every_descendant()
    test_raises_when_class_data_too_short()
    test_raises_when_class_data_too_long()
    test_siblings_from_a_pre_day0_division_get_independent_labels()
    test_lineage_that_dies_before_day0_gets_nan_throughout()
    test_day0_frame_mismatch_raises_clearly()
    print("\nAll tests passed.")
