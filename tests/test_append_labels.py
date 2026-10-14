'''
Run directly: python3 tests/test_append_labels.py
'''
import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pipeline.append_labels import append_labels


def fake_loadmat(class_data):
    def _loadmat(path):
        return {'cloneClassification': class_data}
    return _loadmat


def test_exact_length_match_succeeds():
    # day0_frame=1: every lineage is present at frame 1, so this reduces
    # to the dense lin_id -> row mapping (lin_id i maps to row i-1)
    df = pd.DataFrame({'lin_id': [1, 2, 3], 'frame': [1, 1, 1]})
    class_data = np.array([[3, 1], [1, 2], [3, 5]])  # exactly 3 rows, matching max(lin_id)=3

    with patch('pipeline.append_labels.loadmat', side_effect=fake_loadmat(class_data)):
        result = append_labels(df, '/fake/path.mat', day0_frame=1)

    assert list(result['m_label']) == [3, 1, 3]
    assert list(result['k_label']) == [1, 2, 5]
    print("PASS: exact length match succeeds and labels map correctly (day0_frame=1)")


def test_raises_when_class_data_too_short():
    df = pd.DataFrame({'lin_id': [1, 2, 3], 'frame': [1, 1, 1]})
    class_data = np.array([[3, 1], [1, 2]])  # only 2 rows, but lin_id goes up to 3

    with patch('pipeline.append_labels.loadmat', side_effect=fake_loadmat(class_data)):
        try:
            append_labels(df, '/fake/path.mat', day0_frame=1)
        except AssertionError as e:
            assert 'does not exactly match' in str(e)
            print("PASS: too-short cloneClassification raises clearly")
            return
    raise AssertionError("expected append_labels to raise when cloneClassification is shorter than lin_id needs")


def test_raises_when_class_data_too_long():
    # regression test: this direction previously passed silently (the old
    # check was lin_id.max() <= len(class_data)) - extra unused trailing
    # rows are exactly the kind of symptom that should raise suspicion
    # about lineage-numbering misalignment, not be ignored
    df = pd.DataFrame({'lin_id': [1, 2, 3], 'frame': [1, 1, 1]})
    class_data = np.array([[3, 1], [1, 2], [3, 5], [2, 4]])  # 4 rows, but lin_id only goes up to 3

    with patch('pipeline.append_labels.loadmat', side_effect=fake_loadmat(class_data)):
        try:
            append_labels(df, '/fake/path.mat', day0_frame=1)
        except AssertionError as e:
            assert 'does not exactly match' in str(e)
            print("PASS: too-long cloneClassification now raises too, instead of silently passing")
            return
    raise AssertionError("expected append_labels to raise when cloneClassification is longer than lin_id needs")


def test_day0_frame_after_1_compacts_dead_lineages_out_of_the_row_mapping():
    # lineages 1, 2 die before day0_frame=3 (never appear at frame >= 3);
    # lineages 3, 4, 5 survive to day0_frame and beyond. cloneClassification
    # only has 3 rows (one per surviving lineage) - row 0 is lineage 3's
    # label, row 1 is lineage 4's, row 2 is lineage 5's (compacted/re-ranked
    # by ascending lin_id among survivors, NOT lin_id - 1 directly)
    df = pd.DataFrame({
        'lin_id': [1, 2, 3, 3, 4, 5, 5],
        'frame':  [1, 2, 1, 3, 3, 2, 3],
    })
    class_data = np.array([[1, 1], [2, 2], [3, 3]])  # lineage 3 -> row0, 4 -> row1, 5 -> row2

    with patch('pipeline.append_labels.loadmat', side_effect=fake_loadmat(class_data)):
        result = append_labels(df, '/fake/path.mat', day0_frame=3)

    # lineages 1 and 2 (died before day0) get NaN - no classification exists for them
    assert result.loc[result['lin_id'] == 1, 'm_label'].isna().all()
    assert result.loc[result['lin_id'] == 2, 'm_label'].isna().all()

    # lineage 3 (frame 1 AND frame 3 rows) gets row 0's label on BOTH rows -
    # the pre-day0 row is retroactively labeled with its lineage's fate too
    assert list(result.loc[result['lin_id'] == 3, 'm_label']) == [1, 1]
    assert list(result.loc[result['lin_id'] == 3, 'k_label']) == [1, 1]

    # lineage 4 -> row 1, lineage 5 -> row 2
    assert list(result.loc[result['lin_id'] == 4, 'm_label']) == [2]
    assert list(result.loc[result['lin_id'] == 5, 'm_label']) == [3, 3]

    print("PASS: day0_frame > 1 compacts dead-before-day0 lineages out, labels retroactively applied to pre-day0 rows")


def test_day0_frame_mismatch_raises_clearly():
    # day0_frame=3 is wrong for this data (only lineage 5 actually reaches
    # frame 3), so len(day0_lin_ids)=1 but cloneClassification has 3 rows -
    # must raise rather than silently mis-indexing
    df = pd.DataFrame({
        'lin_id': [1, 2, 5],
        'frame':  [1, 2, 3],
    })
    class_data = np.array([[1, 1], [2, 2], [3, 3]])

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
    test_exact_length_match_succeeds()
    test_raises_when_class_data_too_short()
    test_raises_when_class_data_too_long()
    test_day0_frame_after_1_compacts_dead_lineages_out_of_the_row_mapping()
    test_day0_frame_mismatch_raises_clearly()
    print("\nAll tests passed.")
