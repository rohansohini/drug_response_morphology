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
    df = pd.DataFrame({'lin_id': [1, 2, 3]})
    class_data = np.array([[3, 1], [1, 2], [3, 5]])  # exactly 3 rows, matching max(lin_id)=3

    with patch('pipeline.append_labels.loadmat', side_effect=fake_loadmat(class_data)):
        result = append_labels(df, '/fake/path.mat')

    assert list(result['m_label']) == [3, 1, 3]
    assert list(result['k_label']) == [1, 2, 5]
    print("PASS: exact length match succeeds and labels map correctly")


def test_raises_when_class_data_too_short():
    df = pd.DataFrame({'lin_id': [1, 2, 3]})
    class_data = np.array([[3, 1], [1, 2]])  # only 2 rows, but lin_id goes up to 3

    with patch('pipeline.append_labels.loadmat', side_effect=fake_loadmat(class_data)):
        try:
            append_labels(df, '/fake/path.mat')
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
    df = pd.DataFrame({'lin_id': [1, 2, 3]})
    class_data = np.array([[3, 1], [1, 2], [3, 5], [2, 4]])  # 4 rows, but lin_id only goes up to 3

    with patch('pipeline.append_labels.loadmat', side_effect=fake_loadmat(class_data)):
        try:
            append_labels(df, '/fake/path.mat')
        except AssertionError as e:
            assert 'does not exactly match' in str(e)
            print("PASS: too-long cloneClassification now raises too, instead of silently passing")
            return
    raise AssertionError("expected append_labels to raise when cloneClassification is longer than lin_id needs")


if __name__ == '__main__':
    test_exact_length_match_succeeds()
    test_raises_when_class_data_too_short()
    test_raises_when_class_data_too_long()
    print("\nAll tests passed.")
