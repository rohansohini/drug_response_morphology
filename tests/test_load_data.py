'''
Run directly: python3 tests/test_load_data.py
'''
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from helpers.load_data import parse_label, build_dataset


def test_parse_label_handles_underscores_in_condition_name():
    assert parse_label('naive_frame_3_pixel_42') == ('naive', 3, 42)
    assert parse_label('A1_1uM_vem_frame_12_pixel_9') == ('A1_1uM_vem', 12, 9)
    print("PASS: parse_label correctly handles condition names containing underscores")


def test_build_dataset_merges_correctly():
    labels = ['naive_frame_1_pixel_1', 'A1_1uM_vem_frame_1_pixel_2']
    embedding = np.array([[0.1, 0.2], [0.3, 0.4]])
    selected = pd.DataFrame({
        'condition': ['naive', 'A1_1uM_vem'],
        'frame':     [1, 1],
        'pixel_id':  [1, 2],
        'subclass':  ['fast_prolif', 'dying'],
        'm_label':   [3, 1],
        'k_label':   [4, 2],
    })

    df = build_dataset(labels, embedding, selected)

    assert len(df) == 2
    naive_row = df[df['condition'] == 'naive'].iloc[0]
    assert naive_row['subclass'] == 'fast_prolif'
    assert np.isclose(naive_row['umap_0'], 0.1)

    print("PASS: build_dataset merges UMAP coordinates with metadata correctly")


def test_build_dataset_raises_on_unmatched_cell():
    labels = ['naive_frame_1_pixel_999']  # no matching row in `selected`
    embedding = np.array([[0.1, 0.2]])
    selected = pd.DataFrame({
        'condition': ['naive'], 'frame': [1], 'pixel_id': [1],
        'subclass': ['fast_prolif'], 'm_label': [3], 'k_label': [4],
    })

    try:
        build_dataset(labels, embedding, selected)
    except ValueError as e:
        assert 'unmatched' in str(e).lower() or 'no matching' in str(e).lower()
        print("PASS: build_dataset raises clearly when an embedded cell has no metadata match")
        return
    raise AssertionError("expected build_dataset to raise on an unmatched cell")


if __name__ == '__main__':
    test_parse_label_handles_underscores_in_condition_name()
    test_build_dataset_merges_correctly()
    test_build_dataset_raises_on_unmatched_cell()
    print("\nAll tests passed.")
