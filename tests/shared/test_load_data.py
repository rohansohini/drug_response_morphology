'''
Run directly: python3 tests/shared/test_load_data.py
'''
import os
import sys

import shutil
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.load_data import parse_label, build_dataset, match_labels_to_metadata, export_plot_data


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
    assert np.isclose(naive_row['dim_0'], 0.1)

    print("PASS: build_dataset merges embedding coordinates with metadata correctly, into generic dim_0/dim_1 columns")


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


def test_build_dataset_raises_on_duplicate_metadata_key():
    # regression test: a duplicate (condition, frame, pixel_id) key in
    # `selected` would otherwise silently fan out rows in the merge
    # (confirmed directly: a left-merge against a non-unique key
    # duplicates the embedded cell's row with no NaN introduced, so the
    # unmatched-cell check above does NOT catch this class of bug)
    labels = ['naive_frame_1_pixel_1']
    embedding = np.array([[0.1, 0.2]])
    selected = pd.DataFrame({
        'condition': ['naive', 'naive'],       # same key twice
        'frame':     [1, 1],
        'pixel_id':  [1, 1],
        'subclass':  ['fast_prolif', 'dying'],
        'm_label':   [3, 1],
        'k_label':   [4, 2],
    })

    try:
        build_dataset(labels, embedding, selected)
    except AssertionError as e:
        assert 'duplicate' in str(e).lower()
        print("PASS: build_dataset raises clearly on a duplicate metadata key, instead of silently duplicating rows")
        return
    raise AssertionError("expected build_dataset to raise on a duplicate (condition, frame, pixel_id) key")


def test_match_labels_to_metadata_works_without_a_subclass_column():
    # regression test: match_labels_to_metadata used to detect unmatched
    # rows by checking the 'subclass' column for NaN, which breaks for
    # any analysis whose `selected` has no subclass at all (e.g.
    # vem_48hr_pretreatment's frame-1-only metadata, since
    # process_imaging_data doesn't classify frame-1 cells) - must work
    # (and still correctly raise on a genuinely unmatched cell) regardless
    labels = ['naive_frame_1_pixel_1', 'naive_frame_1_pixel_999']  # second has no match
    selected = pd.DataFrame({
        'condition': ['naive'], 'frame': [1], 'pixel_id': [1], 'lin_id': [7],
    })

    try:
        match_labels_to_metadata(labels, selected)
    except ValueError as e:
        assert 'unmatched' in str(e).lower() or 'no matching' in str(e).lower()
        print("PASS: match_labels_to_metadata works (and still raises on unmatched cells) with no subclass column")
        return
    raise AssertionError("expected match_labels_to_metadata to raise on the unmatched cell")


def test_match_labels_to_metadata_carries_through_every_selected_column():
    # regression test: match_labels_to_metadata used to hardcode a 6-column
    # subset (condition, frame, pixel_id, subclass, m_label, k_label),
    # silently dropping anything else `selected` had (e.g. lin_id, needed
    # by vem_48hr_pretreatment's frame-1 fate-mixture analysis to resolve
    # lineage_frame3_to_frame1.mat matches) - every column must now pass through
    labels = ['naive_frame_1_pixel_1']
    selected = pd.DataFrame({
        'condition': ['naive'], 'frame': [1], 'pixel_id': [1],
        'subclass': ['fast_prolif'], 'm_label': [3], 'k_label': [4],
        'lin_id': [42], 'node_id': [7],
    })

    df = match_labels_to_metadata(labels, selected)

    assert df.iloc[0]['lin_id'] == 42
    assert df.iloc[0]['node_id'] == 7
    print("PASS: match_labels_to_metadata carries through every column `selected` has, not just a hardcoded subset")


def test_export_plot_data_adds_modality_and_embedding_method_columns():
    dfs = {
        ('cell', 'umap'): pd.DataFrame({'dim_0': [0.1], 'dim_1': [0.2], 'condition': ['naive']}),
        ('cell', 'phate'): pd.DataFrame({'dim_0': [0.3], 'dim_1': [0.4], 'condition': ['vem_1uM']}),
        ('nuc', 'umap'): pd.DataFrame({'dim_0': [0.5], 'dim_1': [0.6], 'condition': ['naive']}),
    }

    tmpdir = tempfile.mkdtemp()
    try:
        output_path = os.path.join(tmpdir, 'plot_data.csv')
        combined = export_plot_data(dfs, output_path)

        assert len(combined) == 3
        assert set(combined['modality']) == {'cell', 'nuc'}
        assert set(combined['embedding_method']) == {'umap', 'phate'}

        on_disk = pd.read_csv(output_path)
        assert len(on_disk) == 3
        assert list(on_disk.columns[:2]) == ['modality', 'embedding_method']
        print("PASS: export_plot_data combines every (modality, embedding_method) df with those columns added, and saves to disk")
    finally:
        shutil.rmtree(tmpdir)


def test_export_plot_data_creates_missing_output_directory():
    dfs = {('cell', 'umap'): pd.DataFrame({'dim_0': [0.1], 'dim_1': [0.2]})}

    tmpdir = tempfile.mkdtemp()
    try:
        output_path = os.path.join(tmpdir, 'nested', 'dir', 'plot_data.csv')
        export_plot_data(dfs, output_path)
        assert os.path.exists(output_path)
        print("PASS: export_plot_data creates a missing output directory rather than erroring")
    finally:
        shutil.rmtree(tmpdir)


if __name__ == '__main__':
    test_parse_label_handles_underscores_in_condition_name()
    test_build_dataset_merges_correctly()
    test_build_dataset_raises_on_unmatched_cell()
    test_build_dataset_raises_on_duplicate_metadata_key()
    test_match_labels_to_metadata_works_without_a_subclass_column()
    test_match_labels_to_metadata_carries_through_every_selected_column()
    test_export_plot_data_adds_modality_and_embedding_method_columns()
    test_export_plot_data_creates_missing_output_directory()
    print("\nAll tests passed.")
