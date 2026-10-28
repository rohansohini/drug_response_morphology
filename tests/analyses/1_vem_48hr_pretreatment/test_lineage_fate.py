'''
Run directly: python3 tests/analyses/1_vem_48hr_pretreatment/test_lineage_fate.py
'''
import os
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'analyses', '1_vem_48hr_pretreatment')
sys.path.insert(0, ANALYSIS_DIR)
import config

# lineage_fate.py resolves config.PROCESS_IMAGING_DATA_ROOT onto sys.path
# at import time to reuse process_imaging_data's own pipeline code - point
# it at the real nested location instead of config.py's placeholder path,
# so `import lineage_fate` below actually finds `pipeline.*`
config.PROCESS_IMAGING_DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'process_imaging_data')

import lineage_fate


def write_track_table_mat(path, rows):
    '''rows: list of (node_id, frame, _, pred_node_id, x, y, _, _, _, lin_id) - only the columns
    pipeline.track_table.load_track actually reads are populated meaningfully; the rest are 0.'''
    import scipy.io as sio
    arr = np.zeros((len(rows), 10))
    for i, (node_id, frame, pred_node_id, x, y, lin_id) in enumerate(rows):
        arr[i, 0] = node_id
        arr[i, 1] = frame
        arr[i, 3] = pred_node_id
        arr[i, 4] = x
        arr[i, 5] = y
        arr[i, 9] = lin_id
    sio.savemat(path, {'track_table': arr})


def test_build_raw_to_global_lin_id_matches_real_make_master_ids_offsetting():
    # two tracks, each with 2 raw lineages (raw lin_id 1, 2 in both tracks,
    # since raw lin_id is only unique WITHIN a track) - track 2's lineages
    # must get a different global lin_id than track 1's, even though their
    # raw lin_id values collide
    with tempfile.TemporaryDirectory() as drug_dir:
        track1_dir = os.path.join(drug_dir, 'Cropped_1')
        track2_dir = os.path.join(drug_dir, 'Cropped_2')
        os.makedirs(track1_dir)
        os.makedirs(track2_dir)

        # track 1: lineage 1 (node 1, frame 1), lineage 2 (node 2, frame 1)
        write_track_table_mat(os.path.join(track1_dir, 'track_table_1.mat'), [
            (1, 1, 0, 10, 10, 1),
            (2, 1, 0, 20, 20, 2),
        ])
        sio_seg = os.path.join(track1_dir, 'segmentation_info_1.mat')
        import scipy.io as sio
        sio.savemat(sio_seg, {'segmentation_info': np.zeros((1, 1))})

        # track 2: lineage 1 (node 1, frame 1), lineage 2 (node 2, frame 1) - same raw IDs as track 1
        write_track_table_mat(os.path.join(track2_dir, 'track_table_2.mat'), [
            (1, 1, 0, 30, 30, 1),
            (2, 1, 0, 40, 40, 2),
        ])
        sio.savemat(os.path.join(track2_dir, 'segmentation_info_2.mat'), {'segmentation_info': np.zeros((1, 1))})

        raw_to_global = lineage_fate.build_raw_to_global_lin_id(drug_dir)

        g11 = raw_to_global[(1, 1)]
        g12 = raw_to_global[(1, 2)]
        g21 = raw_to_global[(2, 1)]
        g22 = raw_to_global[(2, 2)]

        assert len({g11, g12, g21, g22}) == 4, (
            f"expected 4 distinct global lin_ids (one per raw lineage across both tracks), got "
            f"{{(1,1):{g11}, (1,2):{g12}, (2,1):{g21}, (2,2):{g22}}}"
        )
        print("PASS: build_raw_to_global_lin_id resolves (track, raw lin_id) using real make_master_ids offsetting")


def make_fake_lineage_and_clone_files(tmp_dir, lineage_rows, class_rows):
    import scipy.io as sio
    lineage_path = os.path.join(tmp_dir, 'lineage_frame3_to_frame1.mat')
    clone_path   = os.path.join(tmp_dir, 'clone_kmeans_id.mat')
    sio.savemat(lineage_path, {'lineage_table': np.array(lineage_rows, dtype=float)})
    sio.savemat(clone_path, {'cloneClassification': np.array(class_rows, dtype=float)})
    return lineage_path, clone_path


def test_compute_frame1_fate_percentages_single_descendant_is_100_percent():
    with tempfile.TemporaryDirectory() as tmp:
        # one frame-1 lineage (file_idx=1, old_clone_id_frame1=5) -> exactly
        # one day-0 cell (new_clone_id_global=1), which is dying (m=1,k=1)
        lineage_path, clone_path = make_fake_lineage_and_clone_files(
            tmp, [[1, 1, 5]], [[1, 1]],
        )
        raw_to_global = {(1, 5): 42}

        result = lineage_fate.compute_frame1_fate_percentages(lineage_path, clone_path, raw_to_global)

        assert len(result) == 1
        row = result.iloc[0]
        assert row['lin_id'] == 42
        assert row['n_day0_descendants'] == 1
        assert row['pct_dying'] == 1.0
        assert row['pct_persisting'] == 0.0 and row['pct_slow_prolif'] == 0.0 and row['pct_fast_prolif'] == 0.0
        print("PASS: a frame-1 lineage with a single day-0 descendant gets 100% in that descendant's subclass")


def test_compute_frame1_fate_percentages_mixed_siblings():
    with tempfile.TemporaryDirectory() as tmp:
        # frame-1 lineage (file_idx=2, old_clone_id_frame1=9) has 4 day-0
        # descendants: 1 dying, 1 persisting, 2 fast_prolif
        lineage_path, clone_path = make_fake_lineage_and_clone_files(
            tmp,
            [[2, 1, 9], [2, 2, 9], [2, 3, 9], [2, 4, 9]],
            [[1, 1], [2, 1], [3, 4], [3, 3]],   # dying, persisting, fast_prolif, fast_prolif
        )
        raw_to_global = {(2, 9): 7}

        result = lineage_fate.compute_frame1_fate_percentages(lineage_path, clone_path, raw_to_global)

        assert len(result) == 1
        row = result.iloc[0]
        assert row['n_day0_descendants'] == 4
        assert row['pct_dying'] == 0.25
        assert row['pct_persisting'] == 0.25
        assert row['pct_slow_prolif'] == 0.0
        assert row['pct_fast_prolif'] == 0.5
        print("PASS: a frame-1 lineage's mixed day-0 descendants produce the correct percentage breakdown")


def test_compute_frame1_fate_percentages_raises_on_unresolved_lineage():
    with tempfile.TemporaryDirectory() as tmp:
        lineage_path, clone_path = make_fake_lineage_and_clone_files(tmp, [[1, 1, 99]], [[1, 1]])
        raw_to_global = {}  # (1, 99) is not in here

        try:
            lineage_fate.compute_frame1_fate_percentages(lineage_path, clone_path, raw_to_global)
        except AssertionError as e:
            assert 'no matching' in str(e)
            print("PASS: an unresolved (file_idx, old_clone_id_frame1) raises clearly")
            return
        raise AssertionError("expected compute_frame1_fate_percentages to raise on an unresolved lineage")


def test_compute_frame1_fate_percentages_raises_on_out_of_range_new_clone_id():
    with tempfile.TemporaryDirectory() as tmp:
        # new_clone_id_global=5 but cloneClassification only has 1 row
        lineage_path, clone_path = make_fake_lineage_and_clone_files(tmp, [[1, 5, 1]], [[1, 1]])
        raw_to_global = {(1, 1): 10}

        try:
            lineage_fate.compute_frame1_fate_percentages(lineage_path, clone_path, raw_to_global)
        except AssertionError as e:
            assert 'new_clone_id_global' in str(e)
            print("PASS: a new_clone_id_global outside cloneClassification's row range raises clearly")
            return
        raise AssertionError("expected compute_frame1_fate_percentages to raise on an out-of-range new_clone_id_global")


if __name__ == '__main__':
    test_build_raw_to_global_lin_id_matches_real_make_master_ids_offsetting()
    test_compute_frame1_fate_percentages_single_descendant_is_100_percent()
    test_compute_frame1_fate_percentages_mixed_siblings()
    test_compute_frame1_fate_percentages_raises_on_unresolved_lineage()
    test_compute_frame1_fate_percentages_raises_on_out_of_range_new_clone_id()
    print("\nAll tests passed.")
