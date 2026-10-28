'''
Regression test: track number must come from the caller-supplied mapping
(ultimately the Cropped_XX folder name - see discover_raw_data.py), never
parsed from the .mat filename. The filename's own embedded number is known
to be unreliable, so loadmat() is mocked here with deliberately misleading
filenames to prove the track column reflects the dict key, not the path.

Run directly: python3 tests/test_raw_data_track_numbering.py
'''
import os
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pipeline.track_table import load_track_table
from pipeline.load_nuclear_data import load_nuclear_data


def fake_track_table_mat(file_path):
    # columns: 0=node_id, 1=frame, 3=pred_node_id, 4=x_coord, 5=y_coord, 9=lin_id
    arr = np.zeros((1, 10))
    arr[0] = [1, 1, 0, 0, 10, 10, 0, 0, 0, 1]
    return {'track_table': arr}


def test_load_track_table_uses_dict_key_not_filename():
    file_paths = {
        5: '/data/Cropped_5/track_table_99.mat',   # filename says 99, folder/dict key says 5
        9: '/data/Cropped_9/track_table_02.mat',   # filename says 2, folder/dict key says 9
    }

    with patch('pipeline.track_table.loadmat', side_effect=fake_track_table_mat):
        df = load_track_table(file_paths)

    assert set(df['track']) == {5, 9}, f"expected tracks {{5, 9}}, got {set(df['track'])}"
    print("PASS: load_track_table assigns track numbers from the dict key, ignoring the misleading filename")


def make_fake_cluster(n_nuclei=1):
    centroid   = np.empty((1, n_nuclei), dtype=object)
    pixel_idx  = np.empty((1, n_nuclei), dtype=object)
    for i in range(n_nuclei):
        centroid[0, i]  = np.array([float(i + 1), float(i + 1)])
        pixel_idx[0, i] = np.array([1])
    return {'Centroid': centroid, 'PixelIdxList': pixel_idx}


def fake_nuclear_mat(file_path):
    clusters = np.empty((1, 1), dtype=object)
    clusters[0, 0] = make_fake_cluster()
    return {'clusters': clusters}


def test_load_nuclear_data_uses_dict_key_not_filename():
    segmentation_info_files = {
        5: '/data/Cropped_5/segmentation_info_77.mat',  # filename says 77
        9: '/data/Cropped_9/segmentation_info_03.mat',  # filename says 3
    }

    with patch('pipeline.load_nuclear_data.loadmat', side_effect=fake_nuclear_mat):
        df = load_nuclear_data(segmentation_info_files, img_size=2501)

    assert set(df['track']) == {5, 9}, f"expected tracks {{5, 9}}, got {set(df['track'])}"
    print("PASS: load_nuclear_data assigns track numbers from the dict key, ignoring the misleading filename")


if __name__ == '__main__':
    test_load_track_table_uses_dict_key_not_filename()
    test_load_nuclear_data_uses_dict_key_not_filename()
    print("\nAll tests passed.")
