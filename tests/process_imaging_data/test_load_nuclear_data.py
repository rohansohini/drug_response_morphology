'''
Run directly: python3 tests/process_imaging_data/test_load_nuclear_data.py
'''
import os
import sys
from unittest.mock import patch

import numpy as np

ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'process_imaging_data')
sys.path.insert(0, ANALYSIS_DIR)
from pipeline.load_nuclear_data import load_nuclear_data


def make_cluster(centroids, pixel_lists):
    '''
    A single frame's worth of real nucleus data, matching the exact
    in-memory shape scipy.io.loadmat hands back for a real MATLAB
    struct with Centroid/PixelIdxList cell-array fields (confirmed
    directly - NOT round-tripped through savemat, which doesn't
    reliably reproduce this nested cell-in-struct shape from Python).

    Inputs:
        centroids:   list of (x, y) tuples, one per nucleus
        pixel_lists: list of 1D int sequences (linear pixel indices),
                     one per nucleus, same order as centroids
    Outputs:
        cluster: numpy.void record with dtype.names == ('Centroid', 'PixelIdxList')
    '''
    dt = np.dtype([('Centroid', object), ('PixelIdxList', object)])
    cluster = np.zeros((1,), dtype=dt)[0]

    centroid_cell = np.empty((1, len(centroids)), dtype=object)
    for i, (x, y) in enumerate(centroids):
        centroid_cell[0, i] = np.array([[float(x), float(y)]])
    cluster['Centroid'] = centroid_cell

    pixel_cell = np.empty((1, len(pixel_lists)), dtype=object)
    for i, idxs in enumerate(pixel_lists):
        pixel_cell[0, i] = np.array(idxs).reshape(-1, 1)
    cluster['PixelIdxList'] = pixel_cell

    return cluster


def make_empty_frame():
    '''A frame with zero nuclei detected, saved as [] (not a struct) - the real failure mode.'''
    return np.array([], dtype=np.float64).reshape(0, 0)


def fake_loadmat(clusters):
    def _loadmat(path):
        return {'clusters': clusters}
    return _loadmat


def test_loads_a_normal_frame_with_multiple_nuclei():
    cluster = make_cluster(
        centroids=[(10.0, 20.0), (30.0, 40.0)],
        pixel_lists=[[1, 2, 3], [4, 5]],
    )
    clusters = np.empty((1, 1), dtype=object)
    clusters[0, 0] = cluster

    with patch('pipeline.load_nuclear_data.loadmat', side_effect=fake_loadmat(clusters)):
        df = load_nuclear_data({1: '/fake/seg_info.mat'}, img_size=100)

    assert len(df) == 2
    assert list(df['frame']) == [1, 1]
    assert list(df['track']) == [1, 1]
    # centroid converted from 1-indexed (MATLAB) to 0-indexed (Python): -1
    assert list(df['x_coord']) == [9.0, 29.0]
    assert list(df['y_coord']) == [19.0, 39.0]
    print("PASS: a normal frame with multiple nuclei loads correctly, one row per nucleus")


def test_a_frame_saved_as_empty_matrix_is_skipped_not_a_crash():
    # regression test: a frame with zero nuclei detected for a track is
    # saved as [] (not a struct) - cluster['PixelIdxList'] (string field
    # access) used to raise IndexError on this, since [] has no named
    # fields at all
    normal = make_cluster(centroids=[(10.0, 20.0)], pixel_lists=[[1, 2]])
    clusters = np.empty((2, 1), dtype=object)
    clusters[0, 0] = normal
    clusters[1, 0] = make_empty_frame()

    with patch('pipeline.load_nuclear_data.loadmat', side_effect=fake_loadmat(clusters)):
        df = load_nuclear_data({1: '/fake/seg_info.mat'}, img_size=100)

    assert len(df) == 1, "only the normal frame (frame 1) should contribute a row"
    assert list(df['frame']) == [1]
    print("PASS: a frame saved as an empty matrix (zero nuclei) is skipped gracefully, not a crash")


def test_a_struct_with_an_empty_pixel_idx_list_is_also_skipped():
    # the same "zero nuclei this frame" situation, serialized the OTHER
    # way: an actual struct whose PixelIdxList cell array is itself
    # empty (shape (1, 0)), rather than the whole frame being []
    dt = np.dtype([('Centroid', object), ('PixelIdxList', object)])
    empty_struct_cluster = np.zeros((1,), dtype=dt)[0]
    empty_struct_cluster['Centroid'] = np.empty((1, 0), dtype=object)
    empty_struct_cluster['PixelIdxList'] = np.empty((1, 0), dtype=object)

    normal = make_cluster(centroids=[(10.0, 20.0)], pixel_lists=[[1, 2]])
    clusters = np.empty((2, 1), dtype=object)
    clusters[0, 0] = normal
    clusters[1, 0] = empty_struct_cluster

    with patch('pipeline.load_nuclear_data.loadmat', side_effect=fake_loadmat(clusters)):
        df = load_nuclear_data({1: '/fake/seg_info.mat'}, img_size=100)

    assert len(df) == 1, "only the normal frame (frame 1) should contribute a row"
    print("PASS: a struct with an empty PixelIdxList cell array is also skipped, same as a fully-empty frame")


def test_raises_clearly_when_every_frame_is_empty():
    clusters = np.empty((2, 1), dtype=object)
    clusters[0, 0] = make_empty_frame()
    clusters[1, 0] = make_empty_frame()

    with patch('pipeline.load_nuclear_data.loadmat', side_effect=fake_loadmat(clusters)):
        try:
            load_nuclear_data({1: '/fake/seg_info.mat'}, img_size=100)
        except AssertionError as e:
            assert 'zero nuclei' in str(e)
            print("PASS: every frame being empty raises clearly instead of crashing in pd.concat")
            return
    raise AssertionError("expected load_nuclear_data to raise when every frame across every track is empty")


if __name__ == '__main__':
    test_loads_a_normal_frame_with_multiple_nuclei()
    test_a_frame_saved_as_empty_matrix_is_skipped_not_a_crash()
    test_a_struct_with_an_empty_pixel_idx_list_is_also_skipped()
    test_raises_clearly_when_every_frame_is_empty()
    print("\nAll tests passed.")
