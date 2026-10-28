'''
Run directly: python3 tests/process_imaging_data/test_discover_raw_data.py
'''
import os
import sys
import shutil
import tempfile

ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'process_imaging_data')
sys.path.insert(0, ANALYSIS_DIR)
from pipeline.discover_raw_data import discover_tracks


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'w').close()


def test_discovers_tracks_by_folder_number_not_filename_number():
    # the whole point of discover_tracks: the number inside the filename
    # (e.g. track_table_99.mat) must NOT be trusted - only the Cropped_XX
    # folder's own number is used as the track number
    tmp_dir = tempfile.mkdtemp()
    try:
        _touch(os.path.join(tmp_dir, 'Cropped_1', 'track_table_99.mat'))
        _touch(os.path.join(tmp_dir, 'Cropped_1', 'segmentation_info_42.mat'))
        _touch(os.path.join(tmp_dir, 'Cropped_2', 'track_table_07.mat'))
        _touch(os.path.join(tmp_dir, 'Cropped_2', 'segmentation_info_13.mat'))

        track_table_files, segmentation_info_files = discover_tracks(tmp_dir)

        assert set(track_table_files.keys()) == {1, 2}
        assert set(segmentation_info_files.keys()) == {1, 2}
        assert track_table_files[1].endswith('track_table_99.mat')
        assert track_table_files[2].endswith('track_table_07.mat')
        assert segmentation_info_files[1].endswith('segmentation_info_42.mat')
        assert segmentation_info_files[2].endswith('segmentation_info_13.mat')

        print("PASS: track numbers come from Cropped_XX folder names, not filenames")
    finally:
        shutil.rmtree(tmp_dir)


def test_skips_non_numbered_cropped_folders():
    # a scratch/non-track folder like Cropped_matlab_only can sit alongside
    # the real numbered ones (confirmed on real Quest data) - it must be
    # silently skipped, not treated as an error
    tmp_dir = tempfile.mkdtemp()
    try:
        _touch(os.path.join(tmp_dir, 'Cropped_1', 'track_table_01.mat'))
        _touch(os.path.join(tmp_dir, 'Cropped_1', 'segmentation_info_01.mat'))
        _touch(os.path.join(tmp_dir, 'Cropped_matlab_only', 'some_other_file.mat'))

        track_table_files, segmentation_info_files = discover_tracks(tmp_dir)

        assert set(track_table_files.keys()) == {1}
        assert set(segmentation_info_files.keys()) == {1}

        print("PASS: non-numbered Cropped_* folders (e.g. Cropped_matlab_only) are skipped, not an error")
    finally:
        shutil.rmtree(tmp_dir)


def test_raises_on_missing_file_in_a_cropped_folder():
    tmp_dir = tempfile.mkdtemp()
    try:
        _touch(os.path.join(tmp_dir, 'Cropped_1', 'track_table_01.mat'))
        # no segmentation_info_*.mat in Cropped_1

        try:
            discover_tracks(tmp_dir)
        except AssertionError as e:
            assert 'segmentation_info' in str(e)
            print("PASS: missing file in a Cropped_XX folder raises clearly")
            return
        raise AssertionError("expected discover_tracks to raise on a missing file")
    finally:
        shutil.rmtree(tmp_dir)


def test_raises_on_duplicate_file_in_a_cropped_folder():
    tmp_dir = tempfile.mkdtemp()
    try:
        _touch(os.path.join(tmp_dir, 'Cropped_1', 'track_table_01.mat'))
        _touch(os.path.join(tmp_dir, 'Cropped_1', 'track_table_02.mat'))
        _touch(os.path.join(tmp_dir, 'Cropped_1', 'segmentation_info_01.mat'))

        try:
            discover_tracks(tmp_dir)
        except AssertionError as e:
            assert 'track_table' in str(e)
            print("PASS: duplicate file in a Cropped_XX folder raises clearly")
            return
        raise AssertionError("expected discover_tracks to raise on duplicate files")
    finally:
        shutil.rmtree(tmp_dir)


def test_raises_on_no_cropped_folders():
    tmp_dir = tempfile.mkdtemp()
    try:
        try:
            discover_tracks(tmp_dir)
        except AssertionError as e:
            assert 'Cropped' in str(e)
            print("PASS: no Cropped_XX folders raises clearly")
            return
        raise AssertionError("expected discover_tracks to raise when no Cropped_XX folders exist")
    finally:
        shutil.rmtree(tmp_dir)


def test_raises_when_only_non_numbered_cropped_folders_exist():
    # a Cropped_matlab_only folder alone (no real numbered tracks) must
    # still raise - it's not itself "no data found", it's "no USABLE data
    # found", and should fail just as loudly
    tmp_dir = tempfile.mkdtemp()
    try:
        _touch(os.path.join(tmp_dir, 'Cropped_matlab_only', 'some_other_file.mat'))

        try:
            discover_tracks(tmp_dir)
        except AssertionError as e:
            assert 'Cropped' in str(e)
            print("PASS: only-non-numbered-folders case still raises clearly")
            return
        raise AssertionError("expected discover_tracks to raise when no numbered Cropped_XX folders exist")
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    test_discovers_tracks_by_folder_number_not_filename_number()
    test_skips_non_numbered_cropped_folders()
    test_raises_on_missing_file_in_a_cropped_folder()
    test_raises_on_duplicate_file_in_a_cropped_folder()
    test_raises_on_no_cropped_folders()
    test_raises_when_only_non_numbered_cropped_folders_exist()
    print("\nAll tests passed.")
