import glob
import os
import re

###### Output:
#  - track_table_files:       {track_number: path to that track's track_table_*.mat}
#  - segmentation_info_files: {track_number: path to that track's segmentation_info_*.mat}


def _glob_one(dir_path: str, pattern: str) -> str:
    '''
    Glob a directory for exactly one file matching pattern. Raises if zero
    or more than one match, rather than silently picking one.

    Inputs:
        dir_path: directory to search
        pattern:  glob pattern (e.g. 'track_table_*.mat')
    Outputs:
        the single matching file's path
    '''
    matches = sorted(glob.glob(os.path.join(dir_path, pattern)))
    assert len(matches) == 1, (
        f"expected exactly one file matching '{pattern}' in {dir_path}, "
        f"found {len(matches)}: {matches}"
    )
    return matches[0]


def discover_tracks(drug_dir: str) -> tuple[dict[int, str], dict[int, str]]:
    '''
    Discover each track's track_table and segmentation_info .mat files from
    the Cropped_XX/ folder layout.

    The XX in a Cropped_XX folder name is the only reliable track number -
    the numbers embedded in the track_table_*.mat and segmentation_info_*.mat
    filenames inside that folder do not reliably match the folder's own
    number, so each file is found by globbing for its pattern within the
    folder (expecting exactly one match) rather than by constructing an
    exact filename from the track number.

    Inputs:
        drug_dir: path to one drug's raw data root (contains Cropped_XX/ folders)
    Outputs:
        track_table_files:       {track_number: path to track_table_*.mat}
        segmentation_info_files: {track_number: path to segmentation_info_*.mat}
    '''
    cropped_dirs = sorted(glob.glob(os.path.join(drug_dir, 'Cropped_*')))
    assert cropped_dirs, f"no Cropped_XX folders found under {drug_dir}"

    track_table_files       = {}
    segmentation_info_files = {}

    for cropped_dir in cropped_dirs:
        match = re.match(r'Cropped_(\d+)$', os.path.basename(cropped_dir))
        assert match, f"folder name doesn't match 'Cropped_XX': {cropped_dir}"
        track = int(match.group(1))

        track_table_files[track]       = _glob_one(cropped_dir, 'track_table_*.mat')
        segmentation_info_files[track] = _glob_one(cropped_dir, 'segmentation_info_*.mat')

    return track_table_files, segmentation_info_files
