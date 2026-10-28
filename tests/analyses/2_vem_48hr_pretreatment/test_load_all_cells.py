'''
Run directly: python3 tests/analyses/2_vem_48hr_pretreatment/test_load_all_cells.py
'''
import os
import sys
import tempfile

import numpy as np
import pandas as pd

ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'analyses', '2_vem_48hr_pretreatment')
sys.path.insert(0, ANALYSIS_DIR)
from load_all_cells import load_all_cells


def make_metadata_csv(path):
    # m_label/k_label are NaN for frame 1/2 rows, mirroring real data under
    # the day0-cell classification model (process_imaging_data doesn't
    # classify a cell until day 0) - load_all_cells must not care, since it
    # no longer derives a subclass from them at all
    rows = [
        (1, True,  np.nan, np.nan),   # frame 1 - kept
        (2, True,  np.nan, np.nan),   # frame 2, outside FRAMES=[1] - dropped
        (3, False, np.nan, np.nan),   # frame 3 but has_mask=False - dropped
        (14, True, np.nan, np.nan),   # frame 14, outside FRAMES - dropped
    ]
    df = pd.DataFrame(rows, columns=['frame', 'has_mask', 'm_label', 'k_label'])
    df.to_csv(path, index=False)


def test_restricts_to_frames_and_has_mask_true():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, 'vem_1uM_48hrsbefore_metadata.csv')
        make_metadata_csv(csv_path)

        result = load_all_cells(csv_path, frames=[1], drug='vem_1uM_48hrsbefore')

        assert len(result) == 1, f"expected 1 row (has_mask=True, frame 1), got {len(result)}"
        assert set(result['frame']) == {1}
        print("PASS: load_all_cells restricts to has_mask=True rows within the given frames")


def test_adds_condition_and_sampled_columns_without_subclass():
    # regression test: load_all_cells must NOT derive a subclass - frame-1
    # cells have no m_label/k_label to classify from (classification starts
    # at day 0), so calling classify_subclass on them would raise
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, 'vem_1uM_48hrsbefore_metadata.csv')
        make_metadata_csv(csv_path)

        result = load_all_cells(csv_path, frames=[1], drug='vem_1uM_48hrsbefore')

        assert (result['condition'] == 'vem_1uM_48hrsbefore').all(), \
            "every row's condition must be the drug name itself (single-dataset analysis)"
        assert result['sampled'].all(), "every available cell must be sampled=True - no budget here"
        assert 'subclass' not in result.columns, \
            "load_all_cells must not add a subclass column - frame-1 cells aren't classified"
        print("PASS: load_all_cells adds condition/sampled columns, with no subclass column")


if __name__ == '__main__':
    test_restricts_to_frames_and_has_mask_true()
    test_adds_condition_and_sampled_columns_without_subclass()
    print("\nAll tests passed.")
