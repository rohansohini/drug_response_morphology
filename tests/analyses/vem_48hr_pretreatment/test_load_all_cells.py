'''
Run directly: python3 tests/analyses/vem_48hr_pretreatment/test_load_all_cells.py
'''
import os
import sys
import tempfile

import pandas as pd

ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'analyses', 'vem_48hr_pretreatment')
sys.path.insert(0, ANALYSIS_DIR)
from load_all_cells import load_all_cells


def make_metadata_csv(path):
    rows = [
        # frame, has_mask, m_label, k_label -> everything else is incidental
        (1, True,  3, 4),   # frame 1, fast_prolif - kept
        (2, True,  1, 1),   # frame 2, dying - kept
        (3, True,  2, 1),   # frame 3, persisting - kept
        (3, False, 3, 4),   # frame 3 but has_mask=False - dropped
        (14, True, 3, 4),   # frame 14, outside [1,2,3] - dropped
    ]
    df = pd.DataFrame(rows, columns=['frame', 'has_mask', 'm_label', 'k_label'])
    df.to_csv(path, index=False)


def test_restricts_to_frames_and_has_mask_true():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, 'vem_1uM_48hrsbefore_metadata.csv')
        make_metadata_csv(csv_path)

        result = load_all_cells(csv_path, frames=[1, 2, 3], drug='vem_1uM_48hrsbefore')

        assert len(result) == 3, f"expected 3 rows (has_mask=True, frame in 1-3), got {len(result)}"
        assert set(result['frame']) == {1, 2, 3}
        print("PASS: load_all_cells restricts to has_mask=True rows within the given frames")


def test_adds_condition_subclass_sampled_columns():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, 'vem_1uM_48hrsbefore_metadata.csv')
        make_metadata_csv(csv_path)

        result = load_all_cells(csv_path, frames=[1, 2, 3], drug='vem_1uM_48hrsbefore')

        assert (result['condition'] == 'vem_1uM_48hrsbefore').all(), \
            "every row's condition must be the drug name itself (single-dataset analysis)"
        assert result['sampled'].all(), "every available cell must be sampled=True - no budget here"

        subclass_by_frame = dict(zip(result['frame'], result['subclass']))
        assert subclass_by_frame[1] == 'fast_prolif'
        assert subclass_by_frame[2] == 'dying'
        assert subclass_by_frame[3] == 'persisting'
        print("PASS: load_all_cells adds condition/subclass/sampled columns correctly")


if __name__ == '__main__':
    test_restricts_to_frames_and_has_mask_true()
    test_adds_condition_subclass_sampled_columns()
    print("\nAll tests passed.")
