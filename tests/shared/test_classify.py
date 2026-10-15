'''
Run directly: python3 tests/shared/test_classify.py
'''
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.classify import classify_subclass

# the scheme itself now lives in each analysis's own config.py (not a
# shared default) - these tests exercise classify_subclass() against the
# same scheme every analysis currently uses
DYING_M_LABEL, PERSISTING_M_LABEL, PROLIF_M_LABEL = 1, 2, 3
SLOW_K_LABELS, FAST_K_LABELS = [1, 2], [3, 4, 5, 6]


def test_all_eighteen_combined_labels_map_somewhere():
    rows = []
    for m in [1, 2, 3]:
        for k in [1, 2, 3, 4, 5, 6]:
            rows.append({'m_label': m, 'k_label': k})
    df = pd.DataFrame(rows)

    result = classify_subclass(df, DYING_M_LABEL, PERSISTING_M_LABEL, PROLIF_M_LABEL, SLOW_K_LABELS, FAST_K_LABELS)
    assert not result.isna().any()

    expected = {
        (1, 1): 'dying', (1, 2): 'dying', (1, 3): 'dying', (1, 4): 'dying', (1, 5): 'dying', (1, 6): 'dying',
        (2, 1): 'persisting', (2, 2): 'persisting', (2, 3): 'persisting', (2, 4): 'persisting', (2, 5): 'persisting', (2, 6): 'persisting',
        (3, 1): 'slow_prolif', (3, 2): 'slow_prolif',
        (3, 3): 'fast_prolif', (3, 4): 'fast_prolif', (3, 5): 'fast_prolif', (3, 6): 'fast_prolif',
    }
    for (m, k), expected_label in expected.items():
        actual = result[(df['m_label'] == m) & (df['k_label'] == k)].iloc[0]
        assert actual == expected_label, f"m_label={m}, k_label={k}: expected {expected_label}, got {actual}"

    print("PASS: all 18 (m_label, k_label) combinations map to the correct subclass")


def test_unmapped_combination_raises():
    df = pd.DataFrame({'m_label': [4], 'k_label': [1]})  # m_label=4 doesn't exist
    try:
        classify_subclass(df, DYING_M_LABEL, PERSISTING_M_LABEL, PROLIF_M_LABEL, SLOW_K_LABELS, FAST_K_LABELS)
    except ValueError as e:
        assert 'm_label' in str(e)
        print("PASS: an (m_label, k_label) combo outside the known scheme raises clearly, not silently")
        return
    raise AssertionError("expected classify_subclass to raise on an unmapped combination")


if __name__ == '__main__':
    test_all_eighteen_combined_labels_map_somewhere()
    test_unmapped_combination_raises()
    print("\nAll tests passed.")
