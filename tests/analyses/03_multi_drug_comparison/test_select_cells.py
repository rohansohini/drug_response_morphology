'''
Run directly: python3 tests/analyses/03_multi_drug_comparison/test_select_cells.py
'''
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT    = os.path.join(os.path.dirname(__file__), '..', '..', '..')
ANALYSIS_DIR = os.path.join(REPO_ROOT, 'analyses', '03_multi_drug_comparison')
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, ANALYSIS_DIR)
import config
from select_cells import select_cells


def make_metadata(frames, n_per_frame, pixel_id_start, has_mask_fraction=1.0, rng=None):
    rng = rng or np.random.default_rng(0)
    rows = []
    pixel_id = pixel_id_start
    for frame in frames:
        for _ in range(n_per_frame):
            rows.append({'frame': frame, 'pixel_id': pixel_id, 'has_mask': rng.random() < has_mask_fraction})
            pixel_id += 1
    return pd.DataFrame(rows)


def test_samples_up_to_the_configured_size_per_condition():
    rng = np.random.default_rng(0)
    metadata_by_condition = {
        'naive':   make_metadata([1], n_per_frame=5000, pixel_id_start=1, rng=rng),
        'vem_1uM': make_metadata([1], n_per_frame=5000, pixel_id_start=1_000_000, rng=rng),
    }

    original = config.SAMPLE_SIZE_PER_CONDITION
    config.SAMPLE_SIZE_PER_CONDITION = 100
    try:
        selected = select_cells(metadata_by_condition, frame=1, rng=np.random.default_rng(1))
    finally:
        config.SAMPLE_SIZE_PER_CONDITION = original

    assert (selected['condition'] == 'naive').sum() == 100
    assert (selected['condition'] == 'vem_1uM').sum() == 100
    print("PASS: select_cells samples exactly SAMPLE_SIZE_PER_CONDITION per condition when enough are available")


def test_uses_everything_available_when_fewer_than_the_configured_size():
    metadata_by_condition = {
        'naive': make_metadata([1], n_per_frame=30, pixel_id_start=1),
    }

    original = config.SAMPLE_SIZE_PER_CONDITION
    config.SAMPLE_SIZE_PER_CONDITION = 1000
    try:
        selected = select_cells(metadata_by_condition, frame=1, rng=np.random.default_rng(0))
    finally:
        config.SAMPLE_SIZE_PER_CONDITION = original

    assert len(selected) == 30, f"expected all 30 available cells, got {len(selected)}"
    print("PASS: select_cells uses every available cell when there are fewer than SAMPLE_SIZE_PER_CONDITION")


def test_restricts_to_the_requested_frame_and_has_mask():
    df = pd.concat([
        make_metadata([1], n_per_frame=50, pixel_id_start=1, has_mask_fraction=0.5),
        make_metadata([2], n_per_frame=50, pixel_id_start=10_000),
    ], ignore_index=True)

    selected = select_cells({'naive': df}, frame=1, rng=np.random.default_rng(0))

    assert (selected['frame'] == 1).all(), "a cell from a different frame was selected"
    assert selected['has_mask'].all(), "a has_mask=False cell was selected"
    print("PASS: select_cells restricts to the requested frame and has_mask=True cells only")


def test_zero_available_cells_raises_clearly():
    df = pd.DataFrame({'frame': [2], 'pixel_id': [1], 'has_mask': [True]})   # nothing at frame 1

    try:
        select_cells({'naive': df}, frame=1, rng=np.random.default_rng(0))
    except AssertionError as e:
        assert 'naive' in str(e) and 'frame 1' in str(e)
        print("PASS: a condition with no has_mask=True cell at the requested frame raises clearly")
        return
    raise AssertionError("expected select_cells to raise when a condition has nothing to sample from")


def test_sampling_is_independent_across_equal_sized_buckets():
    '''
    Regression test mirroring naive_vs_vem's own: a SHARED rng, not a
    freshly re-seeded one per condition, must give two equal-sized
    buckets different relative draw positions.
    '''
    original = config.SAMPLE_SIZE_PER_CONDITION
    config.SAMPLE_SIZE_PER_CONDITION = 500
    try:
        metadata_by_condition = {
            'naive':   make_metadata([1], n_per_frame=1000, pixel_id_start=0),
            'vem_1uM': make_metadata([1], n_per_frame=1000, pixel_id_start=100_000),
        }
        rng = np.random.default_rng(42)
        selected = select_cells(metadata_by_condition, frame=1, rng=rng)
    finally:
        config.SAMPLE_SIZE_PER_CONDITION = original

    naive_ranks = sorted((selected[selected['condition'] == 'naive']['pixel_id'] - 0).tolist())
    vem_ranks   = sorted((selected[selected['condition'] == 'vem_1uM']['pixel_id'] - 100_000).tolist())

    assert naive_ranks != vem_ranks, (
        "selected relative positions are IDENTICAL across two equal-sized buckets - "
        "sampling is not independent per bucket (same-seed regression)"
    )
    print("PASS: two equal-sized buckets picked different relative positions - sampling is independent per bucket")


if __name__ == '__main__':
    test_samples_up_to_the_configured_size_per_condition()
    test_uses_everything_available_when_fewer_than_the_configured_size()
    test_restricts_to_the_requested_frame_and_has_mask()
    test_zero_available_cells_raises_clearly()
    test_sampling_is_independent_across_equal_sized_buckets()
    print("\nAll tests passed.")
