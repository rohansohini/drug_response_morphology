'''
Run directly: python3 tests/analyses/02_multi_drug_comparison/test_select_cells.py
'''
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT    = os.path.join(os.path.dirname(__file__), '..', '..', '..')
ANALYSIS_DIR = os.path.join(REPO_ROOT, 'analyses', '02_multi_drug_comparison')
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


def test_samples_up_to_the_configured_size_per_bucket():
    rng = np.random.default_rng(0)
    metadata_by_condition = {
        'naive':   make_metadata([1, 2], n_per_frame=5000, pixel_id_start=1, rng=rng),
        'vem_1uM': make_metadata([1, 2], n_per_frame=5000, pixel_id_start=1_000_000, rng=rng),
    }

    original = config.SAMPLE_SIZE_PER_CONDITION
    config.SAMPLE_SIZE_PER_CONDITION = 100
    try:
        selected = select_cells(metadata_by_condition, rng=np.random.default_rng(1))
    finally:
        config.SAMPLE_SIZE_PER_CONDITION = original

    # 2 frames x 2 conditions x 100 = 400
    assert len(selected) == 400
    for frame in [1, 2]:
        for condition in ['naive', 'vem_1uM']:
            n = ((selected['frame'] == frame) & (selected['condition'] == condition)).sum()
            assert n == 100, f"expected 100 cells for frame {frame}/{condition}, got {n}"
    print("PASS: select_cells samples exactly SAMPLE_SIZE_PER_CONDITION per (frame, condition) bucket")


def test_uses_everything_available_when_fewer_than_the_configured_size():
    metadata_by_condition = {
        'naive':   make_metadata([1], n_per_frame=30, pixel_id_start=1),
        'vem_1uM': make_metadata([1], n_per_frame=30, pixel_id_start=1_000_000),
    }

    original = config.SAMPLE_SIZE_PER_CONDITION
    config.SAMPLE_SIZE_PER_CONDITION = 1000
    try:
        selected = select_cells(metadata_by_condition, rng=np.random.default_rng(0))
    finally:
        config.SAMPLE_SIZE_PER_CONDITION = original

    assert len(selected) == 60, f"expected all 60 available cells (30+30), got {len(selected)}"
    print("PASS: select_cells uses every available cell when there are fewer than SAMPLE_SIZE_PER_CONDITION")


def test_computes_the_true_frame_intersection_when_conditions_differ():
    # naive has frames 1-5, vem only has frames 3-8 - the true overlap is
    # 3-5 only, not naive's or vem's own full range
    metadata_by_condition = {
        'naive':   make_metadata(range(1, 6), n_per_frame=50, pixel_id_start=1),
        'vem_1uM': make_metadata(range(3, 9), n_per_frame=50, pixel_id_start=1_000_000),
    }

    selected = select_cells(metadata_by_condition, rng=np.random.default_rng(0))

    assert set(selected['frame'].unique()) == {3, 4, 5}, (
        f"expected only the true shared frames (3-5), got {sorted(selected['frame'].unique())}"
    )
    print("PASS: select_cells computes the true frame intersection, not just one condition's own range")


def test_respects_max_frame_cap():
    # both conditions actually have frames up to 20, but config.MAX_FRAME
    # should still cut the usable range off at 13
    metadata_by_condition = {
        'naive':   make_metadata(range(1, 21), n_per_frame=30, pixel_id_start=1),
        'vem_1uM': make_metadata(range(1, 21), n_per_frame=30, pixel_id_start=1_000_000),
    }

    assert config.MAX_FRAME == 13, "test assumes the real config.MAX_FRAME is 13 - update this test if that changes"

    selected = select_cells(metadata_by_condition, rng=np.random.default_rng(0))

    assert selected['frame'].max() == 13, (
        f"expected frames capped at config.MAX_FRAME (13), got max frame {selected['frame'].max()}"
    )
    print("PASS: select_cells caps the usable frame range at config.MAX_FRAME even when more is available")


def test_only_has_mask_true_cells_selected():
    metadata_by_condition = {
        'naive':   make_metadata([1], n_per_frame=200, pixel_id_start=1, has_mask_fraction=0.5),
        'vem_1uM': make_metadata([1], n_per_frame=200, pixel_id_start=1_000_000, has_mask_fraction=0.5),
    }

    selected = select_cells(metadata_by_condition, rng=np.random.default_rng(0))

    assert selected['has_mask'].all(), "a has_mask=False cell was selected"
    print("PASS: select_cells only ever selects has_mask=True cells")


def test_no_shared_frames_raises_clearly():
    metadata_by_condition = {
        'naive':   pd.DataFrame({'frame': [1, 2], 'pixel_id': [1, 2], 'has_mask': [True, True]}),
        'vem_1uM': pd.DataFrame({'frame': [10, 11], 'pixel_id': [1, 2], 'has_mask': [True, True]}),
    }

    try:
        select_cells(metadata_by_condition, rng=np.random.default_rng(0))
    except AssertionError as e:
        assert 'shared' in str(e).lower()
        print("PASS: no frames shared across every condition raises clearly instead of silently returning nothing")
        return
    raise AssertionError("expected select_cells to raise when no frame is shared across every condition")


def test_a_bucket_with_zero_cells_is_skipped_not_an_error():
    # unlike "no shared frames at all", a single (frame, condition) bucket
    # legitimately having zero has_mask=True cells (e.g. one frame's
    # segmentation totally failed for one condition) must not crash the
    # whole run - every OTHER bucket should still be selected normally
    metadata_by_condition = {
        'naive':   pd.concat([
            make_metadata([1], n_per_frame=50, pixel_id_start=1),
            pd.DataFrame({'frame': [2], 'pixel_id': [999], 'has_mask': [False]}),  # frame 2: nothing usable
        ], ignore_index=True),
        'vem_1uM': make_metadata([1, 2], n_per_frame=50, pixel_id_start=1_000_000),
    }

    selected = select_cells(metadata_by_condition, rng=np.random.default_rng(0))

    assert ((selected['frame'] == 2) & (selected['condition'] == 'naive')).sum() == 0
    assert ((selected['frame'] == 1) & (selected['condition'] == 'naive')).sum() > 0
    assert ((selected['frame'] == 2) & (selected['condition'] == 'vem_1uM')).sum() > 0
    print("PASS: a (frame, condition) bucket with zero available cells is skipped, not a crash")


def test_sampling_is_independent_across_equal_sized_buckets():
    '''
    Regression test mirroring naive_vs_vem's own: a SHARED rng, not a
    freshly re-seeded one per bucket, must give two equal-sized buckets
    different relative draw positions.
    '''
    original = config.SAMPLE_SIZE_PER_CONDITION
    config.SAMPLE_SIZE_PER_CONDITION = 500
    try:
        metadata_by_condition = {
            'naive':   make_metadata([1], n_per_frame=1000, pixel_id_start=0),
            'vem_1uM': make_metadata([1], n_per_frame=1000, pixel_id_start=100_000),
        }
        selected = select_cells(metadata_by_condition, rng=np.random.default_rng(42))
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
    test_samples_up_to_the_configured_size_per_bucket()
    test_uses_everything_available_when_fewer_than_the_configured_size()
    test_computes_the_true_frame_intersection_when_conditions_differ()
    test_respects_max_frame_cap()
    test_only_has_mask_true_cells_selected()
    test_no_shared_frames_raises_clearly()
    test_a_bucket_with_zero_cells_is_skipped_not_an_error()
    test_sampling_is_independent_across_equal_sized_buckets()
    print("\nAll tests passed.")
