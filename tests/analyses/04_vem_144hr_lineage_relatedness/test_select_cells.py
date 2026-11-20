'''
Run directly: python3 tests/analyses/04_vem_144hr_lineage_relatedness/test_select_cells.py
'''
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT    = os.path.join(os.path.dirname(__file__), '..', '..', '..')
ANALYSIS_DIR = os.path.join(REPO_ROOT, 'analyses', '04_vem_144hr_lineage_relatedness')
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, ANALYSIS_DIR)
import config
from select_cells import select_cells

TEST_DRUG = 'vem_1uM_144hrsbefore'


def make_metadata(frames, n_per_frame, pixel_id_start=1, has_mask_fraction=1.0, rng=None):
    rng = rng or np.random.default_rng(0)
    rows = []
    pixel_id = pixel_id_start
    for frame in frames:
        for _ in range(n_per_frame):
            rows.append({'frame': frame, 'pixel_id': pixel_id, 'has_mask': rng.random() < has_mask_fraction})
            pixel_id += 1
    return pd.DataFrame(rows)


def test_samples_up_to_the_configured_size_per_frame():
    metadata = make_metadata(range(1, 12), n_per_frame=2000, rng=np.random.default_rng(0))

    original = config.SAMPLE_SIZE_PER_FRAME
    config.SAMPLE_SIZE_PER_FRAME = 100
    try:
        selected = select_cells(metadata, TEST_DRUG, max_frame=11, rng=np.random.default_rng(1))
    finally:
        config.SAMPLE_SIZE_PER_FRAME = original

    assert len(selected) == 11 * 100, f"expected 11 frames x 100 = 1100, got {len(selected)}"
    for frame in range(1, 12):
        assert (selected['frame'] == frame).sum() == 100
    print("PASS: select_cells samples exactly SAMPLE_SIZE_PER_FRAME per frame")


def test_uses_everything_available_when_fewer_than_the_configured_size():
    metadata = make_metadata([1], n_per_frame=30)

    original = config.SAMPLE_SIZE_PER_FRAME
    config.SAMPLE_SIZE_PER_FRAME = 1000
    try:
        selected = select_cells(metadata, TEST_DRUG, max_frame=1, rng=np.random.default_rng(0))
    finally:
        config.SAMPLE_SIZE_PER_FRAME = original

    assert len(selected) == 30, f"expected all 30 available cells, got {len(selected)}"
    print("PASS: select_cells uses every available cell when there are fewer than SAMPLE_SIZE_PER_FRAME")


def test_respects_max_frame_cap():
    # cells exist well past max_frame, but only frames 1-max_frame should ever be sampled
    metadata = make_metadata(range(1, 21), n_per_frame=30)

    selected = select_cells(metadata, TEST_DRUG, max_frame=11, rng=np.random.default_rng(0))

    assert selected['frame'].max() == 11, (
        f"expected frames capped at max_frame (11), got max frame {selected['frame'].max()}"
    )
    print("PASS: select_cells caps the usable frame range at the given max_frame")


def test_max_frame_varies_by_condition():
    # the real config.CONDITIONS gives 'vem' and 'naive' different
    # max_frame values - naive's own (much smaller) cap must be
    # respected independently of vem's
    metadata = make_metadata(range(1, 21), n_per_frame=30)

    assert config.CONDITIONS['naive']['max_frame'] == config.BEFORE_DRUG_END_FRAME == 5, (
        "test assumes the real naive max_frame is config.BEFORE_DRUG_END_FRAME (5)"
    )

    selected = select_cells(metadata, 'naive', max_frame=config.CONDITIONS['naive']['max_frame'], rng=np.random.default_rng(0))
    assert selected['frame'].max() == 5, f"expected naive's frames capped at 5, got {selected['frame'].max()}"
    assert (selected['condition'] == 'naive').all()
    print("PASS: select_cells respects each condition's own max_frame independently")


def test_a_missing_frame_is_skipped_not_an_error():
    # frame 6 has zero has_mask=True cells (mirroring the real vem
    # dataset, where frame 6 - day -1, the frame right before day 0 at
    # frame 7 - is currently missing entirely, and falls outside both
    # the before-drug (1-config.BEFORE_DRUG_END_FRAME) and after-drug
    # (config.AFTER_DRUG_START_FRAME-config.AFTER_DRUG_END_FRAME) ranges,
    # though select_cells.py's own loop still visits it) - must not
    # crash, every OTHER frame should still be selected normally
    metadata = pd.concat([
        make_metadata(range(1, 6), n_per_frame=30),
        pd.DataFrame({'frame': [6], 'pixel_id': [99999], 'has_mask': [False]}),
        make_metadata(range(7, 12), n_per_frame=30, pixel_id_start=200_000),
    ], ignore_index=True)

    selected = select_cells(metadata, TEST_DRUG, max_frame=11, rng=np.random.default_rng(0))

    assert (selected['frame'] == 6).sum() == 0
    assert (selected['frame'] == 5).sum() > 0
    assert (selected['frame'] == 7).sum() > 0
    print("PASS: a frame with zero has_mask=True cells is skipped, not a crash")


def test_only_has_mask_true_cells_selected():
    metadata = make_metadata([1], n_per_frame=200, has_mask_fraction=0.5)

    selected = select_cells(metadata, TEST_DRUG, max_frame=1, rng=np.random.default_rng(0))

    assert selected['has_mask'].all(), "a has_mask=False cell was selected"
    print("PASS: select_cells only ever selects has_mask=True cells")


def test_no_available_cells_at_all_raises_clearly():
    metadata = pd.DataFrame({'frame': [1, 2], 'pixel_id': [1, 2], 'has_mask': [False, False]})

    try:
        select_cells(metadata, TEST_DRUG, max_frame=2, rng=np.random.default_rng(0))
    except AssertionError as e:
        assert 'nothing to sample' in str(e)
        print("PASS: no has_mask=True cells at all raises clearly instead of silently returning nothing")
        return
    raise AssertionError("expected select_cells to raise when there are no usable cells at all")


def test_sampling_is_independent_across_equal_sized_frames():
    '''
    Regression test mirroring other analyses' own: a SHARED rng, not a
    freshly re-seeded one per frame, must give two equal-sized frames
    different relative draw positions.
    '''
    original = config.SAMPLE_SIZE_PER_FRAME
    config.SAMPLE_SIZE_PER_FRAME = 500
    try:
        metadata = pd.concat([
            make_metadata([1], n_per_frame=1000, pixel_id_start=0),
            make_metadata([2], n_per_frame=1000, pixel_id_start=100_000),
        ], ignore_index=True)
        selected = select_cells(metadata, TEST_DRUG, max_frame=2, rng=np.random.default_rng(42))
    finally:
        config.SAMPLE_SIZE_PER_FRAME = original

    frame1_ranks = sorted((selected[selected['frame'] == 1]['pixel_id'] - 0).tolist())
    frame2_ranks = sorted((selected[selected['frame'] == 2]['pixel_id'] - 100_000).tolist())

    assert frame1_ranks != frame2_ranks, (
        "selected relative positions are IDENTICAL across two equal-sized frames - "
        "sampling is not independent per frame (same-seed regression)"
    )
    print("PASS: two equal-sized frames picked different relative positions - sampling is independent per frame")


if __name__ == '__main__':
    test_samples_up_to_the_configured_size_per_frame()
    test_uses_everything_available_when_fewer_than_the_configured_size()
    test_respects_max_frame_cap()
    test_max_frame_varies_by_condition()
    test_a_missing_frame_is_skipped_not_an_error()
    test_only_has_mask_true_cells_selected()
    test_no_available_cells_at_all_raises_clearly()
    test_sampling_is_independent_across_equal_sized_frames()
    print("\nAll tests passed.")
