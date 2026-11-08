'''
Run directly: python3 tests/analyses/04_vem_full_timecourse/test_select_cells.py
'''
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT    = os.path.join(os.path.dirname(__file__), '..', '..', '..')
ANALYSIS_DIR = os.path.join(REPO_ROOT, 'analyses', '04_vem_full_timecourse')
sys.path.insert(0, REPO_ROOT)      # for select_cells.py's `from shared.classify import ...`
sys.path.insert(0, ANALYSIS_DIR)   # for `import config` / `from select_cells import ...`
import config
from select_cells import select_cells


def make_metadata(rng, frames, n_per_frame, pixel_id_start, has_mask_fraction=1.0):
    '''Synthetic metadata with a realistic subclass mix, mirroring naive_vs_vem's own test fixture.'''
    rows = []
    pixel_id = pixel_id_start
    for frame in frames:
        for _ in range(n_per_frame):
            r = rng.random()
            if r < 0.05:
                m_label, k_label = 1, rng.integers(1, 7)        # dying - rare
            elif r < 0.15:
                m_label, k_label = 2, rng.integers(1, 7)        # persisting
            elif r < 0.30:
                m_label, k_label = 3, rng.integers(1, 3)        # slow prolif
            else:
                m_label, k_label = 3, rng.integers(3, 7)        # fast prolif - common

            has_mask = rng.random() < has_mask_fraction
            rows.append({
                'frame': frame, 'pixel_id': pixel_id, 'm_label': m_label,
                'k_label': int(k_label), 'has_mask': has_mask,
            })
            pixel_id += 1

    return pd.DataFrame(rows)


def test_uses_every_available_frame_no_cap():
    # regression test: select_cells must NOT cap frames the way
    # naive_vs_vem's own select_cells now does - the whole point of this
    # analysis is every available frame, including well past 13
    rng = np.random.default_rng(0)
    metadata = make_metadata(rng, range(1, 20), n_per_frame=100, pixel_id_start=1)

    selected = select_cells(metadata)

    assert selected['frame'].max() == 19, (
        f"expected frames up to 19 to be usable, got max frame {selected['frame'].max()}"
    )
    print(f"PASS: select_cells uses every available frame, up to {selected['frame'].max()} "
          f"(no cap, unlike naive_vs_vem's own select_cells)")


def test_only_has_mask_true_cells_selected():
    rng = np.random.default_rng(1)
    metadata = make_metadata(rng, range(1, 4), n_per_frame=500, pixel_id_start=1, has_mask_fraction=0.5)

    selected = select_cells(metadata)

    assert selected['has_mask'].all(), "a has_mask=False cell was selected"
    print("PASS: select_cells only ever selects has_mask=True cells")


def test_condition_column_is_always_the_configured_drug():
    rng = np.random.default_rng(2)
    metadata = make_metadata(rng, range(1, 4), n_per_frame=300, pixel_id_start=1)

    selected = select_cells(metadata)

    assert (selected['condition'] == config.DRUG).all()
    print(f"PASS: every selected cell's condition is config.DRUG ('{config.DRUG}')")


def test_rare_subclass_not_starved_or_overrepresented():
    rng = np.random.default_rng(3)
    metadata = make_metadata(rng, range(1, 6), n_per_frame=300, pixel_id_start=1)

    selected = select_cells(metadata)

    # dying cells are rare by construction (~5% of the population) - the
    # selection should include some, but water-filling means they won't be
    # capped at the same per-bucket share as the abundant fast_prolif class
    assert (selected['subclass'] == 'dying').sum() > 0
    assert (selected['subclass'] == 'fast_prolif').sum() > 0
    print("PASS: a rare subclass (dying) still gets represented in the selection, not starved out")


def test_total_selected_respects_budget_when_capacity_allows():
    rng = np.random.default_rng(4)
    metadata = make_metadata(rng, range(1, 8), n_per_frame=5000, pixel_id_start=1)

    selected = select_cells(metadata)

    assert abs(len(selected) - config.TOTAL_CELL_BUDGET) < 50, (
        f"expected ~{config.TOTAL_CELL_BUDGET} selected cells with ample data, got {len(selected)}"
    )
    print(f"PASS: total selected ({len(selected)}) lands close to config.TOTAL_CELL_BUDGET "
          f"({config.TOTAL_CELL_BUDGET}) when every bucket has ample data")


def test_sampling_is_independent_across_equal_sized_buckets():
    '''
    Regression test mirroring naive_vs_vem's own: a SHARED rng, not a
    freshly re-seeded one per bucket, must give two equal-sized buckets
    different relative draw positions.
    '''
    rows = []
    bucket_size = 1000
    for frame, pixel_start in [(1, 0), (2, 100_000)]:
        for i in range(bucket_size):
            rows.append({'frame': frame, 'pixel_id': pixel_start + i, 'm_label': 3, 'k_label': 4, 'has_mask': True})
    metadata = pd.DataFrame(rows)

    original_budget = config.TOTAL_CELL_BUDGET
    config.TOTAL_CELL_BUDGET = 1000   # two equal buckets (fast_prolif at frame 1, frame 2) split exactly 500/500
    try:
        selected = select_cells(metadata)
    finally:
        config.TOTAL_CELL_BUDGET = original_budget

    frame1 = selected[selected['frame'] == 1]
    frame2 = selected[selected['frame'] == 2]
    assert len(frame1) == len(frame2) == 500, (
        f"test setup expected an exact even 500/500 split, got {len(frame1)} and {len(frame2)}"
    )

    rank1 = sorted((frame1['pixel_id'] - 0).tolist())
    rank2 = sorted((frame2['pixel_id'] - 100_000).tolist())
    assert rank1 != rank2, (
        "selected relative positions are IDENTICAL across two equal-sized buckets - "
        "sampling is not independent per bucket (same-seed regression)"
    )
    print("PASS: two equal-sized buckets picked different relative positions - sampling is independent per bucket")


if __name__ == '__main__':
    test_uses_every_available_frame_no_cap()
    test_only_has_mask_true_cells_selected()
    test_condition_column_is_always_the_configured_drug()
    test_rare_subclass_not_starved_or_overrepresented()
    test_total_selected_respects_budget_when_capacity_allows()
    test_sampling_is_independent_across_equal_sized_buckets()
    print("\nAll tests passed.")
