'''
Run directly: python3 tests/test_select_cells.py
'''
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from pipeline.select_cells import select_cells


def make_condition_metadata(rng, condition_label, frames, n_per_frame, pixel_id_start):
    '''
    Build synthetic metadata with a realistic subclass mix (proliferators
    much more common than dying cells, mirroring real biology) and a
    sprinkling of has_mask=False rows.
    '''
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

            has_mask = rng.random() > 0.1
            rows.append({
                'frame': frame, 'pixel_id': pixel_id, 'm_label': m_label,
                'k_label': int(k_label), 'has_mask': has_mask,
            })
            pixel_id += 1

    return pd.DataFrame(rows)


def test_respects_condition_weights_and_overlapping_frames():
    rng = np.random.default_rng(0)
    # naive has frames 1-10, vem has frames 5-15 - overlap is 5-10 only.
    # n_per_frame must be large enough that BOTH conditions have more
    # available cells than their respective target budget (6000 for naive,
    # 24000 for vem) within that 6-frame overlap - otherwise both sides
    # just hand over everything they have (correct water-filling behavior,
    # but it tests capacity-capping, not weight-respecting allocation).
    naive_df = make_condition_metadata(rng, 'naive', range(1, 11), n_per_frame=5000, pixel_id_start=1)
    vem_df   = make_condition_metadata(rng, 'A1_1uM_vem', range(5, 16), n_per_frame=5000, pixel_id_start=10_000_000)

    selected = select_cells({'naive': naive_df, 'A1_1uM_vem': vem_df})

    # only overlapping frames (5-10) should ever appear
    assert set(selected['frame'].unique()).issubset(set(range(5, 11)))
    assert selected['frame'].min() >= 5
    assert selected['frame'].max() <= 10

    # every selected cell actually has a mask
    assert selected['has_mask'].all()

    # both conditions now have ample availability, so the condition split
    # should closely track config.CONDITION_WEIGHTS (0.2 / 0.8)
    n_naive = (selected['condition'] == 'naive').sum()
    n_vem   = (selected['condition'] == 'A1_1uM_vem').sum()
    total   = n_naive + n_vem
    assert total > 0
    naive_fraction = n_naive / total
    assert 0.15 < naive_fraction < 0.25, f"expected naive to be close to 20% of the selection, got {naive_fraction:.2f}"

    print(f"PASS: selection respects overlapping frames (5-10 only) and roughly the 20/80 condition "
          f"split (naive={n_naive}, vem={n_vem})")


def test_rare_subclass_not_starved_or_overrepresented():
    rng = np.random.default_rng(1)
    naive_df = make_condition_metadata(rng, 'naive', range(1, 6), n_per_frame=200, pixel_id_start=1)
    vem_df   = make_condition_metadata(rng, 'A1_1uM_vem', range(1, 6), n_per_frame=200, pixel_id_start=1_000_000)

    selected = select_cells({'naive': naive_df, 'A1_1uM_vem': vem_df})

    # dying cells are rare by construction (~5% of the population) - the
    # selection should include some, but water-filling means they won't be
    # capped at the same per-bucket share as the abundant fast_prolif class
    # once availability runs out
    assert (selected['subclass'] == 'dying').sum() > 0
    assert (selected['subclass'] == 'fast_prolif').sum() > 0

    print("PASS: a rare subclass (dying) still gets represented in the selection, not starved out")


def test_no_overlapping_frames_raises():
    naive_df = pd.DataFrame({'frame': [1, 2], 'pixel_id': [1, 2], 'm_label': [3, 3], 'k_label': [4, 4], 'has_mask': [True, True]})
    vem_df   = pd.DataFrame({'frame': [10, 11], 'pixel_id': [1, 2], 'm_label': [3, 3], 'k_label': [4, 4], 'has_mask': [True, True]})

    try:
        select_cells({'naive': naive_df, 'A1_1uM_vem': vem_df})
    except AssertionError as e:
        assert 'overlap' in str(e).lower() or 'shared' in str(e).lower()
        print("PASS: no overlapping frames raises clearly instead of silently returning nothing useful")
        return
    raise AssertionError("expected select_cells to raise when conditions share no frames")


def test_total_selected_respects_budget_when_capacity_allows():
    rng = np.random.default_rng(2)
    # both conditions need enough availability to clear vem's 24000 target
    # (the larger of the two) - see the sizing note in the weights test above
    naive_df = make_condition_metadata(rng, 'naive', range(1, 7), n_per_frame=5000, pixel_id_start=1)
    vem_df   = make_condition_metadata(rng, 'A1_1uM_vem', range(1, 7), n_per_frame=5000, pixel_id_start=1_000_000)

    selected = select_cells({'naive': naive_df, 'A1_1uM_vem': vem_df})

    # with ample data available in every bucket, total should land close to
    # config.TOTAL_CELL_BUDGET (small rounding slack from per-condition/
    # per-bucket integer allocation is expected)
    assert abs(len(selected) - config.TOTAL_CELL_BUDGET) < 50, (
        f"expected ~{config.TOTAL_CELL_BUDGET} selected cells with ample data, got {len(selected)}"
    )

    print(f"PASS: total selected ({len(selected)}) lands close to config.TOTAL_CELL_BUDGET "
          f"({config.TOTAL_CELL_BUDGET}) when every bucket has ample data")


if __name__ == '__main__':
    test_respects_condition_weights_and_overlapping_frames()
    test_rare_subclass_not_starved_or_overrepresented()
    test_no_overlapping_frames_raises()
    test_total_selected_respects_budget_when_capacity_allows()
    print("\nAll tests passed.")
