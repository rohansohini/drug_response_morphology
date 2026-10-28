'''
Run directly: python3 tests/analyses/0_naive_vs_vem/test_select_cells.py
'''
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT    = os.path.join(os.path.dirname(__file__), '..', '..', '..')
ANALYSIS_DIR = os.path.join(REPO_ROOT, 'analyses', '0_naive_vs_vem')
sys.path.insert(0, REPO_ROOT)      # for select_cells.py's `from shared.classify import ...`
sys.path.insert(0, ANALYSIS_DIR)   # for `import config` / `from select_cells import ...`
import config
from select_cells import select_cells

# tests below use config.NAIVE_CONDITION/config.VEM_CONDITION rather than
# hardcoded strings, so they keep working if the real condition names ever
# change again (as they already did once - process_imaging_data's naive
# condition turned out to be 'A2_naive', not 'naive')
NAIVE = config.NAIVE_CONDITION
VEM   = config.VEM_CONDITION


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
    naive_df = make_condition_metadata(rng, NAIVE, range(1, 11), n_per_frame=5000, pixel_id_start=1)
    vem_df   = make_condition_metadata(rng, VEM, range(5, 16), n_per_frame=5000, pixel_id_start=10_000_000)

    selected = select_cells({NAIVE: naive_df, VEM: vem_df})

    # only overlapping frames (5-10) should ever appear
    assert set(selected['frame'].unique()).issubset(set(range(5, 11)))
    assert selected['frame'].min() >= 5
    assert selected['frame'].max() <= 10

    # every selected cell actually has a mask
    assert selected['has_mask'].all()

    # both conditions now have ample availability, so the condition split
    # should closely track config.CONDITION_WEIGHTS (0.2 / 0.8)
    n_naive = (selected['condition'] == NAIVE).sum()
    n_vem   = (selected['condition'] == VEM).sum()
    total   = n_naive + n_vem
    assert total > 0
    naive_fraction = n_naive / total
    assert 0.15 < naive_fraction < 0.25, f"expected naive to be close to 20% of the selection, got {naive_fraction:.2f}"

    print(f"PASS: selection respects overlapping frames (5-10 only) and roughly the 20/80 condition "
          f"split (naive={n_naive}, vem={n_vem})")


def test_rare_subclass_not_starved_or_overrepresented():
    rng = np.random.default_rng(1)
    naive_df = make_condition_metadata(rng, NAIVE, range(1, 6), n_per_frame=200, pixel_id_start=1)
    vem_df   = make_condition_metadata(rng, VEM, range(1, 6), n_per_frame=200, pixel_id_start=1_000_000)

    selected = select_cells({NAIVE: naive_df, VEM: vem_df})

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
        select_cells({NAIVE: naive_df, VEM: vem_df})
    except AssertionError as e:
        assert 'overlap' in str(e).lower() or 'shared' in str(e).lower()
        print("PASS: no overlapping frames raises clearly instead of silently returning nothing useful")
        return
    raise AssertionError("expected select_cells to raise when conditions share no frames")


def test_total_selected_respects_budget_when_capacity_allows():
    rng = np.random.default_rng(2)
    # both conditions need enough availability to clear vem's 24000 target
    # (the larger of the two) - see the sizing note in the weights test above
    naive_df = make_condition_metadata(rng, NAIVE, range(1, 7), n_per_frame=5000, pixel_id_start=1)
    vem_df   = make_condition_metadata(rng, VEM, range(1, 7), n_per_frame=5000, pixel_id_start=1_000_000)

    selected = select_cells({NAIVE: naive_df, VEM: vem_df})

    # with ample data available in every bucket, total should land close to
    # config.TOTAL_CELL_BUDGET (small rounding slack from per-condition/
    # per-bucket integer allocation is expected)
    assert abs(len(selected) - config.TOTAL_CELL_BUDGET) < 50, (
        f"expected ~{config.TOTAL_CELL_BUDGET} selected cells with ample data, got {len(selected)}"
    )

    print(f"PASS: total selected ({len(selected)}) lands close to config.TOTAL_CELL_BUDGET "
          f"({config.TOTAL_CELL_BUDGET}) when every bucket has ample data")


def test_sampling_is_independent_across_equal_sized_buckets():
    '''
    Regression test: select_cells previously passed the same fixed int
    seed to every bucket's .sample() call, which (confirmed directly,
    before fixing) makes pandas pick the exact same RELATIVE row
    positions from any two equal-sized buckets - a real risk of
    correlated, non-independent sampling if the underlying rows have any
    structural ordering (e.g. sorted by pixel_id). Builds two
    equal-sized, equal-allocation vem buckets (frame 1 and frame 2, both
    all fast_prolif, pixel_id sorted ascending within each) and checks
    the selected pixel_ids are NOT at identical relative ranks within
    their bucket.
    '''
    naive_df = pd.DataFrame({
        'frame': [1, 2], 'pixel_id': [1, 2], 'm_label': [3, 3], 'k_label': [4, 4], 'has_mask': [True, True],
    })

    bucket_size = 1000
    rows = []
    for frame, pixel_start in [(1, 0), (2, 100_000)]:
        for i in range(bucket_size):
            rows.append({'frame': frame, 'pixel_id': pixel_start + i, 'm_label': 3, 'k_label': 4, 'has_mask': True})
    vem_df = pd.DataFrame(rows)

    # TOTAL_CELL_BUDGET=1000 with vem getting the whole budget (weight 1.0)
    # and two equal-sized (1000-cell) buckets divides EXACTLY in half
    # (500/500, no leftover remainder) - this is deliberate: the bug this
    # test targets only manifests when both buckets get the exact same n,
    # since .sample(n=k, random_state=<same int>) only picks identical
    # positions when both k and the seed match. An uneven split (e.g. from
    # allocation.py's leftover redistribution) would mask the bug even if
    # present, since different n naturally gives different draws regardless.
    original_budget, original_weights = config.TOTAL_CELL_BUDGET, config.CONDITION_WEIGHTS
    config.TOTAL_CELL_BUDGET = 1000
    config.CONDITION_WEIGHTS = {NAIVE: 0.0, VEM: 1.0}
    try:
        selected = select_cells({NAIVE: naive_df, VEM: vem_df})
    finally:
        config.TOTAL_CELL_BUDGET, config.CONDITION_WEIGHTS = original_budget, original_weights

    frame1 = selected[(selected['condition'] == VEM) & (selected['frame'] == 1)]
    frame2 = selected[(selected['condition'] == VEM) & (selected['frame'] == 2)]

    assert len(frame1) == len(frame2) == 500, (
        f"test setup expected an exact even 500/500 split, got {len(frame1)} and {len(frame2)} - "
        f"fix the budget numbers above so this test actually isolates the seeding behavior"
    )

    rank1 = sorted((frame1['pixel_id'] - 0).tolist())
    rank2 = sorted((frame2['pixel_id'] - 100_000).tolist())

    assert rank1 != rank2, (
        "selected relative positions are IDENTICAL across two equal-sized buckets - "
        "sampling is not independent per bucket (same-seed regression)"
    )

    print(f"PASS: two equal-sized, equal-allocation buckets ({len(frame1)} cells each) picked "
          f"different relative positions - sampling is independent per bucket")


def test_missing_condition_weight_raises_clearly():
    # 'some_new_condition' is deliberately not a key in the real
    # config.CONDITION_WEIGHTS, to simulate someone adding a condition to
    # config.CONDITIONS without updating CONDITION_WEIGHTS to match
    naive_df   = pd.DataFrame({'frame': [1], 'pixel_id': [1], 'm_label': [3], 'k_label': [4], 'has_mask': [True]})
    unknown_df = pd.DataFrame({'frame': [1], 'pixel_id': [1], 'm_label': [3], 'k_label': [4], 'has_mask': [True]})

    try:
        select_cells({NAIVE: naive_df, 'some_new_condition': unknown_df})
    except AssertionError as e:
        assert 'some_new_condition' in str(e) and 'CONDITION_WEIGHTS' in str(e)
        print("PASS: a condition missing from config.CONDITION_WEIGHTS raises a clear error, not a bare KeyError")
        return
    raise AssertionError("expected select_cells to raise for a condition with no configured weight")


if __name__ == '__main__':
    test_respects_condition_weights_and_overlapping_frames()
    test_rare_subclass_not_starved_or_overrepresented()
    test_no_overlapping_frames_raises()
    test_total_selected_respects_budget_when_capacity_allows()
    test_sampling_is_independent_across_equal_sized_buckets()
    test_missing_condition_weight_raises_clearly()
    print("\nAll tests passed.")
