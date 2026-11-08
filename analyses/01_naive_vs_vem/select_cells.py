import numpy as np
import pandas as pd

import config
from shared.classify import classify_subclass
from shared.allocation import water_fill_allocate


def select_cells(metadata_by_condition: dict[str, pd.DataFrame]) -> pd.DataFrame:
    '''
    Select which cells go through CAJAL, from each condition's full metadata.

    Restricted to has_mask=True rows (only real segmented cells can go
    through mask extraction/CAJAL), to frames present in EVERY condition
    (a cell at a frame only one condition has can't be compared across
    conditions, which is the whole point of this dataset), and to frames
    <= config.MAX_FRAME (process_imaging_data itself no longer caps
    frames - every condition's metadata.csv can now have far more of
    them than this analysis has ever used, so this analysis caps its own
    frame range explicitly instead of silently absorbing whatever's
    newly available upstream).

    config.TOTAL_CELL_BUDGET is split across conditions by
    config.CONDITION_WEIGHTS, then water-filled (see pipeline/allocation.py)
    within each condition across its own (frame, subclass) buckets - so a
    bucket with fewer real cells than an equal share just contributes
    everything it has, and the leftover goes to other buckets, rather than
    demanding cells that don't exist.

    Inputs:
        metadata_by_condition: {condition: full metadata DataFrame (must
                                have: frame, has_mask, m_label, k_label,
                                pixel_id)}
    Outputs:
        selected: concatenated rows chosen across every condition, with a
                  'condition' and 'subclass' column added
                  (dying/persisting/slow_prolif/fast_prolif)
    '''
    overlapping_frames = None
    for df in metadata_by_condition.values():
        frames = set(df['frame'].unique())
        overlapping_frames = frames if overlapping_frames is None else overlapping_frames & frames
    overlapping_frames = {f for f in overlapping_frames if f <= config.MAX_FRAME}

    assert overlapping_frames, (
        "no frame numbers <= config.MAX_FRAME are shared across every condition - nothing to sample"
    )

    weight_sum = sum(config.CONDITION_WEIGHTS.values())
    assert abs(weight_sum - 1.0) < 1e-6, f"config.CONDITION_WEIGHTS must sum to 1.0, got {weight_sum}"

    missing_weights = set(metadata_by_condition) - set(config.CONDITION_WEIGHTS)
    assert not missing_weights, (
        f"condition(s) {missing_weights} have no entry in config.CONDITION_WEIGHTS - "
        f"add a weight for every condition in config.CONDITIONS"
    )

    # a single shared Generator, not a fixed int, is passed to every
    # .sample() call below - passing the same int to pandas .sample()
    # reseeds a fresh RandomState each call, so any two buckets of equal
    # size would deterministically pick the exact same RELATIVE row
    # positions (confirmed directly) - a real risk of correlated,
    # non-independent sampling if the underlying metadata rows have any
    # structural ordering (e.g. sorted by pixel_id). A shared Generator
    # object instead carries its state across calls, giving each bucket an
    # independent draw while the whole selection stays fully reproducible
    # for a given config.UMAP_SEED.
    rng = np.random.default_rng(config.UMAP_SEED)

    selected_parts = []
    for condition, df in metadata_by_condition.items():
        weight = config.CONDITION_WEIGHTS[condition]
        condition_budget = round(config.TOTAL_CELL_BUDGET * weight)

        visible = df[df['has_mask'] & df['frame'].isin(overlapping_frames)].copy()
        visible['condition'] = condition
        visible['subclass']  = classify_subclass(
            visible,
            dying_m_label=config.DYING_M_LABEL, persisting_m_label=config.PERSISTING_M_LABEL,
            prolif_m_label=config.PROLIF_M_LABEL,
            slow_k_labels=config.SLOW_K_LABELS, fast_k_labels=config.FAST_K_LABELS,
        )

        bucket_sizes = visible.groupby(['frame', 'subclass']).size().to_dict()
        bucket_allocation = water_fill_allocate(bucket_sizes, condition_budget)

        for (frame, subclass), n in bucket_allocation.items():
            if n == 0:
                continue
            bucket_df = visible[(visible['frame'] == frame) & (visible['subclass'] == subclass)]
            selected_parts.append(bucket_df.sample(n=n, random_state=rng))

    return pd.concat(selected_parts, ignore_index=True)
