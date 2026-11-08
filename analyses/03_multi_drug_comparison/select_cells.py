import numpy as np
import pandas as pd

import config


def select_cells(metadata_by_condition: dict[str, pd.DataFrame], rng: np.random.Generator) -> pd.DataFrame:
    '''
    Select up to config.SAMPLE_SIZE_PER_CONDITION has_mask=True cells per
    (frame, condition), across every frame shared by ALL conditions and
    <= config.MAX_FRAME. The four conditions don't necessarily have the
    same number of frames, so "shared" is the actual intersection of
    frame numbers present in each condition's own metadata, not assumed.

    Much simpler than naive_vs_vem's select_cells: a flat per-bucket
    sample size, not a total budget split by weight, and no subclass
    stratification - this analysis only ever compares cells by condition,
    never by subclass, so there's no need to balance a budget across
    (frame, subclass) buckets.

    Inputs:
        metadata_by_condition: {condition: full metadata DataFrame (must
                                have: frame, has_mask, pixel_id)}
        rng:                   a single shared np.random.Generator, reused
                                across every (frame, condition) call rather
                                than reseeded per call - passing the same
                                int seed to pandas .sample() instead
                                reseeds a fresh RandomState each call, so
                                any two equal-sized buckets would
                                deterministically pick the same RELATIVE
                                row positions (confirmed directly in
                                naive_vs_vem's own select_cells.py). A
                                shared Generator carries its state across
                                calls instead, giving each bucket an
                                independent draw while the whole selection
                                stays reproducible for a given
                                config.SAMPLE_SEED.
    Outputs:
        selected: concatenated rows chosen across every (frame, condition)
                  bucket, with a 'condition' column added
    '''
    overlapping_frames = None
    for df in metadata_by_condition.values():
        frames = set(df['frame'].unique())
        overlapping_frames = frames if overlapping_frames is None else overlapping_frames & frames
    overlapping_frames = sorted(f for f in overlapping_frames if f <= config.MAX_FRAME)

    assert overlapping_frames, (
        "no frame numbers <= config.MAX_FRAME are shared across every condition - nothing to sample"
    )
    print(f"  Frames shared by every condition, <= config.MAX_FRAME: {overlapping_frames}")

    selected_parts = []
    for frame in overlapping_frames:
        for condition, df in metadata_by_condition.items():
            visible = df[(df['frame'] == frame) & df['has_mask']].copy()
            visible['condition'] = condition

            n = min(config.SAMPLE_SIZE_PER_CONDITION, len(visible))
            if n < config.SAMPLE_SIZE_PER_CONDITION:
                print(f"    {condition} has only {n} has_mask=True cell(s) at frame {frame} "
                      f"(wanted {config.SAMPLE_SIZE_PER_CONDITION}) - using all of them")
            selected_parts.append(visible.sample(n=n, random_state=rng))

    return pd.concat(selected_parts, ignore_index=True)
