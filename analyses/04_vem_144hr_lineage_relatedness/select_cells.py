import numpy as np
import pandas as pd

import config


def select_cells(metadata: pd.DataFrame, drug: str, max_frame: int, rng: np.random.Generator) -> pd.DataFrame:
    '''
    Select up to config.SAMPLE_SIZE_PER_FRAME has_mask=True cells per
    frame, across every frame from 1 to max_frame - the full pooled
    range this condition's one CAJAL run covers. For 'vem' this is every
    frame up to config.AFTER_DRUG_END_FRAME (splitting into before-drug
    vs. after-drug happens later, when the notebook builds sister/cousin
    pairs, not here); 'naive' has no such split at all, just its own
    single comparison window.

    A flat per-frame sample size, no budget-splitting or subclass
    stratification - this analysis never classifies cells by subclass
    at all. A simple, lineage-UNaware subsample: which true sister/
    cousin pairs end up with both members selected is left entirely to
    chance, not biased towards keeping more of them (see config.py).

    A frame with zero has_mask=True cells (e.g. vem's frame 6, currently
    missing from that dataset entirely - still visited by the loop below
    since it iterates every frame up to max_frame, not just the named
    before/after-drug ranges) is skipped gracefully rather than
    erroring - it just contributes no cells and no sister/cousin pairs
    from that frame.

    Inputs:
        metadata:  this condition's full metadata DataFrame (must have:
                  frame, has_mask, pixel_id)
        drug:      this condition's drug name (config.CONDITIONS[condition]['drug']),
                  used as the 'condition' column's value
        max_frame: highest frame to sample from, inclusive
                  (config.CONDITIONS[condition]['max_frame'])
        rng:       a single shared np.random.Generator, reused across
                  every frame's .sample() call rather than reseeded per
                  call - passing the same int seed to pandas .sample()
                  instead reseeds a fresh RandomState each call, so any
                  two equal-sized frames would deterministically pick
                  the same RELATIVE row positions (see naive_vs_vem's
                  own select_cells.py for the same issue)
    Outputs:
        selected: concatenated rows chosen across every frame, with a
                  'condition' column added
    '''
    visible = metadata[metadata['has_mask'] & (metadata['frame'] <= max_frame)].copy()
    visible['condition'] = drug

    assert not visible.empty, (
        f"no has_mask=True cells at all in frames 1-{max_frame} - nothing to sample"
    )

    selected_parts = []
    for frame in range(1, max_frame + 1):
        frame_df = visible[visible['frame'] == frame]
        if frame_df.empty:
            print(f"    frame {frame}: no has_mask=True cells available - skipping")
            continue

        n = min(config.SAMPLE_SIZE_PER_FRAME, len(frame_df))
        if n < config.SAMPLE_SIZE_PER_FRAME:
            print(f"    frame {frame} has only {n} has_mask=True cell(s) "
                  f"(wanted {config.SAMPLE_SIZE_PER_FRAME}) - using all of them")
        selected_parts.append(frame_df.sample(n=n, random_state=rng))

    return pd.concat(selected_parts, ignore_index=True)
