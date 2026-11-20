import numpy as np
import pandas as pd

import config
from shared.classify import classify_subclass
from shared.allocation import water_fill_allocate


def select_cells(metadata: pd.DataFrame) -> pd.DataFrame:
    '''
    Select which vem_1uM cells go through CAJAL, from its full metadata.

    Restricted to has_mask=True rows (only real segmented cells can go
    through mask extraction/CAJAL). Uses EVERY available frame - no
    frame cap - since the whole point of this analysis is to look across
    vem_1uM's entire timecourse, not just the frames naive_vs_vem
    restricts itself to.

    config.TOTAL_CELL_BUDGET is water-filled (see shared/allocation.py)
    across (frame, subclass) buckets - so a bucket with fewer real cells
    than an equal share just contributes everything it has, and the
    leftover goes to other buckets, rather than demanding cells that
    don't exist. Simpler than naive_vs_vem's own select_cells: a single
    condition, so there's no cross-condition budget split to do first.

    Inputs:
        metadata: vem_1uM's full metadata DataFrame (must have: frame,
                  has_mask, m_label, k_label, pixel_id)
    Outputs:
        selected: selected rows, with a 'condition' and 'subclass' column
                  added (dying/persisting/slow_prolif/fast_prolif)
    '''
    visible = metadata[metadata['has_mask']].copy()
    visible['condition'] = config.DRUG
    visible['subclass']  = classify_subclass(
        visible,
        dying_m_label=config.DYING_M_LABEL, persisting_m_label=config.PERSISTING_M_LABEL,
        prolif_m_label=config.PROLIF_M_LABEL,
        slow_k_labels=config.SLOW_K_LABELS, fast_k_labels=config.FAST_K_LABELS,
    )

    bucket_sizes = visible.groupby(['frame', 'subclass']).size().to_dict()
    bucket_allocation = water_fill_allocate(bucket_sizes, config.TOTAL_CELL_BUDGET)

    # a single shared Generator, not a fixed int, is passed to every
    # .sample() call below - see naive_vs_vem's own select_cells.py for
    # why (a fixed int reseeds a fresh RandomState each call, so any two
    # equal-sized buckets would deterministically pick the exact same
    # RELATIVE row positions).
    rng = np.random.default_rng(config.UMAP_SEED)

    selected_parts = []
    for (frame, subclass), n in bucket_allocation.items():
        if n == 0:
            continue
        bucket_df = visible[(visible['frame'] == frame) & (visible['subclass'] == subclass)]
        selected_parts.append(bucket_df.sample(n=n, random_state=rng))

    return pd.concat(selected_parts, ignore_index=True)
