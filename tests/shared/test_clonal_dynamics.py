'''
Run directly: python3 tests/shared/test_clonal_dynamics.py
'''
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.clonal_dynamics import (
    CLONE_COLORS, pick_tightest_clones, plot_clones_on_embedding, export_clones_by_subclass,
)


def _embedded_df(lin_id_cells: dict):
    # lin_id_cells: {lin_id: [(frame, pixel_id, dim_0, dim_1), ...]}
    rows = []
    for lin_id, cells in lin_id_cells.items():
        for frame, pixel_id, dim_0, dim_1 in cells:
            rows.append({'lin_id': lin_id, 'frame': frame, 'pixel_id': pixel_id, 'dim_0': dim_0, 'dim_1': dim_1})
    return pd.DataFrame(rows)


def _block_distance_matrix(n, blocks):
    '''
    Build an (n, n) symmetric distance matrix with zero diagonal, where
    every (i, j) pair within the same `blocks` group gets a small
    distance (tight cluster) and every cross-group pair gets a large one
    (loose) - lets a test construct exactly the "tight vs. loose" clones
    pick_tightest_clones is supposed to tell apart.

    Inputs:
        n:      matrix size
        blocks: {group_label: (indices, within_group_distance)} - indices
                not covered by any block default to being "loose" with
                everything (large distance)
    '''
    dist = np.full((n, n), 10.0)
    np.fill_diagonal(dist, 0.0)
    for indices, within_distance in blocks.values():
        for i in indices:
            for j in indices:
                if i != j:
                    dist[i, j] = within_distance
    return dist


def test_pick_tightest_clones_prefers_lower_mean_pairwise_distance():
    # lin_id 'tight' has a small mean pairwise distance among its own
    # cells, lin_id 'loose' has a large one (both have the same cell
    # count, so count can't be driving the result) - 'tight' must win
    embedded_df = _embedded_df({
        'tight': [(f, f, 0, 0) for f in range(10)],
        'loose': [(f, 100 + f, 0, 0) for f in range(10)],
    })
    label_to_index = {}
    for i, (frame, pixel_id) in enumerate(zip(embedded_df['frame'], embedded_df['pixel_id'])):
        label_to_index[(frame, pixel_id)] = i

    tight_indices = [label_to_index[(f, f)] for f in range(10)]
    loose_indices = [label_to_index[(f, 100 + f)] for f in range(10)]
    dist_matrix = _block_distance_matrix(len(embedded_df), {
        'tight': (tight_indices, 0.1),
        'loose': (loose_indices, 9.0),
    })

    top_lin_ids, summary = pick_tightest_clones(embedded_df, dist_matrix, label_to_index, n_clones=2, min_embedded_cells=5)

    assert top_lin_ids == ['tight', 'loose'], f"expected 'tight' ranked above 'loose', got {top_lin_ids}"
    assert summary.loc['tight', 'mean_pairwise_distance'] < summary.loc['loose', 'mean_pairwise_distance']

    print("PASS: pick_tightest_clones ranks by lowest mean pairwise CAJAL distance (tightest cluster first)")


def test_pick_tightest_clones_filters_by_min_embedded_cells():
    # 'small' is tighter than 'big' but has too few embedded cells to be
    # eligible at all - it must be excluded regardless of how tight it is
    embedded_df = _embedded_df({
        'small': [(f, f, 0, 0) for f in range(3)],
        'big':   [(f, 100 + f, 0, 0) for f in range(10)],
    })
    label_to_index = {}
    for i, (frame, pixel_id) in enumerate(zip(embedded_df['frame'], embedded_df['pixel_id'])):
        label_to_index[(frame, pixel_id)] = i

    small_indices = [label_to_index[(f, f)] for f in range(3)]
    big_indices = [label_to_index[(f, 100 + f)] for f in range(10)]
    dist_matrix = _block_distance_matrix(len(embedded_df), {
        'small': (small_indices, 0.01),
        'big':   (big_indices, 5.0),
    })

    top_lin_ids, summary = pick_tightest_clones(embedded_df, dist_matrix, label_to_index, n_clones=2, min_embedded_cells=5)

    assert top_lin_ids == ['big'], f"expected only 'big' to be eligible ('small' is below the cell-count floor), got {top_lin_ids}"

    print("PASS: pick_tightest_clones excludes a clone below min_embedded_cells, however tight it is")


def test_pick_tightest_clones_skips_labels_missing_from_the_distance_matrix():
    # a cell in embedded_df with no matching (frame, pixel_id) in
    # label_to_index (e.g. CAJAL subsampled it out) must simply not count
    # towards that lin_id's eligibility or mean distance, not raise
    embedded_df = _embedded_df({
        'lin1': [(f, f, 0, 0) for f in range(10)] + [(99, 999, 0, 0)],   # last cell has no distance-matrix entry
    })
    label_to_index = {(f, f): i for i, f in enumerate(range(10))}
    dist_matrix = _block_distance_matrix(10, {'lin1': (list(range(10)), 1.0)})

    top_lin_ids, summary = pick_tightest_clones(embedded_df, dist_matrix, label_to_index, n_clones=1, min_embedded_cells=5)

    assert top_lin_ids == ['lin1']
    assert summary.loc['lin1', 'n_cells'] == 10, "the unmatched cell must not be counted"

    print("PASS: pick_tightest_clones ignores embedded cells with no matching distance-matrix entry")


def test_pick_tightest_clones_returns_fewer_than_n_clones_if_not_enough_eligible():
    embedded_df = _embedded_df({'lin1': [(f, f, 0, 0) for f in range(10)]})
    label_to_index = {(f, f): i for i, f in enumerate(range(10))}
    dist_matrix = _block_distance_matrix(10, {'lin1': (list(range(10)), 1.0)})

    top_lin_ids, summary = pick_tightest_clones(embedded_df, dist_matrix, label_to_index, n_clones=2, min_embedded_cells=5)

    assert top_lin_ids == ['lin1'], f"expected just the 1 eligible clone, got {top_lin_ids}"
    assert len(summary) == 1

    print("PASS: pick_tightest_clones returns fewer than n_clones when fewer are eligible, rather than erroring")


def test_plot_clones_on_embedding_runs_without_error():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    background_df = pd.DataFrame({'dim_0': rng.normal(0, 1, 100), 'dim_1': rng.normal(0, 1, 100)})
    clone_dfs = [
        pd.DataFrame({'dim_0': rng.normal(i, 0.3, 5), 'dim_1': rng.normal(i, 0.3, 5)})
        for i in range(2)
    ]

    fig = plot_clones_on_embedding(background_df, clone_dfs, ['lin1', 'lin2'], CLONE_COLORS[:2], 'UMAP', 'test title')
    assert fig is not None
    assert len(fig.axes) == 2
    plt.close(fig)

    print("PASS: plot_clones_on_embedding runs without error for 2 clones")


def test_plot_clones_on_embedding_handles_a_single_clone():
    # plt.subplots(1, 1, ...) returns a bare Axes, not an array - must not
    # crash when iterating over it (same edge case as
    # representative_shapes.plot_representative_shapes)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    background_df = pd.DataFrame({'dim_0': [0, 1, 2], 'dim_1': [0, 1, 2]})
    clone_dfs = [pd.DataFrame({'dim_0': [0.5], 'dim_1': [0.5]})]

    fig = plot_clones_on_embedding(background_df, clone_dfs, ['lin1'], [CLONE_COLORS[0]], 'UMAP', 'test title')
    assert fig is not None
    assert len(fig.axes) == 1
    plt.close(fig)

    print("PASS: plot_clones_on_embedding handles a single clone without crashing")


def _synthetic_embedded_df_and_distances(seed=0):
    rng = np.random.default_rng(seed)

    # per-subclass clones, with deliberately different tightness so the
    # "pick the tightest 2" behavior is exercised for real - including a
    # subclass ('slow_prolif') with only 1 eligible clone
    CLONE_SPECS = {
        'persisting':  {'p_tight': (0.1, 15), 'p_loose': (8.0, 15), 'p_mid': (3.0, 15)},
        'slow_prolif': {'s_only': (1.0, 12)},
        'fast_prolif': {'f_tight': (0.2, 20), 'f_loose': (7.0, 20)},
    }

    rows = []
    label_to_index = {}
    blocks = {}
    idx = 0
    pixel_counter = 1
    for subclass, clones in CLONE_SPECS.items():
        for lin_id, (within_distance, n) in clones.items():
            indices = []
            for i in range(n):
                frame = i % 5
                pixel_id = pixel_counter
                pixel_counter += 1
                rows.append({
                    'lin_id': lin_id, 'subclass': subclass, 'frame': frame, 'pixel_id': pixel_id,
                    'dim_0': rng.normal(0, 1), 'dim_1': rng.normal(0, 1),
                })
                label_to_index[(frame, pixel_id)] = idx
                indices.append(idx)
                idx += 1
            blocks[lin_id] = (indices, within_distance)

    embedded_df = pd.DataFrame(rows)
    dist_matrix = _block_distance_matrix(idx, blocks)
    return embedded_df, dist_matrix, label_to_index


def test_export_clones_by_subclass_writes_pdf_svg_and_summary_csv_per_subclass():
    embedded_df, dist_matrix, label_to_index = _synthetic_embedded_df_and_distances()

    with tempfile.TemporaryDirectory() as tmp_dir:
        combined = export_clones_by_subclass(
            embedded_df, dist_matrix, label_to_index, tmp_dir, 'cell_umap', 'UMAP', 'cell morphology',
            n_clones=2, min_embedded_cells=10, subclasses=['persisting', 'slow_prolif', 'fast_prolif'],
        )

        assert os.path.isfile(os.path.join(tmp_dir, 'cell_umap_persisting.pdf'))
        assert os.path.isfile(os.path.join(tmp_dir, 'svgs', 'cell_umap_persisting.svg'))
        assert os.path.isfile(os.path.join(tmp_dir, 'cell_umap_fast_prolif.pdf'))
        assert os.path.isfile(os.path.join(tmp_dir, 'svgs', 'cell_umap_fast_prolif.svg'))

        summary_csv_path = os.path.join(tmp_dir, 'cell_umap_clone_summary.csv')
        assert os.path.isfile(summary_csv_path)
        on_disk = pd.read_csv(summary_csv_path)
        assert set(on_disk.columns) == {'subclass', 'lin_id', 'n_cells', 'mean_pairwise_distance'}

        persisting_rows = on_disk[on_disk['subclass'] == 'persisting']
        assert list(persisting_rows['lin_id']) == ['p_tight', 'p_mid'], \
            f"expected the 2 tightest persisting clones in tightness order, got {list(persisting_rows['lin_id'])}"

        slow_prolif_rows = on_disk[on_disk['subclass'] == 'slow_prolif']
        assert len(slow_prolif_rows) == 1, f"expected only 1 eligible slow_prolif clone, got {len(slow_prolif_rows)}"
        assert slow_prolif_rows.iloc[0]['lin_id'] == 's_only'

        fast_prolif_rows = on_disk[on_disk['subclass'] == 'fast_prolif']
        assert list(fast_prolif_rows['lin_id']) == ['f_tight', 'f_loose']

        assert len(combined) == len(on_disk)

    print("PASS: export_clones_by_subclass picks the tightest clones per subclass, writes a PDF+SVG per subclass, "
          "and a combined summary CSV - handling a subclass with fewer than n_clones eligible clones")


def test_clone_colors_do_not_reuse_any_tab_color_or_region_color():
    from shared.representative_shapes import REGION_COLORS

    tab_colors = {
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    }
    taken = tab_colors | {c.lower() for c in REGION_COLORS}

    assert not (set(c.lower() for c in CLONE_COLORS) & taken), (
        "a CLONE_COLORS entry collides with a standard 'tab:*' color or a representative_shapes REGION_COLORS entry"
    )

    print("PASS: CLONE_COLORS doesn't reuse any standard 'tab:*' color or REGION_COLORS entry")


if __name__ == '__main__':
    test_pick_tightest_clones_prefers_lower_mean_pairwise_distance()
    test_pick_tightest_clones_filters_by_min_embedded_cells()
    test_pick_tightest_clones_skips_labels_missing_from_the_distance_matrix()
    test_pick_tightest_clones_returns_fewer_than_n_clones_if_not_enough_eligible()
    test_plot_clones_on_embedding_runs_without_error()
    test_plot_clones_on_embedding_handles_a_single_clone()
    test_export_clones_by_subclass_writes_pdf_svg_and_summary_csv_per_subclass()
    test_clone_colors_do_not_reuse_any_tab_color_or_region_color()
    print("\nAll tests passed.")
