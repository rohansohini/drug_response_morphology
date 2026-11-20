'''
Run directly: python3 tests/shared/test_clonal_dynamics.py
'''
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.clonal_dynamics import (
    CLONE_COLORS, pick_tightest_clones, select_clones_by_subclass, pick_clone_medoid,
    plot_clones_on_embedding, plot_clone_shapes, export_clones_by_subclass,
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


def test_select_clones_by_subclass_matches_calling_pick_tightest_clones_directly():
    # select_clones_by_subclass is just a per-subclass loop around
    # pick_tightest_clones - confirm it gives the exact same answer as
    # calling pick_tightest_clones by hand per subclass, since the whole
    # point of splitting it out is to call it ONCE per modality (see
    # shared/clonal_dynamics.py's own docstring) rather than once per
    # embedding method
    embedded_df = pd.DataFrame({
        'lin_id':  ['a_tight', 'a_tight', 'a_loose', 'a_loose', 'b_only', 'b_only'],
        'subclass': ['persisting'] * 4 + ['fast_prolif'] * 2,
        'frame':   [0, 1, 0, 1, 0, 1],
        'pixel_id': [1, 2, 3, 4, 5, 6],
    })
    label_to_index = {(0, 1): 0, (1, 2): 1, (0, 3): 2, (1, 4): 3, (0, 5): 4, (1, 6): 5}
    dist_matrix = _block_distance_matrix(6, {
        'a_tight': ([0, 1], 0.1),
        'a_loose': ([2, 3], 9.0),
        'b_only':  ([4, 5], 1.0),
    })

    selections = select_clones_by_subclass(
        embedded_df, dist_matrix, label_to_index, n_clones=2, min_embedded_cells=2,
        subclasses=['persisting', 'fast_prolif'],
    )

    expected_persisting = pick_tightest_clones(
        embedded_df[embedded_df['subclass'] == 'persisting'], dist_matrix, label_to_index, 2, 2,
    )
    expected_fast_prolif = pick_tightest_clones(
        embedded_df[embedded_df['subclass'] == 'fast_prolif'], dist_matrix, label_to_index, 2, 2,
    )

    assert selections['persisting'][0] == expected_persisting[0]
    assert selections['fast_prolif'][0] == expected_fast_prolif[0]
    assert set(selections.keys()) == {'persisting', 'fast_prolif'}

    print("PASS: select_clones_by_subclass matches calling pick_tightest_clones directly, once per subclass")


def test_pick_clone_medoid_picks_the_cell_closest_to_the_rest_of_its_own_clone():
    # 4 cells: pixel_id 11 has the lowest mean distance to the other 3
    # (cells 12/13 are clustered near it; cell 10 is far from everyone) -
    # it must win, even though it is NOT the simple average-position
    # pick in 2D (that distinction is the whole point of using the real
    # distance matrix instead of the UMAP/PHATE projection)
    dist_matrix = np.array([
        [0, 5, 5, 5],
        [5, 0, 1, 1],
        [5, 1, 0, 2],
        [5, 1, 2, 0],
    ], dtype=float)
    label_to_index = {(1, 10): 0, (1, 11): 1, (1, 12): 2, (1, 13): 3}
    clone_df = pd.DataFrame({
        'condition': ['vem_1uM'] * 4, 'frame': [1, 1, 1, 1], 'pixel_id': [10, 11, 12, 13],
    })

    medoid = pick_clone_medoid(clone_df, dist_matrix, label_to_index)

    assert medoid['pixel_id'] == 11, f"expected pixel_id 11 (lowest mean distance to the rest), got {medoid['pixel_id']}"

    print("PASS: pick_clone_medoid picks the cell with the lowest mean real distance to the rest of its own clone")


def test_pick_clone_medoid_ignores_cells_missing_from_the_distance_matrix():
    # a clone_df row with no matching (frame, pixel_id) in label_to_index
    # must simply be excluded from the medoid search, not raise
    dist_matrix = np.array([[0, 1], [1, 0]], dtype=float)
    label_to_index = {(1, 10): 0, (1, 11): 1}
    clone_df = pd.DataFrame({
        'condition': ['vem_1uM'] * 3, 'frame': [1, 1, 1], 'pixel_id': [10, 11, 999],   # 999 has no matrix entry
    })

    medoid = pick_clone_medoid(clone_df, dist_matrix, label_to_index)

    assert medoid['pixel_id'] in (10, 11), f"expected the medoid to come from the 2 resolvable cells, got {medoid['pixel_id']}"

    print("PASS: pick_clone_medoid ignores cells with no matching distance-matrix entry")


def test_plot_clone_shapes_runs_without_error():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    crops = {'lin1': np.ones((20, 20), dtype=np.uint16), 'lin2': np.zeros((20, 20), dtype=np.uint16)}
    fig = plot_clone_shapes(crops, ['lin1', 'lin2'], CLONE_COLORS[:2], 'test title')
    assert fig is not None
    assert len(fig.axes) == 2
    plt.close(fig)

    print("PASS: plot_clone_shapes runs without error for 2 clones")


def test_plot_clone_shapes_always_renders_n_panels_with_a_placeholder_for_missing_ones():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    crop = np.ones((20, 20), dtype=np.uint16)
    fig_partial = plot_clone_shapes({'lin1': crop}, ['lin1'], [CLONE_COLORS[0]], 'title', n_panels=2)
    fig_full = plot_clone_shapes({'lin1': crop, 'lin2': crop}, ['lin1', 'lin2'], CLONE_COLORS[:2], 'title', n_panels=2)

    assert len(fig_partial.axes) == 2, f"expected 2 panels even with only 1 clone's shape, got {len(fig_partial.axes)}"
    assert fig_partial.get_size_inches().tolist() == fig_full.get_size_inches().tolist(), (
        "a figure with a missing clone shape must be the exact same size as one with every slot filled"
    )
    assert any('no eligible' in t.get_text() for t in fig_partial.axes[1].texts)

    plt.close(fig_partial)
    plt.close(fig_full)

    print("PASS: plot_clone_shapes always renders n_panels, same size whether or not every slot has a shape")


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


def test_plot_clones_on_embedding_always_renders_n_panels_the_same_size():
    # regression test: a subclass with fewer eligible clones than
    # n_clones must still export the SAME size image as one with every
    # slot filled - any leftover panel gets a placeholder instead of
    # shrinking the whole figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    background_df = pd.DataFrame({'dim_0': rng.normal(0, 1, 100), 'dim_1': rng.normal(0, 1, 100)})
    one_clone_df = [pd.DataFrame({'dim_0': rng.normal(0, 0.3, 5), 'dim_1': rng.normal(0, 0.3, 5)})]
    two_clone_dfs = one_clone_df + [pd.DataFrame({'dim_0': rng.normal(3, 0.3, 5), 'dim_1': rng.normal(3, 0.3, 5)})]

    fig_partial = plot_clones_on_embedding(background_df, one_clone_df, ['lin1'], [CLONE_COLORS[0]], 'UMAP', 'title', n_panels=2)
    fig_full = plot_clones_on_embedding(background_df, two_clone_dfs, ['lin1', 'lin2'], CLONE_COLORS[:2], 'UMAP', 'title', n_panels=2)

    assert len(fig_partial.axes) == 2, f"expected 2 panels even with only 1 eligible clone, got {len(fig_partial.axes)}"
    assert fig_partial.get_size_inches().tolist() == fig_full.get_size_inches().tolist(), (
        "a figure with fewer eligible clones must be the exact same size as one with every slot filled"
    )
    assert fig_partial.axes[1].get_legend() is None, "the placeholder panel must not get a legend"
    assert any('no eligible clone' in t.get_text() for t in fig_partial.axes[1].texts)

    plt.close(fig_partial)
    plt.close(fig_full)

    print("PASS: plot_clones_on_embedding always renders n_panels, same size whether or not every slot has a clone")


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
    # subclass ('slow_prolif') with only 1 eligible clone, and one
    # ('dying') with ZERO eligible clones at all
    CLONE_SPECS = {
        'persisting':  {'p_tight': (0.1, 15), 'p_loose': (8.0, 15), 'p_mid': (3.0, 15)},
        'slow_prolif': {'s_only': (1.0, 12)},
        'fast_prolif': {'f_tight': (0.2, 20), 'f_loose': (7.0, 20)},
        'dying':       {'d_too_small': (0.5, 3)},
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
                    'condition': 'vem_1uM', 'dim_0': rng.normal(0, 1), 'dim_1': rng.normal(0, 1),
                })
                label_to_index[(frame, pixel_id)] = idx
                indices.append(idx)
                idx += 1
            blocks[lin_id] = (indices, within_distance)

    embedded_df = pd.DataFrame(rows)
    dist_matrix = _block_distance_matrix(idx, blocks)
    return embedded_df, dist_matrix, label_to_index


def _write_synthetic_masks(embedded_df, masks_dir, half_width_floor=10):
    '''
    One mask TIFF per frame, per modality ('cell' and 'nuc', identical
    content - the test doesn't care which modality it crops from), with
    every pixel_id present at that frame placed as its own small square
    blob at a distinct grid position, far enough apart that half_width
    crops never bleed into a neighboring cell's own pixels.
    '''
    for subdir in ('cell', 'nuc'):
        os.makedirs(os.path.join(masks_dir, subdir), exist_ok=True)

    spacing = 4 * half_width_floor
    for frame, frame_df in embedded_df.groupby('frame'):
        pixel_ids = sorted(frame_df['pixel_id'].unique())
        n_cols = max(1, int(np.ceil(np.sqrt(len(pixel_ids)))))
        canvas_size = spacing * (n_cols + 2)
        mask = np.zeros((canvas_size, canvas_size), dtype=np.uint16)

        for i, pixel_id in enumerate(pixel_ids):
            row, col = divmod(i, n_cols)
            cy, cx = spacing * (row + 1), spacing * (col + 1)
            mask[cy - half_width_floor:cy + half_width_floor, cx - half_width_floor:cx + half_width_floor] = pixel_id

        condition = frame_df['condition'].iloc[0]
        for subdir in ('cell', 'nuc'):
            path = os.path.join(masks_dir, subdir, f"{condition}_frame_{int(frame):02d}_{subdir}_mask_sampled.tiff")
            tifffile.imwrite(path, mask)


def test_export_clones_by_subclass_writes_png_svg_and_summary_csv_per_subclass():
    embedded_df, dist_matrix, label_to_index = _synthetic_embedded_df_and_distances()
    subclasses = ['persisting', 'slow_prolif', 'fast_prolif', 'dying']

    selections = select_clones_by_subclass(
        embedded_df, dist_matrix, label_to_index, n_clones=2, min_embedded_cells=10, subclasses=subclasses,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        masks_dir = os.path.join(tmp_dir, 'masks')
        _write_synthetic_masks(embedded_df, masks_dir, half_width_floor=10)

        combined = export_clones_by_subclass(
            embedded_df, selections, dist_matrix, label_to_index, masks_dir, 'cell',
            tmp_dir, 'cell_umap', 'UMAP', 'cell morphology', n_clones=2, half_width=10,
        )

        assert os.path.isfile(os.path.join(tmp_dir, 'cell_umap_persisting.png'))
        assert os.path.isfile(os.path.join(tmp_dir, 'svgs', 'cell_umap_persisting.svg'))
        assert os.path.isfile(os.path.join(tmp_dir, 'cell_umap_fast_prolif.png'))
        assert os.path.isfile(os.path.join(tmp_dir, 'svgs', 'cell_umap_fast_prolif.svg'))

        # the medoid-shapes row is a SEPARATE file from the embedding
        # scatter, same "export embedding and shapes separately"
        # convention as shared.representative_shapes
        assert os.path.isfile(os.path.join(tmp_dir, 'cell_umap_persisting_shapes.png'))
        assert os.path.isfile(os.path.join(tmp_dir, 'svgs', 'cell_umap_persisting_shapes.svg'))

        # 'dying' has ZERO eligible clones (its only candidate is below
        # min_embedded_cells) - it must still get PNG/SVGs (all-placeholder
        # panels, for both the embedding scatter and the shapes row), not
        # be silently skipped, so every requested subclass's images stay
        # present and the same size as their siblings
        assert os.path.isfile(os.path.join(tmp_dir, 'cell_umap_dying.png')), \
            "a subclass with zero eligible clones must still get a placeholder PNG"
        assert os.path.isfile(os.path.join(tmp_dir, 'svgs', 'cell_umap_dying.svg'))
        assert os.path.isfile(os.path.join(tmp_dir, 'cell_umap_dying_shapes.png')), \
            "a subclass with zero eligible clones must still get a placeholder shapes PNG too"
        assert os.path.isfile(os.path.join(tmp_dir, 'svgs', 'cell_umap_dying_shapes.svg'))

        summary_csv_path = os.path.join(tmp_dir, 'cell_umap_clone_summary.csv')
        assert os.path.isfile(summary_csv_path)
        on_disk = pd.read_csv(summary_csv_path)
        assert set(on_disk.columns) == {'subclass', 'lin_id', 'n_cells', 'mean_pairwise_distance', 'medoid_frame', 'medoid_pixel_id'}

        persisting_rows = on_disk[on_disk['subclass'] == 'persisting']
        assert list(persisting_rows['lin_id']) == ['p_tight', 'p_mid'], \
            f"expected the 2 tightest persisting clones in tightness order, got {list(persisting_rows['lin_id'])}"
        assert persisting_rows['medoid_pixel_id'].notna().all(), "every eligible clone must have a resolved medoid"

        slow_prolif_rows = on_disk[on_disk['subclass'] == 'slow_prolif']
        assert len(slow_prolif_rows) == 1, f"expected only 1 eligible slow_prolif clone, got {len(slow_prolif_rows)}"
        assert slow_prolif_rows.iloc[0]['lin_id'] == 's_only'

        fast_prolif_rows = on_disk[on_disk['subclass'] == 'fast_prolif']
        assert list(fast_prolif_rows['lin_id']) == ['f_tight', 'f_loose']

        dying_rows = on_disk[on_disk['subclass'] == 'dying']
        assert dying_rows.empty, "dying has zero eligible clones - it should contribute no summary rows, even though it still got placeholder figures"

        assert len(combined) == len(on_disk)

    print("PASS: export_clones_by_subclass picks the tightest clones per subclass, writes embedding+shapes PNG+SVG "
          "pairs per subclass (with medoid info in the combined summary CSV) - handling a subclass with fewer than "
          "n_clones eligible clones")


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
    test_select_clones_by_subclass_matches_calling_pick_tightest_clones_directly()
    test_pick_clone_medoid_picks_the_cell_closest_to_the_rest_of_its_own_clone()
    test_pick_clone_medoid_ignores_cells_missing_from_the_distance_matrix()
    test_plot_clone_shapes_runs_without_error()
    test_plot_clone_shapes_always_renders_n_panels_with_a_placeholder_for_missing_ones()
    test_plot_clones_on_embedding_runs_without_error()
    test_plot_clones_on_embedding_handles_a_single_clone()
    test_plot_clones_on_embedding_always_renders_n_panels_the_same_size()
    test_export_clones_by_subclass_writes_png_svg_and_summary_csv_per_subclass()
    test_clone_colors_do_not_reuse_any_tab_color_or_region_color()
    print("\nAll tests passed.")
