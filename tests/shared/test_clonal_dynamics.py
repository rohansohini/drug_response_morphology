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
    CLONE_COLORS, pick_top_clones, plot_clones_on_embedding, export_clones_by_subclass,
)


def _full_df(lin_id_counts: dict):
    rows = []
    for lin_id, n in lin_id_counts.items():
        rows.extend({'lin_id': lin_id} for _ in range(n))
    return pd.DataFrame(rows)


def _embedded_df(lin_id_cells: dict):
    # lin_id_cells: {lin_id: [(frame, dim_0, dim_1), ...]}
    rows = []
    for lin_id, cells in lin_id_cells.items():
        for frame, dim_0, dim_1 in cells:
            rows.append({'lin_id': lin_id, 'frame': frame, 'dim_0': dim_0, 'dim_1': dim_1})
    return pd.DataFrame(rows)


def test_pick_top_clones_ranks_by_true_count_not_embedded_count():
    # lin_id 'B' has fewer embedded cells than 'A' but far more TRUE
    # cells - it must still win, since "most segmented cells" means the
    # true population, not an artifact of the random CAJAL sampling budget
    full_df = _full_df({'A': 50, 'B': 100})
    embedded_df = _embedded_df({
        'A': [(f, 0, 0) for f in range(20)],   # 20 embedded cells
        'B': [(f, 0, 0) for f in range(6)],    # only 6 embedded cells, but both are >= floor
    })

    top_lin_ids, summary = pick_top_clones(full_df, embedded_df, n_clones=2, min_embedded_cells=5)

    assert top_lin_ids == ['B', 'A'], f"expected B (100 true cells) ranked above A (50), got {top_lin_ids}"
    assert summary.loc['B', 'true_n_cells'] == 100
    assert summary.loc['B', 'embedded_n_cells'] == 6

    print("PASS: pick_top_clones ranks by TRUE cell count, not embedded-sample count")


def test_pick_top_clones_filters_by_min_embedded_cells():
    # lin_id 'C' is the biggest TRUE clone by far, but has only 2 embedded
    # cells (below the floor) - it must be excluded entirely, since a
    # clone with no plottable points can't be painted onto the embedding
    full_df = _full_df({'A': 50, 'C': 1000})
    embedded_df = _embedded_df({
        'A': [(f, 0, 0) for f in range(10)],
        'C': [(0, 0, 0), (0, 0, 0)],   # only 2 embedded cells
    })

    top_lin_ids, summary = pick_top_clones(full_df, embedded_df, n_clones=2, min_embedded_cells=5)

    assert top_lin_ids == ['A'], f"expected only A to be eligible (C is below the embedded-cell floor), got {top_lin_ids}"

    print("PASS: pick_top_clones excludes a clone below min_embedded_cells, however big its true count")


def test_pick_top_clones_breaks_ties_by_frame_span():
    # A and B have identical true AND embedded cell counts, but B's
    # embedded cells are spread across more distinct frames - B should
    # win the tie
    full_df = _full_df({'A': 10, 'B': 10})
    embedded_df = _embedded_df({
        'A': [(0, 0, 0)] * 8,                       # all 8 embedded cells on frame 0
        'B': [(f, 0, 0) for f in range(8)],         # 8 embedded cells across 8 different frames
    })

    top_lin_ids, summary = pick_top_clones(full_df, embedded_df, n_clones=2, min_embedded_cells=5)

    assert top_lin_ids == ['B', 'A'], f"expected B (more frame-spread) to win the tie over A, got {top_lin_ids}"

    print("PASS: pick_top_clones breaks a true-count tie by preferring more distinct embedded frames")


def test_pick_top_clones_returns_fewer_than_n_clones_if_not_enough_eligible():
    full_df = _full_df({'A': 50})
    embedded_df = _embedded_df({'A': [(f, 0, 0) for f in range(10)]})

    top_lin_ids, summary = pick_top_clones(full_df, embedded_df, n_clones=3, min_embedded_cells=5)

    assert top_lin_ids == ['A'], f"expected just the 1 eligible clone, got {top_lin_ids}"
    assert len(summary) == 1

    print("PASS: pick_top_clones returns fewer than n_clones when fewer are eligible, rather than erroring")


def test_plot_clones_on_embedding_runs_without_error():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    background_df = pd.DataFrame({'dim_0': rng.normal(0, 1, 100), 'dim_1': rng.normal(0, 1, 100)})
    clone_dfs = [
        pd.DataFrame({'dim_0': rng.normal(i, 0.3, 5), 'dim_1': rng.normal(i, 0.3, 5)})
        for i in range(3)
    ]

    fig = plot_clones_on_embedding(background_df, clone_dfs, ['lin1', 'lin2', 'lin3'], CLONE_COLORS, 'UMAP', 'test title')
    assert fig is not None
    assert len(fig.axes) == 3
    plt.close(fig)

    print("PASS: plot_clones_on_embedding runs without error for 3 clones")


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


def _synthetic_condition_data(seed=0):
    rng = np.random.default_rng(seed)

    full_rows = []
    embedded_rows = []
    for subclass, lin_id_counts in [
        ('dying',       {'d1': 30, 'd2': 20, 'd3': 10}),
        ('fast_prolif', {'f1': 200, 'f2': 150}),   # only 2 eligible clones here, fewer than n_clones=3
    ]:
        for lin_id, true_n in lin_id_counts.items():
            full_rows.extend({'lin_id': lin_id, 'subclass': subclass} for _ in range(true_n))
            n_embedded = min(true_n, 15)
            for i in range(n_embedded):
                embedded_rows.append({
                    'lin_id': lin_id, 'subclass': subclass, 'frame': i % 5,
                    'dim_0': rng.normal(0, 1), 'dim_1': rng.normal(0, 1),
                })

    return pd.DataFrame(full_rows), pd.DataFrame(embedded_rows)


def test_export_clones_by_subclass_writes_pdf_svg_and_summary_csv_per_subclass():
    full_df, embedded_df = _synthetic_condition_data()

    with tempfile.TemporaryDirectory() as tmp_dir:
        combined = export_clones_by_subclass(
            embedded_df, full_df, tmp_dir, 'cell_umap', 'UMAP', 'cell morphology',
            n_clones=3, min_embedded_cells=5,
        )

        assert os.path.isfile(os.path.join(tmp_dir, 'cell_umap_dying.pdf'))
        assert os.path.isfile(os.path.join(tmp_dir, 'svgs', 'cell_umap_dying.svg'))
        assert os.path.isfile(os.path.join(tmp_dir, 'cell_umap_fast_prolif.pdf'))
        assert os.path.isfile(os.path.join(tmp_dir, 'svgs', 'cell_umap_fast_prolif.svg'))

        summary_csv_path = os.path.join(tmp_dir, 'cell_umap_clone_summary.csv')
        assert os.path.isfile(summary_csv_path)
        on_disk = pd.read_csv(summary_csv_path)
        assert set(on_disk.columns) == {'subclass', 'lin_id', 'true_n_cells', 'embedded_n_cells', 'embedded_n_frames'}

        dying_rows = on_disk[on_disk['subclass'] == 'dying']
        assert len(dying_rows) == 3, f"expected all 3 dying clones to be eligible, got {len(dying_rows)}"
        assert set(dying_rows['lin_id']) == {'d1', 'd2', 'd3'}

        fast_prolif_rows = on_disk[on_disk['subclass'] == 'fast_prolif']
        assert len(fast_prolif_rows) == 2, f"expected only 2 eligible fast_prolif clones (fewer than n_clones=3), got {len(fast_prolif_rows)}"

        assert combined.equals(on_disk) or len(combined) == len(on_disk)

    print("PASS: export_clones_by_subclass writes a PDF+SVG per subclass and a combined summary CSV, "
          "handling a subclass with fewer than n_clones eligible clones")


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
    test_pick_top_clones_ranks_by_true_count_not_embedded_count()
    test_pick_top_clones_filters_by_min_embedded_cells()
    test_pick_top_clones_breaks_ties_by_frame_span()
    test_pick_top_clones_returns_fewer_than_n_clones_if_not_enough_eligible()
    test_plot_clones_on_embedding_runs_without_error()
    test_plot_clones_on_embedding_handles_a_single_clone()
    test_export_clones_by_subclass_writes_pdf_svg_and_summary_csv_per_subclass()
    test_clone_colors_do_not_reuse_any_tab_color_or_region_color()
    print("\nAll tests passed.")
