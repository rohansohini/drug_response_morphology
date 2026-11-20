'''
Run directly: python3 tests/shared/test_representative_shapes.py
'''
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.representative_shapes import (
    assign_kmeans_regions, crop_cell_shape, pick_representative_shapes,
    plot_embedding_by_region, plot_representative_shapes, REGION_COLORS,
)


def test_assign_kmeans_regions_separates_well_separated_clusters():
    rng = np.random.default_rng(0)
    n = 30
    pts = np.vstack([
        rng.normal([0, 0], 0.2, (n, 2)),
        rng.normal([20, 20], 0.2, (n, 2)),
        rng.normal([0, 20], 0.2, (n, 2)),
    ])
    df = pd.DataFrame({'dim_0': pts[:, 0], 'dim_1': pts[:, 1]})

    result = assign_kmeans_regions(df, n_regions=3)

    assert set(result['region'].unique()) == {0, 1, 2}
    # each of the 3 true groups must land entirely in its own region
    for start in [0, n, 2 * n]:
        group_regions = result['region'].iloc[start:start + n]
        assert group_regions.nunique() == 1, f"expected one region per well-separated cluster, got {group_regions.unique()}"

    print("PASS: assign_kmeans_regions puts each well-separated cluster into its own region")


def _write_mask_with_two_cells(path, shape=(400, 400)):
    mask = np.zeros(shape, dtype=np.uint16)
    mask[100:110, 100:110] = 1   # cell 1, near (105, 105)
    mask[300:320, 300:320] = 2   # cell 2, near (310, 310) - bigger, near the corner
    tifffile.imwrite(path, mask)
    return mask


def test_crop_cell_shape_isolates_only_the_requested_cell():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'mask.tiff')
        _write_mask_with_two_cells(path)

        crop = crop_cell_shape(path, pixel_id=1, half_width=50)

        assert crop.shape == (100, 100), f"expected a fixed (100, 100) crop, got {crop.shape}"
        assert set(np.unique(crop)) == {0, 1}, "expected only background (0) and cell 1's own id in the crop"
        assert (crop == 1).sum() == 100, "expected all 100 of cell 1's pixels to appear in the crop"

        print("PASS: crop_cell_shape isolates only the requested cell's own pixels")


def test_crop_cell_shape_same_size_even_near_image_edge():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'mask.tiff')
        mask = np.zeros((200, 200), dtype=np.uint16)
        mask[0:10, 0:10] = 5   # cell right at the top-left corner
        tifffile.imwrite(path, mask)

        crop = crop_cell_shape(path, pixel_id=5, half_width=50)

        assert crop.shape == (100, 100), f"expected the same fixed (100, 100) crop even at the image edge, got {crop.shape}"
        assert (crop == 5).sum() == 100, "expected all of the edge cell's pixels to still appear, just shifted within the crop"

        print("PASS: crop_cell_shape keeps a consistent output size even when the cell sits at the image edge")


def test_crop_cell_shape_warns_when_half_width_too_small():
    import io
    import contextlib

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'mask.tiff')
        mask = np.zeros((400, 400), dtype=np.uint16)
        mask[100:200, 100:200] = 1   # a 100x100 cell
        tifffile.imwrite(path, mask)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            crop_cell_shape(path, pixel_id=1, half_width=10)   # way too small for a 100x100 cell
        assert 'WARNING' in buf.getvalue() and 'too small' in buf.getvalue()

        print("PASS: crop_cell_shape warns when half_width can't fully contain the cell")


def test_pick_representative_shapes_picks_the_cell_closest_to_each_centroid():
    with tempfile.TemporaryDirectory() as tmp:
        masks_dir = os.path.join(tmp, 'masks')
        os.makedirs(os.path.join(masks_dir, 'cell'))
        mask = _write_mask_with_two_cells(os.path.join(masks_dir, 'cell', 'naive_frame_01_cell_mask_sampled.tiff'))

        # two regions, each with points tightly around one of two centers -
        # pixel_id 1 should win region 0 (closest to (0,0)), pixel_id 2
        # should win region 1 (closest to (10,10))
        df = pd.DataFrame({
            'dim_0':    [0.0, 0.1, -0.1, 10.0, 10.1, 9.9],
            'dim_1':    [0.0, 0.1, -0.1, 10.0, 10.1, 9.9],
            'condition': ['naive'] * 6,
            'frame':     [1] * 6,
            'pixel_id':  [1, 1, 1, 2, 2, 2],
        })

        df_r, shapes = pick_representative_shapes(df, masks_dir, 'cell', n_regions=2, half_width=20)

        assert len(shapes) == 2
        assert set(shapes['pixel_id']) == {1, 2}
        for _, row in shapes.iterrows():
            assert row['crop'].shape == (40, 40)
            assert (row['crop'] == row['pixel_id']).sum() > 0

        print("PASS: pick_representative_shapes picks the real cell closest to each region's own centroid")


def test_plot_embedding_by_region_runs_without_error():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    n = 20
    df = pd.DataFrame({
        'dim_0': rng.normal(0, 1, n), 'dim_1': rng.normal(0, 1, n),
        'region': [i % 2 for i in range(n)],
    })

    fig = plot_embedding_by_region(df, 'UMAP', 'cell morphology (representative regions)')
    assert fig is not None
    plt.close(fig)

    print("PASS: plot_embedding_by_region runs without error")


def test_plot_representative_shapes_runs_without_error():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    shapes = pd.DataFrame({
        'region': [0, 1], 'condition': ['naive', 'naive'], 'frame': [1, 1], 'pixel_id': [1, 2],
        'crop': [np.zeros((10, 10), dtype=np.uint16), np.ones((10, 10), dtype=np.uint16)],
    })

    fig = plot_representative_shapes(shapes, 'cell morphology (representative regions)', n_regions=2)
    assert fig is not None
    plt.close(fig)

    print("PASS: plot_representative_shapes runs without error")


def test_plot_representative_shapes_handles_a_single_region():
    # a single-subplot figure (n_regions=1) returns a bare Axes, not an
    # array, from plt.subplots - must not crash when iterating over it
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    shapes = pd.DataFrame({
        'region': [0], 'condition': ['naive'], 'frame': [1], 'pixel_id': [1],
        'crop': [np.ones((10, 10), dtype=np.uint16)],
    })

    fig = plot_representative_shapes(shapes, 'cell morphology', n_regions=1)
    assert fig is not None
    plt.close(fig)

    print("PASS: plot_representative_shapes handles a single region without crashing")


def test_plot_representative_shapes_always_renders_n_regions_panels():
    # regression test: figure size must come from the REQUESTED n_regions,
    # not from however many regions `shapes` actually has - otherwise a
    # region missing its own representative shape (for any reason) would
    # silently shrink the whole image, making it a different size from
    # a sibling plot (e.g. the same analysis's other modality/embedding
    # method) that didn't lose a region.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    shapes = pd.DataFrame({
        'region': [0, 2, 4],   # regions 1 and 3 have no representative shape at all
        'crop': [np.ones((10, 10), dtype=np.uint16)] * 3,
    })

    full_shapes = pd.DataFrame({
        'region': [0, 1, 2, 3, 4],
        'crop': [np.ones((10, 10), dtype=np.uint16)] * 5,
    })

    fig_missing = plot_representative_shapes(shapes, 'cell morphology', n_regions=5)
    fig_full = plot_representative_shapes(full_shapes, 'cell morphology', n_regions=5)

    assert len(fig_missing.axes) == 5, f"expected 5 panels even with 2 regions missing, got {len(fig_missing.axes)}"
    assert fig_missing.get_size_inches().tolist() == fig_full.get_size_inches().tolist(), (
        "a figure with missing regions must be the exact same size as one with every region present"
    )

    missing_titles = [ax.get_title() for ax in fig_missing.axes]
    assert missing_titles == ['region 0', 'region 1', 'region 2', 'region 3', 'region 4']

    plt.close(fig_missing)
    plt.close(fig_full)

    print("PASS: plot_representative_shapes always renders n_regions panels, same size whether or not every region has a shape")


def test_region_colors_do_not_reuse_any_tab_color():
    # every analysis using this module already uses the full matplotlib
    # 'tab:*' palette elsewhere (condition/subclass/group colors) - a
    # region color identical to one of those would make two unrelated
    # categories look like the same thing across different plots
    tab_colors = {
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    }
    assert not (set(c.lower() for c in REGION_COLORS) & tab_colors), (
        "a REGION_COLORS entry collides with a standard matplotlib 'tab:*' color"
    )
    print("PASS: REGION_COLORS doesn't reuse any standard 'tab:*' color")


if __name__ == '__main__':
    test_assign_kmeans_regions_separates_well_separated_clusters()
    test_crop_cell_shape_isolates_only_the_requested_cell()
    test_crop_cell_shape_same_size_even_near_image_edge()
    test_crop_cell_shape_warns_when_half_width_too_small()
    test_pick_representative_shapes_picks_the_cell_closest_to_each_centroid()
    test_plot_embedding_by_region_runs_without_error()
    test_plot_representative_shapes_runs_without_error()
    test_plot_representative_shapes_handles_a_single_region()
    test_plot_representative_shapes_always_renders_n_regions_panels()
    test_region_colors_do_not_reuse_any_tab_color()
    print("\nAll tests passed.")
