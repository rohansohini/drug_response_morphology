'''
Run directly: python3 tests/shared/test_density_plots.py
'''
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.density_plots import plot_embedding_with_kde, export_kde_by_frame, export_kde, plot_embedding_with_weighted_kde


def test_legend_shows_well_sampled_groups_only():
    # regression test: seaborn's kdeplot(fill=True, label=...) does NOT
    # attach a legend-visible label in this seaborn/matplotlib version
    # combo (confirmed directly: the resulting collection gets an
    # auto '_child0' label, which matplotlib's legend() ignores by
    # convention) - plot_embedding_with_kde must build the legend from
    # manual proxy patches instead, or it silently renders with no legend
    # at all.
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'dim_0': np.concatenate([rng.normal(0, 1, 50), rng.normal(5, 1, 50), rng.normal(0, 1, 2)]),
        'dim_1': np.concatenate([rng.normal(0, 1, 50), rng.normal(5, 1, 50), rng.normal(0, 1, 2)]),
        'condition': ['naive'] * 50 + ['A1_1uM_vem'] * 50 + ['rare_group'] * 2,
    })

    fig, ax = plt.subplots()
    plot_embedding_with_kde(
        df, 'condition',
        {'naive': 'tab:blue', 'A1_1uM_vem': 'tab:red', 'rare_group': 'tab:green'},
        ax, 'test title', 'UMAP', min_group_size=5,
    )

    legend = ax.get_legend()
    assert legend is not None, "expected a legend to be created"
    labels = [t.get_text() for t in legend.get_texts()]

    assert len(labels) == 2, f"expected exactly 2 legend entries (rare_group has < min_group_size), got {labels}"
    assert any('naive' in l for l in labels)
    assert any('A1_1uM_vem' in l for l in labels)
    assert not any('rare_group' in l for l in labels)

    print("PASS: legend shows exactly the well-sampled groups, with real labels (not empty/auto-generated)")


def _pdf_page_count(path):
    # no pdf-parsing library is in this project's dependencies - count
    # page objects directly in the raw bytes instead of adding one. Each
    # page object is "/Type /Page" (no trailing word char); the one
    # "/Type /Pages" tree-root object has a trailing 's' so \b excludes it.
    import re
    with open(path, 'rb') as f:
        data = f.read()
    return len(re.findall(rb'/Type\s*/Page\b', data))


def test_export_kde_by_frame_writes_pdf_and_one_svg_per_frame(tmp_dir='/tmp/claude-0/-home-user/8db9bf46-8a41-5caa-bf93-3b3ba7caed73/scratchpad/export_test'):
    import shutil
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    rng = np.random.default_rng(0)
    rows = []
    for frame in [1, 2, 3]:
        for cond, offset in [('naive', 0), ('A1_1uM_vem', 5)]:
            for _ in range(10):
                rows.append({'frame': frame, 'condition': cond,
                             'dim_0': rng.normal(offset, 1), 'dim_1': rng.normal(offset, 1)})
    df = pd.DataFrame(rows)

    export_kde_by_frame(df, 'condition', {'naive': 'tab:blue', 'A1_1uM_vem': 'tab:red'},
                         tmp_dir, 'cell_umap', 'UMAP')

    svg_dir = os.path.join(tmp_dir, 'svgs')
    expected_svgs = {f'cell_umap_frame_0{f}.svg' for f in [1, 2, 3]}

    # PDF and GIF sit directly in tmp_dir (no subfolder) - only the SVGs,
    # since there are many, get their own subfolder
    assert set(os.listdir(tmp_dir)) == {'cell_umap.pdf', 'cell_umap.gif', 'svgs'}, \
        f"expected cell_umap.pdf, cell_umap.gif, and svgs/ directly in {tmp_dir}, got {os.listdir(tmp_dir)}"
    assert set(os.listdir(svg_dir)) == expected_svgs, f"got {os.listdir(svg_dir)}"

    pdf_path = os.path.join(tmp_dir, 'cell_umap.pdf')
    assert os.path.getsize(pdf_path) > 0, "cell_umap.pdf is empty"
    assert _pdf_page_count(pdf_path) == 3, \
        f"expected 3 pages (one per frame) in the combined PDF, got {_pdf_page_count(pdf_path)}"

    for f in expected_svgs:
        assert os.path.getsize(os.path.join(svg_dir, f)) > 0, f"{f} is empty"

    shutil.rmtree(tmp_dir)
    print("PASS: export_kde_by_frame writes a flat PDF+GIF, with only SVGs in their own subfolder")


def test_export_kde_by_frame_gif_has_one_frame_per_input_frame_at_the_right_fps(tmp_dir='/tmp/claude-0/-home-user/8db9bf46-8a41-5caa-bf93-3b3ba7caed73/scratchpad/export_gif_test'):
    import shutil
    from PIL import Image, ImageSequence
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    rng = np.random.default_rng(0)
    rows = []
    for frame in [1, 2, 3, 4]:
        for cond, offset in [('naive', 0), ('A1_1uM_vem', 5)]:
            for _ in range(10):
                rows.append({'frame': frame, 'condition': cond,
                             'dim_0': rng.normal(offset, 1), 'dim_1': rng.normal(offset, 1)})
    df = pd.DataFrame(rows)

    export_kde_by_frame(df, 'condition', {'naive': 'tab:blue', 'A1_1uM_vem': 'tab:red'},
                         tmp_dir, 'cell_umap', 'UMAP', fps=2)

    gif_path = os.path.join(tmp_dir, 'cell_umap.gif')
    assert os.path.getsize(gif_path) > 0, "cell_umap.gif is empty"

    with Image.open(gif_path) as gif:
        gif_frames = list(ImageSequence.Iterator(gif))
        assert len(gif_frames) == 4, f"expected 4 GIF frames (one per input frame), got {len(gif_frames)}"

        sizes = {f.size for f in gif_frames}
        assert len(sizes) == 1, f"expected every GIF frame to have the SAME pixel size, got {sizes}"

        # 2 fps -> 500ms per frame
        durations = {f.info.get('duration') for f in gif_frames}
        assert durations == {500}, f"expected 500ms/frame at fps=2, got {durations}"

    shutil.rmtree(tmp_dir)
    print("PASS: export_kde_by_frame's GIF has one frame per input frame, uniform size, and the right fps")


def test_export_kde_writes_a_single_flat_pdf_and_svg():
    tmp_dir = '/tmp/claude-0/-home-user/8db9bf46-8a41-5caa-bf93-3b3ba7caed73/scratchpad/export_kde_test'
    import shutil
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'dim_0': rng.normal(0, 1, 60), 'dim_1': rng.normal(0, 1, 60),
        'group': ['naive (frames 1-5)'] * 30 + ['vem (frames 10-13)'] * 30,
    })

    export_kde(df, 'group', {'naive (frames 1-5)': 'tab:blue', 'vem (frames 10-13)': 'tab:red'},
               tmp_dir, 'cell_umap', 'UMAP')

    svg_dir = os.path.join(tmp_dir, 'svgs')
    assert set(os.listdir(tmp_dir)) == {'cell_umap.pdf', 'svgs'}, \
        f"expected cell_umap.pdf and svgs/ directly in {tmp_dir}, got {os.listdir(tmp_dir)}"
    assert os.listdir(svg_dir) == ['cell_umap.svg']
    assert os.path.getsize(os.path.join(tmp_dir, 'cell_umap.pdf')) > 0
    assert os.path.getsize(os.path.join(svg_dir, 'cell_umap.svg')) > 0

    shutil.rmtree(tmp_dir)
    print("PASS: export_kde writes a single flat PDF + one SVG in its own subfolder")


def test_kde_contours_are_unfilled_outlines_at_default_levels():
    # regression test: plot_embedding_with_kde must draw unfilled contour
    # lines (fill=False) at exactly the default levels (0.7, 0.8, 0.9),
    # not a filled density blob at 5 auto-spaced levels.
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'dim_0': rng.normal(0, 1, 100),
        'dim_1': rng.normal(0, 1, 100),
        'condition': ['naive'] * 100,
    })

    fig, ax = plt.subplots()
    plot_embedding_with_kde(df, 'condition', {'naive': 'tab:blue'}, ax, 'test title', 'UMAP', min_group_size=5)

    # both fill=True and fill=False produce a QuadContourSet - get_fill()
    # is the actual signal for filled vs. outline-only, and len(.levels)
    # confirms exactly 3 contour lines (0.7, 0.8, 0.9) rather than the old
    # default of 5 auto-spaced levels
    from matplotlib.contour import QuadContourSet
    contour_sets = [c for c in ax.collections if isinstance(c, QuadContourSet)]
    assert len(contour_sets) == 1, f"expected exactly one KDE contour set, got {len(contour_sets)}"
    assert contour_sets[0].get_fill() is False, "expected an unfilled (outline-only) KDE contour"
    assert len(contour_sets[0].levels) == 3, \
        f"expected exactly 3 contour levels (0.7, 0.8, 0.9), got {len(contour_sets[0].levels)}"

    print("PASS: KDE contours are drawn as unfilled outlines at exactly 3 levels")


def test_plot_embedding_with_kde_applies_explicit_xlim_ylim():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'dim_0': rng.normal(0, 1, 50),
        'dim_1': rng.normal(0, 1, 50),
        'condition': ['naive'] * 50,
    })

    fig, ax = plt.subplots()
    plot_embedding_with_kde(df, 'condition', {'naive': 'tab:blue'}, ax, 'test title', 'UMAP',
                             min_group_size=5, xlim=(-10, 10), ylim=(-20, 20))

    assert ax.get_xlim() == (-10, 10), f"expected xlim (-10, 10), got {ax.get_xlim()}"
    assert ax.get_ylim() == (-20, 20), f"expected ylim (-20, 20), got {ax.get_ylim()}"
    print("PASS: plot_embedding_with_kde applies explicit xlim/ylim when given")


def test_export_kde_by_frame_uses_identical_axes_across_frames():
    # regression test: each page must share the SAME axis limits, computed
    # once from every frame's data together - not auto-scale independently
    # per frame. Builds 3 frames with DELIBERATELY very different data
    # ranges/extents (frame 3's cloud is 10x wider than frame 1's) so
    # per-frame auto-scaling would clearly produce different axes if the
    # fix weren't applied.
    rng = np.random.default_rng(0)
    rows = []
    for frame, scale in [(1, 1), (2, 3), (3, 10)]:
        for _ in range(30):
            rows.append({'frame': frame, 'condition': 'naive',
                         'dim_0': rng.normal(0, scale), 'dim_1': rng.normal(0, scale)})
    df = pd.DataFrame(rows)

    import shutil
    tmp_dir = '/tmp/claude-0/-home-user/8db9bf46-8a41-5caa-bf93-3b3ba7caed73/scratchpad/export_axes_test'
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    import shared.density_plots as dp
    seen_limits = {}
    orig = dp.plot_embedding_with_kde

    def spy(df_arg, group_col, group_colors, ax, title, embedding_label, **kwargs):
        seen_limits[title] = (kwargs.get('xlim'), kwargs.get('ylim'))
        return orig(df_arg, group_col, group_colors, ax, title, embedding_label, **kwargs)

    dp.plot_embedding_with_kde = spy
    try:
        dp.export_kde_by_frame(df, 'condition', {'naive': 'tab:blue'}, tmp_dir, 'cell', 'UMAP')
    finally:
        dp.plot_embedding_with_kde = orig

    assert len(seen_limits) == 3, f"expected 3 frame calls, got {len(seen_limits)}"
    distinct_xlims = set(v[0] for v in seen_limits.values())
    distinct_ylims = set(v[1] for v in seen_limits.values())
    assert len(distinct_xlims) == 1, f"expected identical xlim across all frames, got {distinct_xlims}"
    assert len(distinct_ylims) == 1, f"expected identical ylim across all frames, got {distinct_ylims}"
    assert list(distinct_xlims)[0] is not None, "xlim must not be None - axes must actually be fixed"

    shutil.rmtree(tmp_dir)
    print("PASS: export_kde_by_frame applies identical axis limits across every frame's page")


def test_weighted_kde_contours_are_unfilled_outlines_at_default_levels():
    # regression test: plot_embedding_with_weighted_kde must draw unfilled
    # contour lines (fill=False) at exactly the default levels
    # (0.7, 0.8, 0.9), matching the same outline-only convention as the
    # hard-label version.
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({
        'dim_0': rng.normal(0, 1, n),
        'dim_1': rng.normal(0, 1, n),
        'pct_dying': rng.uniform(0, 1, n),
    })

    fig, ax = plt.subplots()
    plot_embedding_with_weighted_kde(df, {'dying': 'pct_dying'}, {'dying': 'tab:red'}, ax, 'test title', 'UMAP')

    from matplotlib.contour import QuadContourSet
    contour_sets = [c for c in ax.collections if isinstance(c, QuadContourSet)]
    assert len(contour_sets) == 1, f"expected exactly one KDE contour set, got {len(contour_sets)}"
    assert contour_sets[0].get_fill() is False, "expected an unfilled (outline-only) KDE contour"
    assert len(contour_sets[0].levels) == 3, \
        f"expected exactly 3 contour levels (0.7, 0.8, 0.9), got {len(contour_sets[0].levels)}"

    print("PASS: weighted KDE contours are drawn as unfilled outlines at exactly 3 levels")


def test_weighted_kde_follows_the_weighted_region_not_just_point_density():
    # the central behavior this function exists for: two well-separated
    # clusters with EQUAL point density, but weight for 'dying' concentrated
    # almost entirely in cluster A and weight for 'persisting' concentrated
    # almost entirely in cluster B - each contour must center on its OWN
    # weighted cluster, not some density-only average of both.
    rng = np.random.default_rng(0)
    n = 150
    pts = np.vstack([rng.normal([0, 0], 0.5, (n, 2)), rng.normal([20, 20], 0.5, (n, 2))])
    df = pd.DataFrame({'dim_0': pts[:, 0], 'dim_1': pts[:, 1]})
    df['pct_dying']      = np.concatenate([np.full(n, 0.9), np.full(n, 0.1)])
    df['pct_persisting'] = 1 - df['pct_dying']

    fig, ax = plt.subplots()
    plot_embedding_with_weighted_kde(
        df, {'dying': 'pct_dying', 'persisting': 'pct_persisting'},
        {'dying': 'tab:red', 'persisting': 'tab:blue'}, ax, 'test title', 'UMAP',
    )

    from matplotlib.contour import QuadContourSet
    contour_sets = [c for c in ax.collections if isinstance(c, QuadContourSet)]
    assert len(contour_sets) == 2, f"expected 2 contour sets (dying, persisting), got {len(contour_sets)}"

    def centroid(cs):
        verts = np.concatenate([p.vertices for p in cs.get_paths()])
        return verts.mean(axis=0)

    dying_centroid, persisting_centroid = centroid(contour_sets[0]), centroid(contour_sets[1])

    assert np.allclose(dying_centroid, [0, 0], atol=2), \
        f"expected 'dying' contour centered near (0,0) where its weight concentrates, got {dying_centroid}"
    assert np.allclose(persisting_centroid, [20, 20], atol=2), \
        f"expected 'persisting' contour centered near (20,20) where its weight concentrates, got {persisting_centroid}"

    print("PASS: each weighted KDE contour follows its own weight's concentration, not shared point density")


def test_weighted_kde_skips_a_group_below_min_total_weight():
    rng = np.random.default_rng(0)
    n = 50
    df = pd.DataFrame({
        'dim_0': rng.normal(0, 1, n),
        'dim_1': rng.normal(0, 1, n),
        'pct_dying':      np.full(n, 0.5),
        'pct_persisting': np.full(n, 0.01),  # total weight 0.5 - well below min_total_weight
    })

    fig, ax = plt.subplots()
    plot_embedding_with_weighted_kde(
        df, {'dying': 'pct_dying', 'persisting': 'pct_persisting'},
        {'dying': 'tab:red', 'persisting': 'tab:blue'}, ax, 'test title', 'UMAP', min_total_weight=5,
    )

    legend = ax.get_legend()
    labels = [t.get_text() for t in legend.get_texts()]
    assert len(labels) == 1 and 'dying' in labels[0], \
        f"expected only 'dying' in the legend (persisting's total weight is below min_total_weight), got {labels}"

    print("PASS: a group whose total weight is below min_total_weight is skipped entirely")


def test_weighted_kde_legend_shows_total_weight():
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({
        'dim_0': rng.normal(0, 1, n),
        'dim_1': rng.normal(0, 1, n),
        'pct_dying': np.full(n, 0.8),
    })

    fig, ax = plt.subplots()
    plot_embedding_with_weighted_kde(df, {'dying': 'pct_dying'}, {'dying': 'tab:red'}, ax, 'test title', 'UMAP')

    legend = ax.get_legend()
    labels = [t.get_text() for t in legend.get_texts()]
    assert len(labels) == 1
    assert 'total weight=32.0' in labels[0], f"expected total weight 32.0 (40*0.8) in label, got {labels}"

    print("PASS: weighted KDE legend reports each group's total weight")


if __name__ == '__main__':
    test_legend_shows_well_sampled_groups_only()
    test_export_kde_by_frame_writes_pdf_and_one_svg_per_frame()
    test_export_kde_by_frame_gif_has_one_frame_per_input_frame_at_the_right_fps()
    test_export_kde_writes_a_single_flat_pdf_and_svg()
    test_kde_contours_are_unfilled_outlines_at_default_levels()
    test_plot_embedding_with_kde_applies_explicit_xlim_ylim()
    test_export_kde_by_frame_uses_identical_axes_across_frames()
    test_weighted_kde_contours_are_unfilled_outlines_at_default_levels()
    test_weighted_kde_follows_the_weighted_region_not_just_point_density()
    test_weighted_kde_skips_a_group_below_min_total_weight()
    test_weighted_kde_legend_shows_total_weight()
    print("\nAll tests passed.")
