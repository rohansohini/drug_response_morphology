'''
Run directly: python3 tests/test_density_plots.py
'''
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from helpers.density_plots import plot_umap_with_kde, export_kde_by_frame


def test_legend_shows_well_sampled_groups_only():
    # regression test: seaborn's kdeplot(fill=True, label=...) does NOT
    # attach a legend-visible label in this seaborn/matplotlib version
    # combo (confirmed directly: the resulting collection gets an
    # auto '_child0' label, which matplotlib's legend() ignores by
    # convention) - plot_umap_with_kde must build the legend from manual
    # proxy patches instead, or it silently renders with no legend at all.
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'umap_0': np.concatenate([rng.normal(0, 1, 50), rng.normal(5, 1, 50), rng.normal(0, 1, 2)]),
        'umap_1': np.concatenate([rng.normal(0, 1, 50), rng.normal(5, 1, 50), rng.normal(0, 1, 2)]),
        'condition': ['naive'] * 50 + ['A1_1uM_vem'] * 50 + ['rare_group'] * 2,
    })

    fig, ax = plt.subplots()
    plot_umap_with_kde(
        df, 'condition',
        {'naive': 'tab:blue', 'A1_1uM_vem': 'tab:red', 'rare_group': 'tab:green'},
        ax, 'test title', min_group_size=5,
    )

    legend = ax.get_legend()
    assert legend is not None, "expected a legend to be created"
    labels = [t.get_text() for t in legend.get_texts()]

    assert len(labels) == 2, f"expected exactly 2 legend entries (rare_group has < min_group_size), got {labels}"
    assert any('naive' in l for l in labels)
    assert any('A1_1uM_vem' in l for l in labels)
    assert not any('rare_group' in l for l in labels)

    print("PASS: legend shows exactly the well-sampled groups, with real labels (not empty/auto-generated)")


def test_export_kde_by_frame_writes_matching_pdf_and_svg_per_frame(tmp_dir='/tmp/claude-0/-home-user/8db9bf46-8a41-5caa-bf93-3b3ba7caed73/scratchpad/export_test'):
    import shutil
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    rng = np.random.default_rng(0)
    rows = []
    for frame in [1, 2, 3]:
        for cond, offset in [('naive', 0), ('A1_1uM_vem', 5)]:
            for _ in range(10):
                rows.append({'frame': frame, 'condition': cond,
                             'umap_0': rng.normal(offset, 1), 'umap_1': rng.normal(offset, 1)})
    df = pd.DataFrame(rows)

    export_kde_by_frame(df, 'condition', {'naive': 'tab:blue', 'A1_1uM_vem': 'tab:red'},
                         tmp_dir, 'cell')

    pdf_dir = os.path.join(tmp_dir, 'pdfs')
    svg_dir = os.path.join(tmp_dir, 'svgs')
    expected_pdfs = {f'cell_frame_0{f}.pdf' for f in [1, 2, 3]}
    expected_svgs = {f'cell_frame_0{f}.svg' for f in [1, 2, 3]}

    assert set(os.listdir(pdf_dir)) == expected_pdfs, f"got {os.listdir(pdf_dir)}"
    assert set(os.listdir(svg_dir)) == expected_svgs, f"got {os.listdir(svg_dir)}"
    for f in expected_pdfs | expected_svgs:
        folder = pdf_dir if f.endswith('.pdf') else svg_dir
        assert os.path.getsize(os.path.join(folder, f)) > 0, f"{f} is empty"

    shutil.rmtree(tmp_dir)
    print("PASS: export_kde_by_frame writes one matching PDF and SVG per frame, in separate subfolders")


if __name__ == '__main__':
    test_legend_shows_well_sampled_groups_only()
    test_export_kde_by_frame_writes_matching_pdf_and_svg_per_frame()
    print("\nAll tests passed.")
