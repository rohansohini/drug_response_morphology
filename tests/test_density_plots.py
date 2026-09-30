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
from helpers.density_plots import plot_umap_with_kde


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


if __name__ == '__main__':
    test_legend_shows_well_sampled_groups_only()
    print("\nAll tests passed.")
