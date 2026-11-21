'''
Run directly: python3 tests/shared/test_pair_similarity.py
'''
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.pair_similarity import pair_distance_permutation_test, plot_pair_similarity_test


def make_distance_matrix(n, rng):
    dist = rng.random((n, n))
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    return dist


def test_real_pairs_much_closer_than_pool_give_a_small_p_value():
    rng = np.random.default_rng(0)
    n = 50
    dist = make_distance_matrix(n, rng) + 0.5   # pool-wide distances are all >= 0.5 (off-diagonal)
    # make a handful of pairs artificially very close (near 0) - "sisters"
    real_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    for i, j in real_pairs:
        dist[i, j] = dist[j, i] = 0.01

    observed_distances, null_distances, p_value = pair_distance_permutation_test(
        dist, pool_indices=np.arange(n), real_pairs=real_pairs, n_null_samples=2000, seed=1,
    )

    assert observed_distances.shape == (4,)
    assert (observed_distances < 0.05).all()
    assert p_value < 0.05, f"expected a small p-value for much-closer-than-random pairs, got {p_value}"
    print("PASS: real pairs that are much closer than the pool give a small p-value")


def test_real_pairs_much_farther_than_pool_give_a_large_p_value():
    rng = np.random.default_rng(0)
    n = 50
    dist = make_distance_matrix(n, rng) * 0.1   # pool-wide distances are small
    real_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    for i, j in real_pairs:
        dist[i, j] = dist[j, i] = 5.0   # artificially far apart

    observed_distances, null_distances, p_value = pair_distance_permutation_test(
        dist, pool_indices=np.arange(n), real_pairs=real_pairs, n_null_samples=2000, seed=1,
    )

    assert (observed_distances > 4.0).all()
    assert p_value > 0.95, f"expected p-value near 1 for much-farther-than-random pairs, got {p_value}"
    print("PASS: real pairs that are much farther apart than the pool give a large p-value")


def test_null_never_includes_a_self_pair_zero_distance():
    # regression test: a and b are drawn independently with replacement,
    # so some draws would otherwise accidentally pick the same index
    # twice (distance 0 on the diagonal) and silently bias the null
    # downward - with a small pool this would happen often if not guarded
    rng = np.random.default_rng(0)
    n = 4   # small pool -> frequent collisions if unguarded
    dist = make_distance_matrix(n, rng) + 1.0   # every off-diagonal distance >= 1.0
    np.fill_diagonal(dist, 0.0)

    _, null_distances, _ = pair_distance_permutation_test(
        dist, pool_indices=np.arange(n), real_pairs=[(0, 1)], n_null_samples=500, seed=2,
    )

    assert (null_distances >= 1.0).all(), (
        "a null sample's distance dropped below the smallest real off-diagonal distance - "
        "a self-pair (distance 0) was likely included instead of being redrawn"
    )
    print("PASS: the null never silently includes a zero-distance self-pair, even with a tiny pool")


def test_plot_pair_similarity_test_runs_with_multiple_markers():
    rng = np.random.default_rng(0)
    null_distances = rng.normal(0.5, 0.1, 1000)
    fig = plot_pair_similarity_test(null_distances, {
        'sisters': (rng.normal(0.3, 0.05, 40), 0.01),
        'cousins': (rng.normal(0.45, 0.05, 30), 0.2),
    }, title='test')
    assert fig is not None
    print("PASS: plot_pair_similarity_test runs without error with multiple observed distributions")


def test_plot_pair_similarity_test_draws_one_violin_per_distribution_with_no_legend():
    # regression test: this used to overlay alpha-blended histograms on
    # one shared x-axis, which becomes hard to read once there are 2-3
    # distributions all piled on top of each other - now each
    # distribution (null + every observed group) gets its own violin at
    # its own x position instead, with the p-value in that violin's own
    # x-tick label rather than a legend (so there's no floating UI
    # element left that could ever overlap anything)
    rng = np.random.default_rng(0)
    null_distances = rng.normal(0.5, 0.1, 1000)
    fig = plot_pair_similarity_test(null_distances, {
        'sisters': (rng.normal(0.3, 0.05, 40), 0.0083),
        'cousins': (rng.normal(0.45, 0.05, 30), 0.2),
    }, title='test')
    ax = fig.axes[0]

    # matplotlib's violinplot draws each violin body as its own PolyCollection
    # (a FillBetweenPolyCollection, a subclass of it)
    from matplotlib.collections import PolyCollection
    violin_bodies = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert len(violin_bodies) == 3, f"expected 3 violins (null, sisters, cousins), got {len(violin_bodies)}"

    tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    assert any('null' in l for l in tick_labels)
    assert any('sisters' in l and '0.0083' in l for l in tick_labels)
    assert any('cousins' in l and '0.2' in l for l in tick_labels)

    assert ax.get_legend() is None, "a per-violin x-tick label replaces the legend entirely - there should be none"

    print("PASS: plot_pair_similarity_test draws one violin per distribution, p-values in the x-tick labels, no legend")


def test_plot_pair_similarity_test_rejects_too_many_labels():
    rng = np.random.default_rng(0)
    null_distances = rng.normal(0.5, 0.1, 100)
    observed = {f'group{i}': (rng.normal(0.3, 0.05, 10), 0.1) for i in range(6)}
    try:
        plot_pair_similarity_test(null_distances, observed)
    except AssertionError as e:
        assert 'colors' in str(e)
        print("PASS: plot_pair_similarity_test rejects more labels than it has colors for")
        return
    raise AssertionError("expected plot_pair_similarity_test to raise with too many labels")


if __name__ == '__main__':
    test_real_pairs_much_closer_than_pool_give_a_small_p_value()
    test_real_pairs_much_farther_than_pool_give_a_large_p_value()
    test_null_never_includes_a_self_pair_zero_distance()
    test_plot_pair_similarity_test_runs_with_multiple_markers()
    test_plot_pair_similarity_test_draws_one_violin_per_distribution_with_no_legend()
    test_plot_pair_similarity_test_rejects_too_many_labels()
    print("\nAll tests passed.")
