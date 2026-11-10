'''
Run directly: python3 tests/shared/test_pair_similarity.py
'''
import os
import sys

import numpy as np

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

    observed_mean_distance, null_means, p_value = pair_distance_permutation_test(
        dist, pool_indices=np.arange(n), real_pairs=real_pairs,
        n_pairs_per_draw=len(real_pairs), n_permutations=2000, seed=1,
    )

    assert observed_mean_distance < 0.05
    assert p_value < 0.05, f"expected a small p-value for much-closer-than-random pairs, got {p_value}"
    print("PASS: real pairs that are much closer than the pool give a small p-value")


def test_real_pairs_much_farther_than_pool_give_a_large_p_value():
    rng = np.random.default_rng(0)
    n = 50
    dist = make_distance_matrix(n, rng) * 0.1   # pool-wide distances are small
    real_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    for i, j in real_pairs:
        dist[i, j] = dist[j, i] = 5.0   # artificially far apart

    observed_mean_distance, null_means, p_value = pair_distance_permutation_test(
        dist, pool_indices=np.arange(n), real_pairs=real_pairs,
        n_pairs_per_draw=len(real_pairs), n_permutations=2000, seed=1,
    )

    assert observed_mean_distance > 4.0
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

    _, null_means, _ = pair_distance_permutation_test(
        dist, pool_indices=np.arange(n), real_pairs=[(0, 1)],
        n_pairs_per_draw=3, n_permutations=500, seed=2,
    )

    assert (null_means >= 1.0).all(), (
        "a null draw's mean distance dropped below the smallest real off-diagonal distance - "
        "a self-pair (distance 0) was likely included instead of being redrawn"
    )
    print("PASS: the null never silently includes a zero-distance self-pair, even with a tiny pool")


def test_plot_pair_similarity_test_runs_with_multiple_markers():
    rng = np.random.default_rng(0)
    null_means = rng.normal(0.5, 0.1, 1000)
    fig = plot_pair_similarity_test(null_means, {
        'sisters': (0.3, 0.01),
        'cousins': (0.45, 0.2),
    }, title='test')
    assert fig is not None
    print("PASS: plot_pair_similarity_test runs without error with multiple observed markers")


if __name__ == '__main__':
    test_real_pairs_much_closer_than_pool_give_a_small_p_value()
    test_real_pairs_much_farther_than_pool_give_a_large_p_value()
    test_null_never_includes_a_self_pair_zero_distance()
    test_plot_pair_similarity_test_runs_with_multiple_markers()
    print("\nAll tests passed.")
