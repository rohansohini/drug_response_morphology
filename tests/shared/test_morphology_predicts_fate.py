'''
Run directly: python3 tests/shared/test_morphology_predicts_fate.py
'''
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.morphology_predicts_fate import (
    nearest_neighbor_indices, knn_predict_mixture, prediction_errors, mean_prediction_error,
    knn_prediction_permutation_test, plot_permutation_test,
)


def test_nearest_neighbor_indices_excludes_self_and_picks_the_true_k_nearest():
    # 5 points on a line: 0, 1, 2, 10, 20 - point 0's 2 nearest (excluding
    # itself) must be 1 and 2, not itself
    points = np.array([0, 1, 2, 10, 20])
    dist = np.abs(points[:, None] - points[None, :]).astype(float)

    neighbor_idx = nearest_neighbor_indices(dist, k=2)
    assert neighbor_idx.shape == (5, 2)
    assert set(neighbor_idx[0]) == {1, 2}, f"expected point 0's neighbors to be {{1, 2}}, got {set(neighbor_idx[0])}"
    assert 0 not in neighbor_idx[0], "a point must never be its own neighbor"

    print("PASS: nearest_neighbor_indices excludes self and finds the true k nearest neighbors")


def test_knn_predict_mixture_averages_neighbor_mixtures():
    neighbor_idx = np.array([[1, 2], [0, 2], [0, 1]])
    mixture = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])

    predicted = knn_predict_mixture(neighbor_idx, mixture)
    # point 0's neighbors are 1, 2 -> average of [0,1] and [0.5,0.5] = [0.25, 0.75]
    assert np.allclose(predicted[0], [0.25, 0.75]), f"got {predicted[0]}"

    print("PASS: knn_predict_mixture averages the correct neighbors' mixtures")


def test_prediction_errors_is_zero_per_point_for_a_perfect_prediction():
    mixture = np.array([[0.2, 0.8], [0.9, 0.1], [0.5, 0.5]])
    errors = prediction_errors(mixture, mixture)
    assert errors.shape == (3,)
    assert np.allclose(errors, 0.0)
    print("PASS: prediction_errors is 0 for every point when true == predicted")


def test_prediction_errors_matches_known_euclidean_distances():
    true_mixture = np.array([[0.0, 0.0], [1.0, 0.0]])
    predicted = np.array([[3.0, 4.0], [1.0, 0.0]])
    errors = prediction_errors(true_mixture, predicted)
    assert np.allclose(errors, [5.0, 0.0]), f"got {errors}"
    print("PASS: prediction_errors matches known per-point Euclidean distances")


def test_mean_prediction_error_is_zero_for_a_perfect_prediction():
    mixture = np.array([[0.2, 0.8], [0.9, 0.1]])
    assert mean_prediction_error(mixture, mixture) == 0.0
    print("PASS: mean_prediction_error is 0 for identical true/predicted mixtures")


def test_permutation_test_finds_strong_morphology_fate_relationship_when_one_exists():
    # two well-separated morphological clusters, each with its OWN fixed
    # mixture - a real cell's nearest neighbors are almost always in the
    # same cluster, so KNN prediction should be near-perfect and the real
    # errors should sit far below the null errors
    rng = np.random.default_rng(0)
    n_per_cluster = 60
    cluster_a = rng.normal(0, 1, (n_per_cluster, 2))
    cluster_b = rng.normal(50, 1, (n_per_cluster, 2))
    points = np.vstack([cluster_a, cluster_b])
    dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)

    mixture = np.vstack([
        np.tile([1.0, 0.0], (n_per_cluster, 1)),
        np.tile([0.0, 1.0], (n_per_cluster, 1)),
    ])

    observed_errors, null_errors, p_value, predicted = knn_prediction_permutation_test(
        dist, mixture, k=5, n_permutations=50, seed=0,
    )

    assert observed_errors.shape == (2 * n_per_cluster,)
    assert null_errors.shape == (50 * 2 * n_per_cluster,)
    assert observed_errors.mean() < 0.05, \
        f"expected near-perfect prediction (well-separated clusters), got mean error={observed_errors.mean()}"
    assert p_value < 0.01, f"expected a small p-value (real morphology clearly predicts fate here), got {p_value}"
    assert observed_errors.mean() < null_errors.mean(), "observed errors should be far below the null's mean"

    print("PASS: the permutation test correctly detects a strong real morphology-fate relationship")


def test_permutation_test_finds_no_relationship_when_fate_is_random():
    # mixture values assigned independently of position - no real
    # relationship, so the observed errors should look just like the
    # null, not stochastically smaller
    rng = np.random.default_rng(1)
    n = 150
    points = rng.normal(0, 1, (n, 2))
    dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    mixture = rng.dirichlet([1, 1], size=n)

    observed_errors, null_errors, p_value, predicted = knn_prediction_permutation_test(
        dist, mixture, k=5, n_permutations=50, seed=1,
    )

    assert p_value > 0.05, f"expected no significant relationship for random fate assignment, got p={p_value}"

    print("PASS: the permutation test correctly finds no relationship when fate is assigned at random")


def test_plot_permutation_test_runs_without_error():
    rng = np.random.default_rng(0)
    observed_errors = rng.normal(0.3, 0.05, 60)
    null_errors = rng.normal(0.5, 0.05, 6000)

    fig = plot_permutation_test(observed_errors, null_errors, p_value=0.085)
    assert fig is not None
    plt.close(fig)

    print("PASS: plot_permutation_test runs without error")


def test_plot_permutation_test_sets_the_given_title():
    rng = np.random.default_rng(0)
    observed_errors = rng.normal(0.3, 0.05, 60)
    null_errors = rng.normal(0.5, 0.05, 6000)

    fig = plot_permutation_test(observed_errors, null_errors, p_value=0.085, title='cell morphology')
    assert fig.axes[0].get_title() == 'cell morphology'
    plt.close(fig)

    fig = plot_permutation_test(observed_errors, null_errors, p_value=0.085)
    assert fig.axes[0].get_title() == '', "expected no title when title=None"
    plt.close(fig)

    print("PASS: plot_permutation_test sets the given title, or none at all when not given")


def test_plot_permutation_test_handles_a_p_value_of_exactly_zero():
    # Mann-Whitney U's p-value can (rarely) underflow to exactly 0.0 for
    # an extremely large effect - must still render something sensible,
    # not crash or silently report a misleadingly precise "p = 0.0000"
    rng = np.random.default_rng(0)
    observed_errors = rng.normal(0.3, 0.05, 60)
    null_errors = rng.normal(0.5, 0.05, 6000)

    fig = plot_permutation_test(observed_errors, null_errors, p_value=0.0)
    ax = fig.axes[0]
    text = ' '.join(t.get_text() for t in ax.texts)
    assert '<' in text, f"expected a '< ...' fallback for p_value=0, got: {text!r}"
    plt.close(fig)

    print("PASS: plot_permutation_test handles p_value == 0 without crashing or overstating precision")


if __name__ == '__main__':
    test_nearest_neighbor_indices_excludes_self_and_picks_the_true_k_nearest()
    test_knn_predict_mixture_averages_neighbor_mixtures()
    test_prediction_errors_is_zero_per_point_for_a_perfect_prediction()
    test_prediction_errors_matches_known_euclidean_distances()
    test_mean_prediction_error_is_zero_for_a_perfect_prediction()
    test_permutation_test_finds_strong_morphology_fate_relationship_when_one_exists()
    test_permutation_test_finds_no_relationship_when_fate_is_random()
    test_plot_permutation_test_runs_without_error()
    test_plot_permutation_test_sets_the_given_title()
    test_plot_permutation_test_handles_a_p_value_of_exactly_zero()
    print("\nAll tests passed.")
