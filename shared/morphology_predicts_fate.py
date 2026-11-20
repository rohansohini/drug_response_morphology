import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


def nearest_neighbor_indices(dist_matrix, k):
    '''
    The k nearest neighbors of every point in a square distance matrix,
    excluding each point itself.

    Inputs:
        dist_matrix: (n, n) symmetric distance matrix
        k:           number of neighbors per point (must be < n)
    Outputs:
        neighbor_idx: (n, k) int array - row i holds the indices of
                      point i's k nearest neighbors (not sorted by
                      distance within the row - every neighbor
                      contributes equally in knn_predict_mixture below,
                      so their relative order doesn't matter)
    '''
    n = dist_matrix.shape[0]
    assert 0 < k < n, f"k={k} must be between 1 and n-1={n - 1}"

    dist = dist_matrix.copy()
    np.fill_diagonal(dist, np.inf)
    return np.argpartition(dist, k, axis=1)[:, :k]


def knn_predict_mixture(neighbor_idx, mixture_matrix):
    '''
    Predict each point's own mixture as the plain average of its
    neighbors' (real) mixtures.

    Inputs:
        neighbor_idx:   (n, k) from nearest_neighbor_indices
        mixture_matrix: (n, n_categories) each point's own mixture
    Outputs:
        predicted: (n, n_categories)
    '''
    return mixture_matrix[neighbor_idx].mean(axis=1)


def prediction_errors(true_mixture, predicted_mixture):
    '''
    Per-point Euclidean distance between each point's true and predicted
    mixture - one number PER POINT, not collapsed to a single mean (see
    mean_prediction_error for that summary).
    '''
    return np.linalg.norm(true_mixture - predicted_mixture, axis=1)


def mean_prediction_error(true_mixture, predicted_mixture):
    '''Average (across points) of prediction_errors - one number summarizing how far off the predictions are overall.'''
    return prediction_errors(true_mixture, predicted_mixture).mean()


def knn_prediction_permutation_test(dist_matrix, mixture_matrix, k, n_permutations=200, seed=42):
    '''
    Test whether points close together in `dist_matrix` actually have
    more similar `mixture_matrix` rows than chance - not by collapsing
    everything to a single mean prediction error and comparing THAT
    against a null of means (which hides whether the effect holds
    consistently across cells, or is driven by just a few), but by
    comparing the FULL distribution of real per-cell prediction errors
    against a null distribution of individual per-cell errors, via a
    one-sided Mann-Whitney U test (are real errors stochastically
    SMALLER than null errors).

    The null is still built by permutation, the same mechanism as
    before: randomly reassign which mixture belongs to which point,
    breaking any real link between distance and mixture, while leaving
    the neighbor graph itself (computed once from the REAL distances,
    reused for every permutation - morphology never changes, only which
    label sits where) and the actual set of mixture values exactly as
    they are. What changed is that each permutation now contributes its
    own n individual per-point errors to the null, not just their mean -
    so far fewer permutations are needed than before (previously 10,000,
    since each one only contributed ONE number to the null; a few
    hundred is plenty now, since each one contributes n).

    Inputs:
        dist_matrix:     (n, n) morphological distance matrix (e.g. CAJAL
                         GW distances) - NOT a UMAP/PHATE projection,
                         which distorts distances and so is the wrong
                         representation for anything quantitative
        mixture_matrix:  (n, n_categories) each point's own fate mixture
        k:               number of nearest neighbors to average
        n_permutations:  number of random label reassignments pooled
                         into the null distribution (each contributes n
                         individual errors)
        seed:            RNG seed, for reproducibility
    Outputs:
        observed_errors: (n,) real per-point prediction errors
        null_errors:     (n_permutations * n,) pooled per-point errors
                         across every random reassignment
        p_value:         one-sided Mann-Whitney U p-value (alternative=
                         'less') - small means real morphology predicts
                         fate with consistently lower error than a
                         random reassignment typically manages
        predicted:       (n, n_categories) the real (unpermuted) KNN
                         predictions
    '''
    neighbor_idx = nearest_neighbor_indices(dist_matrix, k)

    predicted = knn_predict_mixture(neighbor_idx, mixture_matrix)
    observed_errors = prediction_errors(mixture_matrix, predicted)

    rng = np.random.default_rng(seed)
    n = mixture_matrix.shape[0]
    null_errors = np.empty(n_permutations * n)
    for i in range(n_permutations):
        shuffled = mixture_matrix[rng.permutation(n)]
        null_errors[i * n:(i + 1) * n] = prediction_errors(shuffled, knn_predict_mixture(neighbor_idx, shuffled))

    p_value = mannwhitneyu(observed_errors, null_errors, alternative='less').pvalue
    return observed_errors, null_errors, p_value, predicted


def plot_permutation_test(observed_errors, null_errors, p_value, title=None):
    '''
    The null distribution of individual prediction errors under random
    fate-mixture reassignment, overlaid with the real distribution of
    individual prediction errors (knn_prediction_permutation_test) - lets
    you see whether real morphology predicts fate consistently across
    cells, or only for a subset of them, rather than a single collapsed
    mean hiding that distinction. Both are density-normalized (not raw
    counts), since the null is typically pooled from many more points
    than there are real cells.

    Inputs:
        observed_errors: (n,) from knn_prediction_permutation_test
        null_errors:     from knn_prediction_permutation_test
        p_value:         from knn_prediction_permutation_test (one-sided
                         Mann-Whitney U testing real < null)
        title:           plot title, or None for no title
    Outputs:
        fig: caller saves/closes this
    '''
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.hist(null_errors, bins=50, density=True, color='lightgrey', edgecolor='grey', alpha=0.8, label='null (random reassignment)')
    ax.hist(observed_errors, bins=50, density=True, color='tab:red', edgecolor='darkred', alpha=0.5, label='real')

    p_label = f'< {np.finfo(float).tiny:.1e}' if p_value == 0 else f'= {p_value:.4g}'
    ax.text(0.97, 0.95, f'Mann-Whitney U\np {p_label}', transform=ax.transAxes, ha='right', va='top')

    ax.set_xlabel('prediction error')
    ax.set_ylabel('density')
    ax.legend()
    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    return fig
