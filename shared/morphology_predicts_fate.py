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
    One violin per distribution - the null (random reassignment) and the
    real (knn_prediction_permutation_test) distribution of individual
    prediction errors - side by side, not overlaid (same design as
    shared/pair_similarity.py's plot_pair_similarity_test: two overlaid
    histograms become hard to read once the real distribution's shift
    away from the null is subtle relative to either one's own spread,
    and both distributions are still shown in FULL - not collapsed to a
    single mean - so you can see whether the effect holds consistently
    across cells, or only for a subset of them).

    The p-value goes into the real violin's own x-tick label, not a
    legend - with both distributions already at their own x position,
    there's no separate floating UI element (a legend box, a text
    annotation) that could ever end up overlapping anything else.

    Inputs:
        observed_errors: (n,) from knn_prediction_permutation_test
        null_errors:     from knn_prediction_permutation_test
        p_value:         from knn_prediction_permutation_test (one-sided
                         Mann-Whitney U testing real < null)
        title:           plot title, or None for no title
    Outputs:
        fig: caller saves/closes this
    '''
    p_label = f'< {np.finfo(float).tiny:.1e}' if p_value == 0 else f'= {p_value:.4g}'
    labels = ['null\n(random reassignment)', f'real\n(p {p_label})']
    datasets = [null_errors, observed_errors]
    violin_colors = ['lightgrey', 'tab:red']

    fig, ax = plt.subplots(figsize=(5, 5))
    parts = ax.violinplot(datasets, showmedians=True)

    for body, color in zip(parts['bodies'], violin_colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.6)
    for key in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
        parts[key].set_color('black')
        parts[key].set_linewidth(1)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    # log2 scale, same reasoning/treatment as shared/pair_similarity.py's
    # plot_pair_similarity_test: prediction error is a strictly positive,
    # right-skewed distance, so a linear axis can compress the low end -
    # where the real shift between the null and real distributions
    # actually lives - into a sliver too small to see by eye
    ax.set_yscale('log', base=2)
    ax.set_ylabel('prediction error (log₂ scale)')
    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    return fig
