import numpy as np
import matplotlib.pyplot as plt


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


def mean_prediction_error(true_mixture, predicted_mixture):
    '''
    Average (across points) Euclidean distance between each point's true
    and predicted mixture - one number summarizing how far off the
    predictions are overall.
    '''
    return np.linalg.norm(true_mixture - predicted_mixture, axis=1).mean()


def knn_prediction_permutation_test(dist_matrix, mixture_matrix, k, n_permutations=1000, seed=42):
    '''
    Test whether points close together in `dist_matrix` actually have
    more similar `mixture_matrix` rows than chance, by comparing
    k-nearest-neighbor mixture prediction (see knn_predict_mixture)
    against a null built by randomly reassigning which mixture belongs
    to which point - breaking any real link between distance and
    mixture, while leaving the neighbor graph itself (computed once from
    the REAL distances, reused for every permutation - morphology never
    changes, only which label sits where) and the actual set of mixture
    values exactly as they are.

    Inputs:
        dist_matrix:     (n, n) morphological distance matrix (e.g. CAJAL
                         GW distances) - NOT a UMAP/PHATE projection,
                         which distorts distances and so is the wrong
                         representation for anything quantitative
        mixture_matrix:  (n, n_categories) each point's own fate mixture
        k:               number of nearest neighbors to average
        n_permutations:  number of random label reassignments making up
                         the null distribution
        seed:            RNG seed, for reproducibility
    Outputs:
        observed_error: mean_prediction_error using the REAL mixture assignment
        null_errors:     (n_permutations,) mean_prediction_error for each
                         random reassignment
        p_value:         fraction of null_errors <= observed_error - small
                         means real morphology predicts fate far better
                         than a random reassignment ever manages to
        predicted:       (n, n_categories) the real (unpermuted) KNN
                         predictions, for plotting predicted vs. actual
    '''
    neighbor_idx = nearest_neighbor_indices(dist_matrix, k)

    predicted = knn_predict_mixture(neighbor_idx, mixture_matrix)
    observed_error = mean_prediction_error(mixture_matrix, predicted)

    rng = np.random.default_rng(seed)
    n = mixture_matrix.shape[0]
    null_errors = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = mixture_matrix[rng.permutation(n)]
        null_errors[i] = mean_prediction_error(shuffled, knn_predict_mixture(neighbor_idx, shuffled))

    p_value = (null_errors <= observed_error).mean()
    return observed_error, null_errors, p_value, predicted


def plot_prediction_results(true_mixture, predicted_mixture, null_errors, observed_error, p_value,
                             category_names, category_colors, title):
    '''
    One figure, two panels, summarizing knn_prediction_permutation_test's
    result:
      left:  predicted vs. true mixture fraction, pooled across every
             category (colored by category), with a y=x reference line -
             how GOOD the morphology-based prediction is
      right: the null distribution of mean_prediction_error under random
             label reassignment, with the real (observed) error marked -
             how SIGNIFICANT that is compared to chance

    Inputs:
        true_mixture:      (n, n_categories)
        predicted_mixture: (n, n_categories), same column order
        null_errors:       (n_permutations,) from knn_prediction_permutation_test
        observed_error:    from knn_prediction_permutation_test
        p_value:           from knn_prediction_permutation_test
        category_names:    column names in display order, e.g.
                           ['dying', 'persisting', 'slow_prolif', 'fast_prolif']
        category_colors:   {category_name: color}
        title:             figure title
    Outputs:
        fig: caller saves/closes this
    '''
    # a permutation test can't resolve a p-value finer than 1/n_permutations
    # (0 of n_permutations beating the observed error doesn't mean p is
    # exactly 0) - report that resolution limit honestly instead
    if p_value == 0:
        p_label = f'< {1 / len(null_errors):.4f}'
    else:
        p_label = f'= {p_value:.4f}'

    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(11, 5))

    for i, category in enumerate(category_names):
        ax_scatter.scatter(true_mixture[:, i], predicted_mixture[:, i], s=6, alpha=0.3,
                            color=category_colors[category], label=category)
    ax_scatter.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=1, zorder=0)
    ax_scatter.set_xlim(-0.05, 1.05)
    ax_scatter.set_ylim(-0.05, 1.05)
    ax_scatter.set_xlabel('actual fraction')
    ax_scatter.set_ylabel('predicted fraction (k nearest morphological neighbors)')
    ax_scatter.set_title('prediction accuracy')
    ax_scatter.legend(fontsize=8)

    ax_hist.hist(null_errors, bins=30, color='lightgrey', edgecolor='grey')
    ax_hist.axvline(observed_error, color='tab:red', linewidth=2)
    ax_hist.text(observed_error, ax_hist.get_ylim()[1] * 0.92, f'  observed\n  p {p_label}',
                 color='tab:red', va='top')
    ax_hist.set_xlabel('mean prediction error')
    ax_hist.set_ylabel('permutations')
    ax_hist.set_title('observed vs. randomly reassigned fate labels')

    fig.suptitle(title)
    fig.tight_layout()
    return fig
