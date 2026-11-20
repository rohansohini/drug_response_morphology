import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


def pair_distance_permutation_test(dist_matrix: np.ndarray, pool_indices, real_pairs: list[tuple[int, int]],
                                    n_null_samples: int = 10_000, seed: int = 42) -> tuple[np.ndarray, np.ndarray, float]:
    '''
    Test whether a given set of real pairs (e.g. sister cell pairs, or
    cousin cell pairs) are morphologically more similar than a random
    pair drawn from the same population - not by collapsing to a single
    observed mean distance and comparing THAT against a null of means
    (which hides whether the effect holds for most real pairs or is
    driven by just a few), but by comparing the FULL distribution of
    real pairs' own distances against a null distribution of INDIVIDUAL
    random pair distances, via a one-sided Mann-Whitney U test (are real
    distances stochastically SMALLER than random-pair distances).

    This is meant to be called once per relationship being tested
    (sisters, cousins, ...) with the SAME `pool_indices`, `n_null_samples`
    and `seed` each time, so every call's null distribution is directly
    comparable and can be drawn on the same plot (see
    plot_pair_similarity_test).

    Inputs:
        dist_matrix:     (n, n) morphological distance matrix (e.g.
                         CAJAL GW distances) - NOT a UMAP/PHATE
                         projection, which distorts distances
        pool_indices:    indices into dist_matrix a random pair may be
                         drawn from - e.g. every embedded cell within the
                         frame range being tested, regardless of
                         relationship. Real pairs are assumed to be drawn
                         from this same population, so the test reads as
                         "more similar than a random pair from this same
                         pool", not some other population.
        real_pairs:      [(i, j), ...] index pairs into dist_matrix - the
                         actual sister (or cousin) pairs
        n_null_samples:  number of individual random pairs making up the
                         null distribution - NOT frame-matched (a null
                         pair, like a real cousin pair, can come from any
                         two frames the pool covers)
        seed:            RNG seed, for reproducibility
    Outputs:
        observed_distances: (len(real_pairs),) the real pairs' own distances
        null_distances:      (n_null_samples,) individual random-pair distances
        p_value:              one-sided Mann-Whitney U p-value (alternative=
                              'less') - small means the real pairs are
                              consistently more similar than a random
                              pairing typically manages
    '''
    observed_distances = np.array([dist_matrix[i, j] for i, j in real_pairs])

    rng = np.random.default_rng(seed)
    pool = np.asarray(pool_indices)
    a = rng.choice(pool, size=n_null_samples)
    b = rng.choice(pool, size=n_null_samples)
    # a drawn pair where both sides happen to be the SAME cell has
    # distance 0 by definition, which would bias the null downward -
    # redraw just those few positions rather than counting them
    collision = a == b
    while collision.any():
        b[collision] = rng.choice(pool, size=collision.sum())
        collision = a == b
    null_distances = dist_matrix[a, b]

    p_value = mannwhitneyu(observed_distances, null_distances, alternative='less').pvalue
    return observed_distances, null_distances, p_value


def plot_pair_similarity_test(null_distances: np.ndarray, observed: dict[str, tuple[np.ndarray, float]],
                               title: str = None) -> plt.Figure:
    '''
    The null distribution of individual random-pair distances, overlaid
    with one or more real relationships' own distance distributions
    (e.g. 'sisters', 'cousins', 'naive') - so several relationships can
    be compared against the exact same null on one plot. All histograms
    are density-normalized (not raw counts), since the null is typically
    built from many more samples than there are real pairs.

    Inputs:
        null_distances: (n_null_samples,) from pair_distance_permutation_test -
                        must be the SAME null every `observed` entry was
                        actually compared against
        observed:       {label: (observed_distances, p_value)}, e.g.
                        {'sisters': (array([...]), 0.0083), 'cousins': (array([...]), 0.11)}
        title:          plot title, or None for no title
    Outputs:
        fig: caller saves/closes this
    '''
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:orange']
    assert len(observed) <= len(colors), f"plot_pair_similarity_test only has {len(colors)} colors defined"

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.hist(null_distances, bins=50, density=True, color='lightgrey', edgecolor='grey', alpha=0.8, label='null (random pairs)')

    for i, (label, (distances, p_value)) in enumerate(observed.items()):
        color = colors[i % len(colors)]
        p_label = f'< {np.finfo(float).tiny:.1e}' if p_value == 0 else f'= {p_value:.4g}'
        ax.hist(distances, bins=50, density=True, color=color, edgecolor=color, alpha=0.4, label=f'{label} (p {p_label})')

    ax.set_xlabel('pairwise CAJAL distance')
    ax.set_ylabel('density')
    if title is not None:
        ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    return fig
