import numpy as np
import matplotlib.pyplot as plt


def pair_distance_permutation_test(dist_matrix: np.ndarray, pool_indices, real_pairs: list[tuple[int, int]],
                                    n_pairs_per_draw: int, n_permutations: int = 10_000,
                                    seed: int = 42) -> tuple[float, np.ndarray, float]:
    '''
    Test whether a given set of real pairs (e.g. sister cell pairs, or
    cousin cell pairs) are morphologically more similar than a random
    pair drawn from the same population would typically be, by
    comparing their observed mean distance against a null built from
    many random pairs - NOT frame-matched (a null pair, like a real
    cousin pair, can come from any two frames the pool covers).

    This is meant to be called once per relationship being tested
    (sisters, cousins, ...) with the SAME `n_pairs_per_draw`, `pool_indices`,
    `n_permutations` and `seed` each time, so every call's null
    distribution is directly comparable and can be drawn on the same
    plot (see plot_pair_similarity_test) - `n_pairs_per_draw` in
    particular should be fixed once (e.g. the smaller of the sister/
    cousin pair counts) rather than matched separately to each
    relationship's own real pair count, which would give each
    relationship a differently-calibrated null.

    Inputs:
        dist_matrix:      (n, n) morphological distance matrix (e.g.
                          CAJAL GW distances) - NOT a UMAP/PHATE
                          projection, which distorts distances
        pool_indices:     indices into dist_matrix a random pair may be
                          drawn from - e.g. every embedded cell within
                          the frame range being tested, regardless of
                          relationship. Real pairs are assumed to be
                          drawn from this same population, so the test
                          reads as "more similar than a random pair from
                          this same pool", not some other population.
        real_pairs:       [(i, j), ...] index pairs into dist_matrix -
                          the actual sister (or cousin) pairs
        n_pairs_per_draw: how many random pairs make up one null draw's
                          mean distance
        n_permutations:   number of random draws making up the null
        seed:             RNG seed, for reproducibility
    Outputs:
        observed_mean_distance: mean distance across real_pairs
        null_means:             (n_permutations,) mean distance of each
                                random draw
        p_value:                fraction of null_means <= observed_mean_distance -
                                small means the real pairs are more
                                similar than chance pairing manages
    '''
    real_distances = [dist_matrix[i, j] for i, j in real_pairs]
    observed_mean_distance = float(np.mean(real_distances))

    rng = np.random.default_rng(seed)
    pool = np.asarray(pool_indices)
    null_means = np.empty(n_permutations)

    for p in range(n_permutations):
        a = rng.choice(pool, size=n_pairs_per_draw)
        b = rng.choice(pool, size=n_pairs_per_draw)
        # a drawn pair where both sides happen to be the SAME cell has
        # distance 0 by definition, which would bias the null downward -
        # redraw just those few positions rather than counting them
        collision = a == b
        while collision.any():
            b[collision] = rng.choice(pool, size=collision.sum())
            collision = a == b
        null_means[p] = dist_matrix[a, b].mean()

    p_value = (null_means <= observed_mean_distance).mean()
    return observed_mean_distance, null_means, p_value


def plot_pair_similarity_test(null_means: np.ndarray, observed: dict[str, tuple[float, float]],
                               title: str = None) -> plt.Figure:
    '''
    The shared null distribution of mean pairwise distance under random
    pairing (pair_distance_permutation_test), with one or more real
    relationships (e.g. 'sisters', 'cousins') each marked as their own
    vertical line - so several relationships can be compared against
    the exact same null on one plot.

    Inputs:
        null_means: (n_permutations,) from pair_distance_permutation_test -
                    must be the SAME null every `observed` entry was
                    actually compared against
        observed:   {label: (observed_mean_distance, p_value)}, e.g.
                    {'sisters': (0.42, 0.0083), 'cousins': (0.51, 0.11)}
        title:      plot title, or None for no title
    Outputs:
        fig: caller saves/closes this
    '''
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:orange']
    assert len(observed) <= len(colors), f"plot_pair_similarity_test only has {len(colors)} marker colors defined"

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(null_means, bins=30, color='lightgrey', edgecolor='grey')

    y_top = ax.get_ylim()[1]
    for i, (label, (value, p_value)) in enumerate(observed.items()):
        color = colors[i % len(colors)]
        # a permutation test can't resolve a p-value finer than
        # 1/n_permutations (0 of n_permutations beating the observed
        # value doesn't mean p is exactly 0) - report that resolution
        # limit honestly instead
        if p_value == 0:
            p_label = f'< {1 / len(null_means):.4f}'
        else:
            p_label = f'= {p_value:.4f}'
        ax.axvline(value, color=color, linewidth=2)
        ax.text(value, y_top * (0.92 - 0.12 * i), f'  {label}\n  p {p_label}', color=color, va='top')

    ax.set_xlabel('mean pairwise CAJAL distance')
    ax.set_ylabel('number of random draws')
    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    return fig
