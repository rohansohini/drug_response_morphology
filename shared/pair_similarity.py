import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


def pair_distance_permutation_test(dist_matrix: np.ndarray, pool_indices, real_pairs: list[tuple[int, int]],
                                    n_null_samples: int = 10_000, seed: int = 42) -> tuple[np.ndarray, np.ndarray, float, float]:
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

    Also returns a rank-biserial correlation effect size alongside the
    p-value - a p-value conflates effect size with sample size (with
    enough pairs, even a tiny real shift in distance becomes extremely
    significant, which can look surprising next to two violins that
    otherwise look similar - see plot_pair_similarity_test), so it is
    NOT a stand-in for "how big is the effect", and is not comparable in
    magnitude across tests run on different n. The rank-biserial
    correlation is: bounded to [-1, 1] regardless of sample size, so IS
    directly comparable across relationships/groups/degrees, and is
    derived for free from the same Mann-Whitney U statistic already
    being computed here.

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
        effect_size:          rank-biserial correlation, r = 1 - 2U/(n1*n2) -
                              positive means real pairs are stochastically
                              CLOSER than random pairs (the expected
                              direction), 0 means no difference from
                              random, negative means real pairs are
                              stochastically FARTHER than random pairs
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

    result = mannwhitneyu(observed_distances, null_distances, alternative='less')
    p_value = result.pvalue
    effect_size = 1 - 2 * result.statistic / (len(observed_distances) * len(null_distances))
    return observed_distances, null_distances, p_value, effect_size


def plot_pair_similarity_test(null_distances: np.ndarray, observed: dict[str, tuple[np.ndarray, float]],
                               title: str = None) -> plt.Figure:
    '''
    One violin per distribution - the null (random pairs) plus one or
    more real relationships' own distance distributions (e.g. 'sisters',
    'cousins', 'naive') - side by side, not overlaid. Several
    relationships can still be compared against the exact same null on
    one plot, but as separate violins sharing one y-axis (pairwise
    distance) rather than alpha-blended histograms sharing one x-axis -
    the latter becomes hard to read once there are 2-3 distributions
    all piled on top of each other in the same region.

    Each real group's own p-value goes into its own x-tick label, not a
    legend - with every distribution already at its own x position,
    there's no separate floating UI element (a legend box, a text
    annotation) that could ever end up overlapping anything else.

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

    labels = ['null\n(random pairs)']
    datasets = [null_distances]
    violin_colors = ['lightgrey']
    for i, (label, (distances, p_value)) in enumerate(observed.items()):
        p_label = f'< {np.finfo(float).tiny:.1e}' if p_value == 0 else f'= {p_value:.4g}'
        labels.append(f'{label}\n(p {p_label})')
        datasets.append(distances)
        violin_colors.append(colors[i % len(colors)])

    fig, ax = plt.subplots(figsize=(1.8 * len(datasets) + 1.5, 5))
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
    # log2 scale - CAJAL distances are strictly positive and right-skewed
    # (a handful of very dissimilar pairs stretch the tail out far), which
    # on a linear axis compresses the low end - where the real median
    # shift between distributions actually lives - into a sliver too
    # small to see by eye, even when Mann-Whitney correctly detects it
    ax.set_yscale('log', base=2)
    ax.set_ylabel('pairwise CAJAL distance (log₂ scale)')
    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    return fig
