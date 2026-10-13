'''
Permutation-based tests for whether a categorical grouping (e.g. subclass)
explains structure in a CAJAL Gromov-Wasserstein distance matrix.

These operate directly on the distance matrix, never on a UMAP projection
of it - UMAP distorts distances by construction (it optimizes a 2D layout
for visual cluster separation, not for preserving actual distance values),
so it is the wrong representation for any quantitative/statistical claim.

Two complementary tests are provided, because a significant PERMANOVA alone
is ambiguous between two different effects:
  - permanova:  do groups sit in different LOCATIONS in distance space?
                (Anderson 2001 - a distance-based analogue of ANOVA)
  - permdisp:   do groups differ in their SPREAD/dispersion?
                (Anderson 2006 - "PERMDISP")
A PERMANOVA hit with no PERMDISP hit indicates a genuine location
difference. A PERMANOVA hit WITH a PERMDISP hit is ambiguous - PERMANOVA's
pseudo-F is sensitive to dispersion differences even when centroids
coincide, so that combination should be reported as such, not read as
clean evidence of a location effect.
'''
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import eigh


def validate_distance_matrix(dist_matrix: np.ndarray, labels=None) -> int:
    '''
    Check that dist_matrix is a well-formed dissimilarity matrix before any
    test touches it - square, symmetric, zero diagonal, non-negative, no
    NaN/inf. Every public function in this module calls this first, so a
    malformed matrix fails loudly here rather than producing a silently
    wrong statistic downstream.

    Inputs:
        dist_matrix: (n, n) array
        labels:      optional, checked to have exactly one entry per row/col
    Outputs:
        n: dist_matrix's size
    '''
    dist_matrix = np.asarray(dist_matrix, dtype=float)
    assert dist_matrix.ndim == 2 and dist_matrix.shape[0] == dist_matrix.shape[1], (
        f"dist_matrix must be square 2D, got shape {dist_matrix.shape}"
    )
    n = dist_matrix.shape[0]
    assert n >= 2, f"dist_matrix must have at least 2 points, got {n}"

    assert np.isfinite(dist_matrix).all(), "dist_matrix contains NaN or inf values"

    asymmetry = np.abs(dist_matrix - dist_matrix.T).max()
    assert asymmetry < 1e-6, (
        f"dist_matrix is not symmetric (max |D - D.T| = {asymmetry:.3g}) - PERMANOVA/PERMDISP "
        f"require a symmetric dissimilarity matrix"
    )

    max_diag = np.abs(np.diag(dist_matrix)).max()
    assert max_diag < 1e-6, f"dist_matrix diagonal is not all zero (max |diag| = {max_diag:.3g})"

    min_val = dist_matrix.min()
    assert min_val >= -1e-8, f"dist_matrix contains a negative distance ({min_val:.3g})"

    if labels is not None:
        assert len(labels) == n, f"len(labels) ({len(labels)}) != dist_matrix size ({n})"

    return n


def _encode_groups(labels: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    '''
    Factorize labels into integer codes 0..K-1 and the sorted unique group
    values, with the shared validation (>=2 groups, warn on tiny groups)
    every test needs.
    '''
    labels = np.asarray(labels)
    codes, uniques = pd.factorize(labels, sort=True)
    n_groups = len(uniques)
    assert n_groups >= 2, f"need >= 2 groups to test, got {n_groups}: {list(uniques)}"

    group_sizes = np.bincount(codes, minlength=n_groups)
    tiny = uniques[group_sizes < 3]
    if len(tiny) > 0:
        warnings.warn(
            f"group(s) {list(tiny)} have fewer than 3 members - their within-group variance "
            f"estimate is unstable, and a singleton group contributes zero within-group "
            f"variation by construction"
        )

    return codes, uniques


def _group_indicator(codes: np.ndarray, n_groups: int) -> np.ndarray:
    n = len(codes)
    G = np.zeros((n, n_groups))
    G[np.arange(n), codes] = 1.0
    return G


def _within_group_ss(dist_sq: np.ndarray, codes: np.ndarray, n_groups: int) -> tuple[float, np.ndarray]:
    '''
    SS_within = sum over groups g of (sum_{i,j in g} dist_sq[i,j]) / (2 * n_g).

    Vectorized via a one-hot indicator matrix G (n x n_groups) instead of a
    per-group Python loop: (dist_sq @ G) is (n x n_groups), its column k
    holding each point's summed squared distance to every point currently
    in group k; multiplying elementwise by G and summing over rows isolates
    exactly the i,j in-group pairs, for every group in one matrix multiply.
    '''
    G = _group_indicator(codes, n_groups)
    sizes = G.sum(axis=0)
    block_sums = np.einsum('ik,ik->k', G, dist_sq @ G)
    with np.errstate(invalid='ignore', divide='ignore'):
        ss_per_group = block_sums / (2 * sizes)
    return ss_per_group.sum(), sizes


def permanova(dist_matrix: np.ndarray, labels, n_permutations: int = 999, seed: int = 42) -> dict:
    '''
    PERMANOVA (Anderson 2001): permutation test of whether group membership
    explains variance in a distance matrix - i.e. whether groups occupy
    different locations in distance space.

    Inputs:
        dist_matrix:    (n, n) symmetric dissimilarity matrix (e.g. a CAJAL
                        GW distance matrix, NEVER a UMAP projection)
        labels:         length-n group label per row/col of dist_matrix
        n_permutations: number of label permutations for the null distribution
        seed:           RNG seed, for reproducibility
    Outputs:
        dict with: pseudo_F, R2 (SS_among / SS_total - effect size,
        independent of n/permutations), p_value, n_permutations, n_groups,
        n, group_sizes, df_among, df_within
    '''
    n = validate_distance_matrix(dist_matrix, labels)
    codes, uniques = _encode_groups(labels, n)
    n_groups = len(uniques)

    dist_sq = np.asarray(dist_matrix, dtype=float) ** 2
    ss_total = dist_sq.sum() / (2 * n)
    df_among  = n_groups - 1
    df_within = n - n_groups

    def pseudo_F(codes_):
        ss_within, _ = _within_group_ss(dist_sq, codes_, n_groups)
        ss_among = ss_total - ss_within
        with np.errstate(invalid='ignore', divide='ignore'):
            F = (ss_among / df_among) / (ss_within / df_within)
        return F, ss_among

    F_obs, ss_among_obs = pseudo_F(codes)
    R2 = ss_among_obs / ss_total

    rng = np.random.default_rng(seed)
    F_perm = np.empty(n_permutations)
    for p in range(n_permutations):
        F_perm[p], _ = pseudo_F(rng.permutation(codes))

    n_exceed = np.sum(F_perm >= F_obs)
    p_value = (n_exceed + 1) / (n_permutations + 1)

    group_sizes = np.bincount(codes, minlength=n_groups)
    return {
        'pseudo_F': F_obs,
        'R2': R2,
        'p_value': p_value,
        'n_permutations': n_permutations,
        'n_groups': n_groups,
        'n': n,
        'group_sizes': dict(zip(uniques.tolist(), group_sizes.tolist())),
        'df_among': df_among,
        'df_within': df_within,
    }


def permanova_pairwise(dist_matrix: np.ndarray, labels, n_permutations: int = 999, seed: int = 42) -> pd.DataFrame:
    '''
    Run permanova() on every pair of groups separately (restricting the
    distance matrix to just that pair's rows/cols), then Benjamini-Hochberg
    correct the resulting p-values across all pairs. A significant omnibus
    PERMANOVA only says "not all groups are the same" - this identifies
    which specific pairs differ.

    Inputs:
        dist_matrix, n_permutations, seed: see permanova()
        labels: length-n group label per row/col of dist_matrix (>= 3
                distinct groups expected - with exactly 2, this duplicates
                the omnibus test with no multiple-comparison correction to do)
    Outputs:
        DataFrame, one row per unordered pair, columns: group_a, group_b,
        n_a, n_b, pseudo_F, R2, p_value, q_value (BH-corrected), sorted by
        q_value ascending
    '''
    n = validate_distance_matrix(dist_matrix, labels)
    labels = np.asarray(labels)
    dist_matrix = np.asarray(dist_matrix, dtype=float)
    uniques = sorted(pd.unique(labels).tolist())
    assert len(uniques) >= 2, f"need >= 2 groups, got {len(uniques)}"

    rows = []
    for i in range(len(uniques)):
        for j in range(i + 1, len(uniques)):
            a, b = uniques[i], uniques[j]
            mask = np.isin(labels, [a, b])
            sub_dist   = dist_matrix[np.ix_(mask, mask)]
            sub_labels = labels[mask]
            res = permanova(sub_dist, sub_labels, n_permutations=n_permutations, seed=seed)
            rows.append({
                'group_a': a, 'group_b': b,
                'n_a': res['group_sizes'][a], 'n_b': res['group_sizes'][b],
                'pseudo_F': res['pseudo_F'], 'R2': res['R2'], 'p_value': res['p_value'],
            })

    df = pd.DataFrame(rows)
    df['q_value'] = benjamini_hochberg(df['p_value'].to_numpy())
    return df.sort_values('q_value').reset_index(drop=True)


def _pcoa(dist_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    '''
    Classical MDS / Principal Coordinates Analysis via Gower's double
    centering. A Gromov-Wasserstein distance matrix is not guaranteed to be
    Euclidean-embeddable, so negative eigenvalues can occur - they are
    clipped to zero (dropping those axes) rather than "corrected" (e.g.
    Lingoes/Cailliez), so the embedding undercounts total variance in
    exchange for staying in a real coordinate space.

    Outputs:
        coords:  (n, n) PCoA coordinates (axes with eigval <= 0 are all-zero)
        eigvals: (n,) eigenvalues, descending
        n_neg:   count of eigenvalues clipped to zero
    '''
    n = dist_matrix.shape[0]
    D2 = np.asarray(dist_matrix, dtype=float) ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    B = (B + B.T) / 2  # symmetrize away float round-off before eigh

    eigvals, eigvecs = eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    n_neg = int(np.sum(eigvals < -1e-8))
    eigvals_clipped = np.clip(eigvals, 0, None)
    coords = eigvecs * np.sqrt(eigvals_clipped)

    return coords, eigvals, n_neg


def _anova_F(z: np.ndarray, codes: np.ndarray, n_groups: int) -> float:
    n = len(z)
    grand_mean = z.mean()
    ss_between = 0.0
    ss_within  = 0.0
    for g in range(n_groups):
        zg = z[codes == g]
        ss_between += len(zg) * (zg.mean() - grand_mean) ** 2
        ss_within  += np.sum((zg - zg.mean()) ** 2)
    df_between = n_groups - 1
    df_within  = n - n_groups
    with np.errstate(invalid='ignore', divide='ignore'):
        return (ss_between / df_between) / (ss_within / df_within)


def permdisp(dist_matrix: np.ndarray, labels, n_permutations: int = 999, seed: int = 42) -> dict:
    '''
    PERMDISP (Anderson 2006): tests homogeneity of multivariate dispersion
    across groups - distinct from (and a check against) PERMANOVA, which
    tests location, not spread.

    Method: embed dist_matrix via PCoA (see _pcoa), compute each point's
    Euclidean distance to its OWN group's centroid in that PCoA space
    (its "dispersion" z_i), then run a one-way ANOVA F-test of z against
    group, with a permutation null built by reassigning group labels to
    the fixed z_i values (999 shuffles by default). This tests whether the
    true grouping explains more variance in per-point dispersion than a
    random grouping would - it does NOT recompute per-permutation centroids
    (vegan's permutest.betadisper does, under some of its permutation
    methods), so treat this as a direct, slightly simpler permutational
    ANOVA on a fixed dispersion measure, not a byte dependent of vegan's.

    Inputs:
        dist_matrix, n_permutations, seed: see permanova()
        labels: length-n group label per row/col of dist_matrix
    Outputs:
        dict with: F, p_value, n_permutations, n_groups, n, group_sizes,
        mean_dispersion ({group: mean distance-to-own-centroid}),
        df_between, df_within, n_negative_eigenvalues
    '''
    n = validate_distance_matrix(dist_matrix, labels)
    codes, uniques = _encode_groups(labels, n)
    n_groups = len(uniques)

    coords, eigvals, n_neg = _pcoa(dist_matrix)
    if n_neg > 0:
        total_mag = np.abs(eigvals).sum()
        neg_frac  = np.abs(eigvals[eigvals < 0]).sum() / total_mag if total_mag > 0 else 0.0
        warnings.warn(
            f"PCoA of dist_matrix produced {n_neg} negative eigenvalue(s) "
            f"({neg_frac:.1%} of total eigenvalue magnitude) - clipped to zero, as expected for a "
            f"non-Euclidean (e.g. Gromov-Wasserstein) distance matrix; dispersion is computed in "
            f"the resulting positive-semidefinite approximation, which discards that fraction of "
            f"variance"
        )

    def distances_to_centroid(codes_):
        z = np.empty(n)
        for g in range(n_groups):
            mask = codes_ == g
            centroid = coords[mask].mean(axis=0)
            z[mask] = np.linalg.norm(coords[mask] - centroid, axis=1)
        return z

    z_obs = distances_to_centroid(codes)
    F_obs = _anova_F(z_obs, codes, n_groups)

    rng = np.random.default_rng(seed)
    F_perm = np.empty(n_permutations)
    for p in range(n_permutations):
        F_perm[p] = _anova_F(z_obs, rng.permutation(codes), n_groups)

    n_exceed = np.sum(F_perm >= F_obs)
    p_value = (n_exceed + 1) / (n_permutations + 1)

    group_sizes = np.bincount(codes, minlength=n_groups)
    mean_dispersion = {
        u: float(z_obs[codes == i].mean()) for i, u in enumerate(uniques.tolist())
    }

    return {
        'F': F_obs,
        'p_value': p_value,
        'n_permutations': n_permutations,
        'n_groups': n_groups,
        'n': n,
        'group_sizes': dict(zip(uniques.tolist(), group_sizes.tolist())),
        'mean_dispersion': mean_dispersion,
        'df_between': n_groups - 1,
        'df_within': n - n_groups,
        'n_negative_eigenvalues': n_neg,
    }


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    '''
    Benjamini-Hochberg FDR correction, implemented directly (no statsmodels
    dependency) to control the false discovery rate across a family of
    p-values from the same comparison set (e.g. all pairwise subclass
    comparisons within one frame/modality).

    Inputs:
        p_values: (m,) raw p-values
    Outputs:
        q_values: (m,) BH-adjusted p-values, same order as input, each
                  clipped to [0, 1] and enforced monotone non-decreasing
                  when sorted by p-value (the standard BH step-up correction)
    '''
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    assert m >= 1, "need at least one p-value"
    assert np.all((p >= 0) & (p <= 1)), "p-values must be in [0, 1]"

    order = np.argsort(p)
    ranked = p[order]
    ranks = np.arange(1, m + 1)
    q_ranked = ranked * m / ranks

    # step-up: enforce monotone non-decreasing from the largest p-value down
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)

    q = np.empty(m)
    q[order] = q_ranked
    return q
