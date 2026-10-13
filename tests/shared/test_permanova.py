'''
Run directly: python3 tests/shared/test_permanova.py
'''
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.permanova import (
    validate_distance_matrix, permanova, permanova_pairwise, permdisp,
    benjamini_hochberg, _pcoa,
)


def _dist_matrix(points):
    return np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)


def test_permanova_detects_separated_clusters():
    rng = np.random.default_rng(0)
    pts = np.vstack([
        rng.normal([0, 0], 0.3, (20, 2)),
        rng.normal([10, 10], 0.3, (20, 2)),
        rng.normal([0, 10], 0.3, (20, 2)),
    ])
    labels = ['A'] * 20 + ['B'] * 20 + ['C'] * 20
    D = _dist_matrix(pts)

    res = permanova(D, labels, n_permutations=199, seed=1)

    assert res['n'] == 60 and res['n_groups'] == 3
    assert res['R2'] > 0.9, f"expected near-total variance explained, got R2={res['R2']}"
    assert res['p_value'] < 0.01, f"expected a strongly significant result, got p={res['p_value']}"
    assert res['df_among'] == 2 and res['df_within'] == 57
    print("PASS: permanova detects well-separated clusters with high R2 and low p-value")


def test_permanova_null_on_random_labels():
    # same data as above, but labels shuffled so group membership carries
    # no real spatial information - PERMANOVA should find nothing
    rng = np.random.default_rng(0)
    pts = np.vstack([
        rng.normal([0, 0], 0.3, (20, 2)),
        rng.normal([10, 10], 0.3, (20, 2)),
        rng.normal([0, 10], 0.3, (20, 2)),
    ])
    D = _dist_matrix(pts)

    shuffler = np.random.default_rng(7)
    labels = shuffler.permutation(['A'] * 20 + ['B'] * 20 + ['C'] * 20)

    res = permanova(D, labels, n_permutations=199, seed=1)

    assert res['R2'] < 0.1, f"expected a small effect size under random labeling, got R2={res['R2']}"
    assert res['p_value'] > 0.05, f"expected a non-significant result, got p={res['p_value']}"
    print("PASS: permanova finds no effect when group labels carry no spatial information")


def test_permanova_pairwise_identifies_different_pairs_and_corrects_pvalues():
    rng = np.random.default_rng(0)
    # A and B are well-separated from each other and from C; C sits roughly
    # at A's location too, so the A-vs-C comparison should come out weak
    pts = np.vstack([
        rng.normal([0, 0], 0.5, (20, 2)),
        rng.normal([10, 10], 0.5, (20, 2)),
        rng.normal([0, 0], 0.5, (20, 2)),
    ])
    labels = ['A'] * 20 + ['B'] * 20 + ['C'] * 20
    D = _dist_matrix(pts)

    df = permanova_pairwise(D, labels, n_permutations=199, seed=1)

    assert set(df.columns) == {'group_a', 'group_b', 'n_a', 'n_b', 'pseudo_F', 'R2', 'p_value', 'q_value'}
    assert len(df) == 3, f"expected 3 pairs for 3 groups, got {len(df)}"
    assert (df['q_value'] >= df['p_value'] - 1e-12).all(), "BH q-values must be >= raw p-values"

    ab = df[(df['group_a'] == 'A') & (df['group_b'] == 'B')].iloc[0]
    ac = df[(df['group_a'] == 'A') & (df['group_b'] == 'C')].iloc[0]
    assert ab['q_value'] < 0.05, f"expected A vs B (well separated) to be significant, got q={ab['q_value']}"
    assert ac['q_value'] > ab['q_value'], "A vs C (same location) should be far less significant than A vs B"
    print("PASS: permanova_pairwise distinguishes separated vs. coincident pairs and BH-corrects p-values")


def test_permdisp_detects_dispersion_difference_with_same_centroid():
    rng = np.random.default_rng(0)
    n_per = 50
    pts = np.vstack([
        rng.normal([0, 0], 0.2, (n_per, 2)),
        rng.normal([0, 0], 3.0, (n_per, 2)),
    ])
    labels = ['tight'] * n_per + ['spread'] * n_per
    D = _dist_matrix(pts)

    disp = permdisp(D, labels, n_permutations=299, seed=2)
    loc  = permanova(D, labels, n_permutations=299, seed=2)

    assert disp['p_value'] < 0.01, f"expected PERMDISP to detect the dispersion difference, got p={disp['p_value']}"
    assert disp['mean_dispersion']['spread'] > disp['mean_dispersion']['tight']
    assert loc['p_value'] > 0.3, (
        f"expected PERMANOVA to find no location effect when centroids coincide, got p={loc['p_value']}"
    )
    print("PASS: permdisp detects a pure dispersion difference that permanova correctly misses (same centroid)")


def test_permdisp_null_when_dispersion_equal():
    rng = np.random.default_rng(3)
    n_per = 40
    pts = np.vstack([
        rng.normal([0, 0], 1.0, (n_per, 2)),
        rng.normal([10, 10], 1.0, (n_per, 2)),
    ])
    labels = ['A'] * n_per + ['B'] * n_per
    D = _dist_matrix(pts)

    disp = permdisp(D, labels, n_permutations=299, seed=2)
    assert disp['p_value'] > 0.05, f"expected no dispersion effect when groups have equal spread, got p={disp['p_value']}"
    print("PASS: permdisp finds no dispersion effect when groups have genuinely equal spread")


def test_permdisp_warns_on_negative_eigenvalues():
    # shortest-path distances on a 5-cycle graph are a classic example of a
    # valid metric (symmetric, triangle inequality holds) that is NOT
    # Euclidean-embeddable - classical MDS of it necessarily yields
    # negative eigenvalues, which permdisp must warn about rather than
    # silently discard
    n = 5
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = min((i - j) % n, (j - i) % n)
    labels = ['A', 'A', 'A', 'B', 'B']

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        permdisp(D, labels, n_permutations=49, seed=1)
        assert any('negative eigenvalue' in str(w.message) for w in caught), (
            "expected a warning about negative eigenvalues from a non-Euclidean distance matrix"
        )
    print("PASS: permdisp warns when PCoA of a non-Euclidean distance matrix yields negative eigenvalues")


def test_pcoa_clips_negative_eigenvalues_to_zero():
    n = 5
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = min((i - j) % n, (j - i) % n)

    coords, eigvals, n_neg = _pcoa(D)
    assert n_neg > 0, "expected the 5-cycle metric to produce at least one negative eigenvalue"
    neg_axes = np.where(eigvals <= 1e-8)[0]
    assert np.allclose(coords[:, neg_axes], 0), "axes with non-positive eigenvalue must contribute zero coordinates"
    print("PASS: _pcoa clips negative-eigenvalue axes to all-zero coordinates")


def test_benjamini_hochberg_matches_hand_computed_values():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074,
                  0.205, 0.212, 0.216, 0.222, 0.251, 0.278, 0.299, 0.3])
    expected = np.array([0.015, 0.06, 0.126, 0.126, 0.126, 0.15, 0.1585714285714286,
                         0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
    q = benjamini_hochberg(p)
    assert np.allclose(q, expected, atol=1e-9), f"expected {expected}, got {q}"
    print("PASS: benjamini_hochberg matches hand-computed step-up BH correction")


def test_benjamini_hochberg_monotone_and_bounded():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 50)
    q = benjamini_hochberg(p)

    assert np.all((q >= 0) & (q <= 1)), "q-values must be in [0, 1]"
    assert np.all(q >= p - 1e-12), "BH q-values must never be smaller than the raw p-value"

    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12), "q-values must be monotone non-decreasing when sorted by p-value"
    print("PASS: benjamini_hochberg output is bounded, >= raw p-values, and monotone in sorted p-value order")


def test_validate_distance_matrix_rejects_asymmetric():
    D = np.array([[0, 1], [2, 0]], dtype=float)
    try:
        validate_distance_matrix(D)
    except AssertionError as e:
        assert 'symmetric' in str(e)
        print("PASS: validate_distance_matrix rejects an asymmetric matrix")
        return
    raise AssertionError("expected validate_distance_matrix to reject an asymmetric matrix")


def test_validate_distance_matrix_rejects_nonzero_diagonal():
    D = np.array([[0, 1], [1, 0.5]], dtype=float)
    try:
        validate_distance_matrix(D)
    except AssertionError as e:
        assert 'diagonal' in str(e)
        print("PASS: validate_distance_matrix rejects a non-zero diagonal")
        return
    raise AssertionError("expected validate_distance_matrix to reject a non-zero diagonal")


def test_validate_distance_matrix_rejects_negative_distance():
    D = np.array([[0, -1], [-1, 0]], dtype=float)
    try:
        validate_distance_matrix(D)
    except AssertionError as e:
        assert 'negative' in str(e)
        print("PASS: validate_distance_matrix rejects a negative distance")
        return
    raise AssertionError("expected validate_distance_matrix to reject a negative distance")


def test_validate_distance_matrix_rejects_nan():
    D = np.array([[0, np.nan], [np.nan, 0]])
    try:
        validate_distance_matrix(D)
    except AssertionError as e:
        assert 'NaN' in str(e)
        print("PASS: validate_distance_matrix rejects NaN values")
        return
    raise AssertionError("expected validate_distance_matrix to reject NaN values")


def test_validate_distance_matrix_rejects_non_square():
    D = np.zeros((3, 4))
    try:
        validate_distance_matrix(D)
    except AssertionError as e:
        assert 'square' in str(e)
        print("PASS: validate_distance_matrix rejects a non-square matrix")
        return
    raise AssertionError("expected validate_distance_matrix to reject a non-square matrix")


def test_validate_distance_matrix_rejects_label_length_mismatch():
    D = np.array([[0, 1], [1, 0]], dtype=float)
    try:
        validate_distance_matrix(D, labels=['A'])
    except AssertionError as e:
        assert 'len(labels)' in str(e)
        print("PASS: validate_distance_matrix rejects a labels/dist_matrix length mismatch")
        return
    raise AssertionError("expected validate_distance_matrix to reject a labels/dist_matrix length mismatch")


def test_permanova_rejects_single_group():
    D = np.array([[0, 1], [1, 0]], dtype=float)
    try:
        permanova(D, ['A', 'A'])
    except AssertionError as e:
        assert '>= 2 groups' in str(e)
        print("PASS: permanova rejects data with only a single group")
        return
    raise AssertionError("expected permanova to reject a single-group input")


def test_permdisp_rejects_single_group():
    D = np.array([[0, 1], [1, 0]], dtype=float)
    try:
        permdisp(D, ['A', 'A'])
    except AssertionError as e:
        assert '>= 2 groups' in str(e)
        print("PASS: permdisp rejects data with only a single group")
        return
    raise AssertionError("expected permdisp to reject a single-group input")


def test_permanova_warns_on_tiny_group():
    rng = np.random.default_rng(0)
    pts = np.vstack([
        rng.normal([0, 0], 0.3, (20, 2)),
        rng.normal([10, 10], 0.3, (20, 2)),
        rng.normal([5, 5], 0.3, (2, 2)),  # a 2-member group
    ])
    labels = ['A'] * 20 + ['B'] * 20 + ['tiny'] * 2
    D = _dist_matrix(pts)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        permanova(D, labels, n_permutations=49, seed=1)
        assert any('fewer than 3 members' in str(w.message) for w in caught), (
            "expected a warning about the tiny group's unstable within-group variance"
        )
    print("PASS: permanova warns when a group has fewer than 3 members")


def test_permanova_reproducible_with_same_seed():
    rng = np.random.default_rng(0)
    pts = np.vstack([
        rng.normal([0, 0], 1.0, (15, 2)),
        rng.normal([3, 3], 1.0, (15, 2)),
    ])
    labels = ['A'] * 15 + ['B'] * 15
    D = _dist_matrix(pts)

    res1 = permanova(D, labels, n_permutations=99, seed=42)
    res2 = permanova(D, labels, n_permutations=99, seed=42)
    assert res1['p_value'] == res2['p_value'], "same seed must give an identical p-value (reproducibility)"
    print("PASS: permanova is exactly reproducible given the same seed")


if __name__ == '__main__':
    test_permanova_detects_separated_clusters()
    test_permanova_null_on_random_labels()
    test_permanova_pairwise_identifies_different_pairs_and_corrects_pvalues()
    test_permdisp_detects_dispersion_difference_with_same_centroid()
    test_permdisp_null_when_dispersion_equal()
    test_permdisp_warns_on_negative_eigenvalues()
    test_pcoa_clips_negative_eigenvalues_to_zero()
    test_benjamini_hochberg_matches_hand_computed_values()
    test_benjamini_hochberg_monotone_and_bounded()
    test_validate_distance_matrix_rejects_asymmetric()
    test_validate_distance_matrix_rejects_nonzero_diagonal()
    test_validate_distance_matrix_rejects_negative_distance()
    test_validate_distance_matrix_rejects_nan()
    test_validate_distance_matrix_rejects_non_square()
    test_validate_distance_matrix_rejects_label_length_mismatch()
    test_permanova_rejects_single_group()
    test_permdisp_rejects_single_group()
    test_permanova_warns_on_tiny_group()
    test_permanova_reproducible_with_same_seed()
    print("\nAll tests passed.")
