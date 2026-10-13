'''
Run directly: python3 tests/shared/test_embed.py
'''
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# cajal itself needs a full conda environment and won't install in a bare
# sandbox - stub it out so embed.py's top-level `import cajal.utilities`
# succeeds, then override the specific functions each test needs.
_fake_cajal = types.ModuleType('cajal')
_fake_cajal.utilities = types.ModuleType('cajal.utilities')
sys.modules['cajal'] = _fake_cajal
sys.modules['cajal.utilities'] = _fake_cajal.utilities

from shared.cajal.embed import reformat_name, get_umap_embedding
import shared.cajal.embed as embed_module


def test_reformat_name_handles_underscores_in_condition_name():
    # condition names like 'A1_1uM_vem' contain underscores themselves -
    # the regex must still split at the LAST '_frame_' boundary correctly
    cases = {
        'naive_frame_03_cell_mask_sampled_1234':       'naive_frame_03_pixel_1234',
        'A1_1uM_vem_frame_12_nuc_mask_sampled_9':       'A1_1uM_vem_frame_12_pixel_9',
        'A1_1uM_vem_frame_02_cell_mask_sampled_555':    'A1_1uM_vem_frame_02_pixel_555',
    }
    for raw, expected in cases.items():
        actual = reformat_name(raw)
        assert actual == expected, f"{raw!r} -> {actual!r}, expected {expected!r}"

    print("PASS: reformat_name correctly handles condition names containing underscores")


def test_get_umap_embedding_skips_if_output_exists(tmp_path='/tmp/claude-0/-home-user/8db9bf46-8a41-5caa-bf93-3b3ba7caed73/scratchpad/embed_test'):
    os.makedirs(tmp_path, exist_ok=True)
    out_path = os.path.join(tmp_path, 'fake_umap.npz')

    fake_labels = np.array(['naive_frame_01_pixel_1', 'A1_1uM_vem_frame_01_pixel_2'])
    fake_embedding = np.array([[0.1, 0.2], [0.3, 0.4]])
    np.savez(out_path, labels=fake_labels, embedding=fake_embedding)

    # dist_csv doesn't even need to exist - if skip-if-exists works, it's never read
    labels, embedding = get_umap_embedding('/nonexistent/path.csv', out_path, seed=42)

    assert list(labels) == list(fake_labels)
    assert np.allclose(embedding, fake_embedding)

    print("PASS: get_umap_embedding skips recomputation and reloads existing output correctly")


def test_get_umap_embedding_fits_umap_on_precomputed_distances(tmp_path='/tmp/claude-0/-home-user/8db9bf46-8a41-5caa-bf93-3b3ba7caed73/scratchpad/embed_test2'):
    os.makedirs(tmp_path, exist_ok=True)
    out_path = os.path.join(tmp_path, 'umap.npz')
    if os.path.exists(out_path):
        os.remove(out_path)

    # monkeypatch cajal.utilities.read_gw_dists/dist_mat_of_dict to avoid
    # needing a real CAJAL distance CSV - synthesize a small, valid
    # precomputed distance matrix (symmetric, zero diagonal) for two
    # well-separated clusters, so we can sanity-check UMAP actually ran
    rng = np.random.default_rng(0)
    n = 40
    names = (
        [f'A1_1uM_vem_frame_01_cell_mask_sampled_{i}' for i in range(n // 2)] +
        [f'naive_frame_01_cell_mask_sampled_{i}' for i in range(n // 2, n)]
    )
    points = np.vstack([
        rng.normal(loc=[0, 0], scale=0.3, size=(n // 2, 2)),
        rng.normal(loc=[10, 10], scale=0.3, size=(n // 2, 2)),
    ])
    dist_matrix = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    dist_dict = {(names[i], names[j]): dist_matrix[i, j] for i in range(n) for j in range(n) if i != j}

    class FakeUtilities:
        @staticmethod
        def read_gw_dists(path, header=True):
            return names, dist_dict
        @staticmethod
        def dist_mat_of_dict(d, names_):
            return dist_matrix

    original_utilities = embed_module.cajal.utilities
    embed_module.cajal.utilities = FakeUtilities()
    try:
        labels, embedding = get_umap_embedding('/fake/dist.csv', out_path, seed=42)
    finally:
        embed_module.cajal.utilities = original_utilities

    assert len(labels) == n
    assert embedding.shape == (n, 2)
    assert os.path.exists(out_path)
    # the two well-separated input clusters should still be separated in the UMAP output
    from scipy.spatial.distance import pdist, squareform
    umap_dist = squareform(pdist(embedding))
    within_a  = umap_dist[:n//2, :n//2][np.triu_indices(n//2, k=1)].mean()
    between   = umap_dist[:n//2, n//2:].mean()
    assert between > within_a, "expected the two input clusters to remain separated in the UMAP embedding"

    print("PASS: get_umap_embedding fits UMAP on a precomputed distance matrix and preserves cluster separation")


if __name__ == '__main__':
    test_reformat_name_handles_underscores_in_condition_name()
    test_get_umap_embedding_skips_if_output_exists()
    test_get_umap_embedding_fits_umap_on_precomputed_distances()
    print("\nAll tests passed.")
