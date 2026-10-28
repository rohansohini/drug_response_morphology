import re
import os
import time

import numpy as np
import umap
import phate

import cajal.utilities


def reformat_name(name: str) -> str:
    '''
    Reformat a CAJAL point name from
    '{condition}_frame_{frame}_{cell|nuc}_mask_sampled_{pixel_id}'
    to '{condition}_frame_{frame}_pixel_{pixel_id}'.

    Inputs:
        name: raw point name from read_gw_dists
    Outputs:
        reformatted name
    '''
    match = re.match(r'(.+_frame_\d+)_(?:cell|nuc)_mask_sampled_(\d+)', name)
    condition_frame, pixel_id = match.groups()
    return f'{condition_frame}_pixel_{pixel_id}'


def load_distance_matrix(dist_csv: str) -> tuple[list[str], np.ndarray]:
    '''
    Load a CAJAL quantized-GW distance matrix and its point labels from disk.
    Used by get_umap_embedding/get_phate_embedding (which fit an embedding
    for plotting) and by anything that needs the distance matrix itself
    for quantitative work
    (e.g. PERMANOVA/PERMDISP, which must NOT operate on the UMAP projection -
    UMAP distorts distances, so it's the wrong representation for anything
    quantitative).

    Inputs:
        dist_csv: path to a *_qgw_*_dists.csv file (cell or nuc), covering
                  every selected cell from every condition together (so
                  distances are directly comparable across conditions)
    Outputs:
        labels:      list of node identifiers '{condition}_frame_{frame}_pixel_{pixel_id}'
        dist_matrix: (n_nodes, n_nodes) symmetric GW distance matrix, same
                     row/col order as labels
    '''
    print("  Loading distance matrix...")
    t0 = time.time()
    names, dist_dict = cajal.utilities.read_gw_dists(dist_csv, header=True)
    dist_matrix = cajal.utilities.dist_mat_of_dict(dist_dict, names)
    labels = [reformat_name(n) for n in names]
    print(f"  Loaded {len(labels)} points ({time.time() - t0:.1f}s)")
    return labels, dist_matrix


def get_umap_embedding(dist_csv: str, out_path: str, seed: int = 42) -> tuple[list[str], np.ndarray]:
    '''
    Fit a 2D UMAP directly on a CAJAL quantized-GW distance matrix.
    Skips computation if the output file already exists.

    Inputs:
        dist_csv: path to a *_qgw_*_dists.csv file (cell or nuc), covering
                  every selected cell from every condition together (so
                  distances are directly comparable across conditions)
        out_path: path to save the .npz output
        seed:     UMAP random state, for reproducibility
    Outputs:
        labels:    list of node identifiers '{condition}_frame_{frame}_pixel_{pixel_id}'
        embedding: (n_nodes, 2) UMAP coordinates, same row order as labels
    '''
    if os.path.exists(out_path):
        print(f"  Skipping UMAP: {out_path} already exists")
        data = np.load(out_path, allow_pickle=True)
        return list(data['labels']), data['embedding']

    labels, dist_matrix = load_distance_matrix(dist_csv)

    print("  Fitting UMAP (precomputed distance)...")
    t0 = time.time()
    reducer   = umap.UMAP(n_components=2, metric='precomputed', random_state=seed)
    embedding = reducer.fit_transform(dist_matrix)
    print(f"  UMAP done ({time.time() - t0:.1f}s)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, labels=np.array(labels), embedding=embedding)
    print(f"  Saved embedding to {out_path}")

    return labels, embedding


def get_phate_embedding(dist_csv: str, out_path: str, seed: int = 42) -> tuple[list[str], np.ndarray]:
    '''
    Fit a 2D PHATE embedding directly on a CAJAL quantized-GW distance matrix.
    Skips computation if the output file already exists.

    Inputs:
        dist_csv: path to a *_qgw_*_dists.csv file (cell or nuc), covering
                  every selected cell from every condition together (so
                  distances are directly comparable across conditions)
        out_path: path to save the .npz output
        seed:     PHATE random state, for reproducibility
    Outputs:
        labels:    list of node identifiers '{condition}_frame_{frame}_pixel_{pixel_id}'
        embedding: (n_nodes, 2) PHATE coordinates, same row order as labels
    '''
    if os.path.exists(out_path):
        print(f"  Skipping PHATE: {out_path} already exists")
        data = np.load(out_path, allow_pickle=True)
        return list(data['labels']), data['embedding']

    labels, dist_matrix = load_distance_matrix(dist_csv)

    print("  Fitting PHATE (precomputed distance)...")
    t0 = time.time()
    # knn_dist='precomputed_distance' (rather than the ambiguous
    # 'precomputed') because PHATE would otherwise try to auto-detect
    # distance vs. affinity from whether the diagonal is all-zero - our GW
    # distance matrix already is, so being explicit skips that guess
    reducer   = phate.PHATE(n_components=2, knn_dist='precomputed_distance', random_state=seed, verbose=0)
    embedding = reducer.fit_transform(dist_matrix)
    print(f"  PHATE done ({time.time() - t0:.1f}s)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, labels=np.array(labels), embedding=embedding)
    print(f"  Saved embedding to {out_path}")

    return labels, embedding
