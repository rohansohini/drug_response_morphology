import os

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.cluster import KMeans

from shared.cajal.extract_masks import extracted_mask_paths

REGION_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                  'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


def assign_kmeans_regions(df, n_regions, seed=42):
    '''
    K-means cluster df's 2D embedding into n_regions groups.

    Inputs:
        df:        must have: dim_0, dim_1
        n_regions: number of clusters
        seed:      KMeans random_state, for reproducibility
    Outputs:
        df: a copy of the input, plus an int 'region' column (0-indexed cluster id)
    '''
    km = KMeans(n_clusters=n_regions, random_state=seed, n_init=10)
    labels = km.fit_predict(df[['dim_0', 'dim_1']].to_numpy())

    df = df.copy()
    df['region'] = labels
    return df


def crop_cell_shape(mask_path, pixel_id, half_width=150):
    '''
    Isolate one cell's own shape out of a shared per-frame mask TIFF
    (shared.cajal.extract_masks' own output - every sampled cell in that
    frame is in the same file, labeled by pixel_id), cropped to a fixed
    (2*half_width, 2*half_width) box centered on that cell, with every
    other cell's pixels zeroed out. Always returns the same shape
    regardless of where the cell sits in the frame (zero-padded at the
    image's edges if the box would otherwise run off it), so every
    region's representative shape is directly comparable/stackable.

    Inputs:
        mask_path:  path to the extracted {drug}_frame_NN_{cell,nuc}_mask_sampled.tiff
        pixel_id:   this cell's label within that mask
        half_width: half the crop's side length in pixels - must be large
                    enough to contain the cell's own full extent; prints
                    a warning (does not raise) if it wasn't, since a
                    missed cell shouldn't crash the whole plot
    Outputs:
        crop: (2*half_width, 2*half_width) array, same dtype as the mask,
              zero everywhere except this cell's own pixels
    '''
    mask = tifffile.imread(mask_path)
    coords = np.argwhere(mask == pixel_id)
    assert len(coords) > 0, f"pixel_id {pixel_id} not found in {mask_path}"

    row_min, col_min = coords.min(axis=0)
    row_max, col_max = coords.max(axis=0)
    center_row, center_col = coords.mean(axis=0).round().astype(int)

    side = 2 * half_width
    crop = np.zeros((side, side), dtype=mask.dtype)

    # the box we'd like, centered on the cell, before clipping to the image
    row0, row1 = center_row - half_width, center_row + half_width
    col0, col1 = center_col - half_width, center_col + half_width

    # clip to the image's actual bounds, and shift the destination slice
    # in `crop` by the same amount, so a box that runs off the edge still
    # lands in the right place rather than wrapping or misaligning
    src_row0, src_row1 = max(row0, 0), min(row1, mask.shape[0])
    src_col0, src_col1 = max(col0, 0), min(col1, mask.shape[1])
    dst_row0, dst_col0 = src_row0 - row0, src_col0 - col0

    window = mask[src_row0:src_row1, src_col0:src_col1]
    cell_only = np.where(window == pixel_id, window, 0)
    crop[dst_row0:dst_row0 + cell_only.shape[0], dst_col0:dst_col0 + cell_only.shape[1]] = cell_only

    if row_min < row0 or row_max >= row1 or col_min < col0 or col_max >= col1:
        print(
            f"  WARNING: half_width={half_width} is too small to fully contain pixel_id "
            f"{pixel_id} in {mask_path} - its shape is clipped in this crop. Increase "
            f"half_width and re-run to fix."
        )

    return crop


def pick_representative_shapes(df, masks_dir, modality, n_regions, half_width=150, seed=42):
    '''
    K-means cluster df's embedding into n_regions, then for each region
    pick the real cell closest to that region's centroid and crop its
    own shape out of its (condition, frame)'s extracted mask TIFF.

    Inputs:
        df:         must have: dim_0, dim_1, condition, frame, pixel_id
        masks_dir:  an analysis's own config.MASKS_DIR
        modality:   'cell' or 'nuc' - selects which of the two mask files
                    extracted_mask_paths points at
        n_regions:  number of k-means clusters (= number of representative shapes)
        half_width: see crop_cell_shape
        seed:       KMeans random_state, for reproducibility
    Outputs:
        df:     a copy of the input, plus a 'region' column
        shapes: one row per region, columns: region, condition, frame,
                pixel_id (the representative cell's own identity), crop
    '''
    df = assign_kmeans_regions(df, n_regions, seed)

    rows = []
    for region in sorted(df['region'].unique()):
        subset = df[df['region'] == region]
        center_x, center_y = subset['dim_0'].mean(), subset['dim_1'].mean()

        dist_sq = (subset['dim_0'] - center_x) ** 2 + (subset['dim_1'] - center_y) ** 2
        nearest = subset.loc[dist_sq.idxmin()]

        cell_path, nuc_path = extracted_mask_paths(nearest['condition'], int(nearest['frame']), masks_dir)
        mask_path = cell_path if modality == 'cell' else nuc_path
        crop = crop_cell_shape(mask_path, int(nearest['pixel_id']), half_width)

        rows.append({
            'region': region, 'condition': nearest['condition'], 'frame': int(nearest['frame']),
            'pixel_id': int(nearest['pixel_id']), 'crop': crop,
        })

    return df, pd.DataFrame(rows)


def plot_representative_shapes(df, shapes, embedding_label, title):
    '''
    One figure: a scatter of df's embedding colored by region (with each
    region's centroid marked), next to a row of that region's actual
    representative shape, bordered in the same color.

    Inputs:
        df:              from pick_representative_shapes - must have:
                         dim_0, dim_1, region
        shapes:          from pick_representative_shapes
        embedding_label: axis label text only, e.g. 'UMAP' or 'PHATE'
        title:           figure title
    Outputs:
        fig: caller saves/closes this
    '''
    n_regions = len(shapes)
    fig = plt.figure(figsize=(6 + 2 * n_regions, 6))
    grid = fig.add_gridspec(1, 2, width_ratios=[6, 2 * n_regions])
    scatter_ax = fig.add_subplot(grid[0])
    shape_axes = [fig.add_subplot(grid[1].subgridspec(1, n_regions)[0, i]) for i in range(n_regions)]

    for region, group in df.groupby('region'):
        color = REGION_COLORS[region % len(REGION_COLORS)]
        scatter_ax.scatter(group['dim_0'], group['dim_1'], s=3, color=color, alpha=0.4,
                            label=f'region {region} (n={len(group)})')
    for _, shape_row in shapes.iterrows():
        color = REGION_COLORS[shape_row['region'] % len(REGION_COLORS)]
        center = df[df['region'] == shape_row['region']][['dim_0', 'dim_1']].mean()
        scatter_ax.scatter(center['dim_0'], center['dim_1'], s=200, color=color,
                            edgecolor='black', marker='X', zorder=3)

    scatter_ax.set_title(title)
    scatter_ax.set_xlabel(f'{embedding_label} 1')
    scatter_ax.set_ylabel(f'{embedding_label} 2')
    scatter_ax.legend(fontsize=8)

    for ax, (_, shape_row) in zip(shape_axes, shapes.iterrows()):
        color = REGION_COLORS[shape_row['region'] % len(REGION_COLORS)]
        ax.imshow(shape_row['crop'] > 0, cmap='gray_r')
        ax.set_title(f"region {shape_row['region']}", color=color)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    fig.tight_layout()
    return fig


def export_representative_shapes(df, masks_dir, modality, n_regions, embedding_label, title, output_dir, file_prefix, half_width=150, seed=42):
    '''
    Full pipeline: cluster, pick each region's representative shape, plot,
    and save. See pick_representative_shapes and plot_representative_shapes.

    output_dir/{file_prefix}.pdf
    output_dir/svgs/{file_prefix}.svg

    Also displays the figure directly in the notebook's output, so you
    don't have to open the saved files just to see it.

    Outputs:
        df: from pick_representative_shapes, for notebook inspection
    '''
    df, shapes = pick_representative_shapes(df, masks_dir, modality, n_regions, half_width, seed)
    fig = plot_representative_shapes(df, shapes, embedding_label, title)

    svg_dir = os.path.join(output_dir, 'svgs')
    os.makedirs(svg_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f'{file_prefix}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(svg_dir, f'{file_prefix}.svg'), bbox_inches='tight')
    display(fig)
    plt.close(fig)
    print(f"  Saved to {output_dir}")

    return df
