import os

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.cluster import KMeans

from shared.cajal.extract_masks import extracted_mask_paths

# every analysis using this module already uses all 10 matplotlib
# 'tab:*' colors elsewhere for something else (condition, subclass, or
# group labels - e.g. CONDITION_COLORS/SUBCLASS_COLORS/GROUP_COLORS in
# various notebooks) - reusing any of them here for k-means REGIONS
# would make two unrelated categories look like the same thing across
# different plots. These 5 (yellow, magenta, teal, navy, maroon) were
# picked to stay visually distinct from every 'tab:*' color already in
# use, and from each other.
REGION_COLORS = ['#ffe119', '#f032e6', '#008080', '#000075', '#800000']


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


def plot_embedding_by_region(df, embedding_label, title):
    '''
    A scatter of df's embedding colored by region, with each region's
    centroid marked with an X in that same color.

    Inputs:
        df:              from pick_representative_shapes - must have:
                         dim_0, dim_1, region
        embedding_label: axis label text only, e.g. 'UMAP' or 'PHATE'
        title:           figure title
    Outputs:
        fig: caller saves/closes this
    '''
    fig, ax = plt.subplots(figsize=(7, 6))

    for region, group in df.groupby('region'):
        color = REGION_COLORS[region % len(REGION_COLORS)]
        ax.scatter(group['dim_0'], group['dim_1'], s=3, color=color, alpha=0.4,
                   label=f'region {region} (n={len(group)})')
    for region in sorted(df['region'].unique()):
        color = REGION_COLORS[region % len(REGION_COLORS)]
        center = df[df['region'] == region][['dim_0', 'dim_1']].mean()
        ax.scatter(center['dim_0'], center['dim_1'], s=200, color=color,
                   edgecolor='black', marker='X', zorder=3)

    ax.set_title(title)
    ax.set_xlabel(f'{embedding_label} 1')
    ax.set_ylabel(f'{embedding_label} 2')
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_representative_shapes(shapes, title, n_regions):
    '''
    One row of each region's actual representative cell shape, bordered
    in that region's own color (the same colors plot_embedding_by_region
    marks its centroid with).

    ALWAYS renders exactly n_regions panels, the caller's own requested
    region count - NOT len(shapes) (how many regions actually got a
    representative shape). These are almost always equal (k-means
    relocates empty clusters during fitting, so an empty final region is
    not expected in practice), but deriving the image size from the
    REQUESTED count rather than the data keeps every export at the same,
    predictable size regardless: a region missing its own shape for any
    reason gets an empty "no cells in this region" placeholder panel
    instead of silently shrinking the whole figure.

    Inputs:
        shapes:    from pick_representative_shapes - may have fewer than
                  n_regions rows
        title:     figure title
        n_regions: the number of panels to render (the region count this
                  plot was actually clustered into - see
                  assign_kmeans_regions/pick_representative_shapes)
    Outputs:
        fig: caller saves/closes this
    '''
    fig, axes = plt.subplots(1, n_regions, figsize=(2.2 * n_regions, 2.6))
    if n_regions == 1:
        axes = [axes]

    shape_by_region = {row['region']: row for _, row in shapes.iterrows()}

    for region, ax in enumerate(axes):
        shape_row = shape_by_region.get(region)
        color = REGION_COLORS[region % len(REGION_COLORS)]

        if shape_row is not None:
            ax.imshow(shape_row['crop'] > 0, cmap='gray_r')
            ax.set_title(f"region {region}", color=color)
        else:
            ax.text(0.5, 0.5, 'no cells\nin this region', ha='center', va='center',
                     fontsize=8, color='grey', transform=ax.transAxes)
            ax.set_title(f"region {region}", color='grey')

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color if shape_row is not None else 'lightgrey')
            spine.set_linewidth(3)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def export_representative_shapes(df, masks_dir, modality, n_regions, embedding_label, title, output_dir, file_prefix, half_width=150, seed=42):
    '''
    Full pipeline: cluster, pick each region's representative shape, plot,
    and save - as two SEPARATE figures/files, not one combined plot (see
    plot_embedding_by_region and plot_representative_shapes):

    output_dir/{file_prefix}_embedding.png, output_dir/svgs/{file_prefix}_embedding.svg
    output_dir/{file_prefix}_shapes.png,    output_dir/svgs/{file_prefix}_shapes.svg

    Also displays both figures directly in the notebook's output, so you
    don't have to open the saved files just to see them.

    half_width is a crop size in pixels around each representative
    cell's own centroid - keep it the SAME across every region within
    one call (every region here always gets the one half_width passed
    in), but pick a SEPARATE, smaller value for 'nuc' than for 'cell'
    when calling this per modality: a nucleus is much smaller than its
    whole cell, so the same half_width that comfortably fits a cell
    crop leaves a nucleus shape too small to make out, surrounded by
    mostly empty padding.

    Outputs:
        df: from pick_representative_shapes, for notebook inspection
    '''
    df, shapes = pick_representative_shapes(df, masks_dir, modality, n_regions, half_width, seed)

    svg_dir = os.path.join(output_dir, 'svgs')
    os.makedirs(svg_dir, exist_ok=True)

    embedding_fig = plot_embedding_by_region(df, embedding_label, title)
    embedding_fig.savefig(os.path.join(output_dir, f'{file_prefix}_embedding.png'), dpi=200, bbox_inches='tight')
    embedding_fig.savefig(os.path.join(svg_dir, f'{file_prefix}_embedding.svg'), bbox_inches='tight')
    display(embedding_fig)
    plt.close(embedding_fig)

    shapes_fig = plot_representative_shapes(shapes, title, n_regions)
    shapes_fig.savefig(os.path.join(output_dir, f'{file_prefix}_shapes.png'), dpi=200, bbox_inches='tight')
    shapes_fig.savefig(os.path.join(svg_dir, f'{file_prefix}_shapes.svg'), bbox_inches='tight')
    display(shapes_fig)
    plt.close(shapes_fig)

    print(f"  Saved to {output_dir}")

    return df
