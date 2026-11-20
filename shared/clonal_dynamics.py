import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

# distinct from every condition/subclass/group 'tab:*' color already used
# elsewhere in this repo, AND from shared.representative_shapes'
# REGION_COLORS (#ffe119/#f032e6/#008080/#000075/#800000 - see that
# module's own comment for the full tab10 survey this follows) - a clone
# color identical to either would make an unrelated category look like
# the same thing across different plots.
CLONE_COLORS = ['#46f0f0', '#9a6324', '#000000']

SUBCLASSES = ['dying', 'persisting', 'slow_prolif', 'fast_prolif']


def pick_tightest_clones(embedded_df, dist_matrix, label_to_index, n_clones=2, min_embedded_cells=10,
                          lin_id_col='lin_id', frame_col='frame', pixel_id_col='pixel_id'):
    '''
    Pick the n_clones lin_ids ("clones") whose own embedded cells are
    most tightly clustered morphologically - lowest mean pairwise CAJAL
    GW distance among themselves, using the REAL distance matrix, not
    the UMAP/PHATE 2D projection (which distorts distances - the same
    rule this repo already follows everywhere else a quantitative claim
    is made, e.g. shared/pair_similarity.py) - among lin_ids with enough
    cells in the embedded sample to make that mean meaningful.

    Inputs:
        embedded_df:        the CAJAL-embedded subset (has lin_id,
                            frame, pixel_id), already restricted by the
                            caller to one subclass
        dist_matrix:        (n, n) real CAJAL GW distance matrix, e.g.
                            from shared.cajal.embed.load_distance_matrix
        label_to_index:     {(frame, pixel_id): row/col index into
                            dist_matrix} - e.g. built from
                            load_distance_matrix's own labels via
                            shared.load_data.parse_label
        n_clones:           how many lin_ids to pick (fewer are returned
                            if fewer than n_clones are eligible)
        min_embedded_cells: floor on embedded-cell count for eligibility
                            (must be >= 2 - a mean pairwise distance
                            needs at least 2 cells to exist at all)
        lin_id_col, frame_col, pixel_id_col: column names
    Outputs:
        top_lin_ids: list of lin_ids, ranked tightest (lowest mean
                     pairwise distance) first - length <= n_clones
        summary:     DataFrame indexed by lin_id, same rows/order as
                     top_lin_ids, columns [n_cells,
                     mean_pairwise_distance] - for CSV export/transparency
    '''
    min_embedded_cells = max(2, min_embedded_cells)

    rows = []
    for lin_id, group in embedded_df.groupby(lin_id_col):
        indices = [
            label_to_index[(frame, pixel_id)]
            for frame, pixel_id in zip(group[frame_col], group[pixel_id_col])
            if (frame, pixel_id) in label_to_index
        ]
        if len(indices) < min_embedded_cells:
            continue

        indices = np.array(indices)
        sub = dist_matrix[np.ix_(indices, indices)]
        iu = np.triu_indices(len(indices), k=1)
        rows.append({'lin_id': lin_id, 'n_cells': len(indices), 'mean_pairwise_distance': sub[iu].mean()})

    summary = pd.DataFrame(rows).set_index('lin_id') if rows else pd.DataFrame(
        columns=['n_cells', 'mean_pairwise_distance']
    )
    summary = summary.sort_values('mean_pairwise_distance', ascending=True)

    top = summary.head(n_clones)
    return top.index.tolist(), top


def plot_clones_on_embedding(background_df, clone_dfs, lin_ids, colors, embedding_label, title):
    '''
    One stacked panel per clone (same layout as Figure 1g of Shaffer et
    al., "Diverse clonal fates emerge upon drug treatment of homogeneous
    cancer cells"): background_df's full population in light grey behind
    every panel, with that panel's own clone highlighted in its own color.

    This is purely a VISUALIZATION on the 2D UMAP/PHATE embedding - which
    clones get highlighted is decided upstream (see pick_tightest_clones)
    on the real CAJAL distance matrix, never on dim_0/dim_1 here.

    Inputs:
        background_df:   the full embedding population to show as grey
                         context in every panel (dim_0, dim_1)
        clone_dfs:       [df, ...] - one DataFrame per clone, each
                         already restricted to that one lin_id's own
                         embedded cells (dim_0, dim_1)
        lin_ids:         [lin_id, ...] - same order/length as clone_dfs, for labeling
        colors:          [color, ...] - same order/length as clone_dfs
        embedding_label: axis label text only, e.g. 'UMAP' or 'PHATE'
        title:           figure suptitle
    Outputs:
        fig: caller saves/closes this
    '''
    n = len(clone_dfs)
    fig, axes = plt.subplots(n, 1, figsize=(5.5, 4.4 * n), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, clone_df, lin_id, color in zip(axes, clone_dfs, lin_ids, colors):
        ax.scatter(background_df['dim_0'], background_df['dim_1'], s=3, color='lightgrey', alpha=0.3, zorder=1)
        ax.scatter(clone_df['dim_0'], clone_df['dim_1'], s=14, color=color, zorder=2,
                   label=f'lin_id {lin_id} (n={len(clone_df)})')
        ax.set_xlabel(f'{embedding_label} 1')
        ax.set_ylabel(f'{embedding_label} 2')
        ax.legend(loc='best', fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def export_clones_by_subclass(embedded_df, dist_matrix, label_to_index, output_dir, file_prefix, embedding_label,
                               title_prefix, n_clones=2, min_embedded_cells=10, colors=CLONE_COLORS,
                               subclasses=SUBCLASSES):
    '''
    For each subclass, pick the n_clones tightest-clustered lin_ids (see
    pick_tightest_clones) and paint them onto embedded_df's embedding
    (see plot_clones_on_embedding) - one PDF/SVG figure per subclass:

    output_dir/{file_prefix}_{subclass}.pdf, output_dir/svgs/{file_prefix}_{subclass}.svg

    Also writes one combined output_dir/{file_prefix}_clone_summary.csv
    (every chosen clone, every subclass, with its n_cells/
    mean_pairwise_distance) so the exact selection stays reproducible/
    inspectable without re-running this function, and displays every
    figure directly in the notebook's output.

    Inputs:
        embedded_df:     the CAJAL-embedded subset for one condition
                         (has dim_0, dim_1, subclass, lin_id, frame,
                         pixel_id) - e.g. from load_full_dataset. Also
                         used as EVERY panel's own grey background, so
                         clones are shown in context of the whole
                         condition's embedding, not just their own subclass.
        dist_matrix, label_to_index: see pick_tightest_clones - the SAME
                         condition/modality this embedded_df came from
        output_dir, file_prefix: as elsewhere in this repo (see shared.density_plots, shared.representative_shapes)
        embedding_label: 'UMAP' or 'PHATE'
        title_prefix:    e.g. 'cell morphology' - the subclass name is appended automatically
        n_clones, min_embedded_cells: see pick_tightest_clones
        colors:          [color, ...], at least n_clones long
        subclasses:      which subclass values to produce a figure for
    Outputs:
        summary: combined clone_summary DataFrame across every subclass
                 that had an eligible clone, for notebook inspection
    '''
    svg_dir = os.path.join(output_dir, 'svgs')
    os.makedirs(svg_dir, exist_ok=True)

    summary_parts = []
    for subclass in subclasses:
        embedded_subset = embedded_df[embedded_df['subclass'] == subclass]

        lin_ids, summary = pick_tightest_clones(embedded_subset, dist_matrix, label_to_index, n_clones, min_embedded_cells)
        if not lin_ids:
            print(f"  No clone in subclass '{subclass}' has >= {max(2, min_embedded_cells)} embedded cells - skipping")
            continue
        if len(lin_ids) < n_clones:
            print(f"  Only {len(lin_ids)}/{n_clones} eligible clone(s) in subclass '{subclass}' "
                  f"(>= {max(2, min_embedded_cells)} embedded cells)")

        clone_dfs = [embedded_subset[embedded_subset['lin_id'] == lin_id] for lin_id in lin_ids]
        fig = plot_clones_on_embedding(
            embedded_df, clone_dfs, lin_ids, colors[:len(lin_ids)], embedding_label,
            f'{title_prefix} - {subclass} clones',
        )
        fig.savefig(os.path.join(output_dir, f'{file_prefix}_{subclass}.pdf'), bbox_inches='tight')
        fig.savefig(os.path.join(svg_dir, f'{file_prefix}_{subclass}.svg'), bbox_inches='tight')
        display(fig)
        plt.close(fig)

        subclass_summary = summary.reset_index()
        subclass_summary.insert(0, 'subclass', subclass)
        summary_parts.append(subclass_summary)

    combined = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    if not combined.empty:
        combined.to_csv(os.path.join(output_dir, f'{file_prefix}_clone_summary.csv'), index=False)

    print(f"  Saved to {output_dir}")
    return combined
