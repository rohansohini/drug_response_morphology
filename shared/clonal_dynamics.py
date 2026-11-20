import os

import matplotlib.pyplot as plt
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


def pick_top_clones(full_df, embedded_df, n_clones=3, min_embedded_cells=5,
                     lin_id_col='lin_id', frame_col='frame'):
    '''
    Pick the n_clones lin_ids ("clones") with the most TRUE segmented
    cells, among those with enough cells actually in the CAJAL-embedded
    sample to be worth plotting.

    full_df is the true population a lin_id's size is measured against -
    e.g. every has_mask vem_1uM cell of one subclass, straight from
    {condition}_metadata.csv, NOT just the randomly budgeted subset that
    went through CAJAL (an analysis's embedding only ever covers a
    sampled fraction of all segmented cells). embedded_df is that
    CAJAL-embedded subset (has dim_0/dim_1), same condition and subclass
    restriction - only lin_ids with at least min_embedded_cells cells
    there are eligible at all, since a clone with the most true cells but
    zero embedded ones can't be painted onto the embedding.

    Ties in true cell count are broken by how many distinct frames that
    lin_id's EMBEDDED cells span (more spread over time wins) - a
    secondary criterion only; "most cells" is still the primary one.

    Inputs:
        full_df:            true per-lin_id population, already
                            restricted by the caller to one condition,
                            has_mask=True, and one subclass
        embedded_df:        the CAJAL-embedded subset, same condition and
                            subclass restriction (has dim_0, dim_1)
        n_clones:           how many lin_ids to pick (fewer are returned
                            if fewer than n_clones are eligible)
        min_embedded_cells: floor on embedded-cell count for eligibility
        lin_id_col, frame_col: column names
    Outputs:
        top_lin_ids: list of lin_ids, ranked (true count desc, then
                     embedded frame-span desc) - length <= n_clones
        summary:     DataFrame indexed by lin_id, same rows/order as
                     top_lin_ids, columns [true_n_cells,
                     embedded_n_cells, embedded_n_frames] - for CSV
                     export/transparency
    '''
    true_counts     = full_df.groupby(lin_id_col).size().rename('true_n_cells')
    embedded_counts = embedded_df.groupby(lin_id_col).size().rename('embedded_n_cells')
    embedded_frames = embedded_df.groupby(lin_id_col)[frame_col].nunique().rename('embedded_n_frames')

    summary = pd.concat([true_counts, embedded_counts, embedded_frames], axis=1)
    for col in ['true_n_cells', 'embedded_n_cells', 'embedded_n_frames']:
        summary[col] = summary[col].fillna(0).astype(int)

    eligible = summary[summary['embedded_n_cells'] >= min_embedded_cells]
    eligible = eligible.sort_values(['true_n_cells', 'embedded_n_frames'], ascending=False)

    top = eligible.head(n_clones)
    return top.index.tolist(), top


def plot_clones_on_embedding(background_df, clone_dfs, lin_ids, colors, embedding_label, title):
    '''
    One stacked panel per clone (same layout as Figure 1g of Shaffer et
    al., "Diverse clonal fates emerge upon drug treatment of homogeneous
    cancer cells"): background_df's full population in light grey behind
    every panel, with that panel's own clone highlighted in its own color.

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


def export_clones_by_subclass(embedded_df, full_metadata_df, output_dir, file_prefix, embedding_label, title_prefix,
                               n_clones=3, min_embedded_cells=5, colors=CLONE_COLORS, subclasses=SUBCLASSES):
    '''
    For each subclass, pick the top n_clones lin_ids (see pick_top_clones)
    and paint them onto embedded_df's embedding (see
    plot_clones_on_embedding) - one PDF/SVG figure per subclass:

    output_dir/{file_prefix}_{subclass}.pdf, output_dir/svgs/{file_prefix}_{subclass}.svg

    Also writes one combined output_dir/{file_prefix}_clone_summary.csv
    (every chosen clone, every subclass, with its true_n_cells/
    embedded_n_cells/embedded_n_frames) so the exact selection stays
    reproducible/inspectable without re-running this function, and
    displays every figure directly in the notebook's output.

    Inputs:
        embedded_df:       the CAJAL-embedded subset for one condition
                           (has dim_0, dim_1, subclass, lin_id, frame) -
                           e.g. vem_df from load_full_dataset, already
                           restricted to the vem condition. Also used as
                           EVERY panel's own grey background, so clones
                           are shown in context of the whole condition's
                           embedding, not just their own subclass.
        full_metadata_df:  the SAME condition's full has_mask metadata,
                           with a 'subclass' column already added (e.g.
                           via shared.classify.classify_subclass) - the
                           true population each lin_id's count is
                           measured against, not just the CAJAL-sampled
                           subset
        output_dir, file_prefix: as elsewhere in this repo (see shared.density_plots, shared.representative_shapes)
        embedding_label:   'UMAP' or 'PHATE'
        title_prefix:      e.g. 'cell morphology' - the subclass name is appended automatically
        n_clones, min_embedded_cells: see pick_top_clones
        colors:            [color, ...], at least n_clones long
        subclasses:        which subclass values to produce a figure for
    Outputs:
        summary: combined clone_summary DataFrame across every subclass
                 that had an eligible clone, for notebook inspection
    '''
    svg_dir = os.path.join(output_dir, 'svgs')
    os.makedirs(svg_dir, exist_ok=True)

    summary_parts = []
    for subclass in subclasses:
        embedded_subset = embedded_df[embedded_df['subclass'] == subclass]
        full_subset     = full_metadata_df[full_metadata_df['subclass'] == subclass]

        lin_ids, summary = pick_top_clones(full_subset, embedded_subset, n_clones, min_embedded_cells)
        if not lin_ids:
            print(f"  No clone in subclass '{subclass}' has >= {min_embedded_cells} embedded cells - skipping")
            continue
        if len(lin_ids) < n_clones:
            print(f"  Only {len(lin_ids)}/{n_clones} eligible clone(s) in subclass '{subclass}' "
                  f"(>= {min_embedded_cells} embedded cells)")

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
