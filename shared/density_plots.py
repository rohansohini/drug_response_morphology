import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from IPython.display import display
import seaborn as sns


def plot_embedding_with_kde(df, group_col, group_colors, ax, title, embedding_label, min_group_size=5, levels=(0.7, 0.8, 0.9), bw_adjust=1, xlim=None, ylim=None):
    '''
    Scatter every cell in df (light grey background), then overlay a KDE
    contour per group in group_col. Works on any 2D embedding (UMAP,
    PHATE, ...) - it only ever reads the generic dim_0/dim_1 columns that
    shared.load_data.build_dataset always produces, regardless of which
    method computed them; embedding_label only controls the axis labels.

    Inputs:
        df:              must have: dim_0, dim_1, and group_col
        group_col:       column identifying the group to contour per (e.g.
                         'condition' or 'subclass')
        group_colors:    {group_value: color} - also determines which
                         groups get plotted and their legend order
        ax:              matplotlib axis to plot on
        title:           plot title
        embedding_label: name of the embedding method, for axis labels
                         only (e.g. 'UMAP' or 'PHATE') - purely cosmetic,
                         has no effect on what gets plotted
        min_group_size:  skip a group's KDE contour if it has fewer points
                         than this (a KDE from a handful of points is
                         noise, not signal - better to omit it than show a
                         misleading shape)
        levels:          iso-proportion levels to draw as unfilled contour
                         lines (outline only, no fill) - level p encloses
                         the densest (1-p) fraction of that group's mass,
                         so higher values are tighter, closer-to-the-peak
                         rings
        bw_adjust:       KDE bandwidth multiplier - above 1 smooths over
                         sampling/distance-tie noise, below 1 hews closer
                         to the raw point density
        xlim, ylim:      (min, max) tuples to fix the axes to, or None to
                         auto-scale to this call's own data (see
                         export_kde_by_frame, which fixes these across
                         every frame's page so they're directly comparable)
    '''
    ax.scatter(df['dim_0'], df['dim_1'], s=3, color='lightgrey', alpha=0.3, zorder=1)

    # seaborn's kdeplot doesn't attach a legend-visible label to its contour
    # collections, filled or not (confirmed: they get an auto '_child0' label
    # that matplotlib's legend ignores by convention) - build the legend from
    # manual proxy handles instead of relying on kdeplot's own label kwarg,
    # which silently produces an empty legend otherwise.
    legend_handles = []
    for group, color in group_colors.items():
        subset = df[df[group_col] == group]
        if len(subset) < min_group_size:
            continue
        sns.kdeplot(
            x=subset['dim_0'], y=subset['dim_1'], ax=ax,
            color=color, fill=False, levels=list(levels), linewidths=1.5, zorder=2,
            bw_adjust=bw_adjust,
        )
        legend_handles.append(Line2D([0], [0], color=color, linewidth=1.5, label=f'{group} (n={len(subset)})'))

    ax.set_title(title)
    ax.set_xlabel(f'{embedding_label} 1')
    ax.set_ylabel(f'{embedding_label} 2')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(handles=legend_handles, fontsize=8)


def plot_embedding_with_weighted_kde(df, weight_cols, group_colors, ax, title, embedding_label, min_total_weight=5, levels=(0.7, 0.8, 0.9), bw_adjust=1, xlim=None, ylim=None):
    '''
    Like plot_embedding_with_kde, but for cells with a MIXTURE of group
    membership instead of one hard label - e.g. each cell has its own
    %dying/%persisting/%slow_prolif/%fast_prolif fate breakdown, rather
    than being classified as exactly one of those. One weighted KDE
    contour per group, where every cell contributes to EVERY group's
    contour in proportion to its own percentage in that group (via
    seaborn's native `weights` support), instead of being forced into
    exactly one group's contour and omitted from the rest.

    Inputs:
        df:               must have: dim_0, dim_1, and one column per
                          weight_cols value
        weight_cols:      {group_value: column name in df holding that
                          group's per-cell weight (e.g. 0-1 fraction or
                          0-100 percentage - only relative scale matters,
                          seaborn normalizes internally)} - also
                          determines which groups get plotted and their
                          legend order
        group_colors:     {group_value: color}
        ax:               matplotlib axis to plot on
        title:            plot title
        embedding_label:  see plot_embedding_with_kde
        min_total_weight: skip a group's KDE contour if its weights sum
                          to less than this across every cell (mirrors
                          plot_embedding_with_kde's min_group_size - a KDE
                          built from almost no weight is noise, not signal)
        levels:           see plot_embedding_with_kde
        bw_adjust:        see plot_embedding_with_kde
        xlim, ylim:       see plot_embedding_with_kde
    '''
    ax.scatter(df['dim_0'], df['dim_1'], s=3, color='lightgrey', alpha=0.3, zorder=1)

    legend_handles = []
    for group, color in group_colors.items():
        weights = df[weight_cols[group]].to_numpy()
        total_weight = weights.sum()
        if total_weight < min_total_weight:
            continue
        sns.kdeplot(
            x=df['dim_0'], y=df['dim_1'], weights=weights, ax=ax,
            color=color, fill=False, levels=list(levels), linewidths=1.5, zorder=2,
            bw_adjust=bw_adjust,
        )
        legend_handles.append(Line2D([0], [0], color=color, linewidth=1.5,
                                      label=f'{group} (total weight={total_weight:.1f})'))

    ax.set_title(title)
    ax.set_xlabel(f'{embedding_label} 1')
    ax.set_ylabel(f'{embedding_label} 2')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(handles=legend_handles, fontsize=8)


def export_kde_by_frame(df, group_col, group_colors, output_dir, file_prefix, embedding_label, title_prefix=None, min_group_size=5, levels=(0.7, 0.8, 0.9), bw_adjust=1):
    '''
    One combined, multi-page PDF and one SVG per frame (in their own
    svgs/ subfolder, since there are many of those but only one PDF) -
    each page a plot_embedding_with_kde call restricted to that frame's
    cells. Shared by every analysis notebook - they differ only in which
    df/group_col/colors/output_dir they pass in, and which embedding's
    coordinates df holds.

    output_dir/{file_prefix}.pdf
    output_dir/svgs/{file_prefix}_frame_NN.svg

    Also displays each frame's figure directly in the notebook's output
    as it's generated, so you don't have to open the saved files just to
    see them.

    Every page shares the same, fixed axis limits, computed once from
    every frame's cells in df together and padded 5% on each side,
    rather than each page auto-scaling to just its own frame's data
    range. Without this, two pages can independently end up with very
    similar axis ranges whenever their data happens to span a similar
    extent, making two actually different points land in very similar
    relative screen positions purely from the auto-scaling, not because
    the underlying data matches. Fixed axes remove that ambiguity and
    make frames directly visually comparable besides.

    Inputs:
        df:              must have: frame, dim_0, dim_1, group_col
        group_col:       column to contour per (e.g. 'condition' or 'subclass')
        group_colors:    {group_value: color}
        output_dir:      directory to save into (and create a svgs/ subfolder in)
        file_prefix:     filename prefix (e.g. 'cell_umap' -> 'cell_umap.pdf', 'cell_umap_frame_01.svg')
        embedding_label: see plot_embedding_with_kde, purely cosmetic
                         (axis labels only, e.g. 'UMAP' or 'PHATE')
        title_prefix:    prefix for each plot's title (defaults to file_prefix if not given)
        min_group_size:  passed through to plot_embedding_with_kde
        levels:          passed through to plot_embedding_with_kde
        bw_adjust:       passed through to plot_embedding_with_kde
    '''
    title_prefix = file_prefix if title_prefix is None else title_prefix

    frames = sorted(df['frame'].unique())
    assert len(frames) > 0, (
        f"df has no rows at all for '{file_prefix}' - check whatever filtering produced it "
        f"(e.g. a condition/subclass name that no longer matches the real data)"
    )
    svg_dir = os.path.join(output_dir, 'svgs')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)

    pdf_path = os.path.join(output_dir, f'{file_prefix}.pdf')

    x_min, x_max = df['dim_0'].min(), df['dim_0'].max()
    y_min, y_max = df['dim_1'].min(), df['dim_1'].max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    xlim  = (x_min - x_pad, x_max + x_pad)
    ylim  = (y_min - y_pad, y_max + y_pad)

    with PdfPages(pdf_path) as pdf:
        for frame in frames:
            subset = df[df['frame'] == frame]
            fig, ax = plt.subplots(figsize=(6, 6))
            plot_embedding_with_kde(
                subset, group_col, group_colors, ax,
                f'{title_prefix}: frame {frame}', embedding_label, min_group_size=min_group_size,
                levels=levels, bw_adjust=bw_adjust, xlim=xlim, ylim=ylim,
            )
            pdf.savefig(fig, bbox_inches='tight')
            fig.savefig(os.path.join(svg_dir, f'{file_prefix}_frame_{frame:02d}.svg'), bbox_inches='tight')
            display(fig)   # show it in the notebook's own output, not just save it to disk
            plt.close(fig)

    print(f"  Saved {len(frames)} frame(s) to {pdf_path} and {svg_dir}")
