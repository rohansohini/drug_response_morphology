import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import seaborn as sns


def plot_embedding_with_kde(df, group_col, group_colors, ax, title, embedding_label, min_group_size=5, levels=(0.3, 0.5, 0.7), xlim=None, ylim=None):
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
                         lines (outline only, no fill)
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
            bw_adjust=1.2,  # smooth over sampling/distance-tie noise a bit, since we're
                            # now contouring tight around the peak (see levels above) -
                            # a noisy, under-smoothed KDE shows more spurious bumps there
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


def plot_embedding_with_weighted_kde(df, weight_cols, group_colors, ax, title, embedding_label, min_total_weight=5, levels=(0.3, 0.5, 0.7), xlim=None, ylim=None):
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
        levels:           iso-proportion levels to draw as unfilled
                          contour lines (outline only, no fill)
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
            bw_adjust=1.2,  # see plot_embedding_with_kde
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


def export_kde_by_frame(df, group_col, group_colors, output_dir, file_prefix, embedding_label, title_prefix=None, min_group_size=5):
    '''
    One combined, multi-page PDF (one page per frame present in df), plus
    one SVG per frame, each a plot_embedding_with_kde call restricted to
    that frame's cells. PDF is saved to output_dir/pdfs/{file_prefix}.pdf;
    SVGs to output_dir/svgs/{file_prefix}_frame_NN.svg - SVG has no
    multi-page format, so it stays one file per frame while the PDF is
    combined into a single file. Shared by every analysis notebook - they
    differ only in which df/group_col/colors/output_dir they pass in, and
    (now) which embedding's coordinates df holds.

    Every page shares the same, fixed axis limits - computed once from
    every frame's cells in df together, padded 5% on each side - rather
    than each page auto-scaling to just its own frame's data range. Without
    this, two pages can independently end up with very similar axis
    ranges whenever their data happens to span a similar extent, making
    two actually-different points land in very similar relative screen
    positions purely from the auto-scaling, not because the underlying
    data matches - fixed axes remove that ambiguity and make frames
    directly visually comparable besides.

    Inputs:
        df:              must have: frame, dim_0, dim_1, group_col
        group_col:       column to contour per (e.g. 'condition' or 'subclass')
        group_colors:    {group_value: color}
        output_dir:      directory to create pdfs/ and svgs/ subfolders in
        file_prefix:     filename prefix (e.g. 'cell' -> 'cell.pdf', 'cell_frame_01.svg')
        embedding_label: see plot_embedding_with_kde - purely cosmetic
                         (axis labels only, e.g. 'UMAP' or 'PHATE')
        title_prefix:    prefix for each plot's title (defaults to file_prefix if not given)
        min_group_size:  passed through to plot_embedding_with_kde
    '''
    title_prefix = file_prefix if title_prefix is None else title_prefix

    frames  = sorted(df['frame'].unique())
    pdf_dir = os.path.join(output_dir, 'pdfs')
    svg_dir = os.path.join(output_dir, 'svgs')
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)

    pdf_path = os.path.join(pdf_dir, f'{file_prefix}.pdf')

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
                f'{title_prefix} - frame {frame}', embedding_label, min_group_size=min_group_size,
                xlim=xlim, ylim=ylim,
            )
            pdf.savefig(fig, bbox_inches='tight')
            fig.savefig(os.path.join(svg_dir, f'{file_prefix}_frame_{frame:02d}.svg'), bbox_inches='tight')
            plt.close(fig)

    print(f"  Saved {len(frames)} frame(s) to {pdf_path} and {svg_dir}")
