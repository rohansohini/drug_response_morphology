import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns


def plot_umap_with_kde(df, group_col, group_colors, ax, title, min_group_size=5, kde_alpha=0.4):
    '''
    Scatter every cell in df (light grey background), then overlay a KDE
    contour per group in group_col.

    Inputs:
        df:             must have: umap_0, umap_1, and group_col
        group_col:      column identifying the group to contour per (e.g.
                        'condition' or 'subclass')
        group_colors:   {group_value: color} - also determines which
                        groups get plotted and their legend order
        ax:             matplotlib axis to plot on
        title:          plot title
        min_group_size: skip a group's KDE contour if it has fewer points
                        than this (a KDE from a handful of points is
                        noise, not signal - better to omit it than show a
                        misleading shape)
        kde_alpha:      opacity for the KDE contour fill
    '''
    ax.scatter(df['umap_0'], df['umap_1'], s=3, color='lightgrey', alpha=0.3, zorder=1)

    # seaborn's kdeplot(fill=True) doesn't attach a legend-visible label to
    # its filled contour collections (confirmed: they get an auto '_child0'
    # label that matplotlib's legend ignores by convention) - build the
    # legend from manual proxy patches instead of relying on kdeplot's own
    # label kwarg, which silently produces an empty legend otherwise.
    legend_handles = []
    for group, color in group_colors.items():
        subset = df[df[group_col] == group]
        if len(subset) < min_group_size:
            continue
        sns.kdeplot(
            x=subset['umap_0'], y=subset['umap_1'], ax=ax,
            color=color, fill=True, alpha=kde_alpha, levels=5, zorder=2,
        )
        legend_handles.append(mpatches.Patch(color=color, label=f'{group} (n={len(subset)})'))

    ax.set_title(title)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(handles=legend_handles, fontsize=8)


def export_kde_by_frame(df, group_col, group_colors, output_dir, file_prefix, title_prefix=None, min_group_size=5):
    '''
    One PDF and one matching SVG per frame present in df, each a
    plot_umap_with_kde call restricted to that frame's cells. Saved to
    output_dir/pdfs/ and output_dir/svgs/ respectively - SVG has no
    multi-page format, so both formats are written one file per frame
    (rather than one multi-page PDF), so the two folders contain the same
    set of frames. Shared by every analysis notebook - they differ only in
    which df/group_col/colors/output_dir they pass in.

    Inputs:
        df:             must have: frame, umap_0, umap_1, group_col
        group_col:      column to contour per (e.g. 'condition' or 'subclass')
        group_colors:   {group_value: color}
        output_dir:     directory to create pdfs/ and svgs/ subfolders in
        file_prefix:    filename prefix, frame number appended (e.g. 'cell' -> 'cell_frame_01.pdf')
        title_prefix:   prefix for each plot's title (defaults to file_prefix if not given)
        min_group_size: passed through to plot_umap_with_kde
    '''
    title_prefix = file_prefix if title_prefix is None else title_prefix

    frames  = sorted(df['frame'].unique())
    pdf_dir = os.path.join(output_dir, 'pdfs')
    svg_dir = os.path.join(output_dir, 'svgs')
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)

    for frame in frames:
        subset = df[df['frame'] == frame]
        fig, ax = plt.subplots(figsize=(6, 6))
        plot_umap_with_kde(
            subset, group_col, group_colors, ax,
            f'{title_prefix} - frame {frame}', min_group_size=min_group_size,
        )
        filename = f'{file_prefix}_frame_{frame:02d}'
        fig.savefig(os.path.join(pdf_dir, f'{filename}.pdf'), bbox_inches='tight')
        fig.savefig(os.path.join(svg_dir, f'{filename}.svg'), bbox_inches='tight')
        plt.close(fig)

    print(f"  Saved {len(frames)} frame(s) to {pdf_dir} and {svg_dir}")
