import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import seaborn as sns


def plot_umap_with_kde(df, group_col, group_colors, ax, title, min_group_size=5, levels=(0.7, 0.8, 0.9)):
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
        levels:         iso-proportion levels to draw as unfilled contour
                        lines (outline only, no fill)
    '''
    ax.scatter(df['umap_0'], df['umap_1'], s=3, color='lightgrey', alpha=0.3, zorder=1)

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
            x=subset['umap_0'], y=subset['umap_1'], ax=ax,
            color=color, fill=False, levels=list(levels), linewidths=1.5, zorder=2,
        )
        legend_handles.append(Line2D([0], [0], color=color, linewidth=1.5, label=f'{group} (n={len(subset)})'))

    ax.set_title(title)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(handles=legend_handles, fontsize=8)


def export_kde_by_frame(df, group_col, group_colors, output_dir, file_prefix, title_prefix=None, min_group_size=5):
    '''
    One combined, multi-page PDF (one page per frame present in df), plus
    one SVG per frame, each a plot_umap_with_kde call restricted to that
    frame's cells. PDF is saved to output_dir/pdfs/{file_prefix}.pdf; SVGs
    to output_dir/svgs/{file_prefix}_frame_NN.svg - SVG has no multi-page
    format, so it stays one file per frame while the PDF is combined into
    a single file. Shared by every analysis notebook - they differ only in
    which df/group_col/colors/output_dir they pass in.

    Inputs:
        df:             must have: frame, umap_0, umap_1, group_col
        group_col:      column to contour per (e.g. 'condition' or 'subclass')
        group_colors:   {group_value: color}
        output_dir:     directory to create pdfs/ and svgs/ subfolders in
        file_prefix:    filename prefix (e.g. 'cell' -> 'cell.pdf', 'cell_frame_01.svg')
        title_prefix:   prefix for each plot's title (defaults to file_prefix if not given)
        min_group_size: passed through to plot_umap_with_kde
    '''
    title_prefix = file_prefix if title_prefix is None else title_prefix

    frames  = sorted(df['frame'].unique())
    pdf_dir = os.path.join(output_dir, 'pdfs')
    svg_dir = os.path.join(output_dir, 'svgs')
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)

    pdf_path = os.path.join(pdf_dir, f'{file_prefix}.pdf')

    with PdfPages(pdf_path) as pdf:
        for frame in frames:
            subset = df[df['frame'] == frame]
            fig, ax = plt.subplots(figsize=(6, 6))
            plot_umap_with_kde(
                subset, group_col, group_colors, ax,
                f'{title_prefix} - frame {frame}', min_group_size=min_group_size,
            )
            pdf.savefig(fig, bbox_inches='tight')
            fig.savefig(os.path.join(svg_dir, f'{file_prefix}_frame_{frame:02d}.svg'), bbox_inches='tight')
            plt.close(fig)

    print(f"  Saved {len(frames)} frame(s) to {pdf_path} and {svg_dir}")
