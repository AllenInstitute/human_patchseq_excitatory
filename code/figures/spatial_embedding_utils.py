"""
Spatial and Embedding Visualization Utilities

This module contains functions for visualizing spatial transcriptomics data,
UMAP embeddings, and density distributions.

Main Functions:
- plot_umap: Standard UMAP scatter plot
- plot_ref_umap: Reference UMAP with overlays
- plot_ref_umap_crop: Cropped/zoomed reference UMAP
- plot_spatial_slice: Spatial transcriptomics slice visualization
- plot_spatial_density_kde: KDE density plot with peak labels

Dependencies:
- Data: numpy, pandas
- Visualization: matplotlib, seaborn
- Image processing: scipy.ndimage
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.ndimage import rotate

from core_utils import get_seaad_colors

SEAAD_COLORS = get_seaad_colors()


def plot_umap(ax, df, group_col, groups, color_dict, x_col='umap_1', y_col='umap_2',
              umap1_label='UMAP1', umap2_label='UMAP2', plot_axis_arrows=False, label_dict=None):
    """
    Plot UMAP embedding colored by groups.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    df : pandas.DataFrame
        Data with UMAP coordinates
    group_col : str
        Column name for grouping variable
    groups : list
        Ordered list of groups to plot
    color_dict : dict
        Mapping of groups to colors
    x_col, y_col : str, optional
        Column names for UMAP coordinates (default='umap_1', 'umap_2')
    umap1_label, umap2_label : str, optional
        Axis labels (default='UMAP1', 'UMAP2')
    plot_axis_arrows : bool, optional
        If True, add directional arrows for axes (default=False)
    label_dict : dict, optional
        Dictionary of label positions with format:
        {label: {'x': x_pos, 'y': y_pos, 'ha': 'left', 'va': 'center'}}

    Notes
    -----
    Creates a scatter plot with legend removed and equal aspect ratio.
    """
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=group_col, hue_order=groups,
                   palette=color_dict, s=5, ax=ax)

    if label_dict is not None:
        for label, label_formatting_dict in label_dict.items():
            color = SEAAD_COLORS.get(label, 'black')
            plt.text(label_formatting_dict['x'], label_formatting_dict['y'], label,
                    color=color, ha=label_formatting_dict['ha'],
                    va=label_formatting_dict['va'], fontsize=5)

    if plot_axis_arrows:
        xmin, _ = ax.get_xlim()
        ymin, _ = ax.get_ylim()

        # Add small x and y axis arrows
        arrowprops = dict(arrowstyle='-|>', color='grey', lw=1, mutation_scale=5)
        length = 1
        ax.annotate('', xy=(xmin + length, ymin), xytext=(xmin, ymin), arrowprops=arrowprops)
        ax.annotate('', xy=(xmin, ymin + length), xytext=(xmin, ymin), arrowprops=arrowprops)

        # Add axis labels
        ax.text(xmin, ymin-0.5, umap1_label, va='center', ha='left', fontsize=5, color='grey')
        ax.text(xmin-0.5, ymin, umap2_label, va='bottom', ha='center', fontsize=5, color='grey', rotation=90)

    ax.legend().remove()
    ax.set_aspect('equal')
    ax.axis('off')


def plot_ref_umap(reference_umap, ax, label_dict=None, line_list=None, plot_axis_arrows=False):
    """
    Plot reference UMAP with patch-seq overlay.

    Parameters
    ----------
    reference_umap : pandas.DataFrame
        Reference data with columns: 'UMAP_1', 'UMAP_2', 'subclass',
        'platform', 'color'
    ax : matplotlib.axes.Axes
        Axis to plot on
    label_dict : dict, optional
        Text labels to overlay (see plot_umap for format)
    line_list : list of dict, optional
        Lines to draw, each with 'x_values' and 'y_values'
    plot_axis_arrows : bool, optional
        If True, add directional arrows for axes (default=False)

    Notes
    -----
    Plots reference points in grey and patch-seq points in color.
    Useful for showing patch-seq cells in context of larger dataset.
    """
    # Plot all non-patch-seq points
    for hann_type, group in reference_umap[reference_umap['platform'] != 'patch-seq'].groupby('subclass'):
        plt.scatter(group['UMAP_1'], group['UMAP_2'],
                   s=1, alpha=0.7, label=hann_type,
                   c='lightgrey', edgecolor='none')

    # Plot all patch-seq points
    for hann_type, group in reference_umap[reference_umap['platform'] == 'patch-seq'].groupby('subclass'):
        plt.scatter(group['UMAP_1'], group['UMAP_2'],
                   s=3, alpha=0.9, label='patch-seq',
                   c=group['color'], marker='o', edgecolor='white', linewidths=0.05)

    ax.set_aspect('equal')
    plt.axis('off')

    if label_dict is not None:
        for label, label_formatting_dict in label_dict.items():
            color = SEAAD_COLORS.get(label, 'grey')
            plt.text(label_formatting_dict['x'], label_formatting_dict['y'], label,
                    color=color, ha=label_formatting_dict['ha'],
                    va=label_formatting_dict['va'], fontsize=5)

    if line_list is not None:
        for line in line_list:
            plt.plot(line['x_values'], line['y_values'], lw=0.5, color='lightgrey')

    if plot_axis_arrows:
        xmin, _ = ax.get_xlim()
        ymin, _ = ax.get_ylim()

        arrowprops = dict(arrowstyle='-|>', color='grey', lw=1, mutation_scale=5)
        length = 5
        ax.annotate('', xy=(xmin + length, ymin), xytext=(xmin, ymin), arrowprops=arrowprops)
        ax.annotate('', xy=(xmin, ymin + length), xytext=(xmin, ymin), arrowprops=arrowprops)

        ax.text(xmin, ymin-0.5, 't-UMAP1', va='center', ha='left', fontsize=5, color='grey')
        ax.text(xmin-0.5, ymin, 't-UMAP2', va='bottom', ha='center', fontsize=5, color='grey', rotation=90)


def plot_ref_umap_crop(reference_umap, ax, x_min, x_max, y_min, y_max, color_dict, supertypes,
                       label_dict=None, line_list=None, legend_loc=None, legend_length=1, plot_axis_arrows=False):
    """
    Plot cropped/zoomed section of reference UMAP.

    Parameters
    ----------
    reference_umap : pandas.DataFrame
        Reference data (see plot_ref_umap)
    ax : matplotlib.axes.Axes
        Axis to plot on
    x_min, x_max, y_min, y_max : float
        Crop boundaries
    color_dict : dict
        Mapping of supertypes to colors
    supertypes : list
        Supertypes to highlight in color
    label_dict : dict, optional
        Text labels to overlay
    line_list : list of dict, optional
        Lines to draw
    legend_loc : tuple, optional
        (x, y) position for axis arrows if plot_axis_arrows=True
    legend_length : float, optional
        Length of axis arrows (default=1)
    plot_axis_arrows : bool, optional
        If True, add directional arrows for axes (default=False)

    Notes
    -----
    Only plots the specified rectangular region of the UMAP.
    Supertypes in the list are colored; others are grey.
    """
    # Plot all reference points
    for hann_type, group in reference_umap[reference_umap['platform'] != 'patch-seq'].groupby('subclass'):
        plt.scatter(group['UMAP_1'], group['UMAP_2'],
                   s=1, alpha=0.7, label=hann_type,
                   c=group['color'], edgecolor='none')

    # Plot all patch-seq points
    for hann_type, group in reference_umap[reference_umap['platform'] == 'patch-seq'].groupby('subclass'):
        plt.scatter(group['UMAP_1'], group['UMAP_2'],
                   s=3, alpha=0.9, label='patch-seq',
                   facecolors=group['color'], edgecolors='white', marker='o', linewidths=0.05)

    # Crop
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_aspect('equal')
    plt.axis('off')

    if label_dict is not None:
        for label, label_formatting_dict in label_dict.items():
            color = color_dict[label] if label in supertypes else 'lightgrey'
            plt.text(label_formatting_dict['x'], label_formatting_dict['y'], label,
                    color=color, ha=label_formatting_dict['ha'],
                    va=label_formatting_dict['va'], fontsize=5)

    if line_list is not None:
        for line in line_list:
            plt.plot(line['x_values'], line['y_values'], lw=0.5, color='lightgrey')

    if plot_axis_arrows:
        if legend_loc is None:
            xmin, _ = ax.get_xlim()
            ymin, _ = ax.get_ylim()
        else:
            xmin, ymin = legend_loc

        arrowprops = dict(arrowstyle='-|>', color='grey', lw=1, mutation_scale=2)
        ax.annotate('', xy=(xmin + legend_length, ymin), xytext=(xmin, ymin), arrowprops=arrowprops)
        ax.annotate('', xy=(xmin, ymin + legend_length), xytext=(xmin, ymin), arrowprops=arrowprops)

        ax.text(xmin, ymin-0.25, 't-UMAP1', va='center', ha='left', fontsize=5, color='grey')
        ax.text(xmin-0.25, ymin, 't-UMAP2', va='bottom', ha='center', fontsize=5, color='grey', rotation=90)


def plot_spatial_slice(data, ax, groups, color_dict, rot_deg=0, ncols=-1, label_layers=True,
                       skip_subplot=[], overlay_layers=False, overlay_path=None):
    """
    Plot spatial transcriptomics data by groups.

    Parameters
    ----------
    data : AnnData
        Spatial transcriptomics data with:
        - data.obsm['spatial_cirro']: spatial coordinates
        - data.obs['SEAAD_Supertype_name']: group labels
    ax : matplotlib.axes.Axes
        Axis to plot on
    groups : list
        Groups to plot (each in separate subplot)
    color_dict : dict
        Mapping of groups to colors
    rot_deg : float, optional
        Rotation angle in degrees (default=0)
    ncols : int, optional
        Number of columns for subplot layout (default=-1, single row)
    label_layers : bool, optional
        If True, add cortical layer labels (default=True)
    skip_subplot : list, optional
        Indices of subplots to skip (for layout control)
    overlay_layers : bool, optional
        If True, overlay layer boundary image (default=False)
    overlay_path : str, optional
        Path to layer boundary overlay image

    Notes
    -----
    Creates a tiled layout with one subplot per group.
    Each subplot shows all cells in grey with the focal group in color.
    """
    layer_label_formatting_dict = {
        'L1': [250, 200],
        'L2': [50, 700],
        'L3': [-200, 1400],
        'L4': [-500, 1950],
        'L5': [-750, 2600],
        'L6': [-1300, 4000]
    }

    if overlay_layers:
        overlay = mpimg.imread(overlay_path)
        overlay_rotated = np.flipud(rotate(overlay, angle=221))

    # Slice rotation (degrees ccw)
    theta = np.deg2rad(rot_deg)
    rotation_matrix = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])

    xmin = min(data.obsm['spatial_cirro'] @ rotation_matrix.T[:, 0])
    xmax = max(data.obsm['spatial_cirro'] @ rotation_matrix.T[:, 0])
    xbuffer = abs(xmax - xmin) - 1000
    xoffset = xmin * (-1)

    ymin = min(data.obsm['spatial_cirro'] @ rotation_matrix.T[:, 1])
    ymax = max(data.obsm['spatial_cirro'] @ rotation_matrix.T[:, 1])
    ybuffer = abs(ymax - ymin) + 1000
    yoffset = ymin * (-1)

    groups = [g for g in groups if g in data.obs['SEAAD_Supertype_name'].unique()]
    plot_idx = -1
    for i, group in enumerate(groups):
        plot_idx += 1

        if i in skip_subplot:
            plot_idx += 1
            xoffset += xbuffer

        if (plot_idx > 0) & (plot_idx % ncols == 0):
            # Go to next y level
            yoffset -= ybuffer
            xoffset = xmin * (-1)

        if overlay_layers:
            # Overlay layer lines
            overlay_ymin = ymin + yoffset - 1800
            overlay_ymax = ymax + yoffset + 1300
            overlay_xmin = xmin + xoffset - 300
            overlay_xmax = xmax + xoffset + 1300
            ax.imshow(
                overlay_rotated,
                extent=[overlay_xmin, overlay_xmax, overlay_ymin, overlay_ymax],
                aspect='auto',
                alpha=1,
                zorder=10,
                interpolation='none'
            )

        # Get group data
        group_data = data[data.obs['SEAAD_Supertype_name'] == group]
        if len(group_data) > 0:
            non_group_data = data[data.obs['SEAAD_Supertype_name'] != group]

            # Plot rotated data
            ax.scatter(non_group_data.obsm['spatial_cirro'] @ rotation_matrix.T[:, 0] + xoffset,
                      non_group_data.obsm['spatial_cirro'] @ rotation_matrix.T[:, 1] + yoffset,
                      c="#b9b9b9", alpha=1, s=.05, marker='.', rasterized=True, linewidths=0)
            ax.scatter(group_data.obsm['spatial_cirro'] @ rotation_matrix.T[:, 0] + xoffset,
                      group_data.obsm['spatial_cirro'] @ rotation_matrix.T[:, 1] + yoffset,
                      c=color_dict[group], alpha=1, s=1, marker='o', rasterized=True, linewidths=0)

            # Add centered label in color
            non_group_coords = non_group_data.obsm['spatial_cirro'] @ rotation_matrix.T
            x_center = non_group_coords[:, 0].max() + xoffset
            y_center = non_group_coords[:, 1].max() + yoffset + 200
            ax.text(x_center, y_center, group, fontsize=5, ha='right', va='center', color=color_dict[group])

            is_last_in_row = ((plot_idx + 1) % ncols == 0) or (i == len(groups) - 1) or (plot_idx + 1 in skip_subplot)
            if label_layers and is_last_in_row:
                right_edge = (data.obsm['spatial_cirro'] @ rotation_matrix.T[:, 0]).max() + xoffset
                top_edge = (data.obsm['spatial_cirro'] @ rotation_matrix.T[:, 1]).max() + yoffset
                for layer_label, [layer_x, layer_y] in layer_label_formatting_dict.items():
                    ax.text(right_edge + layer_x, top_edge - layer_y, layer_label,
                           color='lightgrey', fontsize=5, ha='left', va='center')

            xoffset += xbuffer

        ax.axis('off')
        ax.set_aspect('equal')


def plot_spatial_density_kde(df, ax, y_label, hue_label, hue_order, color_dict,
                             label_peaks=True, label_y=True, label_x=True, fill=True, alpha=0.1, common_norm=True):
    """
    Plot KDE density distribution along cortical depth with peak labels.

    Parameters
    ----------
    df : pandas.DataFrame
        Data with depth measurements
    ax : matplotlib.axes.Axes
        Axis to plot on
    y_label : str
        Column name for depth/y-axis variable
    hue_label : str
        Column name for grouping variable
    hue_order : list
        Ordered list of groups
    color_dict : dict
        Mapping of groups to colors
    label_peaks : bool, optional
        If True, annotate peak positions (default=True)
    label_y : bool, optional
        If True, show y-axis labels (default=True)
    label_x : bool, optional
        If True, show x-axis label (default=True)
    fill : bool, optional
        If True, fill under KDE curves (default=True)
    alpha : float, optional
        Fill transparency (default=0.1)
    common_norm : bool, optional
        If True, normalize KDE across all groups (default=True)

    Notes
    -----
    Creates vertical KDE plots (density on x-axis, depth on y-axis).
    Peak positions are automatically detected and labeled.
    Y-axis is inverted so shallow depths are at top.
    """
    if label_peaks:
        # Initial plot (no fill) to get the peaks from
        ax = sns.kdeplot(
            data=df,
            y=y_label,
            hue=hue_label,
            hue_order=hue_order,
            palette=color_dict,
            fill=False,
            common_norm=common_norm,
            alpha=0.2,
            linewidth=0.5,
            legend=False,
            ax=ax
        )

        # Map each drawn line back to its hue
        peaks = {}
        used_hues = set()

        # Precompute RGBA for palette entries
        palette_rgba = {h: mcolors.to_rgba(color_dict[h]) for h in hue_order if h in color_dict}

        for line in ax.lines:
            x, y = line.get_data()
            if len(x) == 0:
                continue
            peak_idx = np.argmax(x)
            peak_x = float(x[peak_idx])
            peak_y = float(y[peak_idx])

            # Try to get hue from line label
            label = line.get_label()
            mapped_hue = None
            if isinstance(label, str) and label in hue_order:
                mapped_hue = label

            # If label didn't help, match by color (RGBA)
            if mapped_hue is None:
                line_rgba = mcolors.to_rgba(line.get_color())
                candidates = [h for h in palette_rgba.keys() if h not in used_hues]
                if not candidates:
                    continue
                # Prefer exact-ish match
                for h in candidates:
                    if np.allclose(line_rgba, palette_rgba[h], atol=1e-2):
                        mapped_hue = h
                        break
                # Fallback: nearest by Euclidean distance
                if mapped_hue is None:
                    mapped_hue = min(candidates, key=lambda h: np.linalg.norm(np.array(line_rgba) - np.array(palette_rgba[h])))

            peaks[mapped_hue] = {'peak_x': peak_x, 'peak_y': peak_y}
            used_hues.add(mapped_hue)

        # Convert to DataFrame
        peak_df = (
            pd.DataFrame.from_dict(peaks, orient='index')
            .reset_index()
            .rename(columns={'index': hue_label})
        )

        # Clear and replot kde with fill
        ax.cla()

    ax = sns.kdeplot(
        data=df,
        y=y_label,
        hue=hue_label,
        hue_order=hue_order,
        palette=color_dict,
        fill=fill,
        common_norm=common_norm,
        alpha=alpha,
        linewidth=0.2,
        legend=False,
        ax=ax
    )

    ax.set_ylim(0, 3500)  # Standard isodepth range

    ax.set_xticks([])
    if label_x:
        ax.set_xlabel("Density", fontsize=6)
    else:
        ax.set_xlabel("")

    if label_y:
        ax.set_yticks([0, 2800])
        ax.set_yticklabels(['pia', '~wm'])
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])
    ax.set_ylabel("")

    sns.despine(ax=ax)
    plt.gca().invert_yaxis()
    ax.tick_params(axis='both', colors='grey', labelsize=6)
    for spine in ax.spines.values():
        spine.set_color('grey')

    if label_peaks:
        # Overlay text on the peaks
        for _, row in peak_df.iterrows():
            hue = row[hue_label]
            peak_x = row['peak_x']
            peak_y = row['peak_y']

            text_x = peak_x + 0.00001
            text_y = peak_y

            ax.text(
                text_x,
                text_y,
                hue,
                color=color_dict.get(hue, 'k'),
                fontsize=5,
                fontweight='bold',
                rotation=15,
                rotation_mode='anchor',
                va='center',
                ha='left',
                zorder=10,
            )


def get_spatial_density_kde(df, y_label, hue_label, hue_order, color_dict):

    fig, ax = plt.subplots()
    
    # Initial plot (no fill) to get the peaks from
    ax = sns.kdeplot(
        data=df,
        y=y_label,
        hue=hue_label, 
        hue_order=hue_order,
        palette=color_dict,
        fill=False,
        common_norm=True,
        alpha=0.2,
        linewidth=2,
        legend=False
    )

    # Map each drawn line back to its hue (line order from sns.kdeplot does not match hue_order)
    peaks = {}
    used_hues = set()

    # Precompute RGBA for your palette entries
    palette_rgba = {h: mcolors.to_rgba(color_dict[h]) for h in hue_order if h in color_dict}

    for line in ax.lines:
        x, y = line.get_data()
        if len(x) == 0:
            continue

        # scale by group’s global proportion
        line.set_data(x, y)

        peak_idx = np.argmax(x)
        peak_x = float(x[peak_idx])
        peak_y = float(y[peak_idx])

        # Try to get hue from line label if seaborn embedded it
        label = line.get_label()
        mapped_hue = None
        if isinstance(label, str) and label in hue_order:
            mapped_hue = label

        # If label didn't help, match by color (RGBA)
        if mapped_hue is None:
            line_rgba = mcolors.to_rgba(line.get_color())
            # find unmatched hue with closest color
            candidates = [h for h in palette_rgba.keys() if h not in used_hues]
            if not candidates:
                continue
            # prefer exact-ish match
            for h in candidates:
                if np.allclose(line_rgba, palette_rgba[h], atol=1e-2):
                    mapped_hue = h
                    break
            # fallback: pick nearest by Euclidean distance in RGBA space
            if mapped_hue is None:
                mapped_hue = min(candidates, key=lambda h: np.linalg.norm(np.array(line_rgba) - np.array(palette_rgba[h])))

        peaks[mapped_hue] = {'peak_x': peak_x, 'peak_y': peak_y, 
                             'line_x' : x,     'line_y' : y}
        used_hues.add(mapped_hue)

    # Convert to DataFrame
    peak_df = pd.DataFrame.from_dict(peaks, orient='index').reset_index().rename(columns={'index': hue_label}).set_index(hue_label)

    # delete plot
    plt.close(fig)  
    
    return peak_df


def plot_precomputed_kde(ax, df, ttype_order, color_dict, label_x=True, label_y=True, label_peaks=True):

    for t in ttype_order:
        if t in df.index:
            ax.plot(df.loc[t].line_x, 
                    df.loc[t].line_y, 
                    c=color_dict[t])

    ax.set_ylim(0, 3500) #standard isodepth range 
    ax.set_xticks([])   
    if label_x:
        ax.set_xlabel("Density", fontsize=6)
    else:
        ax.set_xlabel("")

    if label_y:
        ax.set_yticks([0, 2800])  
        ax.set_yticklabels(['pia', '~wm']) 
    else:
        ax.set_yticks([])  
        ax.set_yticklabels([]) 
    ax.set_ylabel("")

    sns.despine(ax=ax)
    plt.gca().invert_yaxis()  # optional: deeper depths lower on plot
    ax.tick_params(axis='both', colors='grey', labelsize=6)  # Set ticks color to greycolors='grey', 
    for spine in ax.spines.values(): spine.set_color('grey')

    if label_peaks:
        # Overlay text on the peaks
        for label in ttype_order:
            if label in df.index:
                peak_x = df.loc[label]['peak_x']
                peak_y = df.loc[label]['peak_y']

                # offset a bit to the right and up for visibility
                text_x = peak_x + 0.00001   # move 5% right of the peak
                text_y = peak_y          # same depth value

                ax.text(
                    text_x,
                    text_y,
                    label,  # the label text
                    color=color_dict[label],
                    fontsize=5,
                    fontweight='bold',
                    rotation=15,
                    rotation_mode='anchor',
                    va='center',
                    ha='left',
                    zorder=10,
                )
