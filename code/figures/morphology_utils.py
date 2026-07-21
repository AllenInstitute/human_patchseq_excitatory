"""
Morphology Visualization Utilities

This module contains functions for visualizing neuronal morphologies,
depth profiles, hairball overlays, and structural properties.

Main Functions:
- plot_morph_lineup_swb: Gallery display of aligned morphologies
- plot_dendrite_depth_profiles_swb: Dendrite depth distributions
- plot_axon_depth_profiles_swb: Axon depth distributions
- plot_prop_for_mets_swb: Morphological property stripplots
- plot_morpho_gallery: Gallery interface with soma depths
- overlay_morphologies: Overlay multiple morphologies
- plot_distal_apicals: Distal apical tip visualization
- process_distal_apicals: Extract distal apical locations
- plot_subclass_hairballs: Morphology hairball overlays
- plot_subclass_heatmap: KDE heatmap with masking
- basic_morph_plot: Core morphology plotting function
- read_and_normalize_swc: SWC reading and normalization
- compartment_list_by_type: Helper for compartment filtering

Dependencies:
- Data: numpy, pandas
- Visualization: matplotlib, seaborn
- Morphology: allensdk, neuron_morphology, morph_utils
"""

import os
import numpy as np
import pandas as pd
import warnings
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar
import seaborn as sns

import allensdk.core.swc as swc
from neuron_morphology.swc_io import morphology_from_swc
from morph_utils.measurements import leftextent, rightextent

# Import from core_utils
from core_utils import hex_to_gray, adjust_lightness, get_seaad_colors

SEAAD_COLORS = get_seaad_colors()

# Morphology constants
AVERAGE_DISTANCE_BETWEEN_NODES = 1.14  # μm, average distance between nodes for IVSCC data


def add_swc_paths(data, directory, column_name='swc_file_path', drop_missing=True):
    """
    Add SWC file paths to DataFrame by specimen ID.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing a 'specimen_id' column
    directory : str
        Path to the directory with SWC files
    column_name : str, optional
        Name of the new column for SWC paths (default='swc_file_path')
    drop_missing : bool, optional
        If True, removes rows without a matching SWC file (default=True)

    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with the SWC file paths added

    Notes
    -----
    Searches for SWC files named '{specimen_id}.swc' in the specified directory.
    Prints warning message listing any specimens without matching SWC files.
    """
    data[column_name] = np.nan
    for idx, row in data.iterrows():
        if os.path.isfile(os.path.join(directory, str(int(row.specimen_id)) + '.swc')):
            data.loc[idx, column_name] = os.path.join(directory, str(int(row.specimen_id)) + '.swc')

    if drop_missing:
        # Don't plot cells without swc files
        missing_idxs = list(data.loc[pd.isna(data[column_name]), :].index)
        if len(missing_idxs) > 0:
            dropped_cells = list(data.specimen_id[missing_idxs])
            data = data.drop(index=missing_idxs)
            print('Warning: could not get swcs for some cells. Not plotting these specimen ids: ', dropped_cells)

    return data


def plot_morph_lineup_swb(ax, merge_df, type_column, types, aligned_swc_dir, layer_edges, layer_labels, color_dict,
                          plot_quantiles=None, path_color="#cccccc", morph_spacing=300, plot_scalebar=True,
                          plot_layer_labels=True, compartment_type='dendrite'):
    """
    Plot aligned morphologies in a horizontal lineup.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    merge_df : pandas.DataFrame
        Data with specimen info
    type_column : str
        Column name for type grouping
    types : list
        Types to plot (in order)
    aligned_swc_dir : str
        Directory containing layer-aligned SWC files
    layer_edges : dict
        Layer boundary depths (keys: layer names, values: depths in μm)
    layer_labels : dict
        Layer label positions (keys: layer names, values: depths for labels)
    color_dict : dict
        Mapping of types to colors
    plot_quantiles : array, optional
        Quantiles to plot (not currently used)
    path_color : str, optional
        Color for layer lines (default='#cccccc')
    morph_spacing : int, optional
        Horizontal spacing between morphologies in μm (default=300)
    plot_scalebar : bool, optional
        If True, add scale bar (default=True)
    plot_layer_labels : bool, optional
        If True, add layer labels (default=True)
    compartment_type : {'dendrite', 'axon'}, optional
        Which compartments to plot (default='dendrite')

    Notes
    -----
    Morphologies are colored by type for dendrites (basal darker, apical lighter).
    For axons, greyscale colors are used.
    Grey stars mark non-TemL specimens.
    """
    if plot_quantiles is None:
        plot_quantiles = np.linspace(0, 1, 5)

    xoffset = 0
    for type in types:
        spec_ids = merge_df[merge_df[type_column] == type].sort_values("soma aligned distance from pia").specimen_id

        color = color_dict[type]

        morph_locs = []
        for spec_id in spec_ids:
            swc_path = os.path.join(aligned_swc_dir, f"{spec_id}.swc")
            morph = swc.read_swc(swc_path)
            if compartment_type == 'dendrite':
                morph_colors = {3: adjust_lightness(color, 0.5), 4: color}
            else:  # axon
                morph_colors = {2: hex_to_gray(color)}

            basic_morph_plot(morph, ax=ax, xoffset=xoffset, morph_colors=morph_colors)
            morph_locs.append(xoffset)

            # Add a grey mark if not TemL
            if merge_df[merge_df.specimen_id == spec_id]['lobe'].values[0] != 'TemL':
                ax.plot([xoffset], [-list(layer_edges)[-1]-100], marker='o', color='grey', markersize=1, zorder=5)

            xoffset += morph_spacing

        title_str = type
        if title_str.count(" ") > 1:
            title_split = title_str.split(" ")
            title_str = title_split[0] + " " + title_split[1] + "\n" + title_split[2]
        ax.text(np.mean(morph_locs), 100, title_str, fontsize=6, color=color, ha="center")

    sns.despine(ax=ax, bottom=True)
    ax.set_xticks([])
    ax.set_xlim(-morph_spacing / 1.25, xoffset - morph_spacing / 2)
    ax.set_aspect("equal")

    for e in layer_edges:
        ax.axhline(-e, zorder=-5, color=path_color, linewidth=0.75)
    ax.axis('off')

    if plot_scalebar:
        scalebar = ScaleBar(1, "um", location='lower left', frameon=False, fixed_value=500, color='grey', font_properties={'size': 6})
        ax.add_artist(scalebar)

    if plot_layer_labels:
        for ll in layer_labels:
            layer_label_buffer = 0
            if ll == 'L1':
                layer_label_buffer = 50
            if ll == 'L2':
                layer_label_buffer = -50
            ax.text(-700, -layer_labels[ll]+layer_label_buffer, "{}".format(ll),
                   verticalalignment='center', horizontalalignment='left', fontsize=6, color='lightgrey')


def plot_dendrite_depth_profiles_swb(ax, merge_df, type_column, types, hist_df, basal_cols, apical_cols,
                                     layer_edges, color_dict, plot_quantiles=None, path_color="#cccccc",
                                     spacing=200, bin_width=5, plot_scalebar=True):
    """
    Plot dendrite depth profiles by type.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    merge_df : pandas.DataFrame
        Specimen metadata
    type_column : str
        Column for type grouping
    types : list
        Types to plot
    hist_df : pandas.DataFrame
        Histogram data (specimen_id index, depth bins as columns)
    basal_cols : list
        Column names for basal dendrite bins
    apical_cols : list
        Column names for apical dendrite bins
    layer_edges : dict
        Layer boundaries
    color_dict : dict
        Type to color mapping
    plot_quantiles : array, optional
        Not currently used
    path_color : str, optional
        Layer line color
    spacing : int, optional
        Horizontal spacing between profiles (default=200)
    bin_width : int, optional
        Histogram bin width in μm (default=5)
    plot_scalebar : bool, optional
        If True, add scale bar

    Returns
    -------
    dict
        Mapping of types to x-offsets (for alignment with other plots)
    """
    if plot_quantiles is None:
        plot_quantiles = np.linspace(0, 1, 5)

    xoffset = 0
    xoffsets = {}
    for type in types:
        spec_ids = merge_df.loc[merge_df[type_column] == type, :].sort_values("soma aligned distance from pia").index

        color = color_dict[type]
        sub_depth_df = hist_df.loc[
            hist_df.index.intersection(spec_ids), basal_cols + apical_cols]
        avg_depth = sub_depth_df.mean(axis=0)
        basal_avg = avg_depth[basal_cols].values
        apical_avg = avg_depth[apical_cols].values
        all_avg = basal_avg + apical_avg
        zero_mask = all_avg > 0
        ax.plot(
            all_avg[zero_mask] + xoffset,
            -np.arange(len(all_avg))[zero_mask] * bin_width,
            c=color, linewidth=1, zorder=10
        )
        xoffsets[type] = xoffset
        xoffset += spacing

    sns.despine(ax=ax, bottom=True, left=True)
    ax.set_xticks([])
    ax.tick_params(axis="y", left=False, labelleft=False)
    for e in layer_edges:
        ax.axhline(-e, zorder=-5, color=path_color, linewidth=0.75)

    if plot_scalebar:
        scalebar = ScaleBar(1, "um", location='lower left', frameon=False, fixed_value=100, color='grey', font_properties={'size': 6})
        ax.add_artist(scalebar)

    return xoffsets


def plot_axon_depth_profiles_swb(ax, merge_df, type_column, types, hist_df, axon_cols,
                                 layer_edges, plot_quantiles=None, path_color="#cccccc",
                                 spacing=200, bin_width=5, plot_scalebar=True):
    """
    Plot axon depth profiles by type (in greyscale).

    Parameters
    ----------
    Similar to plot_dendrite_depth_profiles_swb, but:
    - axon_cols : list of axon bin columns (instead of basal/apical)
    - Plots in greyscale (converted from type colors)

    Returns
    -------
    None
    """
    if plot_quantiles is None:
        plot_quantiles = np.linspace(0, 1, 5)

    xoffset = 0
    for type in types:
        spec_ids = merge_df.loc[merge_df[type_column] == type, :].sort_values("soma aligned distance from pia").index

        color = hex_to_gray(SEAAD_COLORS[type])
        sub_depth_df = hist_df.loc[
            hist_df.index.intersection(spec_ids), axon_cols]
        avg_depth = sub_depth_df.mean(axis=0)
        axon_avg = avg_depth[axon_cols].values
        zero_mask = axon_avg > 0
        ax.plot(
            axon_avg[zero_mask] + xoffset,
            -np.arange(len(axon_avg))[zero_mask] * bin_width,
            c=color, linewidth=1, zorder=10
        )
        xoffset += spacing

    sns.despine(ax=ax, bottom=True, left=True)
    ax.set_xticks([])
    ax.tick_params(axis="y", left=False, labelleft=False)
    for e in layer_edges:
        ax.axhline(-e, zorder=-5, color=path_color, linewidth=0.75)

    if plot_scalebar:
        scalebar = ScaleBar(1, "um", location='lower left', frameon=False, fixed_value=100, color='grey', font_properties={'size': 6})
        ax.add_artist(scalebar)


def plot_dendrite_depth_profiles_subclass(ax, merge_df, type_column, types, hist_df, basal_cols, apical_cols,
        layer_edges, plot_quantiles=None, path_color="#cccccc", spacing=200, bin_width=5, plot_scalebar=False): #xoffsets
    if plot_quantiles is None:
        plot_quantiles = np.linspace(0, 1, 5)

    xoffset = 0
    xoffsets = {}
    for type in types:
        spec_ids = merge_df.loc[merge_df[type_column] == type, :].sort_values("soma aligned distance from pia").index

        color = SEAAD_COLORS[type]
        sub_depth_df = hist_df.loc[
            hist_df.index.intersection(spec_ids), basal_cols + apical_cols]
        avg_depth = sub_depth_df.mean(axis=0)
        basal_avg = avg_depth[basal_cols].values
        apical_avg = avg_depth[apical_cols].values
        all_avg = basal_avg + apical_avg
        zero_mask = all_avg > 0
        ax.plot(
            all_avg[zero_mask] + xoffset,
            -np.arange(len(all_avg))[zero_mask] * bin_width,
            c=color, linewidth=1, zorder=10
        )
        xoffsets[type] = xoffset
        xoffset += spacing

    sns.despine(ax=ax, bottom=True, left=True)
    ax.set_xticks([])
    ax.tick_params(axis="y", left=False, labelleft=False)
    for e in layer_edges:
        ax.axhline(-e, zorder=-5, color=path_color, linewidth=0.75)

    if plot_scalebar:
        scalebar = ScaleBar(1, "um", location='lower left', frameon=False, fixed_value=250, color = 'grey', font_properties={'size': 6})
        ax.add_artist(scalebar)

    return xoffsets


def plot_axon_depth_profiles_subclass(ax, merge_df, type_column, types, hist_df, axon_cols,
        layer_edges, plot_quantiles=None, path_color="#cccccc", spacing=200, bin_width=5, plot_scalebar=True): #xoffsets, 
    if plot_quantiles is None:
        plot_quantiles = np.linspace(0, 1, 5)

    xoffset = 0 
    for type in types:
        # xoffset = xoffsets[type]
        spec_ids = merge_df.loc[merge_df[type_column] == type, :].sort_values("soma aligned distance from pia").index

        color = SEAAD_COLORS[type]
        sub_depth_df = hist_df.loc[
            hist_df.index.intersection(spec_ids), axon_cols]
        avg_depth = sub_depth_df.mean(axis=0)
        axon_avg = avg_depth[axon_cols].values
        zero_mask = axon_avg > 0
        ax.plot(
            axon_avg[zero_mask] + xoffset,
            -np.arange(len(axon_avg))[zero_mask] * bin_width,
            c=color, linewidth=1, zorder=10
        )
        xoffset += spacing

    sns.despine(ax=ax, bottom=True, left=True)
    ax.set_xticks([])
    ax.tick_params(axis="y", left=False, labelleft=False)
    for e in layer_edges:
        ax.axhline(-e, zorder=-5, color=path_color, linewidth=0.75)

    if plot_scalebar:
        scalebar = ScaleBar(1, "um", location='lower left', frameon=False, fixed_value=250, color = 'grey', font_properties={'size': 6})
        ax.add_artist(scalebar)


def plot_prop_for_mets_swb(ax, types, merge_df, type_column, prop_name, prop_label,
                           show_yticklabels=True, xlim=None, color_column=None, color_dict=SEAAD_COLORS,
                           mark_paradigm=False, mark_paradigm_type='xo'):
    """
    Plot morphological property stripplot with pointplot overlay.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    types : list
        Types to plot
    merge_df : pandas.DataFrame
        Data with morphological features
    type_column : str
        Column for type grouping
    prop_name : str
        Column name for property to plot
    prop_label : str
        Label for x-axis
    show_yticklabels : bool, optional
        If True, show y-tick labels
    xlim : tuple, optional
        X-axis limits
    color_column : str, optional
        Column for coloring (default: same as type_column)
    color_dict : dict, optional
        Type to color mapping
    mark_paradigm : bool, optional
        If True, use different markers for acute/culture
    mark_paradigm_type : {'xo', 'oo'}, optional
        Marker style for paradigms

    Notes
    -----
    Creates stripplot for individual values with pointplot showing means.
    Y-tick labels are colored by type.
    """
    if color_column is None:
        color_column = type_column

    if mark_paradigm:
        acute_df = merge_df[merge_df.paradigm == 'acute'].loc[merge_df[type_column].isin(types), :]
        culture_df = merge_df[merge_df.paradigm == 'culture'].loc[merge_df[type_column].isin(types), :]
        if mark_paradigm_type == 'xo':
            sns.stripplot(data=acute_df, y=type_column, x=prop_name, hue=color_column, palette=color_dict,
                         s=1.5, alpha=0.5, order=types, ax=ax, marker='o')
            sns.stripplot(data=culture_df, y=type_column, x=prop_name, hue=color_column, palette=color_dict,
                         s=1.5, alpha=0.5, order=types, ax=ax, marker='x', linewidth=0.5)
        elif mark_paradigm_type == 'oo':
            sns.stripplot(data=acute_df, y=type_column, x=prop_name, hue=color_column, palette=color_dict,
                         s=1.5, alpha=0.5, order=types, ax=ax, marker='o')
            sns.stripplot(data=culture_df, y=type_column, x=prop_name, hue=color_column, palette=color_dict,
                         s=1.5, alpha=0.5, order=types, ax=ax, marker='$\circ$')
    else:
        sns.stripplot(data=merge_df.loc[merge_df[type_column].isin(types), :], y=type_column, x=prop_name,
                     hue=color_column, palette=color_dict, s=1.5, alpha=0.5, order=types, ax=ax)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.pointplot(data=merge_df.loc[merge_df[type_column].isin(types), :], y=type_column, x=prop_name,
                     hue=type_column, palette=color_dict, markers="|", scale=0.8, errorbar=None, order=types, ax=ax)

    ax.set_xlabel(prop_label, size=5)
    ax.set_ylabel("")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.tick_params(axis="both", colors='grey', labelsize=5, length=3, left=show_yticklabels, labelleft=show_yticklabels)

    sns.despine(ax=ax)
    if ax.legend_ is not None:
        ax.legend_.remove()

    ax.figure.canvas.draw()

    # Color y-tick labels by feature type
    yticklabels = ax.get_yticklabels()
    for label in yticklabels:
        feature = label.get_text()
        if color_dict and feature in color_dict:
            label.set_color(color_dict[feature])

    # Set axis and ticks to grey
    ax.spines['bottom'].set_color('grey')
    ax.spines['top'].set_color('grey')
    ax.spines['left'].set_color('grey')
    ax.spines['right'].set_color('grey')


def plot_morpho_gallery(ax, gallery_cells, data, soma_depths_df, layer_info, layer_labels, color_dict, ttype_order,
                       cluster_col='t_type', buffer=100, plot_scalebar=True, plot_somadepth=False):
    """
    Plot morphology gallery with optional soma depth scatter.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    gallery_cells : pandas.DataFrame
        Cells to plot (with specimen_id)
    data : pandas.DataFrame
        Full dataset (indexed by specimen_id)
    soma_depths_df : pandas.DataFrame
        Soma depth data
    layer_info : dict
        Layer boundaries
    layer_labels : dict
        Layer label positions
    color_dict : dict
        Type to color mapping
    ttype_order : list
        Order of types
    cluster_col : str, optional
        Column for type grouping (default='t_type')
    buffer : int, optional
        Horizontal spacing (default=100)
    plot_scalebar : bool, optional
        If True, add scale bar
    plot_somadepth : bool, optional
        If True, add soma depth scatter at end

    Returns
    -------
    float
        Fractional x-position where soma depth plot starts (if plotted)
    """
    group_title_start_height = 100
    group_title_font_size = 5
    compartment_list = [1, 3, 4]

    xoffset_dict = {}
    title_count = 0

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)

        sps_list = list(gallery_cells.specimen_id.values)
        sps_list.reverse()  # To maintain CSV order

        ax.axis('off')
        xoffset = 0
        ax.set_anchor('W')
        ax.axhline(0, c="lightgrey", linewidth=0.50)

        layer_colors = {'2': 'lightgrey', '3': 'lightgrey', '4': 'lightgrey', '5': 'lightgrey', '6': 'lightgrey', 'wm': 'darkgrey', '1': 'darkgrey'}
        layer_labels_fixed = {'L1': 123.97, 'L2': 329.01, 'L3': 866.19, 'L4': 1440.05, 'L5': 1873.59, 'L6': 2633.37}

        for l in layer_info:
            ax.axhline(-layer_info[l], c=layer_colors[l], linewidth=0.20)
        for l in layer_labels_fixed:
            ax.text(-500, -layer_labels_fixed[l], "{}".format(l), va='center', ha='left', fontsize=group_title_font_size, color='lightgrey')

        while sps_list:
            sp = sps_list.pop()

            ttype = data.loc[sp, 't_type']
            title_count += 1
            try:
                hex_color = color_dict[ttype][1:]
                contrast_level = 0.45
                rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                dark_rgb = tuple(int(contrast_level * x) for x in rgb)
                grey_hex = '#8f8f8f'
                dark_hex = '#%02x%02x%02x' % dark_rgb
                hex_color = '#' + hex_color
                cell_type_colors_dict = {3: dark_hex, 4: hex_color, 2: grey_hex}
                if compartment_list == [1, 2]:
                    cell_type_colors_dict = {3: grey_hex, 4: grey_hex, 2: hex_color}
            except:
                cell_type_colors_dict = {1: 'black', 2: "firebrick", 4: "orange", 3: "steelblue"}

            morph_colors = {k: v for k, v in cell_type_colors_dict.items() if k in compartment_list}

            swc_path = data.loc[sp, 'SWC_layer_aligned']
            nrn = morphology_from_swc(swc_path)
            morph = swc.read_swc(swc_path)
            xoffset += leftextent(nrn, compartment_list)

            ax.text(xoffset, group_title_start_height,
                   "{}".format(ttype.replace('Car3', '\nCar3')),
                   ha='center', fontsize=group_title_font_size, color=hex_color, rotation=30)

            basic_morph_plot(morph, morph_colors=morph_colors, ax=ax, xoffset=xoffset)
            xoffset_dict[ttype] = xoffset

            xoffset += rightextent(nrn, compartment_list)
            xoffset += buffer

            if plot_scalebar:
                scalebar = ScaleBar(1, "um", location='center left', frameon=False, fixed_value=250, color='grey', font_properties={'size': 6})
                ax.add_artist(scalebar)

        xoffset += buffer * 5

        if plot_somadepth:
            soma_depth_start_xoffset = xoffset
            for ttype in ttype_order:
                ttype_depths = [-1 * d for d in soma_depths_df[soma_depths_df[cluster_col] == ttype]['fig1_soma_depth'].tolist()]
                if len(ttype_depths) > 0:
                    xoffset += buffer * 2
                    ax.scatter(
                        [xoffset] * len(ttype_depths),
                        ttype_depths,
                        facecolors='none',
                        edgecolors=color_dict[ttype],
                        marker='.',
                        s=1
                    )

            soma_depth_end_xoffset = xoffset
            ax.text(
                soma_depth_start_xoffset + (soma_depth_end_xoffset - soma_depth_start_xoffset) / 2,
                group_title_start_height,
                "Soma depth",
                ha='center',
                va='bottom',
                fontsize=group_title_font_size,
                color='grey'
            )

        ax.set_aspect("equal")
        ax.set_xlim(0, xoffset)

        return soma_depth_start_xoffset / xoffset if plot_somadepth else 0


def overlay_morphologies(ax, specimen_dict, xoffsets, data, color_dict, compartment_type='dendrite', alpha=0.5, color_grey=False):
    """
    Overlay selected morphologies at specified x-offsets.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    specimen_dict : dict
        Mapping of group names to specimen IDs
    xoffsets : dict
        Mapping of group names to x-offsets
    data : pandas.DataFrame
        Specimen data (indexed by specimen_id)
    color_dict : dict
        Group to color mapping
    compartment_type : {'dendrite', 'axon'}, optional
        Which compartments to plot
    alpha : float, optional
        Transparency (default=0.5)
    color_grey : bool, optional
        If True, use greyscale colors
    """
    if compartment_type == 'axon':
        compartment_list = [1, 2]
    else:
        compartment_list = [1, 3, 4]

    for group, sp in specimen_dict.items():
        xoffset = xoffsets[group]

        if color_grey:
            grey_hex = '#8f8f8f'
            dark_hex = "#333333"
            cell_type_colors_dict = {1: 'black', 2: grey_hex, 3: grey_hex, 4: dark_hex}
        else:
            try:
                hex_color = color_dict[group][1:]
                contrast_level = 0.45
                rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                dark_rgb = tuple(int(contrast_level * x) for x in rgb)
                grey_hex = '#8f8f8f'
                dark_hex = '#%02x%02x%02x' % dark_rgb
                hex_color = '#' + hex_color
                cell_type_colors_dict = {3: dark_hex, 4: hex_color, 2: grey_hex}

                if compartment_list == [1, 2]:
                    cell_type_colors_dict = {3: grey_hex, 4: grey_hex, 2: hex_color}
            except:
                cell_type_colors_dict = {1: 'black', 2: "firebrick", 4: "orange", 3: "steelblue"}

        morph_colors = {k: v for k, v in cell_type_colors_dict.items() if k in compartment_list}

        swc_path = data.loc[sp, 'SWC_layer_aligned']
        morph = swc.read_swc(swc_path)
        basic_morph_plot(morph, morph_colors=morph_colors, ax=ax, xoffset=xoffset, alpha=alpha, lw=0.05)


def plot_distal_apicals(data_dict, groups, ax, xoffsets, buffer_dict, color_dict, plot_legend=True):
    """
    Plot soma locations and distal apical tips with jitter.

    Parameters
    ----------
    data_dict : dict
        Nested dict: {group: {'somas': [...], 'apicals': [...], 'label': str}}
    groups : list
        Groups to plot
    ax : matplotlib.axes.Axes
        Axis to plot on
    xoffsets : dict
        x-positions for each group
    buffer_dict : dict
        Additional x-offsets per group
    color_dict : dict
        Group to color mapping
    plot_legend : bool, optional
        If True, add legend

    Notes
    -----
    Somas plotted as circles, apical tips as triangles.
    Random horizontal jitter added for visibility.
    """
    for group in groups:
        xoffset = int(xoffsets[group]) + buffer_dict[group]

        # Colors for subclass
        hex_color = color_dict[group][1:]
        contrast_level = 0.6
        lighter_color_rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        darker_color_rgb = tuple([int((contrast_level * x)) for x in lighter_color_rgb])
        dark_hex = '#%02x%02x%02x' % darker_color_rgb
        hex_color = '#' + hex_color

        # Plot soma depths first
        xoffset += buffer_dict[group]
        soma_depths_y = np.array(data_dict[group]['somas']) * -1
        soma_depths_x = np.ones(len(soma_depths_y)) * xoffset
        jitter = np.array([random.randint(-2000, 2000) * 0.01 for n in soma_depths_y])
        soma_depths_x = soma_depths_x + jitter
        sns.scatterplot(x=soma_depths_x, y=soma_depths_y, ax=ax, color=hex_color, edgecolor=hex_color,
                       s=0.5, marker="$\circ$", zorder=10)

        # Plot distal apicals second
        distal_apicals_y = data_dict[group]['apicals']
        distal_apicals_x = np.ones(len(distal_apicals_y)) * xoffset
        jitter = np.array([random.randint(-2000, 2000) * 0.01 for n in distal_apicals_y])
        distal_apicals_x = distal_apicals_x + jitter
        sns.scatterplot(x=distal_apicals_x, y=distal_apicals_y, ax=ax, color=dark_hex, edgecolor=dark_hex,
                       s=1, marker=r"$\bigtriangleup$", zorder=10)

    if plot_legend:
        # Create custom legend handles
        legend_elements = [
            Line2D([0], [0], color='grey', marker=r'$\bigtriangleup$', linestyle='None', label='apical', markersize=1),
            Line2D([0], [0], color='lightgrey', marker=r'$\circ$', linestyle='None', label='soma', markersize=1)
        ]

        # Add legend outside right
        ax.legend(
            handles=legend_elements,
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            frameon=False,
            fontsize=5,
            handletextpad=0.4,
        )
        for text in ax.get_legend().get_texts():
            if text.get_text() == 'apical':
                text.set_color('grey')
            else:
                text.set_color('lightgrey')


def process_distal_apicals(df, groups, group_col):
    """
    Extract distal apical dendrite locations from morphologies.

    Parameters
    ----------
    df : pandas.DataFrame
        Data with SWC paths and soma depths (indexed by specimen_id)
    groups : list
        Groups to process
    group_col : str
        Column name for grouping

    Returns
    -------
    dict
        Nested dict: {group: {'somas': y-coords, 'apicals': y-coords, 'label': str}}

    Notes
    -----
    Finds the most distal (furthest from soma in y) apical dendrite node.
    Only processes cells that have apical dendrites (type 4).
    """
    group_data_dict = {}
    for group in groups:
        # Get group data
        group_data = df[df[group_col] == group]

        soma_depths_y = []
        distal_apicals_y = []
        num_nodes = 0
        for sp in group_data.index.values:
            swc_pth = df.SWC_layer_aligned[df.index == sp].iloc[0]
            nrn = morphology_from_swc(swc_pth)
            if nrn.has_type(4):
                # Neuron has apical, get the most distal one
                apical_nodes = nrn.get_node_by_types([4])
                most_distal_apical = -100000000000000
                for node in apical_nodes:
                    most_distal_apical = max(node['y'], most_distal_apical)
                distal_apicals_y.append(most_distal_apical)
                soma_depths_y.append(df.soma_distance_from_pia[df.index == sp].iloc[0])
                num_nodes += 1
        group_data_dict[group] = {
            'somas': soma_depths_y,
            'apicals': distal_apicals_y,
            'label': "{} (n={})".format(group, num_nodes)
        }

    return group_data_dict


def plot_subclass_hairballs(ax, morpho_sorted_clean_data, subclass_order, layer_info, layer_colors, layer_labels,
                            color_dict, xoffsets, compartment_list, plot_scalebar=True, plot_subclass=True, ylabel=''):
    """
    Plot overlaid morphologies ("hairball" visualization) by subclass.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    morpho_sorted_clean_data : pandas.DataFrame
        Morphology data with SWC paths
    subclass_order : list
        Subclasses to plot (in order)
    layer_info : dict
        Layer boundaries
    layer_colors : dict
        Layer line colors
    layer_labels : dict
        Layer label positions
    color_dict : dict
        Subclass to color mapping
    xoffsets : dict
        x-positions for each subclass
    compartment_list : list
        Compartment types to plot (e.g., [1, 3, 4])
    plot_scalebar : bool, optional
        If True, add scale bar
    plot_subclass : bool, optional
        If True, label subclasses
    ylabel : str, optional
        Y-axis label

    Notes
    -----
    All morphologies for a subclass are overlaid at the same x-position.
    Creates a dense "hairball" showing overall morphology distribution.
    """
    ax.axhline(0, c="lightgrey", linewidth=0.50)
    for l in layer_info:
        ax.axhline(-layer_info[l], c=layer_colors[l], linewidth=0.50)
    for l in layer_labels:
        ax.text(-100, -layer_labels[l], "{}".format(l), verticalalignment='center',
               horizontalalignment='left', fontsize=6, color='lightgrey')

    # Plot
    for i, subclass in enumerate(subclass_order):
        xoffset = xoffsets[subclass]

        group_data = morpho_sorted_clean_data[morpho_sorted_clean_data.subclass_label == subclass]
        this_num_cells = len(group_data)

        try:
            hex_color = color_dict[subclass][1:]
            contrast_level = 0.45
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            dark_rgb = tuple(int(contrast_level * x) for x in rgb)
            grey_hex = '#8f8f8f'
            dark_hex = '#%02x%02x%02x' % dark_rgb
            hex_color = '#' + hex_color
            cell_type_colors_dict = {3: dark_hex, 4: hex_color, 2: grey_hex}

            if compartment_list == [1, 2]:
                cell_type_colors_dict = {3: grey_hex, 4: grey_hex, 2: hex_color}
        except:
            cell_type_colors_dict = {1: 'black', 2: "firebrick", 4: "orange", 3: "steelblue"}

        morph_colors = {k: v for k, v in cell_type_colors_dict.items() if k in compartment_list}

        for swc_path in group_data.SWC_layer_aligned.tolist():
            nrn = read_and_normalize_swc(swc_path)
            basic_morph_plot(nrn, morph_colors=morph_colors, ax=ax, xoffset=xoffset, alpha=1, lw=0.05)

        # Save label
        if plot_subclass:  # Title subclass and number of cells
            ax.text(xoffset, 50,
                   "{}\n(n={}) ".format(subclass, this_num_cells),
                   horizontalalignment='center', fontsize=5, color=color_dict[subclass])
        else:  # Just title number of cells
            ax.text(xoffset, 50,
                   "(n={}) ".format(this_num_cells),
                   horizontalalignment='center', fontsize=5, color=color_dict[subclass])

        if plot_scalebar & (i == 0):
            scalebar = ScaleBar(1, "um", location='lower left', frameon=False, fixed_value=500, color='grey', font_properties={"size": 6})
            ax.add_artist(scalebar)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.text(-0.02, 0.5, ylabel, fontsize=5, color='black', va='center', ha='center', rotation='vertical', transform=ax.transAxes)


def plot_subclass_heatmap(ax, kde_group_dict, x_var_order, layer_info, layer_colors, layer_labels, color_dict, xoffsets,
                          plot_scalebar=True, plot_subclass=True, mask_min_dict=None, colormap='gist_heat_r',
                          ylabel='', xlim=None, ylim=None, color_seaad=False):
    """
    Plot KDE heatmap of morphology density by subclass.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    kde_group_dict : dict
        KDE data: {group: {'xi': x-coords, 'yi': y-coords, 'zi': density, 'num_cells': int}}
    x_var_order : list
        Groups to plot (in order)
    layer_info : dict
        Layer boundaries
    layer_colors : dict
        Layer line colors
    layer_labels : dict
        Layer label positions
    color_dict : dict
        Group to color mapping
    xoffsets : dict
        x-positions for each group
    plot_scalebar : bool, optional
        If True, add scale bar
    plot_subclass : bool, optional
        If True, label groups
    mask_min_dict : dict, optional
        Minimum x values for masking heatmap per group
    colormap : str, optional
        Colormap name (default='gist_heat_r')
    ylabel : str, optional
        Y-axis label
    xlim, ylim : tuple, optional
        Axis limits
    color_seaad : bool, optional
        If True, use group-specific colormaps

    Notes
    -----
    Creates pcolormesh heatmaps at specified x-offsets.
    Can mask regions below a minimum x value per group.
    """
    ax.axhline(0, c="lightgrey", linewidth=0.50)
    for l in layer_info:
        ax.axhline(-layer_info[l], c=layer_colors[l], linewidth=0.50)
    for l in layer_labels:
        ax.text(-100, -layer_labels[l], "{}".format(l), verticalalignment='center',
               horizontalalignment='left', fontsize=6, color='lightgrey')

    # Plot
    for i, subclass in enumerate([x for x in x_var_order if x in kde_group_dict.keys()]):
        xoffset = xoffsets[subclass]

        this_xi = kde_group_dict[subclass]['xi']
        this_yi = kde_group_dict[subclass]['yi']
        this_zi = kde_group_dict[subclass]['zi']
        this_num_cells = kde_group_dict[subclass]['num_cells']

        this_xi = this_xi + xoffset
        this_zi = this_zi.reshape(this_xi.shape)

        # Custom remove heatmap below x value
        if mask_min_dict is not None:
            if subclass in mask_min_dict.keys():
                x_min = mask_min_dict[subclass]
                x_mask = this_xi[:, 0] >= x_min
                this_xi = this_xi[x_mask, :]
                this_yi = this_yi[x_mask, :]
                this_zi = this_zi[x_mask, :]

        if color_seaad:
            colormap = mcolors.LinearSegmentedColormap.from_list("custom_seaad", list(zip([0, 1], ["#FFFFFF", color_dict[subclass]])))

        ax.pcolormesh(this_xi, this_yi, this_zi, shading='auto', cmap=colormap)

        # Save label
        if plot_subclass:  # Title subclass and number of cells
            ax.text(xoffset, 50,
                   "{}\n(n={}) ".format(subclass, this_num_cells),
                   horizontalalignment='center', fontsize=5, color=color_dict[subclass])
        else:  # Just title number of cells
            ax.text(xoffset, 50,
                   "(n={}) ".format(this_num_cells),
                   horizontalalignment='center', fontsize=5, color=color_dict[subclass])

        if plot_scalebar & (i == 0):
            scalebar = ScaleBar(1, "um", location='lower left', frameon=False, fixed_value=500, color='grey', font_properties={"size": 6})
            ax.add_artist(scalebar)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.text(-0.02, 0.5, ylabel, fontsize=5, color='black', va='center', ha='center', rotation='vertical', transform=ax.transAxes)


# Core morphology plotting functions

def basic_morph_plot(morph, ax, morph_colors={3: "firebrick", 4: "salmon", 2: "steelblue"},
                    side=False, xoffset=0, alpha=1.0, lw=0.25):
    """
    Core function to plot neuron morphology.

    Parameters
    ----------
    morph : Morphology object
        SWC morphology from allensdk.core.swc
    ax : matplotlib.axes.Axes
        Axis to plot on
    morph_colors : dict, optional
        Compartment type to color mapping:
        - 1: soma
        - 2: axon
        - 3: basal dendrite
        - 4: apical dendrite
    side : bool, optional
        If True, plot side view (z vs y) instead of top view (x vs y)
    xoffset : float, optional
        Horizontal offset for placement
    alpha : float, optional
        Transparency (default=1.0)
    lw : float, optional
        Line width (default=0.25)

    Returns
    -------
    matplotlib.axes.Axes
        The axis with morphology plotted

    Notes
    -----
    Plots each compartment as connected line segments from parent to child nodes.
    Z-order is set by compartment type for proper layering.
    """
    for compartment, color in morph_colors.items():
        lines_x = []
        lines_y = []
        for c in morph.compartment_list_by_type(compartment):
            if c["parent"] == -1:
                continue
            p = morph.compartment_index[c["parent"]]
            if side:
                lines_x += [p["z"] + xoffset, c["z"] + xoffset, None]
            else:
                lines_x += [p["x"] + xoffset, c["x"] + xoffset, None]
            lines_y += [p["y"], c["y"], None]
        ax.plot(lines_x, lines_y, c=color, linewidth=lw, zorder=compartment, alpha=alpha)
    return ax


def read_and_normalize_swc(filepath):
    """
    Read SWC file and normalize coordinates so soma is at origin.

    Parameters
    ----------
    filepath : str
        Path to SWC file

    Returns
    -------
    Morphology object
        SWC morphology with soma at (0, 0, 0)

    Raises
    ------
    ValueError
        If no soma/root is found in SWC

    Notes
    -----
    Only translates x-coordinate; y and z are kept as-is.
    This maintains depth alignment while centering laterally.
    """
    morph = swc.read_swc(filepath)

    soma = morph.soma
    if soma is None:
        raise ValueError("No soma/root found in SWC; cannot normalize.")

    # Soma coordinates
    sx = soma['x']

    # Shift all nodes
    for seg in morph.compartment_list:
        seg['x'] -= sx

    return morph


def compartment_list_by_type(self, compartment_type):
    """
    Return list of all compartments with specified type.

    Parameters
    ----------
    self : Morphology object
        Morphology instance
    compartment_type : int
        Compartment type:
        - 1: soma
        - 2: axon
        - 3: basal dendrite
        - 4: apical dendrite

    Returns
    -------
    list
        List of compartment dictionaries matching the type

    Notes
    -----
    This is a helper method that would typically be added to a Morphology class.
    """
    return [x for x in self._compartment_list if x['type'] == compartment_type]
