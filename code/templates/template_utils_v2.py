import os
import warnings
import numpy as np
import seaborn as sns
import re
import allensdk.core.swc as swc
import matplotlib.colors as mc
import colorsys
from matplotlib_scalebar.scalebar import ScaleBar

import sys 
sys.path.append(r'..\utils') 
from utils import get_seaad_colors


SEAAD_COLORS = get_seaad_colors()

def hex_to_gray(hexcol, method='601'):
    """
    Convert hex color to grayscale hex.
    hexcol: string like '#aabbcc', 'aabbcc', or shorthand '#abc'
    method: '601' (default) or '709' or 'avg' (simple mean)
    Returns: grayscale hex string '#RRGGBB' and integer gray value 0-255
    """
    s = hexcol.lstrip('#')
    if len(s) == 3:
        s = ''.join([c*2 for c in s])
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)

    if method == '601':
        lum = 0.299*r + 0.587*g + 0.114*b
    elif method == '709':
        lum = 0.2126*r + 0.7152*g + 0.0722*b
    else:
        lum = (r + g + b) / 3.0

    v = int(round(lum))
    gray_hex = "#{:02x}{:02x}{:02x}".format(v, v, v)
    return gray_hex


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]

def plot_prop_for_mets_swb(ax, types, merge_df, type_column, prop_name, prop_label,
    show_yticklabels=True, xlim=None, color_column=None, color_dict=SEAAD_COLORS, 
    mark_paradigm=False, mark_paradigm_type='xo'):
    if color_column is None:
        color_column = type_column

    if mark_paradigm:
        acute_df = merge_df[merge_df.paradigm == 'acute'].loc[merge_df[type_column].isin(types), :]
        culture_df = merge_df[merge_df.paradigm == 'culture'].loc[merge_df[type_column].isin(types), :]
        if mark_paradigm_type == 'xo':
            sns.stripplot(data=acute_df, y=type_column, x=prop_name, hue=color_column, palette=color_dict, s=1.5, alpha=0.5, order=types, ax=ax, marker='o')
            sns.stripplot(data=culture_df, y=type_column, x=prop_name, hue=color_column, palette=color_dict, s=1.5, alpha=0.5, order=types, ax=ax, marker='x', linewidth=0.5)

        elif mark_paradigm_type == 'oo':
            sns.stripplot(data=acute_df, y=type_column, x=prop_name, hue=color_column, palette=color_dict, s=1.5, alpha=0.5, order=types, ax=ax, marker='o')
            sns.stripplot(data=culture_df, y=type_column, x=prop_name, hue=color_column, palette=color_dict, s=1.5, alpha=0.5, order=types, ax=ax, marker='$\circ$')
            
    else:
        sns.stripplot(data=merge_df.loc[merge_df[type_column].isin(types), :], y=type_column, x=prop_name, 
                      hue=color_column, palette=color_dict, s=1.5, alpha=0.5, order=types, ax=ax)
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        sns.pointplot( data=merge_df.loc[merge_df[type_column].isin(types), :], y=type_column, x=prop_name, 
                      hue=type_column, palette=color_dict, markers="|", scale=0.8, errorbar=None, order=types, ax=ax)

    ax.set_xlabel(prop_label, size=5) #size=6
    ax.set_ylabel("")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.tick_params(axis="both", colors='grey', labelsize=5, length=3, left=show_yticklabels, labelleft=show_yticklabels)
   
    sns.despine(ax=ax)
    if ax.legend_ is not None:
        ax.legend_.remove()

    ax.figure.canvas.draw()

    # color y-tick labels by feature type
    yticklabels = ax.get_yticklabels()
    for label in yticklabels:
        feature = label.get_text()
        if color_dict and feature in color_dict:
            label.set_color(color_dict[feature])

    # set axis and ticks to grey
    # ax.tick_params(axis='both', colors='grey')  # Set ticks color to grey
    ax.spines['bottom'].set_color('grey')  # Set bottom spine color to grey
    ax.spines['top'].set_color('grey')  # Set top spine color to grey
    ax.spines['left'].set_color('grey')  # Set left spine color to grey
    ax.spines['right'].set_color('grey')  # Set right spine color to grey



def plot_dendrite_depth_profiles_swb(ax, merge_df, type_column, types, hist_df, basal_cols, apical_cols,
        layer_edges, color_dict, plot_quantiles=None, path_color="#cccccc", spacing=200, bin_width=5, plot_scalebar=True):
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
        scalebar = ScaleBar(1, "um", location='lower left', frameon=False, fixed_value=100, color = 'grey', font_properties={'size': 6})
        ax.add_artist(scalebar)

    return xoffsets
    
def plot_axon_depth_profiles_swb(ax, merge_df, type_column, types, hist_df, axon_cols,
        layer_edges, plot_quantiles=None, path_color="#cccccc", spacing=200, bin_width=5, plot_scalebar=True):
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
        scalebar = ScaleBar(1, "um", location='lower left', frameon=False, fixed_value=100, color = 'grey', font_properties={'size': 6})
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


def plot_morph_lineup_swb(ax, merge_df, type_column, types, aligned_swc_dir, layer_edges, layer_labels,
        plot_quantiles=None, path_color="#cccccc", morph_spacing=300, plot_scalebar=True, plot_layer_labels=True):
    if plot_quantiles is None:
        plot_quantiles = np.linspace(0, 1, 5)


    xoffset = 0
    for type in types:
        # print(type)
        # spec_ids = merge_df.loc[merge_df[type_column] == type, :].sort_values("soma_aligned_dist_from_pia").index
        spec_ids = merge_df.loc[merge_df[type_column] == type, :].sort_values("soma aligned distance from pia").index

        # inds = np.arange(len(spec_ids))
        # if len(inds) <= len(plot_quantiles):
        #     plot_inds = inds
        # else:
        #     plot_inds = np.quantile(inds, plot_quantiles).astype(int)
        #     if plot_inds[0] + 1 < plot_inds[1]:
        #         plot_inds[0] += 1
        #     if plot_inds[-1] - 1 > plot_inds[-2]:
        #         plot_inds[-1] -= 1

        # print("Plotted morphs for type", type)
        # print(spec_ids.values[plot_inds])
        color = SEAAD_COLORS[type]
        morph_locs = []
        for spec_id in spec_ids.specimen_id.values(): #[plot_inds]:
            swc_path = os.path.join(aligned_swc_dir, f"{spec_id}.swc")
            morph = swc.read_swc(swc_path)
            basic_morph_plot(morph, ax=ax, xoffset=xoffset,
                morph_colors={3: adjust_lightness(color, 0.5), 4: color})
            morph_locs.append(xoffset)

            # Add a grey mark if not TemL
            if merge_df[merge_df.specimen_id == spec_id]['lobe'].values[0] != 'TemL':
                ax.plot([xoffset], [layer_edges['wm']+100], marker='*', color='grey', markersize=4, zorder=5)

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
    # ax.set_ylabel("µm", rotation=0, fontsize=7)
    # ax.tick_params(axis='y', labelsize=6)
    for e in layer_edges:
        ax.axhline(-e, zorder=-5, color=path_color, linewidth=0.75)
    ax.axis('off')

    if plot_scalebar: 
        scalebar = ScaleBar(1, "um", location='lower left', frameon=False, fixed_value=500, color = 'grey', font_properties={'size': 6})
        ax.add_artist(scalebar)

    if plot_layer_labels:
        for ll in layer_labels:
            layer_label_buffer=0
            if ll == 'L1': layer_label_buffer = 50
            if ll == 'L2': layer_label_buffer = -50
            ax.text(-700,-layer_labels[ll]+layer_label_buffer, "{}".format(ll),verticalalignment='center', horizontalalignment='left',fontsize=6, color='lightgrey') 



def adjust_lightness(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def basic_morph_plot(morph, ax, morph_colors={3: "firebrick", 4: "salmon", 2: "steelblue"},
                     side=False, xoffset=0, alpha=1.0, lw=0.25):
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