import os
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import patches
from matplotlib.path import Path
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.transforms import Affine2D
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

from template_utils_v2 import basic_morph_plot, adjust_lightness

import allensdk.core.swc as swc
from neuron_morphology.swc_io import morphology_from_swc
from morph_utils.measurements import leftextent, rightextent

from svgpathtools import svg2paths
from scipy.ndimage import rotate

import io
import cairosvg  
import warnings
import random

import sys 
sys.path.append(r'..\utils') 
from utils import get_seaad_colors

SEAAD_COLORS = get_seaad_colors()

def plot_svg(ax, svg_path):
    with open(svg_path, 'rb') as f:
        svg_data = f.read()

    # Convert SVG to PNG in memory
    png_bytes = cairosvg.svg2png(bytestring=svg_data)
    img = mpimg.imread(io.BytesIO(png_bytes), format='png')
    ax.imshow(img)
    ax.axis('off')  # Hide axes

def plot_donor_lobe_counts(ax, df, color_dict):

    # Calculate the order based on counts
    order = df['lobe'].value_counts().index

    # Create the countplot with the specified order
    sns.countplot(x='lobe', data=df, order=order, ax=ax, palette=color_dict)

    ax.set_title('Donor cortical region', fontsize=6)
    ax.set_ylabel('Donor count',  fontsize=5)
    ax.set_xlabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')  #Rotate x labels

    # # Add data labels
    # for p in ax.patches:
    #     height = int(p.get_height())  # convert to integer
    #     ax.annotate(f'{height}', 
    #                 (p.get_x() + p.get_width() / 2., p.get_height()),
    #                 ha='center', va='center', xytext=(0, 10), textcoords='offset points',
    #                 fontsize=5, color='grey')

        
    #format axes 
    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(axis="both", labelsize=5)
    ax.tick_params(axis='both', colors='grey')  # Set ticks color to grey
    ax.spines['bottom'].set_color('grey')  # Set bottom spine color to grey
    ax.spines['left'].set_color('grey')  # Set left spine color to grey
        
    # Color lobe labels 
    for tick_label in ax.get_xticklabels():
        label_text = tick_label.get_text()
        if label_text in color_dict:
            tick_label.set_color(color_dict[label_text])

def plot_supertype_counts(ax, df, group_col, group_order, paradigm_colors):

    # Create a count table
    count_df = df.groupby([group_col, 'paradigm']).size().unstack(fill_value=0)
    count_df = count_df.reindex(group_order)

    # Plot
    colors = [paradigm_colors[p] for p in count_df.columns]
    count_df.plot(kind='bar', stacked=True, color=colors, ax=ax)

    ax.set_title('Neuron count by supertype and culturing paradigm', fontsize=6)
    ax.set_ylabel('Count',  fontsize=5)
    ax.set_xlabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')  #Rotate x labels

    #legend
    leg = plt.legend(title='Paradigm', frameon=False, title_fontsize=5, 
                     fontsize=5, loc='upper right', bbox_to_anchor=(0.85, 0.7))
    for text in leg.get_texts():
        text.set_color('grey')

    # # Annotate above each bar
    # totals = count_df.sum(axis=1)
    # for i, total in enumerate(totals):
    #     ax.text(i, total + 2, str(total), ha='center', va='bottom', fontsize=5, color='grey')

    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(axis="both", labelsize=5)
    ax.tick_params(axis='both', colors='grey')  # Set ticks color to grey
    ax.spines['bottom'].set_color('grey')  # Set bottom spine color to grey
    ax.spines['left'].set_color('grey')  # Set left spine color to grey

    # Color supertype labels 
    for tick_label in ax.get_xticklabels():
        label_text = tick_label.get_text()
        if label_text in SEAAD_COLORS:
            tick_label.set_color(SEAAD_COLORS[label_text])

def plot_spatial_slice(data, ax, groups, color_dict, rot_deg=0, ncols=-1, label_layers=True, skip_subplot=[], overlay_layers=False, overlay_path=None):

    if overlay_layers:
        overlay = mpimg.imread(overlay_path)
        overlay_rotated = np.flipud(rotate(overlay, angle=221))

    layer_label_formatting_dict = {
        'L1': [250, 200], #300
        'L2': [50, 700],  #800
        'L3': [-200, 1400], #1400
        'L4': [-500, 1950], #2100
        'L5': [-750, 2600], #3000
        'L6': [-1300, 4000] #4000
    }       

    # slice rotation (degrees ccw)
    theta = np.deg2rad(rot_deg)
    rotation_matrix = np.array([
        [np.cos(theta),  np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])

    xmin = min(data.obsm['spatial_cirro']@ rotation_matrix.T[:,0])
    xmax = max(data.obsm['spatial_cirro']@ rotation_matrix.T[:,0])
    xbuffer = abs(xmax-xmin) -1000
    xoffset = xmin*(-1)

    ymin = min(data.obsm['spatial_cirro']@ rotation_matrix.T[:,1])
    ymax = max(data.obsm['spatial_cirro']@ rotation_matrix.T[:,1])
    ybuffer = abs(ymax-ymin) +1000
    yoffset = ymin*(-1)

    groups = [g for g in groups if g in data.obs['SEAAD_Supertype_name'].unique()]
    plot_idx = -1
    for i, group in enumerate(groups):
        plot_idx +=1

        if i in skip_subplot:
            plot_idx += 1
            xoffset += xbuffer
        
        if (plot_idx > 0) & (plot_idx % ncols == 0):
            #go to next y level
            yoffset -= ybuffer
            xoffset = xmin*(-1)

        if overlay_layers:
            # overlay layer lines
            overlay_ymin = ymin+yoffset-1800
            overlay_ymax = ymax+yoffset+1300
            overlay_xmin = xmin+xoffset-300
            overlay_xmax = xmax+xoffset+1300
            ax.imshow(
                overlay_rotated,
                extent=[overlay_xmin, overlay_xmax, overlay_ymin, overlay_ymax],
                aspect='auto',
                alpha=1,
                zorder=10,
                interpolation='none'
            )

        #get group data
        group_data = data[data.obs['SEAAD_Supertype_name']==group]
        if len(group_data) > 0:
            non_group_data = data[data.obs['SEAAD_Supertype_name']!=group]

            #plot rotated data
            ax.scatter(non_group_data.obsm['spatial_cirro']@ rotation_matrix.T[:,0]+xoffset, 
                       non_group_data.obsm['spatial_cirro']@ rotation_matrix.T[:,1]+yoffset, 
                       c="#b9b9b9", alpha=1, s=.05, marker='.', rasterized=True, linewidths=0)
            ax.scatter(group_data.obsm['spatial_cirro']@ rotation_matrix.T[:,0]+xoffset, 
                       group_data.obsm['spatial_cirro']@ rotation_matrix.T[:,1]+yoffset, 
                       c=color_dict[group], alpha=1, s=1, marker='o', rasterized=True, linewidths=0)   
            
            # Add centered label in color
            non_group_coords = non_group_data.obsm['spatial_cirro'] @ rotation_matrix.T
            x_center = non_group_coords[:, 0].max() + xoffset
            y_center = non_group_coords[:, 1].max() + yoffset + 200
            ax.text(x_center, y_center, group, fontsize=5, ha='right', va='center', color=color_dict[group])

            is_last_in_row = ((plot_idx + 1) % ncols == 0) or (i == len(groups) - 1) or (plot_idx+1 in skip_subplot)
            if label_layers and is_last_in_row:
                right_edge = (data.obsm['spatial_cirro'] @ rotation_matrix.T[:, 0]).max() + xoffset
                top_edge = (data.obsm['spatial_cirro'] @ rotation_matrix.T[:, 1]).max() + yoffset
                for layer_label, [layer_x, layer_y] in layer_label_formatting_dict.items():
                    ax.text(right_edge+layer_x, top_edge-layer_y, layer_label,
                        color='lightgrey', fontsize=5, ha='left', va='center')

            xoffset += xbuffer

        ax.axis('off') 
        ax.set_aspect('equal')
  
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

def plot_morph_lineup_swb(ax, merge_df, type_column, types, aligned_swc_dir, layer_edges, layer_labels, color_dict,
        plot_quantiles=None, path_color="#cccccc", morph_spacing=300, plot_scalebar=True, plot_layer_labels=True, compartment_type='dendrite'):
    if plot_quantiles is None:
        plot_quantiles = np.linspace(0, 1, 5)


    xoffset = 0
    for type in types:
        spec_ids = merge_df[merge_df[type_column] == type].sort_values("soma aligned distance from pia").specimen_id

        color = color_dict[type]

        morph_locs = []
        for spec_id in spec_ids: #.specimen_id.values(): #[plot_inds]:
            swc_path = os.path.join(aligned_swc_dir, f"{spec_id}.swc")
            morph = swc.read_swc(swc_path)
            if compartment_type == 'dendrite':
                morph_colors = {3: adjust_lightness(color, 0.5), 4: color}
            else: #axon
                morph_colors = {2: hex_to_gray(color)}

            basic_morph_plot(morph, ax=ax, xoffset=xoffset,
                morph_colors=morph_colors)
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
        scalebar = ScaleBar(1, "um", location='lower left', frameon=False, fixed_value=500, color = 'grey', font_properties={'size': 6})
        ax.add_artist(scalebar)

    if plot_layer_labels:
        for ll in layer_labels:
            layer_label_buffer=0
            if ll == 'L1': layer_label_buffer = 50
            if ll == 'L2': layer_label_buffer = -50
            ax.text(-700,-layer_labels[ll]+layer_label_buffer, "{}".format(ll),verticalalignment='center', horizontalalignment='left',fontsize=6, color='lightgrey') 

def plot_isodepth(layer_isodepth_df, ax, color_dict, ttype_order):

    ttype_order = [t for t in ttype_order if t in layer_isodepth_df.SEAAD_Supertype_name.unique()]

    sns.stripplot(
        data=layer_isodepth_df,
        x='SEAAD_Supertype_name',               # Categories on x-axis
        y='scaled_gaston_isodepth',                 # Values on y-axis
        hue='SEAAD_Supertype_name',
        palette=color_dict,
        s=1.5,
        alpha=0.5,
        order=ttype_order,           # order applies to x-axis now
        ax=ax,
        zorder=-5
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.pointplot(
            data=layer_isodepth_df,
            x='SEAAD_Supertype_name',
            y='scaled_gaston_isodepth',
            color='black',   
            # palette=color_dict,
            markers="_",
            scale=0.5,
            errorbar=None,
            order=ttype_order,
            linestyles="None",   # <-- disables connecting lines
            ax=ax
        )

    ax.legend_.remove()
    ax.set_xlabel('')
    ax.set_ylabel('Isodepth', fontsize=5)  # Or whatever size you want
    sns.despine(ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')
    ax.tick_params(axis='both', colors='grey', labelsize=4)  # Set ticks color to grey
    ax.spines['bottom'].set_color('grey')  # Set bottom spine color to grey
    ax.spines['left'].set_color('grey')  # Set left spine color to grey
    ax.set_ylim(0, 4500) #standard isodepth range 
    ax.invert_yaxis()

    for label in ax.get_xticklabels():
        text = label.get_text()
        if text in color_dict:
            label.set_color(color_dict[text])
        # label.set_fontsize(4)  # Optional: shrink if crowded

def plot_spatial_density_kde(df, ax, y_label, hue_label, hue_order, color_dict, label_peaks=True, label_y=True, label_x=True, fill=True, alpha=0.1, common_norm=True):

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

        # Map each drawn line back to its hue (line order from sns.kdeplot does not match hue_order)
        peaks = {}
        used_hues = set()

        # Precompute RGBA for your palette entries
        palette_rgba = {h: mcolors.to_rgba(color_dict[h]) for h in hue_order if h in color_dict}

        for line in ax.lines:
            x, y = line.get_data()
            if len(x) == 0:
                continue
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
    )

    ax.set_ylim(0, 3500) #standard isodepth range 

    ax.set_xticks([])      # remove x-axis ticks
    if label_x:
        ax.set_xlabel("Density", fontsize=6)
    else:
        ax.set_xlabel("")

    if label_y:
        ax.set_yticks([0, 2800])      # remove x-axis ticks
        ax.set_yticklabels(['pia', '~wm'])      # remove x-axis ticks
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
        for _, row in peak_df.iterrows():
            hue = row[hue_label]
            peak_x = row['peak_x']
            peak_y = row['peak_y']

            # offset a bit to the right and up for visibility
            text_x = peak_x + 0.00001   # move 5% right of the peak
            text_y = peak_y          # same depth value

            ax.text(
                text_x,
                text_y,
                hue,  # the label text
                color=color_dict.get(hue, 'k'),
                fontsize=5,
                fontweight='bold',
                rotation=15,
                rotation_mode='anchor',
                va='center',
                ha='left',
                zorder=10,
            )

    # plt.tight_layout()

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

def plot_abundance(df, ax, color_dict, frac_col = 'supertype_frac_of_layer', label_supertypes=True, radius=1, center=(0,0), title='', label_threshold=0.01, title_bottom=False):

    # if label: labels = df['supertype'][color_dict[st] for st in layer_df['supertype']]
    # else: labels = None

    wedges, texts = ax.pie(df[frac_col], 
                        labels=None, 
                        colors=[color_dict[st] for st in df['supertype']], 
                        startangle=180, #90, 
                        counterclock=False, 
                        rotatelabels=True,
                        textprops={'fontsize': 4},
                        radius = radius,
                        center = center)
    
    if label_supertypes:
        labels = df['supertype']
        for w, label in zip(wedges, labels):
            angle_span = w.theta2 - w.theta1
            if angle_span / 360 > label_threshold:  # only show label if wedge > label_threshold
                theta = (w.theta2 + w.theta1) / 2
                x = center[0] + 1.05 * radius * np.cos(np.deg2rad(theta))
                y = center[1] + 1.05 * radius * np.sin(np.deg2rad(theta))
                
                rotation = theta
                if 90 < theta < 270:
                    rotation = (theta + 180) % 360
                    ha = 'right'
                else:
                    ha = 'left'

                if rotation > 180:
                    rotation -= 360

                ax.text(x, y, label, ha=ha, va='center', fontsize=4, rotation=rotation, 
                        rotation_mode='anchor', color=color_dict[label])


    ax.set_aspect('equal')
    if title_bottom:
        ax.text(0.5, -0.03, title, transform=ax.transAxes, ha='center', va='top', fontsize=6)
    else: 
        ax.set_title(title, size=6, pad=2)

def abundance_barplot(df, type_order, ax, color_dict, x_col= 'supertype_frac_of_class', y_col='supertype', title='Fraction of glutamatergic types'):

    sns.barplot(
        data=df, 
        y=y_col, 
        x=x_col,
        order=type_order,
        palette=color_dict,
        ax=ax
    )

    # Label styling
    ax.set_ylabel('')
    ax.set_xlabel(title, color='Black', fontsize=5)

    # Make all axis lines and tick marks grey
    ax.tick_params(axis='x', colors='grey', labelsize=5)
    ax.tick_params(axis='y', colors='grey', labelsize=5)
    for spine in ax.spines.values():
        spine.set_color('grey')


    # Color y tick labels by supertype
    for tick_label in ax.get_yticklabels():
        text = tick_label.get_text()
        if text in color_dict:
            tick_label.set_color(color_dict[text])

    sns.despine()
    plt.tight_layout()

def plot_ref_umap_crop(reference_umap, ax, x_min, x_max, y_min, y_max, color_dict, supertypes, label_dict=None, line_list=None, legend_loc=None, legend_length=1, plot_axis_arrows=False):


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

    #crop
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_aspect('equal')
    plt.axis('off')

    if not label_dict is None:
        for label, label_formatting_dict in label_dict.items():
            if label in supertypes:  
                color = color_dict[label]
            else: 
                color= 'lightgrey'
            plt.text(label_formatting_dict['x'], label_formatting_dict['y'], label, 
                     color=color, ha=label_formatting_dict['ha'], va=label_formatting_dict['va'], fontsize=5)

    if not line_list is None: 
        for line in line_list:
            plt.plot(line['x_values'], line['y_values'], lw=0.5, color='lightgrey') 
    
    if plot_axis_arrows:
        #add inset axes labels 
        if legend_loc is None:
            xmin, _ = ax.get_xlim()
            ymin, _ = ax.get_ylim()
        else:
            xmin = legend_loc[0]
            ymin = legend_loc[1]

        # Add small x and y axis arrows
        arrowprops = dict(arrowstyle='-|>', color='grey', lw=1, mutation_scale=2)
        ax.annotate('', xy=(xmin + legend_length, ymin), xytext=(xmin, ymin), arrowprops=arrowprops)  # x arrow 1 unit right
        ax.annotate('', xy=(xmin, ymin + legend_length), xytext=(xmin, ymin), arrowprops=arrowprops)  # y arrow 1 unit up

        # Optional: add axis labels
        ax.text(xmin, ymin-0.25, 't-UMAP1', va='center', ha='left', fontsize=5, color='grey')
        ax.text(xmin-0.25, ymin, 't-UMAP2', va='bottom', ha='center', fontsize=5, color='grey', rotation=90)

def plot_ref_umap(reference_umap, ax, label_dict=None, line_list=None, plot_axis_arrows=False):

    #TODO combine this and the crop function - they're so similar 

    # Plot all non-patch-seq points
    for hann_type, group in reference_umap[reference_umap['platform'] != 'patch-seq'].groupby('subclass'):
        plt.scatter(group['UMAP_1'], group['UMAP_2'], 
                    s=1, alpha=0.7, label=hann_type, 
                    c='lightgrey', edgecolor='none') #c=group['color']
        
    # Plot all patch-seq points
    for hann_type, group in reference_umap[reference_umap['platform'] == 'patch-seq'].groupby('subclass'):
        plt.scatter(group['UMAP_1'], group['UMAP_2'], 
            s=3, alpha=0.9, label='patch-seq', 
            c=group['color'], marker='o', edgecolor='white', linewidths=0.05)

    ax.set_aspect('equal')
    plt.axis('off')

    if not label_dict is None:
        for label, label_formatting_dict in label_dict.items():
            if label in SEAAD_COLORS.keys():
                color = SEAAD_COLORS[label]
            else: 
                color = 'grey'
            plt.text(label_formatting_dict['x'], label_formatting_dict['y'], label, 
                     color=color, ha=label_formatting_dict['ha'], va=label_formatting_dict['va'], fontsize=5)

    if not line_list is None: 
        for line in line_list:
            plt.plot(line['x_values'], line['y_values'], lw=0.5, color='lightgrey') 

    if plot_axis_arrows:

        #add inset axes labels 
        xmin, _ = ax.get_xlim()
        ymin, _ = ax.get_ylim()

        # Add small x and y axis arrows
        arrowprops = dict(arrowstyle='-|>', color='grey', lw=1, mutation_scale=5)
        length = 5
        ax.annotate('', xy=(xmin + length, ymin), xytext=(xmin, ymin), arrowprops=arrowprops)  # x arrow 1 unit right
        ax.annotate('', xy=(xmin, ymin + length), xytext=(xmin, ymin), arrowprops=arrowprops)  # y arrow 1 unit up

        # Optional: add axis labels
        ax.text(xmin, ymin-0.5, 't-UMAP1', va='center', ha='left', fontsize=5, color='grey')
        ax.text(xmin-0.5, ymin, 't-UMAP2', va='bottom', ha='center', fontsize=5, color='grey', rotation=90)

def process_fi_curves(ephys_feat_df):
    amp_cols = ephys_feat_df.columns[ephys_feat_df.columns.str.startswith("stimulus_amplitude_")]
    rate_cols = ephys_feat_df.columns[ephys_feat_df.columns.str.startswith("nwg_avg_rate_")]
    sub_df = ephys_feat_df[rate_cols.union(amp_cols)].reset_index()
    sub_df.columns = sub_df.columns.str.replace("_long_square", "")
    sub_df_long = pd.wide_to_long(
        sub_df,
        stubnames=["nwg_avg_rate", "stimulus_amplitude"],
        i="specimen_id", j="step",
        sep="_").dropna().reset_index().set_index("specimen_id"
    )

    bins = np.arange(-20, 460, 20)
    sub_df_long["amp_bin"] = pd.cut(sub_df_long.stimulus_amplitude, bins=bins)

    new_zero_rows = []
    for n, g in sub_df_long.groupby("specimen_id"):
        # print(g)
        min_bin = g["amp_bin"].min()
        if not pd.isna(min_bin):
            lower_bins = sub_df_long["amp_bin"].cat.categories[sub_df_long["amp_bin"].cat.categories < min_bin]
            for lb in lower_bins:
                new_zero_rows.append({
                    "specimen_id": n,
                    "step": -1,
                    "stimulus_amplitude": np.nan,
                    "amp_bin": lb,
                    "avg_rate": 0,
                })
    new_df = pd.DataFrame(new_zero_rows).set_index("specimen_id")

    #set type to Interval for concat column type matching -swb
    sub_df_long['amp_bin'] = sub_df_long['amp_bin'].cat.categories.take(sub_df_long['amp_bin'].cat.codes).values
    sub_df_long['amp_bin'] = sub_df_long['amp_bin'].astype('interval[int64, right]')
    sub_df_long = sub_df_long.rename(columns={'nwg_avg_rate' : 'avg_rate'})

    sub_df_long_filled = pd.concat([sub_df_long, new_df])
    return sub_df_long_filled, bins
    
def plot_avg_fi_for_mets(ax, ephys_fi_data_formatted, group_col, groups,
        min_n=5, xlim=(0, 250), ylim=(0, 25), show_ylabel=True, show_xlabel=True, color_dict = SEAAD_COLORS):

    grouped = ephys_fi_data_formatted.groupby(group_col)
    for n in groups:
        try: 
            g = grouped.get_group(n)
            avg_by_cell = g.groupby(["amp_bin", "specimen_id"])["avg_rate"].mean().reset_index()
            avg_rates = avg_by_cell.groupby(["amp_bin"])["avg_rate"].mean()
            count_mask = avg_by_cell.groupby("amp_bin")["avg_rate"].count() >= min_n
            err_rates = avg_by_cell.groupby("amp_bin")["avg_rate"].std() / np.sqrt(avg_by_cell.groupby("amp_bin")["avg_rate"].count())
            within_xlim_mask = avg_rates.index.right.values < xlim[1]

            ax.errorbar(
                x=avg_rates.index.right[count_mask & within_xlim_mask],
                y=avg_rates[count_mask & within_xlim_mask],
                yerr=err_rates[count_mask & within_xlim_mask],
                capsize=1,
                capthick=0.5,
                color=color_dict[n],
                lw=0.5,
            )
            ax.scatter(x=avg_rates.index.right[count_mask], s=1, c=color_dict[n], y=avg_rates[count_mask])
        except:
            continue
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    sns.despine(ax=ax)
    ax.tick_params(axis="both", labelsize=6, length=3)
    if show_ylabel:
        ax.set_ylabel("Firing rate (spikes/s)", size=5)
    if show_xlabel:
        # ax.set_xlabel("Stimulus amplitude (pA)", size=5)
        ax.set_xlabel("Stimulus current - \nrheobase (pA)", size=5)

    # set axis and ticks to grey
    ax.tick_params(axis='both', colors='grey')  # Set ticks color to grey
    ax.spines['bottom'].set_color('grey')  # Set bottom spine color to grey
    ax.spines['left'].set_color('grey')  # Set left spine color to grey
       
def align_to_peak(voltage, dv):
    """Align traces so their peaks occur at the same relative time"""
    peak_idx = np.argmax(voltage)
    total_len = len(voltage)
    
    # Define pre-peak and post-peak portions
    pre_peak = peak_idx
    post_peak = total_len - peak_idx - 1  # -1 because peak_idx is inclusive
    
    return pre_peak, post_peak, peak_idx

def resample_around_peak(voltage_arrays, dv_arrays): #, target_points=1000):
    """
    Resample action potential traces aligned to their peaks
    
    Parameters:
    -----------
    voltage_arrays : list of arrays
        List of voltage traces
    dv_arrays : list of arrays  
        List of corresponding dV/dt traces
    target_points : int
        Desired number of points in resampled traces
    
    Returns:
    --------
    resampled_voltages : array
        Array of resampled voltage traces (n_traces x target_points)
    resampled_dvs : array
        Array of resampled dV/dt traces (n_traces x target_points)
    """
    
    # Find peak information for all traces
    peak_info = []
    for v, dv in zip(voltage_arrays, dv_arrays):
        pre, post, peak_idx = align_to_peak(v, dv)
        peak_info.append((pre, post, peak_idx))
    
    # Find common pre/post peak lengths (most restrictive)
    pre_peaks = [info[0] for info in peak_info]
    post_peaks = [info[1] for info in peak_info]
    
    common_pre = min(pre_peaks)
    common_post = min(post_peaks)
    
    resampled_voltages = []
    resampled_dvs = []
    
    for i, (v, dv) in enumerate(zip(voltage_arrays, dv_arrays)):
        pre, post, peak_idx = peak_info[i]
        
        # Extract the common window around the peak
        start_idx = peak_idx - common_pre
        end_idx = peak_idx + common_post + 1  # +1 to include peak
        
        # Ensure indices are valid
        start_idx = max(0, start_idx)
        end_idx = min(len(v), end_idx)
        
        v_window = v[start_idx:end_idx]
        dv_window = dv[start_idx:end_idx]
        
        resampled_voltages.append(v_window)
        resampled_dvs.append(dv_window)
    
    return np.array(resampled_voltages), np.array(resampled_dvs)

def create_average_phase_plot(voltage_arrays, dv_arrays, class_name=None, class_color=None,
                             show_individual=False, show_direction=True, axes=None, show_legend=True):
    """
    Create an averaged action potential phase plot
    
    Parameters:
    -----------
    voltage_arrays : list of arrays
        List of voltage traces
    dv_arrays : list of arrays
        List of corresponding dV/dt traces
    target_points : int
        Number of points for resampling
    show_individual : bool
        Whether to show individual traces
    show_direction : bool
        Whether to show direction arrows
    """
    
    # Resample all traces
    resampled_v, resampled_dv = resample_around_peak(voltage_arrays, dv_arrays)
    
    # Calculate mean and standard error
    mean_voltage = np.mean(resampled_v, axis=0)
    mean_dv = np.mean(resampled_dv, axis=0)
    
    sem_voltage = np.std(resampled_v, axis=0) / np.sqrt(len(voltage_arrays))
    sem_dv = np.std(resampled_dv, axis=0) / np.sqrt(len(voltage_arrays))
    
    # Create the plot
    if axes is None:
        fig, ax1 = plt.subplots(figsize=(10, 6))
    else:
        if type(axes) is tuple:
            ax1, ax2 = axes  # Assuming axes is a tuple of two axes
            t_arr = np.arange(len(mean_voltage)) / 50000  
            if show_individual:
                for i in range(len(resampled_v)):
                    ax2.plot(t_arr, resampled_v[i], 'black', alpha=0.3, linewidth=0.7)
            if class_color:
                ax2.plot(t_arr, mean_voltage, class_color, linewidth=2, label=class_name)
                ax2.fill_between(t_arr, mean_voltage - sem_voltage, mean_voltage + sem_voltage, 
                                alpha=0.3, color=class_color)
                ax2.set_xlabel('Time (s)', fontsize=12)
                ax2.set_ylabel('Voltage (mV)', fontsize=12)
                ax2.set_title('Average Voltage Traces', fontsize=14)
                sns.despine(ax=ax2, top=True, right=True)
        else:
            ax1 = axes
    
    # Phase plot
    if show_individual:
        for i in range(len(resampled_v)):
            ax1.plot(resampled_v[i], resampled_dv[i], 'black', alpha=0.3, linewidth=0.7)
    if class_color:
        ax1.plot(mean_voltage, mean_dv, class_color, linewidth=3, label=class_name)
        ax1.fill_between(mean_voltage, mean_dv - sem_dv, mean_dv + sem_dv, 
                        alpha=0.3, color=class_color)
    
    # Add direction arrows
    if show_direction:
        arrow_step = len(mean_voltage) // 8
        arrow_indices = np.arange(arrow_step, len(mean_voltage)-arrow_step, arrow_step)
        for i in arrow_indices:
            dx = mean_voltage[i+arrow_step//4] - mean_voltage[i]
            dy = mean_dv[i+arrow_step//4] - mean_dv[i]
            ax1.annotate('', xy=(mean_voltage[i+arrow_step//4], mean_dv[i+arrow_step//4]), 
                        xytext=(mean_voltage[i], mean_dv[i]),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.8, lw=1.5))
    
    ax1.set_xlabel('Voltage (mV)', fontsize=12)
    ax1.set_ylabel('dV/dt (mV/ms)', fontsize=12)
    ax1.set_title('Average Action Potential Phase Plots', fontsize=14)
    if show_legend:
        ax1.legend(bbox_to_anchor=(1.2, -0.25), loc='lower right') 

    sns.despine(ax=ax1, top=True, right=True)

    return mean_voltage, mean_dv, sem_voltage, sem_dv

def plot_ap_width(voltage_arrays, dv_arrays, class_name=None, class_color=None,
                  show_individual=False, ax=None, show_legend=True):
    """
    Create an averaged action potential curve
    
    Parameters:
    -----------
    voltage_arrays : list of arrays
        List of voltage traces
    dv_arrays : list of arrays
        List of corresponding dV/dt traces
    target_points : int
        Number of points for resampling
    show_individual : bool
        Whether to show individual traces
    show_direction : bool
        Whether to show direction arrows
    """
    
    # Resample all traces
    resampled_v, _ = resample_around_peak(voltage_arrays, dv_arrays)
    
    # Calculate mean and standard error
    mean_voltage = np.mean(resampled_v, axis=0)
    sem_voltage = np.std(resampled_v, axis=0) / np.sqrt(len(voltage_arrays))
    
    # Create the plot
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
        
    t_arr = np.arange(len(mean_voltage)) / 50000 * 1000  # convert to ms
    if show_individual:
        for i in range(len(resampled_v)):
            ax.plot(t_arr, resampled_v[i], 'black', alpha=0.3, linewidth=0.7)
    if class_color:
        ax.plot(t_arr, mean_voltage, class_color, linewidth=0.5, label=class_name)
        ax.fill_between(t_arr, mean_voltage - sem_voltage, mean_voltage + sem_voltage, 
                        alpha=0.1, color=class_color)
        ax.set_xlabel('Time (ms)', size=5)
        ax.set_ylabel('Voltage (mV)', size=5)
        sns.despine(ax=ax, top=True, right=True)

    if show_legend:
        ax.legend(bbox_to_anchor=(1.2, -0.25), loc='outer right') 

    # set axis and ticks to grey
    ax.tick_params(axis='both', colors='grey', labelsize=6, length=3)  # Set ticks color to grey
    ax.spines['bottom'].set_color('grey')  # Set bottom spine color to grey
    ax.spines['left'].set_color('grey')  # Set left spine color to grey

    
    return mean_voltage, sem_voltage

def plot_ap_phase(voltage_arrays, dv_arrays, class_name=None, class_color=None,
                  show_individual=False, show_direction=True, ax=None, show_legend=True):
    """
    Create an averaged action potential phase plot
    
    Parameters:
    -----------
    voltage_arrays : list of arrays
        List of voltage traces
    dv_arrays : list of arrays
        List of corresponding dV/dt traces
    target_points : int
        Number of points for resampling
    show_individual : bool
        Whether to show individual traces
    show_direction : bool
        Whether to show direction arrows
    """
    
    # Resample all traces
    resampled_v, resampled_dv = resample_around_peak(voltage_arrays, dv_arrays)
    
    # Calculate mean and standard error
    mean_voltage = np.mean(resampled_v, axis=0)
    mean_dv = np.mean(resampled_dv, axis=0)
    sem_dv = np.std(resampled_dv, axis=0) / np.sqrt(len(voltage_arrays))
    
    # Create the plot
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    # Phase plot
    if show_individual:
        for i in range(len(resampled_v)):
            ax.plot(resampled_v[i], resampled_dv[i], 'black', alpha=0.3, linewidth=0.7)
    if class_color:
        ax.plot(mean_voltage, mean_dv, class_color, linewidth=0.5, label=class_name)
        ax.fill_between(mean_voltage, mean_dv - sem_dv, mean_dv + sem_dv, 
                        alpha=0.1, color=class_color)
    
    # Add direction arrows
    if show_direction:
        arrow_step = len(mean_voltage) // 8
        arrow_indices = np.arange(arrow_step, len(mean_voltage)-arrow_step, arrow_step)
        for i in arrow_indices:
            dx = mean_voltage[i+arrow_step//4] - mean_voltage[i]
            dy = mean_dv[i+arrow_step//4] - mean_dv[i]
            ax.annotate('', xy=(mean_voltage[i+arrow_step//4], mean_dv[i+arrow_step//4]), 
                        xytext=(mean_voltage[i], mean_dv[i]),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.8, lw=1.5))
    
    ax.set_xlabel('Voltage (mV)', size=5)
    ax.set_ylabel('dV/dt (mV/ms)', size=5)
    # ax.set_title('Average Action Potential Phase Plots', fontsize=14)
    if show_legend:
        ax.legend(bbox_to_anchor=(1.2, -0.25), loc='outer right') 

    sns.despine(ax=ax, top=True, right=True)

    # set axis and ticks to grey
    ax.tick_params(axis='both', colors='grey', labelsize=6, length=3)  # Set ticks color to grey
    ax.spines['bottom'].set_color('grey')  # Set bottom spine color to grey
    ax.spines['left'].set_color('grey')  # Set left spine color to grey

    return mean_voltage, mean_dv, sem_dv

def plot_ap_subclass(subclass_dict, color_dict, ax, plot_type):

    for subclass_name in subclass_dict.keys():
        sc_varrs = []
        sc_dvdts = []

        for ttype in subclass_dict[subclass_name].keys():
            cell_data = subclass_dict[subclass_name][ttype]
            for cellname in cell_data.keys():
                v_arr = cell_data[cellname]['voltage']
                dv_arr = cell_data[cellname]['dVdt']
                sc_varrs.append(v_arr)
                sc_dvdts.append(dv_arr)

        subclass_color = color_dict[subclass_name]
        if '/' in subclass_name:
            subclass_name = subclass_name.replace('/', '-')

        if plot_type == 'width': 
            plot_ap_width(sc_varrs, sc_dvdts, class_name=subclass_name, class_color=subclass_color,
                            show_individual=False, ax=ax, show_legend=False)

        elif plot_type == 'phase':
            plot_ap_phase(sc_varrs, sc_dvdts, class_name=subclass_name, class_color=subclass_color,
                            show_individual=False, show_direction=False, ax=ax, show_legend=False)

        else: raise ValueError('Unknown plot_type')

def plot_ap_ttype(subclass_dict, ttypes, ttype_to_subclass_dict, color_dict, ax, plot_type):

    for ttype in ttypes:
        subclass_name = ttype_to_subclass_dict[ttype]
        ttp_varrs = []
        ttp_dvdts = []

        try: 
            cell_data = subclass_dict[subclass_name][ttype]
        except: 
            #no data for this ttype
            continue
        for cellname in cell_data.keys():
            v_arr = cell_data[cellname]['voltage']
            dv_arr = cell_data[cellname]['dVdt']
            ttp_varrs.append(v_arr)
            ttp_dvdts.append(dv_arr)

        ttype_color = color_dict[ttype]
        if '/' in ttype:
            ttype = ttype.replace('/', '-')

        if plot_type == 'width': 
            plot_ap_width(ttp_varrs, ttp_dvdts, class_name=ttype, class_color=ttype_color,
                            show_individual=False, ax=ax, show_legend=False)

        elif plot_type == 'phase':
            plot_ap_phase(ttp_varrs, ttp_dvdts, class_name=ttype, class_color=ttype_color,
                      show_individual=False, show_direction=False, ax=ax, show_legend=False)

        else: raise ValueError('Unknown plot_type')

def svg_path_to_mpl_path(svg_path):
    verts = []
    codes = []
    for i, segment in enumerate(svg_path):
        start = segment.start
        end = segment.end
        if i == 0:
            verts.append((start.real, start.imag))
            codes.append(Path.MOVETO)
        verts.append((end.real, end.imag))
        codes.append(Path.LINETO)
    return Path(verts, codes)

def translate_path(mpl_path, dx, dy):
    verts = mpl_path.vertices.copy()
    verts[:, 0] += dx
    verts[:, 1] += dy
    return Path(verts, mpl_path.codes)

def get_svg_bounding_box(paths):
    all_x = []
    all_y = []
    for path in paths:
        for segment in path:
            all_x.extend([segment.start.real, segment.end.real])
            all_y.extend([segment.start.imag, segment.end.imag])
    return min(all_x), max(all_x), min(all_y), max(all_y)

def plot_svg_on_ax(ax, svg_path, xoffset=0, yoffset=0, color='black', invert_y=True):
    if not os.path.exists(svg_path):
        print(f"File not found: {svg_path}")
        return

    paths, _ = svg2paths(svg_path)
    xmin, xmax, ymin, ymax = get_svg_bounding_box(paths)

    for path in paths:
        mpl_path = svg_path_to_mpl_path(path)

        if invert_y:
            # Invert the y-axis of the SVG path (flip vertically within its bounding box)
            flip_transform = Affine2D().scale(1, -1).translate(0, ymin + ymax)
            mpl_path = flip_transform.transform_path(mpl_path)

        shifted_path = translate_path(mpl_path, xoffset - xmin, yoffset - ymin)
        patch = patches.PathPatch(shifted_path, facecolor='none', edgecolor=color, linewidth=0.5)
        ax.add_patch(patch)

    ax.set_aspect('equal')
    ax.autoscale()

def plot_morpho_gallery(ax, gallery_cells, data, soma_depths_df, layer_info, layer_labels, color_dict, cluster_col='t_type', buffer = 100, plot_scalebar=True): # metadata, 

    group_title_start_height = 100 #starting height for group (e.g., region) title 
    group_title_step_height = 250 #how much between staggered title heights
    group_title_font_size = 5 #font size for group (e.g., region) title 
    compartment_list=[1,3,4] 

    xoffset_dict = {}

    title_count = 0

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)

        sps_list = list(gallery_cells.specimen_id.values)
        sps_list.reverse()  # to maintain CSV order

        ax.axis('off')
        xoffset = 0
        ax.set_anchor('W')
        ax.axhline(0, c="lightgrey", linewidth=0.50)

        layer_colors = {'2': 'lightgrey', '3': 'lightgrey', '4': 'lightgrey', '5': 'lightgrey', '6': 'lightgrey', 'wm': 'darkgrey', '1': 'darkgrey'}
        layer_labels = {'L1': 123.97, 'L2': 329.01, 'L3': 866.19, 'L4': 1440.05, 'L5': 1873.59, 'L6': 2633.37}

        for l in layer_info:
            ax.axhline(-layer_info[l], c=layer_colors[l], linewidth=0.20)
        for l in layer_labels:
            ax.text(-500, -layer_labels[l], "{}".format(l), va='center', ha='left', fontsize=group_title_font_size, color='lightgrey')

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

            # title_height = group_title_start_height
            # if not (title_count % 2 == 0): #even 
            #     title_height += group_title_step_height
                
            ax.text(xoffset, group_title_start_height,
                    "{}".format(ttype.replace('Car3', '\nCar3')),
                    ha='center', fontsize=group_title_font_size, color=hex_color, rotation=30)

            basic_morph_plot(morph, morph_colors=morph_colors, ax=ax, xoffset=xoffset)
            xoffset_dict[ttype] = xoffset

            xoffset += rightextent(nrn, compartment_list)
            xoffset += buffer

            soma_depths = [-1 * d for d in soma_depths_df[soma_depths_df[cluster_col] == ttype]['fig1_soma_depth'].tolist()] 
            if soma_depths:
                ax.scatter([xoffset] * len(soma_depths), soma_depths, facecolors='none', edgecolors='#b0aeae', marker='o', s=1)

            xoffset += buffer

        if plot_scalebar:
            scalebar = ScaleBar(1, "um", location='lower right', frameon=False, fixed_value=250, color='grey', font_properties={'size':6})
            ax.add_artist(scalebar)

        ax.set_aspect("equal")
        ax.set_xlim(0, xoffset)

def plot_ephys_gallery(fig, ax_morpho_gallery, svg_root, ttypes, color_dict, plot_scalebar=True):

    valid_ttypes = []
    for ttype in ttypes:
        ttype_reformat = ttype.replace('/', '').replace(' ', '_')
        if os.path.exists(os.path.join(svg_root, f'{ttype_reformat}_width.svg')):
            valid_ttypes.append(ttype)

    n_cells = len(valid_ttypes)
    if plot_scalebar: n_cells += 1
    bbox = ax_morpho_gallery.get_position()
    gap = 0.01  # small vertical gap
    height_ratio = 0.5  # ephys height as fraction of morpho height
    
    for i, ttype in enumerate(valid_ttypes): 

        width = bbox.width / n_cells
        left = bbox.x0 + i * width
        bottom = bbox.y0 - gap - bbox.height * height_ratio
        height = bbox.height * height_ratio

        ax = fig.add_axes([left, bottom, width, height])
        ax.axis('off')

        ax.text(0.5, 1.05, ttype.replace('Car3', '\nCar3'), transform=ax.transAxes, ha='center', va='bottom', fontsize=5, color=color_dict[ttype], rotation=30)

        svg_file_root = ttype.replace('/', '').replace(' ', '_')
        plot_svg_on_ax(ax, os.path.join(svg_root, f'{svg_file_root}_width.svg'), xoffset=0, yoffset=30, color=color_dict[ttype], invert_y=True)
        plot_svg_on_ax(ax, os.path.join(svg_root, f'{svg_file_root}_rate.svg'), xoffset=0, yoffset=-10, color=color_dict[ttype], invert_y=True)

    if plot_scalebar:
        # Add scale bar at end of row
        scalebar_left = bbox.x0 + n_cells * width -0.02
        scalebar_bottom = bbox.y0 - gap - bbox.height * height_ratio
        scalebar_ax = fig.add_axes([scalebar_left, scalebar_bottom, width * 0.8, height])  # slightly narrower
        scalebar_ax.axis('off')
        plot_svg_on_ax(scalebar_ax, os.path.join(svg_root, 'scale.svg'), invert_y=True, color='grey')
        ax.text(1.5, 0.5, '100 mV\n0.5 ms\n1 s', transform=ax.transAxes, ha='left', va='center', fontsize=5, color='grey')

def plot_umap(ax, df, group_col, groups, color_dict, x_col='umap_1', y_col='umap_2', umap1_label='UMAP1', umap2_label='UMAP2', plot_axis_arrows=False, label_dict=None):

    sns.scatterplot(data=df, x=x_col, y=y_col, hue=group_col, hue_order=groups, palette=color_dict, s=5, ax=ax) #s=10

    if not label_dict is None:
        for label, label_formatting_dict in label_dict.items():
            color = SEAAD_COLORS[label]
            plt.text(label_formatting_dict['x'], label_formatting_dict['y'], label, 
                     color=color, ha=label_formatting_dict['ha'], va=label_formatting_dict['va'], fontsize=5)
            

    if plot_axis_arrows:
        xmin, _ = ax.get_xlim()
        ymin, _ = ax.get_ylim()

        # Add small x and y axis arrows
        arrowprops = dict(arrowstyle='-|>', color='grey', lw=1, mutation_scale=5)
        length = 1
        ax.annotate('', xy=(xmin + length, ymin), xytext=(xmin, ymin), arrowprops=arrowprops)  # x arrow 1 unit right
        ax.annotate('', xy=(xmin, ymin + length), xytext=(xmin, ymin), arrowprops=arrowprops)  # y arrow 1 unit up

        # Optional: add axis labels
        ax.text(xmin, ymin-0.5, umap1_label, va='center', ha='left', fontsize=5, color='grey')
        ax.text(xmin-0.5, ymin, umap2_label, va='bottom', ha='center', fontsize=5, color='grey', rotation=90)

    ax.legend().remove()
    ax.set_aspect('equal')
    ax.axis('off')

def process_distal_apicals(df, groups, group_col):

    group_data_dict = {}
    for group in groups:
        #get group data 
        group_data = df[df[group_col] == group]

        #
        soma_depths_y = []
        distal_apicals_y = []
        num_nodes = 0
        for sp in group_data.index.values:
            swc_pth = df.SWC_layer_aligned[df.index == sp].iloc[0]
            nrn = morphology_from_swc(swc_pth)
            if nrn.has_type(4): 
                #neuron has apical, get the most distal one
                apical_nodes = nrn.get_node_by_types([4])
                most_distal_apical = -100000000000000
                for node in apical_nodes: most_distal_apical = max(node['y'], most_distal_apical)
                distal_apicals_y.append(most_distal_apical)
                soma_depths_y.append(df.soma_distance_from_pia[df.index == sp].iloc[0]) #plot the soma for this cell
                num_nodes += 1
        group_data_dict[group] = {'somas': soma_depths_y,
                                'apicals': distal_apicals_y, 
                                'label': "{} (n={})".format(group,num_nodes)}

    return group_data_dict

def plot_distal_apicals(data_dict, groups, ax, xoffsets, buffer_dict, color_dict, plot_legend=True):

    for group in groups:

        xoffset = int(xoffsets[group]) + buffer_dict[group]

        #colors for subclass
        hex_color = color_dict[group][1:]
        contrast_level = 0.6 #0.45 #0.6 
        lighter_color_rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        darker_color_rgb = tuple([int((contrast_level*x)) for x in lighter_color_rgb])
        grey_hex = '#8f8f8f'
        dark_hex = '#%02x%02x%02x' % darker_color_rgb
        hex_color = '#'+hex_color

        #plot soma depths first 
        xoffset += buffer_dict[group] 
        soma_depths_y = np.array(data_dict[group]['somas'])*-1
        soma_depths_x = np.ones(len(soma_depths_y))*xoffset
        jitter = np.array([random.randint(-2000,2000)*0.01 for n in soma_depths_y])
        soma_depths_x = soma_depths_x + jitter
        sns.scatterplot(x=soma_depths_x, y=soma_depths_y, ax=ax, color=hex_color, edgecolor=hex_color, s=0.5, marker="$\circ$", zorder=10) 

        #plot distal apicals second
        distal_apicals_y = data_dict[group]['apicals']
        distal_apicals_x = np.ones(len(distal_apicals_y))*xoffset
        jitter = np.array([random.randint(-2000,2000)*0.01 for n in distal_apicals_y])
        distal_apicals_x = distal_apicals_x + jitter
        sns.scatterplot(x=distal_apicals_x, y=distal_apicals_y, ax=ax, color=dark_hex, edgecolor=dark_hex, s=1, marker=r"$\bigtriangleup$", zorder=10)

    if plot_legend:
        
        # Create custom legend handles
        legend_elements = [
            Line2D([0], [0], color='grey', marker=r'$\bigtriangleup$', linestyle='None', label='apical', markersize=1),
            Line2D([0], [0], color='lightgrey', marker=r'$\circ$', linestyle='None', label='soma', markersize=1)
        ]

        # Add legend outside right
        ax.legend(
            handles=legend_elements,
            loc='upper left', #'upper right', 
            bbox_to_anchor=(1.02, 1), #(1, -0.05), 
            frameon=False,
            fontsize=5,
            handletextpad=0.4,
        )
        for text in ax.get_legend().get_texts(): 
            if text.get_text() == 'apical': text.set_color('grey')
            else: text.set_color('lightgrey')

def overlay_morphologies(ax, specimen_dict, xoffsets, data, color_dict, compartment_type = 'dendrite', alpha=0.5, color_grey=False):

    if compartment_type == 'axon': compartment_list = [1, 2]
    else: compartment_list = [1, 3, 4]

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

    
def compartment_list_by_type(self, compartment_type):
    """ Return an list of all compartments having the specified
    compartment type.

    Parameters
    ----------
    compartment_type: int
        Desired compartment type

    Returns
    -------
    A list of of Morphology Objects
    """
    return [x for x in self._compartment_list if x['type'] == compartment_type]
    

