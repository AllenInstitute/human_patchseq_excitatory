import os
import pandas as pd
import numpy as np
import json
# import math

import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from  matplotlib.patches import PathPatch
# from matplotlib.backends.backend_pdf import PdfPages
from matplotlib_scalebar.scalebar import ScaleBar
# from matplotlib.collections import PathCollection

import seaborn as sns

from neuron_morphology.swc_io import morphology_from_swc
from morph_utils.measurements import leftextent, rightextent
from morph_utils.visuals import basic_morph_plot
# from morph_utils.templates import load_layer_template

# from scipy import stats
# from statsmodels.stats.multitest import fdrcorrection
# from scikit_posthocs import posthoc_dunn
from statannotations.Annotator import Annotator

# import operator

# import warnings
import contextlib
import io

import textwrap




def get_tx_order_dict(subclass_order_file, ttype_order_file):

    with open(r'\\allen\programs\celltypes\workgroups\mousecelltypes\SarahWB\datasets\human_exc\data\formatting_dicts\ttype_subclass_dict.json', 'r') as file: 
        ttype_to_subclass_dict = json.load(file)

    with open(subclass_order_file, 'r') as file: subclass_order = [line.strip() for line in file]
    with open(ttype_order_file, 'r') as file: ttype_order = [line.strip() for line in file]

    tx_order_dict = {}
    for subclass in subclass_order:
        tx_order = [t for t in ttype_order if ttype_to_subclass_dict[t] == subclass] #t.startswith(subclass)]
        # if subclass == 'L6 IT':
        #     tx_order = [t for t in tx_order if not t.startswith('L6 IT Car3')]
        tx_order_dict[subclass] = tx_order

    return tx_order_dict


def suppress_output(func):
    # a function to suppress print statements
    def wrapper(*args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):  # Redirect print statements
            return func(*args, **kwargs)
    return wrapper


def add_layer_aligned_paths(data, #df to add column with layer aligned paths to
                            layer_aligned_paths, #path to layer aligned files 
                            run_sk=False, #run Sk to make layer aligned file if file not found?
                            drop_missing=True): #drop cells that don't have layer aligned file from data?
    
    data["SWC_layer_aligned"] = np.nan

    #if given a path to layer aligned files, search and get paths to swcs here
    for idx, row in data.iterrows():
        if os.path.isfile(os.path.join(layer_aligned_paths, str(row.specimen_id) + '.swc')):
            data.loc[idx, 'SWC_layer_aligned'] = os.path.join(layer_aligned_paths, str(row.specimen_id) + '.swc')

    if drop_missing:
        #don't plot cells without swc files
        missing_idxs = list(data.loc[pd.isna(data["SWC_layer_aligned"]), :].index)
        if len(missing_idxs) > 0:
            dropped_cells = list(data.specimen_id[missing_idxs])
            cells = data.drop(index=missing_idxs)
            print('Warning: could not get swcs for some cells. Not plotting these specimen ids: ', dropped_cells)

    return data


def format_box_plot(ax):
    
    box_patches = [patch for patch in ax.patches if type(patch) == PathPatch]
    if len(box_patches) == 0:  # in matplotlib older than 3.5, the boxes are stored in ax2.artists
        box_patches = ax.artists
    num_patches = len(box_patches)
    lines_per_boxplot = len(ax.lines) // num_patches
    for i, patch in enumerate(box_patches):
        # Set the linecolor on the patch to the facecolor, and set the facecolor to None
        col = patch.get_facecolor()
        patch.set_edgecolor(col)
        patch.set_facecolor('None')

        # Each box has associated Line2D objects (to make the whiskers, fliers, etc.)
        # Loop over them here, and use the same color as above
        for line in ax.lines[i * lines_per_boxplot: (i + 1) * lines_per_boxplot]:
            line.set_color(col)
            line.set_mfc(col)  # facecolor of fliers
            line.set_mec(col)  # edgecolor of fliers

    # # Also fix the legend
    # for legpatch in ax.legend_.get_patches():
    #     col = legpatch.get_facecolor()
    #     legpatch.set_edgecolor(col)
    #     legpatch.set_facecolor('None')
    # sns.despine(left=True)


def format_violin_plot(ax):
    """
    Formats the violin plot so that:
    - The violin lines are colored and the fill is clear.
    - The interior points (if present) match the border color of the violins.
    """
    for patch in ax.collections:
        if hasattr(patch, "set_facecolor"):
            col = patch.get_facecolor()
            patch.set_edgecolor(col)
            patch.set_facecolor("None")
    
    # # Adjust points inside the violin (if inner='point')
    # for child in ax.findobj(match=PathCollection):
    #     if hasattr(child, "get_facecolors"):
    #         facecolors = child.get_facecolors()
    #         if len(facecolors) > 0:  # Ensure the points exist
    #             edge_color = facecolors[0]  # Use the same as violin border
    #             child.set_facecolor(edge_color)
    #             child.set_edgecolor(edge_color)


def plot_ttype_viewer(ax, subclass_cells, subclass_rows, this_layer_info, layer_colors, this_layer_labels, color_dict, soma_depth_col, layer_label_x):

    nrows = len(subclass_rows)
    compartment_list=[1,3,4]
    plot_layer_labels=True
    plot_scalebar=True
    buffer = 100
    subclass_buffer = 150
    histo_buffer = 2500

    layer_label_font_size=8
    group_title_start_height = 100 #starting height for group (e.g., region) title 
    group_title_font_size = 8 #font size for group (e.g., region) title 
    ######

    if nrows > 1:
        subclass_per_axis = {}
        for i, subclass_list in enumerate(subclass_rows):
            subclass_per_axis[i] = subclass_cells[subclass_cells.t_type.isin(subclass_list)].sort_values(by=['t_type', soma_depth_col]) 
    else:
        subclass_per_axis = {
            ax : subclass_cells
        }

    #plot cells on each axis 
    plotted_ttypes = set()
    layer_buffer = max(this_layer_info.values()) + 500
    for j, subclass_cells in enumerate(subclass_per_axis.values()): 
        ax.axis('off')
        xoffset = 0
        yoffset = j * layer_buffer
        ax.set_anchor('W')
        ax.axhline(0, c="lightgrey",linewidth=0.50)

        # this_layer_buffer = j * layer_buffer

        for l in this_layer_info:
            ax.axhline(-this_layer_info[l]-yoffset, c=layer_colors[l],linewidth=0.50)
        if plot_layer_labels:
            for l in this_layer_labels:
                layer_label_buffer=0
                if l == 'L1': layer_label_buffer = 50
                if l == 'L2': layer_label_buffer = -50
                ax.text(layer_label_x,-this_layer_labels[l]+layer_label_buffer-yoffset, "{}".format(l),verticalalignment='center', horizontalalignment='left',fontsize=layer_label_font_size, color='lightgrey') 

        #plot all the cells in this subclass
        for i, (idx, row) in enumerate(subclass_cells.iterrows()):

            ttype = row.t_type

            if not ttype in plotted_ttypes:
                #This is a new ttype
                xoffset += subclass_buffer

                #colors for ttype
                hex_color = color_dict[ttype][1:]
                contrast_level = 0.45
                lighter_color_rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                darker_color_rgb = tuple([int((contrast_level*x)) for x in lighter_color_rgb])
                grey_hex = '#8f8f8f'
                dark_hex = '#%02x%02x%02x' % darker_color_rgb
                hex_color = '#'+hex_color
                cell_type_colors_dict = {3: dark_hex, 4: hex_color, 2: grey_hex}
                morph_colors = {k:v for k,v in cell_type_colors_dict.items() if k in compartment_list} # only show colors we specified in compartment_list

                #plot the title
                ax.text(xoffset,group_title_start_height-yoffset, ttype, horizontalalignment='left', fontsize=group_title_font_size, color=hex_color) #, rotation=45) 

            plotted_ttypes.add(ttype)

            #get the cell 
            sp = subclass_cells.specimen_id[idx]
            swc_pth = subclass_cells.SWC_layer_aligned[subclass_cells.specimen_id == sp].iloc[0]
            nrn = morphology_from_swc(swc_pth)

            #plot cell offset along the x axis from the previous cell 
            xoffset += leftextent(nrn,compartment_list)
            basic_morph_plot(nrn, ax=ax, xoffset=xoffset, yoffset=-yoffset, morph_colors=morph_colors, scatter_roots=False, scatter_soma=False, plot_soma=False)

            xoffset += rightextent(nrn,compartment_list)
            xoffset += buffer

            if j == len(subclass_per_axis.items())-1 and i == 0 and plot_scalebar:
                #add a scale bar 
                scalebar = ScaleBar(1, "um", location='lower left', frameon=False, fixed_value=500, color = 'grey')
                ax.add_artist(scalebar)

            ax.set_aspect("equal") #so neurons don't look squished 


        ax.relim()  # Recalculate limits based on current data
        ax.autoscale_view()  # Rescale view limits


def plot_ttype_histos(ax, subclass_rows, this_layer_info, layer_colors, this_layer_labels, color_dict, aligned_histogram, all_ttype_data, soma_depth_col, avg_dist_btw_nodes, layer_label_x):

    plot_layer_labels=True
    layer_label_font_size=8
    ######

    ax.axis('off')
    xoffset = 0
    ax.set_anchor('W')
    ax.axhline(0, c="lightgrey",linewidth=0.50)

    for l in this_layer_info:
        ax.axhline(-this_layer_info[l], c=layer_colors[l],linewidth=0.50)
    if plot_layer_labels:
        for l in this_layer_labels:
            layer_label_buffer=0
            if l == 'L1': layer_label_buffer = 50
            if l == 'L2': layer_label_buffer = -50
            ax.text(layer_label_x,-this_layer_labels[l]+layer_label_buffer, "{}".format(l),verticalalignment='center', horizontalalignment='left',fontsize=layer_label_font_size, color='lightgrey') 

    #plot histos on seperate xaxis scale
    xoffset=0
    for ttype in sum(subclass_rows, []):

        #get ttype colors
        hex_color = color_dict[ttype][1:]
        contrast_level = 0.45
        lighter_color_rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        darker_color_rgb = tuple([int((contrast_level*x)) for x in lighter_color_rgb])
        grey_hex = '#8f8f8f'
        dark_hex = '#%02x%02x%02x' % darker_color_rgb
        hex_color = '#'+hex_color
        cell_type_colors_dict = {3: dark_hex, 4: hex_color, 2: grey_hex}

        #get this ttype data to plot histos for
        this_ttype_data = all_ttype_data[all_ttype_data.t_type == ttype]

        #get soma depths for all cells in this subclass 
        soma_depths_y = [d*-1 for d in this_ttype_data[soma_depth_col].tolist()]

        # histos of axon, dendrite, and soma distributions 
        histo_compartments = [3,4] #[soma, axon, basal, apical] 

        bin_width = 5 #um
        depth_bins = [int(sub[2:])*bin_width*-1 for sub in aligned_histogram.filter(regex='2_').columns.tolist()] 

        histo_ticks = []
        histo_labels = []
        for compartment_num in histo_compartments:

            #get data for plotting compartment histograms 
            axon_depth_avg = aligned_histogram.filter(regex=str(compartment_num)+'_')[aligned_histogram.specimen_id.isin(this_ttype_data.specimen_id)].mean(axis=0).transpose()
            if compartment_num == 3: axon_depth_avg *= -1
            axon_depth_sem = aligned_histogram.filter(regex=str(compartment_num)+'_')[aligned_histogram.specimen_id.isin(this_ttype_data.specimen_id)].sem(axis=0).transpose()
            
            #multiply by avg distance between nodes to get values in um 
            axon_depth_avg*=avg_dist_btw_nodes
            axon_depth_sem*=avg_dist_btw_nodes

            #turn zeros to nans so they aren't plotted (this will cut out all zeros, even those in the middle, may want to change this to only naning the leading and trailing zeros)
            axon_depth_avg.replace(0, np.nan, inplace=True)
            axon_depth_sem.replace(0, np.nan, inplace=True)

            if compartment_num == 3:
                if not np.isnan(axon_depth_avg-axon_depth_sem).all():
                    xoffset += abs(np.nanmin(axon_depth_avg-axon_depth_sem))
                elif not np.isnan(axon_depth_avg).all():
                    xoffset += abs(np.nanmin(axon_depth_avg))


            #add xoffset for plotting on same subplot as cells
            axon_depth_avg_plot = axon_depth_avg + xoffset 

            #plot
            if not np.isnan(axon_depth_avg_plot).all():
                #plot histos
                ax.plot(axon_depth_avg_plot, depth_bins, color = cell_type_colors_dict[compartment_num], linewidth=1)
                ax.fill_betweenx(depth_bins, axon_depth_avg_plot-axon_depth_sem, axon_depth_avg_plot+axon_depth_sem, alpha=0.5, edgecolor=cell_type_colors_dict[compartment_num], facecolor=cell_type_colors_dict[compartment_num])

                #plot soma depth locations
                if compartment_num == 3:
                    soma_depths_x = np.zeros(len(soma_depths_y))
                    soma_depths_x+=xoffset
                    sns.scatterplot(x=soma_depths_x, y=soma_depths_y, ax=ax, color='black', edgecolor="none", legend=False, size=0.1, marker="$\circ$") 

            if compartment_num == 4:
                if not np.isnan(axon_depth_avg+axon_depth_sem).all():
                    xoffset += abs(np.nanmax(axon_depth_avg+axon_depth_sem))
                elif not np.isnan(axon_depth_avg).all():
                    xoffset += abs(np.nanmax(axon_depth_avg))

            ax.set_xticks(histo_ticks)
            ax.set_xticklabels(histo_labels)
            ax.xaxis.label.set_color('grey')
            ax.tick_params(axis='x', colors='grey')
            ax.set_xlabel(u"\u03bcm")
            # plt.tight_layout()

    #add a scale bar 
    scalebar = ScaleBar(1, "um", location='lower right', frameon=False, fixed_value=100, color = 'grey')
    ax.add_artist(scalebar)


def plot_ttype_viewer_histos(ax, subclass_cells, subclass_ttypes, this_layer_info, layer_colors, this_layer_labels, color_dict, soma_depth_col, layer_label_x, aligned_histogram, all_ttype_data, avg_dist_btw_nodes):

    compartment_list=[1,3,4]
    plot_layer_labels=True
    buffer = 100
    subclass_buffer = 150
    section_buffer = 500

    layer_label_font_size=8
    group_title_start_height = 100 #starting height for group (e.g., region) title 
    group_title_font_size = 10 #font size for group (e.g., region) title 
    ######

    #Axis formatting 
    plotted_ttypes = set()
    ax.axis('off')
    xoffset = 0
    ax.set_anchor('W')
    ax.axhline(0, c="lightgrey",linewidth=0.50)

    #Plot layers 
    for li in this_layer_info:
        ax.axhline(-this_layer_info[li], c=layer_colors[li],linewidth=0.50)
    if plot_layer_labels:
        for ll in this_layer_labels:
            layer_label_buffer=0
            if ll == 'L1': layer_label_buffer = 50
            if ll == 'L2': layer_label_buffer = -50
            ax.text(layer_label_x,-this_layer_labels[ll]+layer_label_buffer, "{}".format(ll),verticalalignment='center', horizontalalignment='left',fontsize=layer_label_font_size, color='lightgrey') 

    #Plot morphologies 
    for i, (idx, row) in enumerate(subclass_cells.iterrows()):

        ttype = row.t_type

        if not ttype in plotted_ttypes:
            #This is a new ttype
            xoffset += subclass_buffer

            #colors for ttype
            hex_color = color_dict[ttype][1:]
            contrast_level = 0.45
            lighter_color_rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            darker_color_rgb = tuple([int((contrast_level*x)) for x in lighter_color_rgb])
            grey_hex = '#8f8f8f'
            dark_hex = '#%02x%02x%02x' % darker_color_rgb
            hex_color = '#'+hex_color
            cell_type_colors_dict = {3: dark_hex, 4: hex_color, 2: grey_hex}
            morph_colors = {k:v for k,v in cell_type_colors_dict.items() if k in compartment_list} # only show colors we specified in compartment_list

            #plot the title
            ax.text(xoffset,group_title_start_height, ttype, horizontalalignment='left', fontsize=group_title_font_size, color=hex_color) #, rotation=45) 

        plotted_ttypes.add(ttype)

        #get the cell 
        sp = subclass_cells.specimen_id[idx]
        swc_pth = subclass_cells.SWC_layer_aligned[subclass_cells.specimen_id == sp].iloc[0]
        nrn = morphology_from_swc(swc_pth)

        #plot cell offset along the x axis from the previous cell 
        xoffset += leftextent(nrn,compartment_list)
        basic_morph_plot(nrn, ax=ax, xoffset=xoffset, morph_colors=morph_colors, scatter_roots=False, scatter_soma=False, plot_soma=False)

        xoffset += rightextent(nrn,compartment_list)
        xoffset += buffer

    #Plot histograms 
    xoffset += section_buffer
    for ttype in subclass_ttypes:

        #get ttype colors
        hex_color = color_dict[ttype][1:]
        contrast_level = 0.45
        lighter_color_rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        darker_color_rgb = tuple([int((contrast_level*x)) for x in lighter_color_rgb])
        grey_hex = '#8f8f8f'
        dark_hex = '#%02x%02x%02x' % darker_color_rgb
        hex_color = '#'+hex_color
        cell_type_colors_dict = {3: dark_hex, 4: hex_color, 2: grey_hex}

        #get this ttype data to plot histos for
        this_ttype_data = all_ttype_data[all_ttype_data.t_type == ttype]

        #get soma depths for all cells in this subclass 
        soma_depths_y = [d*-1 for d in this_ttype_data[soma_depth_col].tolist()]

        # histos of axon, dendrite, and soma distributions 
        histo_compartments = [3,4] #[soma, axon, basal, apical] 

        bin_width = 5 #um
        depth_bins = [int(sub[2:])*bin_width*-1 for sub in aligned_histogram.filter(regex='2_').columns.tolist()] 

        # histo_ticks = []
        # histo_labels = []
        for compartment_num in histo_compartments:

            #get data for plotting compartment histograms 
            axon_depth_avg = aligned_histogram.filter(regex=str(compartment_num)+'_')[aligned_histogram.specimen_id.isin(this_ttype_data.specimen_id)].mean(axis=0).transpose()
            if compartment_num == 3: axon_depth_avg *= -1
            axon_depth_sem = aligned_histogram.filter(regex=str(compartment_num)+'_')[aligned_histogram.specimen_id.isin(this_ttype_data.specimen_id)].sem(axis=0).transpose()
            
            #multiply by avg distance between nodes to get values in um 
            axon_depth_avg*=avg_dist_btw_nodes
            axon_depth_sem*=avg_dist_btw_nodes

            #turn zeros to nans so they aren't plotted (this will cut out all zeros, even those in the middle, may want to change this to only naning the leading and trailing zeros)
            axon_depth_avg.replace(0, np.nan, inplace=True)
            axon_depth_sem.replace(0, np.nan, inplace=True)

            if compartment_num == 3:
                if not np.isnan(axon_depth_avg-axon_depth_sem).all():
                    xoffset += abs(np.nanmin(axon_depth_avg-axon_depth_sem))
                elif not np.isnan(axon_depth_avg).all():
                    xoffset += abs(np.nanmin(axon_depth_avg))


            #add xoffset for plotting on same subplot as cells
            axon_depth_avg_plot = axon_depth_avg + xoffset 

            #plot
            if not np.isnan(axon_depth_avg_plot).all():
                #plot histos
                ax.plot(axon_depth_avg_plot, depth_bins, color = cell_type_colors_dict[compartment_num], linewidth=1)
                ax.fill_betweenx(depth_bins, axon_depth_avg_plot-axon_depth_sem, axon_depth_avg_plot+axon_depth_sem, alpha=0.5, edgecolor=cell_type_colors_dict[compartment_num], facecolor=cell_type_colors_dict[compartment_num])

                #plot soma depth locations
                if compartment_num == 3:
                    soma_depths_x = np.zeros(len(soma_depths_y))
                    soma_depths_x+=xoffset
                    sns.scatterplot(x=soma_depths_x, y=soma_depths_y, ax=ax, color='black', edgecolor="none", legend=False, size=0.1, marker="$\circ$") 

            if compartment_num == 4:
                if not np.isnan(axon_depth_avg+axon_depth_sem).all():
                    xoffset += abs(np.nanmax(axon_depth_avg+axon_depth_sem))
                elif not np.isnan(axon_depth_avg).all():
                    xoffset += abs(np.nanmax(axon_depth_avg))


    #add a scalec bar 
    scalebar = ScaleBar(1, "um", location='lower center', frameon=False, fixed_value=500, color = 'grey')
    ax.add_artist(scalebar)

    ax.set_aspect("equal") #so neurons don't look squished 

# def plot_subclass_eta2_topn(subclass, eta_squared_subclass, ax, n=10): 
def plot_subclass_eta2_topn(eta_squared_ttypes_df, ax, n=10, label_right=True): 
    """
    how much of the variance between subclasses is explained by each feature? 
    a way of getting at which features most 'separate' subclasses. 
    """

    # if not eta_squared_subclass[subclass].empty:
    #     #has eta2 data to show

    #     eta_squared_ttypes_df = eta_squared_subclass[subclass]
    eta_squared_ttypes_df = eta_squared_ttypes_df.nlargest(n, 'eta_squared')

    #color palette
    values = sns.color_palette("ch:s=-.2,r=.6", len(eta_squared_ttypes_df)) # 10 colors
    keys = eta_squared_ttypes_df.feature
    color_palette = dict(zip(keys, values))

    #plot!
    sns.set_theme(style="white")
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True

    #plot pvals 
    sns.barplot(x="eta_squared", y='feature', data=eta_squared_ttypes_df, palette=color_palette)
    sns.despine()

    #edit axis labels 
    ax.set_xlabel("\u03B7\u00b2", size=6)#, color='grey')
    ax.set_ylabel("")
    plt.xticks(ticks=[0,0.5,1], labels=['0', '0.5', '1'], size=6)


    if label_right: 
        # move y-axis to the right
        ax.yaxis.tick_right()  # Move labels to the right side
        label_position = 'right'
        label_ha = 'left'
    else: 
        label_position = 'left'
        label_ha = 'right'
    ax.yaxis.set_label_position(label_position)  # Move the label position to the left/right
    ax.set_yticklabels(ax.get_yticklabels(), ha=label_ha, fontsize=6)  # Align labels to the left/right
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=(not label_right), labelright=label_right) # remove y-ticks and labels

    # # color y-tick labels by feature type
    # yticklabels = ax.get_yticklabels()
    # for label in yticklabels:
    #     feature = label.get_text()
    #     if feature_color_dict and feature in feature_color_dict:
    #         label.set_color(feature_color_dict[feature])

    # if len(yticklabels) > 25: y_font_size = 6
    # elif len(yticklabels) > 20: y_font_size = 8
    # else: y_font_size = 10

    # set axis and ticks to grey
    ax.tick_params(axis='x', colors='grey')  # Set ticks color to grey
    ax.spines['bottom'].set_color('grey')  # Set bottom spine color to grey
    ax.spines['top'].set_color('grey')  # Set top spine color to grey
    ax.spines['left'].set_color('grey')  # Set left spine color to grey
    ax.spines['right'].set_color('grey')  # Set right spine color to grey



def plot_subclass_eta2(subclass, eta_squared_subclass, ax, feature_color_dict,
                       signif_thresh=False, kw_pvals_fdr_subclass=None, #whether to only plot significant features based on fdr corrected kw 
                       value_thresh=False, value=0): #whether to otnly plot geatures with eta2 (abs) above a given threshold (value)
    """
    how much of the variance between subclasses is explained by each feature? 
    a way of getting at which features most 'separate' subclasses. 
    """

    if not eta_squared_subclass[subclass].empty:
        #has eta2 data to show
        eta_squared_ttypes_df = eta_squared_subclass[subclass]

        if signif_thresh: 
            kw_pvals_fdr = kw_pvals_fdr_subclass[subclass] # Get the FDR-corrected p-values for that subclass
            significant_features = [f for f, p in kw_pvals_fdr.items() if p < 0.05]  # Identify significant features based on FDR-corrected p-values (alpha = 0.05)
            eta_squared_ttypes_df = eta_squared_ttypes_df[eta_squared_ttypes_df['feature'].isin(significant_features)] # Filter eta-squared values for significant features
        
        elif value_thresh: 
            eta_squared_ttypes_df = eta_squared_ttypes_df[abs(eta_squared_ttypes_df['eta_squared']) > value]

        if len(eta_squared_ttypes_df) == 0:
            print('No significant eta squared values to plot!')
            # ax.axis('off')
            # plt.text(0.5, 0.5, f'No values to plot.', fontsize=12, color='grey',  ha='center', va='center', transform=plt.gca().transAxes)

            #plot the top four 
            eta_squared_ttypes_df = eta_squared_subclass[subclass]
            eta_squared_ttypes_df = eta_squared_ttypes_df.nlargest(4, 'eta_squared')

        # else: 

        #color palette
        values = sns.color_palette("ch:s=-.2,r=.6", len(eta_squared_ttypes_df)) # 10 colors
        keys = eta_squared_ttypes_df.feature
        color_palette = dict(zip(keys, values))

        #plot!
        sns.set_theme(style="white")
        plt.rcParams['xtick.bottom'] = True
        plt.rcParams['ytick.left'] = True

        #plot pvals 
        sns.barplot(x="eta_squared", y='feature', data=eta_squared_ttypes_df, palette=color_palette)
        sns.despine()

        #edit axis labels 
        ax.set_xlabel("\u03B7\u00b2", size=4)#, color='grey')
        ax.set_ylabel("")
        plt.xticks(ticks=[0,0.5,1], labels=['0', '0.5', '1'], size=6)

        # move y-axis to the right
        ax.yaxis.tick_right()  # Move labels to the right side
        ax.yaxis.set_label_position("right")  # Move the label position to the right

        # # color y-tick labels by feature type
        # yticklabels = ax.get_yticklabels()
        # for label in yticklabels:
        #     feature = label.get_text()
        #     if feature_color_dict and feature in feature_color_dict:
        #         label.set_color(feature_color_dict[feature])

        # if len(yticklabels) > 25: y_font_size = 6
        # elif len(yticklabels) > 20: y_font_size = 8
        # else: y_font_size = 10
        ax.set_yticklabels(ax.get_yticklabels(), ha='left', fontsize=4)  # Align labels to the right

        # remove y-ticks and labels
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=True)

        #plot significance threshold for pvals
        if value_thresh: 
            plt.axvline(x = value, color=(0.8, 0.8, 0.8, 0.6), linewidth=2)

        # set axis and ticks to grey
        ax.tick_params(axis='x', colors='grey')  # Set ticks color to grey
        ax.spines['bottom'].set_color('grey')  # Set bottom spine color to grey
        ax.spines['top'].set_color('grey')  # Set top spine color to grey
        ax.spines['left'].set_color('grey')  # Set left spine color to grey
        ax.spines['right'].set_color('grey')  # Set right spine color to grey


def plot_subclass_boxplot(subclass, subclass_var, feature, 
                          kw_pvals_fdr_subclass, dunn_pvals_fdr_subclass,
                          sorted_clean_data, ttype_order,
                          color_dict, feature_type_color_dict, ax, 
                          show_signif_bars=True, 
                          xlabel_num_cells=True, 
                          n_rows=1,
                          ylabels=True):

    if kw_pvals_fdr_subclass[subclass]:
        # Subclass has results (there were enough ttypes in the subclass, etc.)
        subclass_data = sorted_clean_data.loc[sorted_clean_data[subclass_var] == subclass]
        kw_pvals_fdr = kw_pvals_fdr_subclass[subclass]  
        dunn_pvals_fdr = dunn_pvals_fdr_subclass[subclass]
        x_var = 't_type'  # What to do box plots on
        x_var_order = [t for t in ttype_order if t in subclass_data[x_var].unique()]

        # Color dict 
        color_palette = color_dict  # Colors for x_var
        y_var = feature  # Feature to plot
        
        # Horizontal violin plot
        sns.violinplot(data=subclass_data, y=x_var, x=y_var, ax=ax, order=x_var_order, 
                       palette=color_palette, saturation=1, width=0.5, scale='width', 
                       linewidth=1, inner='point')
        format_violin_plot(ax)  # Custom function to format the violin plot

        if xlabel_num_cells:
            # Add num observations per x_var
            nobs = []
            for t in x_var_order:
                n = len(subclass_data[subclass_data[x_var] == t])
                nobs.append(" (n=%s)" % str(n))
            new_y_tick_labels = [x_var_order[i] + nobs[i] for i in range(len(x_var_order))]  
            ax.set_yticklabels(new_y_tick_labels)

        # Add stats (KW w/ Dunn post hoc FDR corrected)
        if y_var in dunn_pvals_fdr.keys(): 
            # KW FDR value was significant so there are Dunn FDR values to plot
            this_dunn_pvals_fdr = dunn_pvals_fdr[y_var]
            remove = np.tril(np.ones(this_dunn_pvals_fdr.shape), k=0).astype("bool")
            this_dunn_pvals_fdr[remove] = np.nan
            dunn_pvals_fdr_melt = this_dunn_pvals_fdr.melt(ignore_index=False).reset_index().dropna()

            # Set insignificant pvals to NaN 
            p_thresh = 0.05  # Threshold for plotting p values 
            dunn_pvals_fdr_melt.drop(dunn_pvals_fdr_melt[dunn_pvals_fdr_melt['value'] > p_thresh].index, inplace=True)

            if show_signif_bars: 
                # Plot bar for all significant FDR corrected Dunn p values 
                pairs = [(i[1]["index"], i[1]["variable"]) for i in dunn_pvals_fdr_melt.iterrows()]
                if len(pairs): 
                    # There's at least one pair of x_vals for which FDR corrected Dunn test is significant
                    p_values = [i[1]["value"] for i in dunn_pvals_fdr_melt.iterrows()]                    
                    annotator = Annotator(ax, pairs, data=subclass_data, y=x_var, x=y_var, order=x_var_order)
                    annotator.configure(text_format="star", loc="inside")  # text_format="simple"
                annotator.set_pvalues_and_annotate(p_values)

        # # Plot Kruskal-Wallis FDR p-vals
        # if y_var in kw_pvals_fdr.keys():
        #     p_txt = 'KW, p = ' + f"{kw_pvals_fdr[y_var]:.4e}"
        # else: 
        #     p_txt = ''
        # if n_rows == 1: title_height = 1.15
        # elif n_rows == 2: title_height = 1.18
        # else: title_height = 1.2
        # plt.text(1.05, 0.5, y_var, ha='left', va='center', color=feature_type_color_dict[y_var], fontsize=12, transform=plt.gca().transAxes) 
        # plt.text(1.05, 0.5, p_txt, ha='left', va='top', color='grey', fontsize=10, transform=plt.gca().transAxes)
        
        #plot Kruskal-Wallis fdr p-vals
        if y_var in kw_pvals_fdr.keys():
            xlimit = ax.get_xlim()
            ylimit = ax.get_ylim()
            p_txt = 'KW, p = ' + f"{kw_pvals_fdr[y_var]:.4e}"
        else: p_txt = ''
        # plt.title(y_var, color=feature_type_color_dict[y_var])
        if n_rows == 1: title_height = 1.15
        elif n_rows == 2: title_height = 1.18
        else: title_height = 1.2
        # plt.text(0.5, title_height, y_var, ha='center', va='bottom', color=feature_type_color_dict[y_var], fontsize=12, transform=plt.gca().transAxes) 
        # plt.text(0.5, 1.05, p_txt, ha='center', va='bottom', color='grey', fontsize=10, transform=plt.gca().transAxes)

        # Position the y_var as the x-axis label
        max_width = 20  # Maximum character width per line (adjust as needed)
        formatted_y_var = textwrap.fill(y_var, max_width)
        if not feature_type_color_dict is None: 
            plt.text(0.5, -0.2, formatted_y_var, ha='center', va='center', color=feature_type_color_dict[y_var], fontsize=10, transform=plt.gca().transAxes)
        else:
            plt.text(0.5, -0.2, formatted_y_var, ha='center', va='center', color='black', fontsize=10, transform=plt.gca().transAxes)


        # Position the p-value text on the x-axis
        plt.text(0.5, -0.3, p_txt, ha='center', va='top', color='grey', fontsize=10, transform=plt.gca().transAxes)

        # Rotate and format y_var labels 
        ax.set_ylabel(None)
        ax.tick_params(axis='y', rotation=0)
        ax.set_xlabel('', rotation=0)
        ax.spines[['right', 'top']].set_visible(False)
        
        ax.tick_params(axis='x', colors='grey')
        ax.xaxis.set_tick_params(labelcolor='grey')
        ax.spines['bottom'].set_color('grey')
        ax.spines['left'].set_color('grey')

        # Color y-tick labels by ttype color
        if ylabels:
            yticklabels = ax.get_yticklabels()
            for t_label in yticklabels:
                t_label_text = t_label.get_text().split(' (')[0]
                if color_dict and t_label_text in color_dict:
                    t_label.set_color(color_dict[t_label_text])
        else:
            plt.yticks([])
