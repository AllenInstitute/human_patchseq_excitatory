import allensdk.core.swc as swc
from matplotlib_scalebar.scalebar import ScaleBar

def read_and_normalize_swc(filepath):
    """
    Read an SWC file using the Allen Institute Morphology code
    and translate all coordinates so that the soma/root is at (0,0,0).
    
    Returns:
        Morphology object with shifted coordinates
    """
    morph = swc.read_swc(filepath)

    soma = morph.soma
    if soma is None:
        raise ValueError("No soma/root found in SWC; cannot normalize.")

    # Soma coordinates
    sx, sy, sz = soma['x'], soma['y'], soma['z']

    # Shift all nodes
    for seg in morph.compartment_list:
        seg['x'] -= sx
        # seg['y'] -= sy
        # seg['z'] -= sz

    return morph


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

def plot_subclass_hairballs(ax, morpho_sorted_clean_data, subclass_order, layer_info, layer_colors, layer_labels, color_dict, xoffsets, compartment_list, plot_scalebar=True, plot_subclass=True, ylabel=''):

    ax.axhline(0, c="lightgrey",linewidth=0.50)
    for l in layer_info: ax.axhline(-layer_info[l], c=layer_colors[l],linewidth=0.50)
    for l in layer_labels: ax.text(-100,-layer_labels[l], "{}".format(l),verticalalignment='center', horizontalalignment='left',fontsize=6, color='lightgrey') 

    # plot
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

        #save label 
        if plot_subclass: #title subclass and number of cells
            ax.text(xoffset,50,
                "{}\n(n={}) ".format(subclass,this_num_cells),
                horizontalalignment='center',fontsize=5, color=color_dict[subclass]) #, rotation=45)
        else: #just title number of cells
            ax.text(xoffset,50,
                "(n={}) ".format(this_num_cells),
                horizontalalignment='center',fontsize=5, color=color_dict[subclass]) #, rotation=45)

        if plot_scalebar & (i == 0):
            scalebar = ScaleBar(1, "um", location='lower left', frameon=False, fixed_value=500, color='grey', font_properties={"size": 6})
            ax.add_artist(scalebar)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.text(-0.02, 0.5, ylabel, fontsize=5, color='black', va='center', ha='center', rotation='vertical', transform=ax.transAxes)

