import os  
import pandas as pd
import numpy as np
from matplotlib.patches import PathPatch

#basic functions used repeatedly in human glutamateric data analysis.


AVERAGE_DISTANCE_BETWEEN_NODES = 1.14 #avg um between nodes for ivscc data 

def add_swc_paths(data,  directory,  column_name='swc_file_path',  drop_missing=True):
    """ 
    Add file paths to swcs and add to data df. 

    Parameters
    ----------
    data : pandas.DataFrame  
        DataFrame containing a 'specimen_id' column.  

    directory : str  
        Path to the directory with SWC files.  

    column_name : str, optional  
        Name of the new column for SWC paths (default: "swc_file_path").  

    drop_missing : bool, optional  
        If True, removes rows without a matching SWC file (default: True).  

    Returns
    -------
    pandas.DataFrame  
        Updated DataFrame with the SWC file paths added.

    """
    
    #get l
    data[column_name] = np.nan
    for idx, row in data.iterrows():
        if os.path.isfile(os.path.join(directory, str(int(row.specimen_id)) + '.swc')):
            data.loc[idx, column_name] = os.path.join(directory, str(int(row.specimen_id)) + '.swc')
        
    if drop_missing:
        #don't plot cells without swc files
        missing_idxs = list(data.loc[pd.isna(data[column_name]), :].index)
        if len(missing_idxs) > 0:
            dropped_cells = list(data.specimen_id[missing_idxs])
            data = data.drop(index=missing_idxs)
            print('Warning: could not get swcs for some cells. Not plotting these specimen ids: ', dropped_cells)

    return data


def get_seaad_colors():
    seaad_colors_path = r'\\allen\programs\celltypes\workgroups\mousecelltypes\SarahWB\datasets\human_exc\data\seaad_colors\SEAAD_colors.xlsx'
    seaad_colors = pd.read_excel(seaad_colors_path, sheet_name='in')
    seaad_colors = seaad_colors.rename(columns={'Unnamed: 0':'label'})
    color_dict = dict(zip(seaad_colors['label'], seaad_colors['supertype_scANVI_leiden_colors']))
    return color_dict


def get_tx_order(): #cells, ttype_var_name
    # #manually define subclass order 
    # subclass_order = ['L2/3 IT', 'L4 IT', 'L5 ET', 'L5 IT', 'L5/6 NP', 'L6 CT', 'L6b', 'L6 IT', 'L6 IT Car3']
    # subclass_index = {subclass: index for index, subclass in enumerate(subclass_order)}

    # #get ttype order
    # def sort_key(s):
    #     subclass_name, ttype_suffix = s.rsplit('_',1)
    #     subclass_order = subclass_index.get(subclass_name, float('inf'))
    #     try: 
    #         # If the ttype suffix is an int, sort numerically 
    #         ttype_order = int(ttype_suffix)
    #     except ValueError:
    #         # If the ttype suffix is a letter, place at the end (cells we're calling 'ME' type) 
    #         ttype_order = float('inf')
    #     return (subclass_order, ttype_order)
    # ttype_order = sorted(cells[ttype_var_name].unique(), key=sort_key)

    with open(r'\\allen\programs\celltypes\workgroups\mousecelltypes\SarahWB\datasets\human_exc\data\formatting_dicts\subclass_order.txt', 'r') as f:
        subclass_order = f.read().splitlines()

    with open(r'\\allen\programs\celltypes\workgroups\mousecelltypes\SarahWB\datasets\human_exc\data\formatting_dicts\ttype_order.txt', 'r') as f:
        ttype_order = f.read().splitlines()
    
    return ttype_order, subclass_order


def get_tx_order_linux(): 

    with open('/allen/programs/celltypes/workgroups/mousecelltypes/SarahWB/datasets/human_exc/data/formatting_dicts/subclass_order.txt', 'r') as f:
        subclass_order = f.read().splitlines()

    with open('/allen/programs/celltypes/workgroups/mousecelltypes/SarahWB/datasets/human_exc/data/formatting_dicts/ttype_order.txt', 'r') as f:
        ttype_order = f.read().splitlines()
    
    return ttype_order, subclass_order


def format_box_plot(ax):
    
    box_patches = [patch for patch in ax.patches if type(patch) == PathPatch]
    if len(box_patches) == 0:  # in matplotlib older than 3.5, the boxes are stored in ax2.artists
        box_patches = ax.artists
    num_patches = len(box_patches)
    if num_patches == 0:
        return #Nothing to format
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


def find_ttype_key(keys, substring):
    for s in keys:
        if substring in s:
            return s
    return None