"""
Core Utilities for Template Generation

This module contains core utility functions used across multiple visualization modules,
including type ordering, color manipulation, and text sorting.

Main Functions:
- get_tx_order_dict: Load and organize transcriptomic type ordering
- natural_sort_key: Natural (alphanumeric) sorting key function
- hex_to_gray: Convert hex colors to grayscale
- adjust_lightness: Adjust color brightness/saturation

Dependencies:
- Standard library: os, json, re
- Visualization: matplotlib.colors, colorsys
"""

import os
import json
import re
import pandas as pd
from pathlib import Path
from matplotlib.patches import PathPatch
import matplotlib.colors as mc
import colorsys


def get_tx_order_dict(subclass_order_file, ttype_order_file, ttype_to_subclass_file):
    """
    Load transcriptomic type order dictionary organized by subclass.

    Parameters
    ----------
    subclass_order_file : str
        Path to file containing ordered list of subclasses (one per line)
    ttype_order_file : str
        Path to file containing ordered list of types (one per line)

    Returns
    -------
    dict
        Dictionary mapping each subclass to its ordered list of types

    Notes
    -----
    Requires ttype_subclass_dict.json mapping file to link types to subclasses.
    """
    with open(ttype_to_subclass_file, 'r') as file:
        ttype_to_subclass_dict = json.load(file)

    with open(subclass_order_file, 'r') as file:
        subclass_order = [line.strip() for line in file]
    with open(ttype_order_file, 'r') as file:
        ttype_order = [line.strip() for line in file]

    tx_order_dict = {}
    for subclass in subclass_order:
        tx_order = [t for t in ttype_order if ttype_to_subclass_dict[t] == subclass]
        tx_order_dict[subclass] = tx_order

    return tx_order_dict, ttype_order, subclass_order


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    """
    Generate a sorting key for natural (alphanumeric) sorting.

    Parameters
    ----------
    s : str
        String to generate sort key for
    _nsre : re.Pattern, optional
        Compiled regex pattern for extracting numbers

    Returns
    -------
    list
        List of mixed int/str elements for natural sorting

    Examples
    --------
    >>> sorted(['L2', 'L10', 'L1'], key=natural_sort_key)
    ['L1', 'L2', 'L10']
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


def hex_to_gray(hexcol, method='601'):
    """
    Convert hex color to grayscale hex.

    Parameters
    ----------
    hexcol : str
        Hex color string like '#aabbcc', 'aabbcc', or shorthand '#abc'
    method : {'601', '709', 'avg'}, optional
        Luminance calculation method:
        - '601': ITU-R BT.601 (default, NTSC)
        - '709': ITU-R BT.709 (HDTV)
        - 'avg': Simple average of RGB

    Returns
    -------
    str
        Grayscale hex string '#RRGGBB'

    Examples
    --------
    >>> hex_to_gray('#ff0000')  # red
    '#4c4c4c'
    >>> hex_to_gray('#00ff00')  # green
    '#969696'
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


def adjust_lightness(color, amount=0.5):
    """
    Adjust the lightness/brightness of a color.

    Parameters
    ----------
    color : str
        Color name or hex string
    amount : float, optional
        Lightness multiplier (default=0.5)
        - amount < 1: darker
        - amount = 1: unchanged
        - amount > 1: lighter

    Returns
    -------
    tuple
        RGB tuple (r, g, b) with values in [0, 1]

    Examples
    --------
    >>> adjust_lightness('#ff0000', 0.5)  # darker red
    >>> adjust_lightness('blue', 1.5)     # lighter blue
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def get_seaad_colors():
    """
    Load SEAAD color palette from Excel file.

    Returns
    -------
    dict
        Mapping of cell type labels to hex color codes

    Notes
    -----
    Loads colors from SEAAD_colors.xlsx file containing supertype color schemes.
    """
    seaad_colors_path = Path('/root/capsule/data/colors/SEAAD_colors.xlsx')
    seaad_colors = pd.read_excel(seaad_colors_path, sheet_name='in')
    seaad_colors = seaad_colors.rename(columns={'Unnamed: 0':'label'})
    color_dict = dict(zip(seaad_colors['label'], seaad_colors['supertype_scANVI_leiden_colors']))
    return color_dict


def get_tx_order():
    """
    Load transcriptomic type ordering from files.

    Returns
    -------
    tuple
        (ttype_order, subclass_order) where each is a list of ordered labels

    Notes
    -----
    Loads ordering from text files containing one label per line.
    Different from get_tx_order_dict which returns a dictionary structure.
    """
    with open(Path('/root/capsule/data/formatting_dicts/subclass_order.txt'), 'r') as f:
        subclass_order = f.read().splitlines()

    with open(Path('/root/capsule/data/formatting_dicts/ttype_order.txt'), 'r') as f:
        ttype_order = f.read().splitlines()

    return ttype_order, subclass_order


def format_box_plot(ax):
    """
    Format matplotlib box plots with outlined style.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis containing box plot to format

    Notes
    -----
    Converts filled box plots to outlined style by:
    - Setting face color to None
    - Setting edge color to match original face color
    - Applying same color to whiskers, fliers, and other elements
    Compatible with matplotlib versions before and after 3.5.
    """
    box_patches = [patch for patch in ax.patches if type(patch) == PathPatch]
    if len(box_patches) == 0:  # in matplotlib older than 3.5, the boxes are stored in ax.artists
        box_patches = ax.artists
    num_patches = len(box_patches)
    if num_patches == 0:
        return  # Nothing to format
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


def find_ttype_key(keys, substring):
    """
    Find first key containing substring.

    Parameters
    ----------
    keys : list
        List of strings to search
    substring : str
        Substring to find

    Returns
    -------
    str or None
        First matching key containing substring, or None if not found

    Examples
    --------
    >>> find_ttype_key(['L2/3 IT', 'L4 IT', 'L5 IT'], 'L4')
    'L4 IT'
    """
    for s in keys:
        if substring in s:
            return s
    return None
