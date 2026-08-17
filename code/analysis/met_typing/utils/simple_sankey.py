# -*- coding: utf-8 -*-
"""
Produces simple Sankey Diagrams with matplotlib.
@author: Anneya Golob & marcomanz & pierre-sassoulas & jorwoods
                      .-.
                 .--.(   ).--.
      <-.  .-.-.(.->          )_  .--.
       `-`(     )-'             `)    )
         (o  o  )                `)`-'
        (      )                ,)
        ( ()  )                 )
         `---"\    ,    ,    ,/`
               `--' `--' `--'
                |  |   |   |
                |  |   |   |
                '  |   '   |

Modified 2018 Nathan Gouwens

"""

from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


class PySankeyException(Exception):
    pass


class NullsInFrame(PySankeyException):
    pass


class LabelMismatch(PySankeyException):
    pass


def check_data_matches_labels(labels, data, side):
    if len(labels) > 0:
        if isinstance(data, list):
            data = set(data)
        if isinstance(data, pd.Series):
            data = set(data.unique().tolist())
        if isinstance(labels, list):
            labels = set(labels)
        if labels != data:
            msg = "\n"
            if len(labels) <= 20:
                msg = "Labels: " + ",".join(labels) + "\n"
            if len(data) < 20:
                msg += "Data: " + ",".join(data)
            raise LabelMismatch('{0} labels and data do not match.{1}'.format(side, msg))


def arrange_labels(df, left_labels, right_labels):
    # connections between left and right nodes
    mat = pd.crosstab(df["left"], df["right"]).loc[left_labels, right_labels].values
    n_left = len(left_labels)
    n_right = len(right_labels)
    all_labels = np.concatenate([left_labels, right_labels])

    # Form a graph
    # only left to right connections are present
    graph_mat = np.zeros((n_left + n_right, n_left + n_right))
    graph_mat[:n_left, n_left:] = mat
    graph = csr_matrix(graph_mat)

    # First separate by connected components
    n_components, labels = connected_components(csgraph=graph,
        directed=False, return_labels=True)

    new_left = []
    new_right = []
    left_mask = np.zeros_like(all_labels, dtype=bool)
    left_mask[:n_left] = True
    right_mask = np.zeros_like(all_labels, dtype=bool)
    right_mask[n_left:] = True

    new_left = []
    new_right = []
    for i in range(n_components):
        cc_mask = labels == i
        left_cc = all_labels[cc_mask & left_mask]
        right_cc = all_labels[cc_mask & right_mask]

        ind = np.flatnonzero(cc_mask)
        left_ind = ind[ind < n_left]
        right_ind = ind[ind >= n_left] - n_left

        # Reorder within connected component to reduce crossings
        # by barycenter heuristic
        submat = mat[left_ind, :][:, right_ind]
        row_ind, col_ind = barycenter_heuristic(submat)

        new_left += left_cc[row_ind].tolist()
        new_right += right_cc[col_ind].tolist()
    return new_left, new_right


def crossing_count(mat):
    adj_mat = mat > 0

    n = 0
    for i in range(adj_mat.shape[0]):
        for j in range(i + 1, adj_mat.shape[0]):
            for tau in range(adj_mat.shape[1]):
                for rho in range(tau + 1, adj_mat.shape[1]):
                    if adj_mat[j, tau] and adj_mat[i, rho]:
                        n += mat[j, tau] * mat[i, rho]
    return n


def barycenter_heuristic(mat, attempts=20):
    n_row, n_col = mat.shape
    row_ind = np.arange(n_row)
    col_ind = np.arange(n_col)

    keep_going = True
    best_k = np.inf
    while attempts > 0:

        bc_cols = (np.arange(n_row)[:, np.newaxis] * mat).sum(axis=0) / mat.sum(axis=0)
        randomizer = np.random.random(len(bc_cols))
        col_sorter = np.lexsort((randomizer, bc_cols))
        new_mat = mat[:, col_sorter].copy()

        bc_rows = (np.arange(n_col)[np.newaxis, :] * new_mat).sum(axis=1) / new_mat.sum(axis=1)
        randomizer = np.random.random(len(bc_rows))
        row_sorter = np.lexsort((randomizer, bc_rows))
        new_mat = new_mat[row_sorter, :]

        k = crossing_count(new_mat)

        if k > best_k:
            attempts -= 1
            continue
        elif k == best_k:
            attempts -= 1
        else:
            best_k = k

        mat = new_mat.copy()
        row_ind = row_ind[row_sorter]
        col_ind = col_ind[col_sorter]

        if k == 0:
            break

    return row_ind, col_ind



def sankey(left, right, leftWeight=None, rightWeight=None, colorDict=None,
           leftLabels=None, rightLabels=None, aspect=4, rightColor=False,
           fontsize=14, returnLabels=False, rearrange=False,
           orientation="vertical",
           ax=None):
    '''
    Make Sankey Diagram showing flow from left-->right
    Inputs:
        left = NumPy array of object labels on the left of the diagram
        right = NumPy array of corresponding labels on the right of the diagram
            len(right) == len(left)
        leftWeight = NumPy array of weights for each strip starting from the
            left of the diagram, if not specified 1 is assigned
        rightWeight = NumPy array of weights for each strip starting from the
            right of the diagram, if not specified the corresponding leftWeight
            is assigned
        colorDict = Dictionary of colors to use for each label
            {'label':'color'}
        leftLabels = order of the left labels in the diagram
        rightLabels = order of the right labels in the diagram
        aspect = vertical extent of the diagram in units of horizontal extent
        rightColor = If true, each strip in the diagram will be be colored
                    according to its left label
        fontsize = font size of labels
        rearrange = switch order to reduce crossings
        ax = Axes object to draw the plot onto, otherwise uses the current Axes
    Ouput:
        None
    '''

    if ax is None:
        ax = plt.gca()

    if leftWeight is None:
        leftWeight = []
    if rightWeight is None:
        rightWeight = []
    if leftLabels is None:
        leftLabels = []
    if rightLabels is None:
        rightLabels = []
    # Check weights
    if len(leftWeight) == 0:
        leftWeight = np.ones(len(left))

    if len(rightWeight) == 0:
        rightWeight = leftWeight


    # Create Dataframe
    if isinstance(left, pd.Series):
        left = left.reset_index(drop=True)
    if isinstance(right, pd.Series):
        right = right.reset_index(drop=True)
    dataFrame = pd.DataFrame({'left': left, 'right': right, 'leftWeight': leftWeight,
                              'rightWeight': rightWeight}, index=range(len(left)))

    if len(dataFrame[(dataFrame.left.isnull()) | (dataFrame.right.isnull())]):
        raise NullsInFrame('Sankey graph does not support null values.')

    # Identify all labels that appear 'left' or 'right'
    allLabels = pd.Series(np.r_[dataFrame.left.unique(), dataFrame.right.unique()]).unique()

    # Identify left labels
    if len(leftLabels) == 0:
        leftLabels = pd.Series(dataFrame.left.unique()).unique()
    else:
        check_data_matches_labels(leftLabels, dataFrame['left'], 'left')

    # Identify right labels
    if len(rightLabels) == 0:
        rightLabels = pd.Series(dataFrame.right.unique()).unique()
    else:
        check_data_matches_labels(rightLabels, dataFrame['right'], 'right')

    if rearrange:
        leftLabels, rightLabels = arrange_labels(dataFrame, leftLabels, rightLabels)

    # If no colorDict given, make one
    if colorDict is None:
        colorDict = {}
        palette = "hls"
        colorPalette = sns.color_palette(palette, len(allLabels))
        for i, label in enumerate(allLabels):
            colorDict[label] = colorPalette[i]
    else:
        missing = [label for label in allLabels if label not in colorDict.keys()]
        if missing:
            msg = "The colorDict parameter is missing values for the following labels : "
            msg += '{}'.format(', '.join(missing))
            raise ValueError(msg)

    # Determine widths of individual strips
    ns_l = defaultdict()
    ns_r = defaultdict()
    for leftLabel in leftLabels:
        leftDict = {}
        rightDict = {}
        for rightLabel in rightLabels:
            leftDict[rightLabel] = dataFrame[(dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)].leftWeight.sum()
            rightDict[rightLabel] = dataFrame[(dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)].rightWeight.sum()
        ns_l[leftLabel] = leftDict
        ns_r[leftLabel] = rightDict

    # Determine positions of left label patches and total widths
    leftWidths = defaultdict()
    total_gap_size = dataFrame.leftWeight.sum() * 0.2
    if len(leftLabels) > 1:
        gap_size = total_gap_size / ((len(leftLabels) - 1))
    else:
        gap_size = total_gap_size

    for i, leftLabel in enumerate(leftLabels):
        myD = {}
        myD['left'] = dataFrame[dataFrame.left == leftLabel].leftWeight.sum()
        if i == 0:
            myD['bottom'] = 0
            myD['top'] = myD['left']
        else:
#             myD['bottom'] = leftWidths[leftLabels[i - 1]]['top'] + 0.02 * dataFrame.leftWeight.sum()
            myD['bottom'] = leftWidths[leftLabels[i - 1]]['top'] + gap_size
            myD['top'] = myD['bottom'] + myD['left']
            topEdge = myD['top']
        leftWidths[leftLabel] = myD

    # Determine positions of right label patches and total widths
    rightWidths = defaultdict()
    total_gap_size = dataFrame.rightWeight.sum() * 0.2
    if len(rightLabels) > 1:
        gap_size = total_gap_size / ((len(rightLabels) - 1))
    else:
        gap_size = total_gap_size
    for i, rightLabel in enumerate(rightLabels):
        myD = {}
        myD['right'] = dataFrame[dataFrame.right == rightLabel].rightWeight.sum()
        if i == 0:
            myD['bottom'] = 0
            myD['top'] = myD['right']
        else:
#             myD['bottom'] = rightWidths[rightLabels[i - 1]]['top'] + 0.02 * dataFrame.rightWeight.sum()
            myD['bottom'] = rightWidths[rightLabels[i - 1]]['top'] + gap_size
            myD['top'] = myD['bottom'] + myD['right']
            topEdge = myD['top']
        rightWidths[rightLabel] = myD

    # Total vertical extent of diagram
    xMax = topEdge / aspect

    if returnLabels:
        left_labels_info = []
        right_labels_info = []

    # Draw vertical bars on left and right of each  label's section & print label
    for leftLabel in leftLabels:
        if orientation == "vertical":
            ax.fill_between(
                [-0.02 * xMax, 0],
                2 * [leftWidths[leftLabel]['bottom']],
                2 * [leftWidths[leftLabel]['bottom'] + leftWidths[leftLabel]['left']],
                color=colorDict[leftLabel],
                alpha=0.99
            )
            if returnLabels:
                left_labels_info.append({
                    "text": leftLabel,
                    "x": -0.05 * xMax,
                    "y": leftWidths[leftLabel]['bottom'] + 0.5 * leftWidths[leftLabel]['left'],
                    "top_edge": leftWidths[leftLabel]['top'],
                })
            else:
                ax.text(
                    -0.05 * xMax,
                    leftWidths[leftLabel]['bottom'] + 0.5 * leftWidths[leftLabel]['left'],
                    leftLabel,
                    {'ha': 'right', 'va': 'center'},
                    fontsize=fontsize
                )
        elif orientation == "horizontal":
            ax.fill_betweenx(
                [-0.02 * xMax, 0],
                2 * [leftWidths[leftLabel]['bottom']],
                2 * [leftWidths[leftLabel]['bottom'] + leftWidths[leftLabel]['left']],
                color=colorDict[leftLabel],
                alpha=0.99
            )
            if returnLabels:
                left_labels_info.append({
                    "text": leftLabel,
                    "x": leftWidths[leftLabel]['bottom'] + 0.5 * leftWidths[leftLabel]['left'],
                    "y": -0.05 * xMax,
                    "top_edge": leftWidths[leftLabel]['top'],
                })
            else:
                ax.text(
                    leftWidths[leftLabel]['bottom'] + 0.5 * leftWidths[leftLabel]['left'],
                    -0.05 * xMax,
                    leftLabel,
                    {'ha': 'center', 'va': 'top'},
                    fontsize=fontsize,
                    rotation=90
                )

    for rightLabel in rightLabels:
        if orientation == "vertical":
            ax.fill_between(
                [xMax, 1.02 * xMax], 2 * [rightWidths[rightLabel]['bottom']],
                2 * [rightWidths[rightLabel]['bottom'] + rightWidths[rightLabel]['right']],
                color=colorDict[rightLabel],
                alpha=0.99
            )
            if returnLabels:
                right_labels_info.append({
                    "text": rightLabel,
                    "x": 1.05 * xMax,
                    "y": rightWidths[rightLabel]['bottom'] + 0.5 * rightWidths[rightLabel]['right'],
                    "top_edge": rightWidths[rightLabel]['top'],
                })
            else:
                ax.text(
                    1.05 * xMax,
                    rightWidths[rightLabel]['bottom'] + 0.5 * rightWidths[rightLabel]['right'],
                    rightLabel,
                    {'ha': 'left', 'va': 'center'},
                    fontsize=fontsize
                )
        elif orientation == "horizontal":
            ax.fill_betweenx(
                [xMax, 1.02 * xMax], 2 * [rightWidths[rightLabel]['bottom']],
                2 * [rightWidths[rightLabel]['bottom'] + rightWidths[rightLabel]['right']],
                color=colorDict[rightLabel],
                alpha=0.99
            )
            if returnLabels:
                right_labels_info.append({
                    "text": rightLabel,
                    "x": rightWidths[rightLabel]['bottom'] + 0.5 * rightWidths[rightLabel]['right'],
                    "y": 1.05 * xMax,
                    "top_edge": rightWidths[rightLabel]['top'],
                })
            else:
                ax.text(
                    rightWidths[rightLabel]['bottom'] + 0.5 * rightWidths[rightLabel]['right'],
                    1.05 * xMax,
                    rightLabel,
                    {'ha': 'center', 'va': 'bottom'},
                    fontsize=fontsize,
                    rotation=90
                )

    # Plot strips
    for leftLabel in leftLabels:
        for rightLabel in rightLabels:
            labelColor = leftLabel
            if rightColor:
                labelColor = rightLabel
            if len(dataFrame[(dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)]) > 0:
                # Create array of y values for each strip, half at left value,
                # half at right, convolve
                ys_d = np.array(50 * [leftWidths[leftLabel]['bottom']] + 50 * [rightWidths[rightLabel]['bottom']])
                ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode='valid')
                ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode='valid')
                ys_u = np.array(50 * [leftWidths[leftLabel]['bottom'] + ns_l[leftLabel][rightLabel]] + 50 * [rightWidths[rightLabel]['bottom'] + ns_r[leftLabel][rightLabel]])
                ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode='valid')
                ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode='valid')

                # Update bottom edges at each label so next strip starts at the right place
                leftWidths[leftLabel]['bottom'] += ns_l[leftLabel][rightLabel]
                rightWidths[rightLabel]['bottom'] += ns_r[leftLabel][rightLabel]
                if orientation == "vertical":
                    ax.fill_between(
                        np.linspace(0, xMax, len(ys_d)), ys_d, ys_u, alpha=0.65,
                        color=colorDict[labelColor]
                    )
                elif orientation == "horizontal":
                    ax.fill_betweenx(
                        np.linspace(0, xMax, len(ys_d)), ys_d, ys_u, alpha=0.65,
                        color=colorDict[labelColor]
                    )
    ax.axis('off')
    if returnLabels:
        return left_labels_info, right_labels_info


def scale_r(frac, r_inner, r_outer):
    return frac * (r_outer - r_inner) + r_inner

def scale_theta(frac, theta_min=0, theta_max=2 * np.pi - 2 * (np.pi / 180)):
     return frac * (theta_max - theta_min) + theta_min

def sankey_polar(left, right, leftWeight=None, rightWeight=None, colorDict=None,
           leftLabels=None, rightLabels=None, aspect=4, rightColor=False,
           fontsize=14, returnLabels=False, rearrange=False,
           r_inner=1, r_outer=2, bar_fraction=0.02, return_labels=False,
           ax=None):
    '''
    Make circular Sankey Diagram showing flow from inner -> outer
    Inputs:
        left = NumPy array of object labels on the left of the diagram
        right = NumPy array of corresponding labels on the right of the diagram
            len(right) == len(left)
        leftWeight = NumPy array of weights for each strip starting from the
            left of the diagram, if not specified 1 is assigned
        rightWeight = NumPy array of weights for each strip starting from the
            right of the diagram, if not specified the corresponding leftWeight
            is assigned
        colorDict = Dictionary of colors to use for each label
            {'label':'color'}
        leftLabels = order of the left labels in the diagram
        rightLabels = order of the right labels in the diagram
        aspect = vertical extent of the diagram in units of horizontal extent
        rightColor = If true, each strip in the diagram will be be colored
                    according to its left label
        fontsize = font size of labels
        rearrange = switch order to reduce crossings
        ax = Axes object to draw the plot onto, otherwise uses the current Axes
    Ouput:
        None
    '''

    if ax is None:
        ax = plt.subplot(111, projection='polar')

    if leftWeight is None:
        leftWeight = []
    if rightWeight is None:
        rightWeight = []
    if leftLabels is None:
        leftLabels = []
    if rightLabels is None:
        rightLabels = []
    # Check weights
    if len(leftWeight) == 0:
        leftWeight = np.ones(len(left))

    if len(rightWeight) == 0:
        rightWeight = leftWeight


    # Create Dataframe
    if isinstance(left, pd.Series):
        left = left.reset_index(drop=True)
    if isinstance(right, pd.Series):
        right = right.reset_index(drop=True)
    dataFrame = pd.DataFrame({'left': left, 'right': right, 'leftWeight': leftWeight,
                              'rightWeight': rightWeight}, index=range(len(left)))

    if len(dataFrame[(dataFrame.left.isnull()) | (dataFrame.right.isnull())]):
        raise NullsInFrame('Sankey graph does not support null values.')

    # Identify all labels that appear 'left' or 'right'
    allLabels = pd.Series(np.r_[dataFrame.left.unique(), dataFrame.right.unique()]).unique()

    # Identify left labels
    if len(leftLabels) == 0:
        leftLabels = pd.Series(dataFrame.left.unique()).unique()
    else:
        check_data_matches_labels(leftLabels, dataFrame['left'], 'left')

    # Identify right labels
    if len(rightLabels) == 0:
        rightLabels = pd.Series(dataFrame.right.unique()).unique()
    else:
        check_data_matches_labels(rightLabels, dataFrame['right'], 'right')

    if rearrange:
        leftLabels, rightLabels = arrange_labels(dataFrame, leftLabels, rightLabels)

    # If no colorDict given, make one
    if colorDict is None:
        colorDict = {}
        palette = "hls"
        colorPalette = sns.color_palette(palette, len(allLabels))
        for i, label in enumerate(allLabels):
            colorDict[label] = colorPalette[i]
    else:
        missing = [label for label in allLabels if label not in colorDict.keys()]
        if missing:
            msg = "The colorDict parameter is missing values for the following labels : "
            msg += '{}'.format(', '.join(missing))
            raise ValueError(msg)

    # Determine widths of individual strips
    ns_l = defaultdict()
    ns_r = defaultdict()
    for leftLabel in leftLabels:
        leftDict = {}
        rightDict = {}
        for rightLabel in rightLabels:
            leftDict[rightLabel] = dataFrame[(dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)].leftWeight.sum()
            rightDict[rightLabel] = dataFrame[(dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)].rightWeight.sum()
        ns_l[leftLabel] = leftDict
        ns_r[leftLabel] = rightDict

    # Determine positions of left label patches and total widths
    leftWidths = defaultdict()
    total_gap_size = dataFrame.leftWeight.sum() * 0.2
    if len(leftLabels) > 1:
        gap_size = total_gap_size / ((len(leftLabels) - 1))
    else:
        gap_size = total_gap_size

    for i, leftLabel in enumerate(leftLabels):
        myD = {}
        myD['left'] = dataFrame[dataFrame.left == leftLabel].leftWeight.sum()
        if i == 0:
            myD['bottom'] = 0
            myD['top'] = myD['left']
        else:
#             myD['bottom'] = leftWidths[leftLabels[i - 1]]['top'] + 0.02 * dataFrame.leftWeight.sum()
            myD['bottom'] = leftWidths[leftLabels[i - 1]]['top'] + gap_size
            myD['top'] = myD['bottom'] + myD['left']
            topEdge = myD['top']
        leftWidths[leftLabel] = myD

    # Determine positions of right label patches and total widths
    rightWidths = defaultdict()
    total_gap_size = dataFrame.rightWeight.sum() * 0.2
    if len(rightLabels) > 1:
        gap_size = total_gap_size / ((len(rightLabels) - 1))
    else:
        gap_size = total_gap_size
    for i, rightLabel in enumerate(rightLabels):
        myD = {}
        myD['right'] = dataFrame[dataFrame.right == rightLabel].rightWeight.sum()
        if i == 0:
            myD['bottom'] = 0
            myD['top'] = myD['right']
        else:
#             myD['bottom'] = rightWidths[rightLabels[i - 1]]['top'] + 0.02 * dataFrame.rightWeight.sum()
            myD['bottom'] = rightWidths[rightLabels[i - 1]]['top'] + gap_size
            myD['top'] = myD['bottom'] + myD['right']
            topEdge = myD['top']
        rightWidths[rightLabel] = myD

    # Total vertical extent of diagram
    xMax = topEdge / aspect

    if returnLabels:
        left_labels_info = []
        right_labels_info = []

    # Draw vertical bars on left and right of each  label's section & print label
    min_x = -bar_fraction * xMax
    max_x = (1 + bar_fraction) * xMax

    label_positions = {}
    for leftLabel in leftLabels: # left becomes inner
        # Original coordinates
        # x = np.array([-0.02 * xMax, 0])
        # y_lower = np.array(2 * [leftWidths[leftLabel]['bottom']])
        # y_upper = np.array(2 * [leftWidths[leftLabel]['bottom'] + leftWidths[leftLabel]['left']])

        # Polar transformation
        n_points = 100
        inner = np.array(n_points * [scale_r((min_x - min_x) / (max_x - min_x), r_inner, r_outer)])
        outer = np.array(n_points * [scale_r((0 - min_x) / (max_x - min_x), r_inner, r_outer)])
        min_theta = scale_theta(leftWidths[leftLabel]['bottom'] / topEdge)
        max_theta = scale_theta((leftWidths[leftLabel]['bottom'] + leftWidths[leftLabel]['left']) / topEdge)

        theta = np.linspace(min_theta, max_theta, n_points)

        ax.fill_between(
            theta,
            inner,
            outer,
            color=colorDict[leftLabel],
            alpha=0.99,
            zorder=20,
        )

        label_positions[leftLabel] = (min_theta + max_theta) / 2


    for rightLabel in rightLabels:
        # Original coordinates
        # x = xMax, 1.02 * xMax

        # Polar transformation
        n_points = 100
        inner = np.array(n_points * [scale_r((xMax - min_x) / (max_x - min_x), r_inner, r_outer)])
        outer = np.array(n_points * [scale_r((max_x - min_x) / (max_x - min_x), r_inner, r_outer)])
        min_theta = scale_theta(rightWidths[rightLabel]['bottom'] / topEdge)
        max_theta = scale_theta((rightWidths[rightLabel]['bottom'] + rightWidths[rightLabel]['right']) / topEdge)

        theta = np.linspace(min_theta, max_theta, n_points)

        ax.fill_between(
            theta,
            inner,
            outer,
            color=colorDict[rightLabel],
            alpha=0.99,
            zorder=20,
        )

        label_positions[rightLabel] = (min_theta + max_theta) / 2



    # Plot strips
    for leftLabel in leftLabels:
        for rightLabel in rightLabels:
            labelColor = leftLabel
            if rightColor:
                labelColor = rightLabel
            if len(dataFrame[(dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)]) > 0:
                # Create array of y values for each strip, half at left value,
                # half at right, convolve
                ys_d = np.array(50 * [leftWidths[leftLabel]['bottom']] + 50 * [rightWidths[rightLabel]['bottom']])
                ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode='valid')
                ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode='valid')
                ys_u = np.array(50 * [leftWidths[leftLabel]['bottom'] + ns_l[leftLabel][rightLabel]] + 50 * [rightWidths[rightLabel]['bottom'] + ns_r[leftLabel][rightLabel]])
                ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode='valid')
                ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode='valid')

                x = np.linspace(0, xMax, len(ys_d))

                theta_d = scale_theta(ys_d / topEdge)
                theta_u = scale_theta(ys_u / topEdge)

                leftWidths[leftLabel]['bottom'] += ns_l[leftLabel][rightLabel]
                rightWidths[rightLabel]['bottom'] += ns_r[leftLabel][rightLabel]

                r = scale_r(x / xMax, r_inner, r_outer)
                ax.fill_betweenx(
                    r,
                    theta_d,
                    theta_u,
                    alpha=0.35,
                    color=colorDict[labelColor],
                )


    ax.axis('off')
    if return_labels:
        return label_positions