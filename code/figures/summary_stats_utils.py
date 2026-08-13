"""
Summary Statistics and Metadata Visualization Utilities

This module contains functions for visualizing summary statistics, metadata,
and effect sizes for transcriptomic types.

Main Functions:
- plot_donor_lobe_counts: Visualize donor cortical region counts
- plot_subclass_eta2_topn: Plot top N features by eta-squared effect size

Dependencies:
- Data: pandas
- Visualization: matplotlib, seaborn
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl


def plot_donor_lobe_counts(ax, df, color_dict):
    """
    Plot count of specimens by cortical lobe/region.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    df : pandas.DataFrame
        Data with 'lobe' column
    color_dict : dict
        Mapping of lobe names to colors

    Notes
    -----
    Creates a vertical countplot ordered by frequency.
    Labels are colored by lobe and rotated 90 degrees.
    """
    # Calculate the order based on counts
    order = df['lobe'].value_counts().index

    # Create the countplot with the specified order
    sns.countplot(x='lobe', data=df, order=order, ax=ax, palette=color_dict)

    ax.set_title('Donor cortical region', fontsize=6)
    ax.set_ylabel('Donor count', fontsize=5)
    ax.set_xlabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')

    # Format axes
    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(axis="both", labelsize=5)
    ax.tick_params(axis='both', colors='grey')
    ax.spines['bottom'].set_color('grey')
    ax.spines['left'].set_color('grey')

    # Color lobe labels
    for tick_label in ax.get_xticklabels():
        label_text = tick_label.get_text()
        if label_text in color_dict:
            tick_label.set_color(color_dict[label_text])


def plot_subclass_eta2_topn(eta_squared_ttypes_df, ax, n=10, label_right=True):
    """
    Plot top N features by eta-squared (effect size).

    Shows which features most strongly differentiate between types,
    based on the proportion of variance explained (eta²).

    Parameters
    ----------
    eta_squared_ttypes_df : pandas.DataFrame
        DataFrame with columns:
        - 'feature': feature name
        - 'eta_squared': effect size (0-1)
    ax : matplotlib.axes.Axes
        Axis to plot on
    n : int, optional
        Number of top features to show (default=10)
    label_right : bool, optional
        If True, place y-axis labels on right side (default=True)

    Notes
    -----
    Creates a horizontal barplot colored by a sequential palette.
    Eta-squared (η²) represents the proportion of variance in a feature
    explained by group membership.
    """
    eta_squared_ttypes_df = eta_squared_ttypes_df.nlargest(n, 'eta_squared')

    # Color palette
    values = sns.color_palette("ch:s=-.2,r=.6", len(eta_squared_ttypes_df))
    keys = eta_squared_ttypes_df.feature
    color_palette = dict(zip(keys, values))

    # Plot
    sns.set_theme(style="white")
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True

    sns.barplot(x="eta_squared", y='feature', data=eta_squared_ttypes_df, palette=color_palette, ax=ax)
    sns.despine(ax=ax)

    # Edit axis labels
    ax.set_xlabel("\u03B7\u00b2", size=6)  # η²
    ax.set_ylabel("")
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['0', '0.5', '1'], size=6)

    if label_right:
        # Move y-axis to the right
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        label_ha = 'left'
        ax.tick_params(axis='y', which='both', left=False, right=False,
                      labelleft=False, labelright=True)
    else:
        label_ha = 'right'
        ax.tick_params(axis='y', which='both', left=False, right=False,
                      labelleft=True, labelright=False)

    ax.set_yticklabels(ax.get_yticklabels(), ha=label_ha, fontsize=6)

    # Set axis and ticks to grey
    ax.tick_params(axis='x', colors='grey')
    for spine in ['bottom', 'top', 'left', 'right']:
        ax.spines[spine].set_color('grey')


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


def plot_confusion_matrix(ax, cm, labels, color_dict, title="",
                          num_cell_xbuffer=0.4, n_height=0.015,
                          fontsize_diagonal=15, fontsize_counts=15, fontsize_labels=12,
                          fontsize_title=12,
                          fontsize_ticks=10, cmap="rocket", cbar=True):
    """
    Plot row-normalized confusion matrix on existing axis.

    This function creates a standardized confusion matrix visualization
    matching the style used in the classifier notebooks. It handles
    safe normalization (avoiding divide-by-zero), diagonal annotation
    with accuracy percentages, cell count annotations, and color-coded
    tick labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on (must be provided, not created)
    cm : np.ndarray
        Raw confusion matrix (n_labels × n_labels)
        Rows represent true labels, columns represent predicted labels
    labels : list
        Ordered label names (for tick labels)
    color_dict : dict
        Mapping {label: hex_color} for coloring tick labels
    title : str, optional
        Title text for plot (default="")
    num_cell_xbuffer : float, optional
        Horizontal offset for count annotations relative to matrix width
        (default=0.4)
    n_height : float, optional
        Vertical offset for 'n' header annotation (default=0.015)
    fontsize_diagonal : int, optional
        Font size for diagonal accuracy percentages (default=15)
    fontsize_counts : int, optional
        Font size for count annotations on right side (default=15)
    fontsize_ticks : int, optional
        Font size for axis tick labels (default=10)
    cmap : str, optional
        Seaborn colormap name (default='rocket')

    Returns
    -------
    dict
        Dictionary containing:
        - 'normalized_cm': row-normalized confusion matrix (np.ndarray)
        - 'row_acc': per-class accuracy from diagonal (np.ndarray)

    Notes
    -----
    Styling details matching original classifier notebooks:
    - Heatmap: rocket colormap, vmin=0, vmax=1, square cells
    - Normalization: by row (true class), using safe divide to handle
      labels with zero test samples
    - Diagonal annotations: white text if accuracy <50%, black if ≥50%
    - Format: {val*100:.0f} (no decimal places)
    - Count annotations: right side of matrix, grey color, italic 'n' header
    - Tick labels: colored by color_dict
    - Axis labels: 'Predicted' (x), 'Actual' (y), fontsize=12
    - Y-axis ticks: hidden (left=False, right=False)
    - Colorbar: enabled with pad=0.08

    The safe normalization uses np.divide with where clause:
    - Returns 0.0 for labels with no test samples (avoids NaN/Inf)
    - Normalized values represent per-class recall (row-wise accuracy)

    Examples
    --------
    >>> fig, ax = plt.subplots(figsize=(6, 6))
    >>> cm_raw = confusion_matrix(y_test, y_pred, labels=label_list)
    >>> result = plot_confusion_matrix(
    ...     ax=ax,
    ...     cm=cm_raw,
    ...     labels=label_list,
    ...     color_dict=seaad_colors,
    ...     title='Morphology Classifier'
    ... )
    >>> plt.tight_layout()
    >>> plt.savefig('confusion_matrix.pdf')
    """
    import seaborn as sns

    # Safe row normalization (avoid divide-by-zero)
    true_counts = np.sum(cm, axis=1)
    normalized_cm = np.divide(
        cm,
        true_counts[:, np.newaxis],
        out=np.zeros_like(cm, dtype=float),
        where=true_counts[:, np.newaxis] != 0
    )

    # Extract diagonal (per-class accuracy)
    row_acc = np.diag(normalized_cm)

    # Plot heatmap
    sns.heatmap(
        normalized_cm,
        annot=False,
        fmt=".2f",
        cmap=cmap,
        vmin=0,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        cbar=cbar,
        cbar_kws={"pad": 0.8},
        ax=ax
    )

    # Annotate diagonal with accuracy percentages
    for i in range(len(labels)):
        val = row_acc[i]
        # Only annotate if we have samples for this label
        if true_counts[i] > 0:
            ax.text(
                i + 0.5,
                i + 0.5,
                f"{val*100:.0f}",
                ha="center",
                va="center",
                fontsize=fontsize_diagonal,
                color="white" if val < 0.5 else "black"
            )

    # Axis labels
    ax.set_xlabel('Predicted', fontsize=fontsize_labels)
    ax.set_ylabel('Actual', fontsize=fontsize_labels)

    # Set title if provided
    if title:
        ax.set_title(title, fontsize=fontsize_title)

    # Tick label formatting
    ax.tick_params(axis='x', labelsize=fontsize_ticks, rotation=90)
    ax.tick_params(axis='y', labelsize=fontsize_ticks, rotation=0)

    # Annotate cell counts on right side
    for i in range(len(labels)):
        ax.text(
            len(labels) + num_cell_xbuffer,
            i + 0.5,
            int(true_counts[i]),
            ha="center",
            va="center",
            fontsize=fontsize_counts,
            color='grey'
        )

    # Add 'n' header for counts column
    ax.text(
        len(labels) + num_cell_xbuffer,
        n_height,
        'n',
        ha="center",
        va="center",
        fontsize=fontsize_counts,
        color='grey',
        fontstyle='italic'
    )

    # Remove y-axis ticks (keep labels)
    ax.tick_params(axis='y', which='both', left=False, right=False)

    # Color tick labels by cell type
    for label in ax.get_xticklabels():
        text = label.get_text()
        if text in color_dict:
            label.set_color(color_dict[text])

    for label in ax.get_yticklabels():
        text = label.get_text()
        if text in color_dict:
            label.set_color(color_dict[text])

    return {
        'normalized_cm': normalized_cm,
        'row_acc': row_acc
    }


def plot_cbar(ax):

    ax.axis("off")

    ax_cbar = ax.inset_axes([
        -0.7,  # left
        0.05,  # bottom
        0.10,  # width
        0.90,  # height
    ])

    cmap = sns.color_palette("rocket", as_cmap=True)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = plt.colorbar(sm, cax=ax_cbar)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["0", "100"])
    cbar.ax.tick_params(labelsize=5)  
    cbar.outline.set_visible(False)

    for spine in ax_cbar.spines.values():
        spine.set_visible(False)


def plot_abundance_stacked_bar(
    df,
    ax,
    color_dict,
    frac_col='supertype_frac_of_layer',
    title='',
    label_supertypes=True,
    height=0.2,
    vertical=False):
    """
    Plot supertype abundance as a stacked bar chart.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'supertype' column and fraction column.
        Must be sorted by 'supertype' categorical to ensure correct order.
    ax : matplotlib.axes.Axes
        Axis to plot on.
    color_dict : dict
        Mapping of supertype names to colors.
    frac_col : str
        Column containing proportions (0-1 range).
    title : str
        Title for the plot.
    label_supertypes : bool
        Whether to show labels for each segment.
    height : float
        Thickness of the stacked bar.
    vertical : bool
        If False, plot left-to-right.
        If True, plot bottom-to-top.
    """

    categories = df['supertype'].values
    proportions = df[frac_col].values
    colors = [color_dict[cat] for cat in categories]

    position = 0

    for cat, prop, color in zip(categories, proportions, colors):

        if vertical:
            # Bottom-to-top stacked bar
            ax.bar(
                0,
                prop,
                bottom=position,
                width=height,
                color=color,
                edgecolor='white',
                linewidth=0
            )

            if label_supertypes:
                text_x = -0.2
                text_y = position + prop / 2

                ax.text(
                    text_x,
                    text_y,
                    f"{cat}\n{prop:.3f}",
                    ha='center',
                    va='center',
                    fontsize=5,
                    color=color,
                    weight='bold'
                )

        else:
            # Left-to-right stacked bar
            ax.barh(
                0,
                prop,
                left=position,
                height=height,
                color=color,
                edgecolor='white',
                linewidth=0
            )

            if label_supertypes:
                text_x = position + prop / 2
                text_y = -0.2

                ax.text(
                    text_x,
                    text_y,
                    f"{cat}\n{prop:.3f}",
                    ha='center',
                    va='center',
                    fontsize=5,
                    color=color,
                    weight='bold'
                )

        position += prop

    # Axis limits
    if vertical:
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.5, 0.5)
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)

    # Clean styling
    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title, size=6, pad=2)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlabel('')
    ax.set_ylabel('')