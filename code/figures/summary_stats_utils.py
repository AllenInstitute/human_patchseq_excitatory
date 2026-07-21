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
