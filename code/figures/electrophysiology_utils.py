"""
Electrophysiology Visualization Utilities

This module contains functions for visualizing electrophysiological properties
including action potentials, F-I curves, and subthreshold responses.

Main Functions:
- plot_ephys_subthresh: Subthreshold voltage responses
- plot_ephys_gallery: Gallery of ephys traces from SVG files
- plot_avg_fi_for_mets: F-I curve (firing rate vs stimulus) plots
- process_fi_curves: Process and bin F-I curve data
- plot_ap_subclass: Action potential traces by subclass
- plot_ap_ttype: Action potential traces by type
- align_to_peak: Helper for aligning AP traces to peak
- resample_around_peak: Helper for resampling aligned traces

Dependencies:
- Data: numpy, pandas
- Visualization: matplotlib, seaborn, svgpathtools, cairosvg
- Processing: scipy.ndimage
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

from scipy.ndimage import gaussian_filter1d
from svgpathtools import svg2paths

from core_utils import get_seaad_colors

SEAAD_COLORS = get_seaad_colors()


def plot_ephys_subthresh(ax, subthreshold_data, subthreshold_metadata, groups, group_column, color_dict):
    """
    Plot smoothed subthreshold voltage responses by group.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    subthreshold_data : dict
        Nested dictionary: {cell_name: {sweep_num: {'time', 'response', 'samp_rate'}}}
    subthreshold_metadata : pandas.DataFrame
        Metadata with columns: 'cell_specimen_name', group_column
    groups : list
        Groups to plot
    group_column : str
        Column name for grouping variable
    color_dict : dict
        Mapping of groups to colors

    Notes
    -----
    Plots mean normalized responses with Gaussian smoothing (sigma=10).
    Responses are baseline-subtracted and normalized to min value.
    High sampling rate data (200kHz) is downsampled 4x.
    """
    for group in groups:
        # Filter dataframe for current group
        group_cells = subthreshold_metadata[subthreshold_metadata[group_column] == group]['cell_specimen_name'].tolist()

        responses = []
        reference_time = None

        # Accumulate responses
        for cell in group_cells:
            if cell in subthreshold_data:
                cell_data = subthreshold_data[cell]
                sweepnums = list(cell_data.keys())
                sweepnum = sweepnums[-1] if len(sweepnums) > 1 else sweepnums[0]
                traces = cell_data[sweepnum]
                time = traces['time'] - traces['time'][0]
                response = traces['response']
                samp_rate = traces['samp_rate']

                if samp_rate == 200000.0:
                    response = response[::4]
                    time = time[::4]

                if reference_time is None:
                    reference_time = time

                baseline = np.mean(response[0:500])
                response0 = response - baseline
                norm_response = -(response0 / np.min(response0))
                responses.append(norm_response)

        # Plot mean responses
        if responses:
            mean_response = np.mean(responses, axis=0)
            smoothed_response = gaussian_filter1d(mean_response, sigma=10)
            ax.plot(reference_time * 1000, smoothed_response, color=color_dict[group], linewidth=0.5, label=group)

    # Final plot formatting
    ax.set_ylabel('Normalized voltage', size=5)
    ax.set_xlabel('Time (ms)', size=5)

    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(axis="both", labelsize=6)
    ax.tick_params(axis='both', colors='grey')
    ax.spines['bottom'].set_color('grey')
    ax.spines['left'].set_color('grey')


def plot_ephys_gallery(fig, ax_morpho_gallery, svg_root, ttypes, color_dict, end_xoffset=None, plot_scalebar=True):
    """
    Create gallery of ephys traces below morphology gallery.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure containing the galleries
    ax_morpho_gallery : matplotlib.axes.Axes
        Morphology gallery axis (for positioning)
    svg_root : str
        Root directory containing SVG trace files
    ttypes : list
        Types to plot (in order)
    color_dict : dict
        Mapping of types to colors
    end_xoffset : float, optional
        Fractional width to use (default: full width of morpho gallery)
    plot_scalebar : bool, optional
        If True, add scale bar at end (default=True)

    Notes
    -----
    Expects SVG files named: {type}_width.svg, {type}_rate.svg
    (with spaces replaced by underscores, slashes removed).
    Adds scale.svg at end if plot_scalebar=True.
    """
    valid_ttypes = []
    for ttype in ttypes:
        ttype_reformat = ttype.replace('/', '').replace(' ', '_')
        if os.path.exists(os.path.join(svg_root, f'{ttype_reformat}_width.svg')):
            valid_ttypes.append(ttype)

    n_cells = len(valid_ttypes)
    if plot_scalebar:
        n_cells += 1

    bbox = ax_morpho_gallery.get_position()
    gap = 0.01  # Small vertical gap
    height_ratio = 0.5  # Ephys height as fraction of morpho height

    if not end_xoffset:
        # If no x max given, use x width of ax
        end_xoffset = bbox.width

    for i, ttype in enumerate(valid_ttypes):
        width = end_xoffset / n_cells
        left = bbox.x0 + i * width
        bottom = bbox.y0 - gap - bbox.height * height_ratio
        height = bbox.height * height_ratio

        ax = fig.add_axes([left, bottom, width, height])
        ax.axis('off')

        ax.text(0.5, 1.05, ttype.replace('Car3', '\nCar3'), transform=ax.transAxes,
               ha='center', va='bottom', fontsize=5, color=color_dict[ttype], rotation=30)

        svg_file_root = ttype.replace('/', '').replace(' ', '_')
        plot_svg_on_ax(ax, os.path.join(svg_root, f'{svg_file_root}_width.svg'),
                      xoffset=0, yoffset=30, color=color_dict[ttype], invert_y=True)
        plot_svg_on_ax(ax, os.path.join(svg_root, f'{svg_file_root}_rate.svg'),
                      xoffset=0, yoffset=-10, color=color_dict[ttype], invert_y=True)

    if plot_scalebar:
        # Add scale bar at end of row
        scalebar_left = bbox.x0 + n_cells * width - 0.01
        scalebar_bottom = bbox.y0 - gap - bbox.height * height_ratio
        scalebar_ax = fig.add_axes([scalebar_left, scalebar_bottom, width * 0.8, height])
        scalebar_ax.axis('off')
        plot_svg_on_ax(scalebar_ax, os.path.join(svg_root, 'scale.svg'), invert_y=True, color='grey')
        ax.text(1.5, 0.5, '100 mV\n0.5 ms\n1 s', transform=ax.transAxes,
               ha='left', va='center', fontsize=5, color='grey')


def process_fi_curves(ephys_feat_df):
    """
    Process F-I curve data into binned format.

    Parameters
    ----------
    ephys_feat_df : pandas.DataFrame
        Ephys features with columns:
        - specimen_id (index)
        - stimulus_amplitude_*: stimulus currents
        - avg_rate_*: firing rates

    Returns
    -------
    pandas.DataFrame
        Long-format dataframe with columns:
        - specimen_id (index)
        - step: sweep number
        - stimulus_amplitude: current in pA
        - avg_rate: firing rate in Hz
        - amp_bin: binned stimulus amplitude
    bins : array
        Bin edges for stimulus amplitudes

    Notes
    -----
    Bins stimulus amplitudes in 20 pA increments from -20 to 460 pA.
    Fills in zero firing rates for bins below threshold.
    """
    amp_cols = ephys_feat_df.columns[ephys_feat_df.columns.str.startswith("stimulus_amplitude_")]
    rate_cols = ephys_feat_df.columns[ephys_feat_df.columns.str.startswith("avg_rate_")]
    sub_df = ephys_feat_df[rate_cols.union(amp_cols)].reset_index()
    sub_df.columns = sub_df.columns.str.replace("_long_square", "")
    sub_df_long = pd.wide_to_long(
        sub_df,
        stubnames=["avg_rate", "stimulus_amplitude"],
        i="specimen_id", j="step",
        sep="_").dropna().reset_index().set_index("specimen_id"
    )

    bins = np.arange(-20, 460, 20)
    sub_df_long["amp_bin"] = pd.cut(sub_df_long.stimulus_amplitude, bins=bins)

    new_zero_rows = []
    for n, g in sub_df_long.groupby("specimen_id"):
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

    # Set type to Interval for concat column type matching
    sub_df_long['amp_bin'] = sub_df_long['amp_bin'].cat.categories.take(sub_df_long['amp_bin'].cat.codes).values
    sub_df_long['amp_bin'] = sub_df_long['amp_bin'].astype('interval[int64, right]')
    sub_df_long = sub_df_long.rename(columns={'avg_rate': 'avg_rate'})

    sub_df_long_filled = pd.concat([sub_df_long, new_df])
    return sub_df_long_filled, bins


def plot_avg_fi_for_mets(ax, ephys_fi_data_formatted, group_col, groups,
                         min_n=5, xlim=(0, 250), ylim=(0, 25), show_ylabel=True, show_xlabel=True, color_dict=SEAAD_COLORS):
    """
    Plot averaged F-I curves (firing rate vs stimulus current) by group.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    ephys_fi_data_formatted : pandas.DataFrame
        Processed F-I data (output of process_fi_curves)
    group_col : str
        Column name for grouping variable
    groups : list
        Groups to plot
    min_n : int, optional
        Minimum number of cells required per bin (default=5)
    xlim, ylim : tuple, optional
        Axis limits
    show_ylabel, show_xlabel : bool, optional
        If True, show axis labels (default=True)
    color_dict : dict, optional
        Mapping of groups to colors

    Notes
    -----
    Plots mean ± SEM with error bars.
    Only bins with >= min_n cells are plotted.
    """
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
        ax.set_xlabel("Stimulus current - \nrheobase (pA)", size=5)

    # Set axis and ticks to grey
    ax.tick_params(axis='both', colors='grey')
    ax.spines['bottom'].set_color('grey')
    ax.spines['left'].set_color('grey')


def align_to_peak(voltage, dv):
    """
    Find peak position in voltage trace for alignment.

    Parameters
    ----------
    voltage : array
        Voltage trace
    dv : array
        dV/dt trace (not used, kept for API compatibility)

    Returns
    -------
    pre_peak : int
        Number of points before peak
    post_peak : int
        Number of points after peak
    peak_idx : int
        Index of peak
    """
    peak_idx = np.argmax(voltage)
    total_len = len(voltage)

    pre_peak = peak_idx
    post_peak = total_len - peak_idx - 1  # -1 because peak_idx is inclusive

    return pre_peak, post_peak, peak_idx


def resample_around_peak(voltage_arrays, dv_arrays):
    """
    Resample action potential traces aligned to their peaks.

    Aligns all traces so peaks occur at the same time point,
    then trims to the shortest common window around the peak.

    Parameters
    ----------
    voltage_arrays : list of arrays
        List of voltage traces
    dv_arrays : list of arrays
        List of corresponding dV/dt traces

    Returns
    -------
    resampled_voltages : array
        Array of resampled voltage traces (n_traces x n_points)
    resampled_dvs : array
        Array of resampled dV/dt traces (n_traces x n_points)

    Notes
    -----
    All traces are trimmed to the most restrictive common window
    (shortest pre-peak and post-peak lengths across all traces).
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


def plot_ap_subclass(subclass_dict, color_dict, ax, plot_type):
    """
    Plot action potential traces by subclass.

    Parameters
    ----------
    subclass_dict : dict
        Nested dictionary: {subclass: {ttype: {cell: {'voltage', 'dVdt'}}}}
    color_dict : dict
        Mapping of subclasses to colors
    ax : matplotlib.axes.Axes
        Axis to plot on
    plot_type : {'width', 'phase'}
        Type of plot:
        - 'width': voltage vs time (AP waveform)
        - 'phase': dV/dt vs voltage (phase plot)

    Notes
    -----
    Aggregates all cells across all types within each subclass.
    Plots mean ± SEM with no individual traces or legend.
    """
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

        else:
            raise ValueError('Unknown plot_type')


def plot_ap_ttype(subclass_dict, ttypes, ttype_to_subclass_dict, color_dict, ax, plot_type):
    """
    Plot action potential traces by type.

    Parameters
    ----------
    subclass_dict : dict
        Nested dictionary: {subclass: {ttype: {cell: {'voltage', 'dVdt'}}}}
    ttypes : list
        Types to plot (in order)
    ttype_to_subclass_dict : dict
        Mapping of types to subclasses
    color_dict : dict
        Mapping of types to colors
    ax : matplotlib.axes.Axes
        Axis to plot on
    plot_type : {'width', 'phase'}
        Type of plot (see plot_ap_subclass)

    Notes
    -----
    Similar to plot_ap_subclass but plots individual types.
    Skips types with no data.
    """
    for ttype in ttypes:
        subclass_name = ttype_to_subclass_dict[ttype]
        ttp_varrs = []
        ttp_dvdts = []

        try:
            cell_data = subclass_dict[subclass_name][ttype]
        except:
            # No data for this ttype
            continue

        for cellname in cell_data.keys():
            v_arr = cell_data[cellname]['voltage']
            dv_arr = cell_data[cellname]['dVdt']
            if len(v_arr) > 0 and len(dv_arr) > 0:
                ttp_varrs.append(v_arr)
                ttp_dvdts.append(dv_arr)

        # Skip ttypes with no usable traces
        if not ttp_varrs:
            print(f"Skipping {ttype}: no valid AP traces")
            continue

        ttype_color = color_dict[ttype]
        if '/' in ttype:
            ttype = ttype.replace('/', '-')

        if plot_type == 'width':
            plot_ap_width(ttp_varrs, ttp_dvdts, class_name=ttype, class_color=ttype_color,
                         show_individual=False, ax=ax, show_legend=False)

        elif plot_type == 'phase':
            plot_ap_phase(ttp_varrs, ttp_dvdts, class_name=ttype, class_color=ttype_color,
                         show_individual=False, show_direction=False, ax=ax, show_legend=False)

        else:
            raise ValueError('Unknown plot_type')


# Helper functions for AP plotting

def plot_ap_width(voltage_arrays, dv_arrays, class_name=None, class_color=None,
                 show_individual=False, ax=None, show_legend=True):
    """Plot action potential waveform (voltage vs time)."""
    resampled_v, _ = resample_around_peak(voltage_arrays, dv_arrays)

    mean_voltage = np.mean(resampled_v, axis=0)
    sem_voltage = np.std(resampled_v, axis=0) / np.sqrt(len(voltage_arrays))

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    t_arr = np.arange(len(mean_voltage)) / 50000 * 1000  # Convert to ms
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

    ax.tick_params(axis='both', colors='grey', labelsize=6, length=3)
    ax.spines['bottom'].set_color('grey')
    ax.spines['left'].set_color('grey')

    return mean_voltage, sem_voltage


def plot_ap_phase(voltage_arrays, dv_arrays, class_name=None, class_color=None,
                 show_individual=False, show_direction=True, ax=None, show_legend=True):
    """Plot action potential phase plot (dV/dt vs voltage)."""
    resampled_v, resampled_dv = resample_around_peak(voltage_arrays, dv_arrays)

    mean_voltage = np.mean(resampled_v, axis=0)
    mean_dv = np.mean(resampled_dv, axis=0)
    sem_dv = np.std(resampled_dv, axis=0) / np.sqrt(len(voltage_arrays))

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
    if show_legend:
        ax.legend(bbox_to_anchor=(1.2, -0.25), loc='outer right')

    sns.despine(ax=ax, top=True, right=True)

    ax.tick_params(axis='both', colors='grey', labelsize=6, length=3)
    ax.spines['bottom'].set_color('grey')
    ax.spines['left'].set_color('grey')

    return mean_voltage, mean_dv, sem_dv


# SVG helper functions (for ephys gallery)

def svg_path_to_mpl_path(svg_path):
    """Convert SVG path to matplotlib Path."""
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
    """Translate matplotlib path by offset."""
    verts = mpl_path.vertices.copy()
    verts[:, 0] += dx
    verts[:, 1] += dy
    return Path(verts, mpl_path.codes)


def get_svg_bounding_box(paths):
    """Get bounding box of SVG paths."""
    all_x = []
    all_y = []
    for path in paths:
        for segment in path:
            all_x.extend([segment.start.real, segment.end.real])
            all_y.extend([segment.start.imag, segment.end.imag])
    return min(all_x), max(all_x), min(all_y), max(all_y)


def plot_svg_on_ax(ax, svg_path, xoffset=0, yoffset=0, color='black', invert_y=True):
    """Plot SVG paths on matplotlib axis."""
    if not os.path.exists(svg_path):
        print(f"File not found: {svg_path}")
        return

    paths, _ = svg2paths(svg_path)
    xmin, xmax, ymin, ymax = get_svg_bounding_box(paths)

    for path in paths:
        mpl_path = svg_path_to_mpl_path(path)

        if invert_y:
            # Invert the y-axis (flip vertically)
            flip_transform = Affine2D().scale(1, -1).translate(0, ymin + ymax)
            mpl_path = flip_transform.transform_path(mpl_path)

        shifted_path = translate_path(mpl_path, xoffset - xmin, yoffset - ymin)
        patch = patches.PathPatch(shifted_path, facecolor='none', edgecolor=color, linewidth=0.5)
        ax.add_patch(patch)

    ax.set_aspect('equal')
    ax.autoscale()
