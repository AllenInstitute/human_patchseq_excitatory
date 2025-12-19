import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.ndimage import gaussian_filter1d
import matplotlib.colors as mcolors

def plot_ephys_rheo(ax, rheobase_spike_data, rheobase_metadata, groups, group_column, color_dict):

    for group in groups: 
        # Filter dataframe for current group
        group_cells = rheobase_metadata[rheobase_metadata[group_column] == group]['cell_specimen_name'].tolist()
        
        group_spikes = []
        for cell in group_cells:
            if cell in rheobase_spike_data:
                cell_data = rheobase_spike_data[cell]
                spike = cell_data['spike_data']
                time = cell_data['time_data']
                if cell_data['sampling_rate'] == 200000.0:
                    spike = spike[::4]
                    time = time[::4]
                group_spikes.append(spike)

        # Compute and plot mean spike
        if group_spikes:
            group_spikes = np.array(group_spikes)
            mean_spike = np.mean(group_spikes, axis=0)
            t0 = time[0]
            time0 = time - t0
            ax.plot(time0 * 1000, mean_spike, linewidth=0.5, color=color_dict[group], label=group)

    # Final plot formatting
    ax.set_ylabel('Voltage (mV)', size=5)
    ax.set_xlabel('Time (ms)', size=5)
    
    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(axis="both", labelsize=6)
    ax.tick_params(axis='both', colors='grey')  # Set ticks color to grey
    ax.spines['bottom'].set_color('grey')  # Set bottom spine color to grey
    ax.spines['left'].set_color('grey')  # Set left spine color to grey


def plot_ephys_subthresh(ax, subthreshold_data, subthreshold_metadata, groups, group_column, color_dict):

    for group in groups:
        # Filter dataframe for current group
        group_cells = subthreshold_metadata[subthreshold_metadata[group_column] == group]['cell_specimen_name'].tolist()

        responses = []
        count = 0
        reference_time = None  # To store a representative time axis

        #accumulate responses
        for cell in group_cells:
            if cell in subthreshold_data: #swb Q: why need this check?
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
                    reference_time = time  # Save time axis once

                baseline = np.mean(response[0:500])
                response0 = response - baseline
                norm_response = -(response0 / np.min(response0))
                responses.append(norm_response)
                count += 1

        #plot mean responses
        if responses:
            mean_response = np.mean(responses, axis=0)
            smoothed_response = gaussian_filter1d(mean_response, sigma=10)  # Adjust sigma to control smoothness
            ax.plot(reference_time * 1000, smoothed_response, color=color_dict[group], linewidth=0.5, label=group)

            # ax.plot(reference_time*1000, mean_response, color=color_dict[group], linewidth=0.5, label=group)

    # Final plot formatting
    ax.set_ylabel('Normalized voltage', size=5) #same as ephys_rheo so don't label
    ax.set_xlabel('Time (ms)', size=5)
    
    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(axis="both", labelsize=6)
    ax.tick_params(axis='both', colors='grey')  # Set ticks color to grey
    ax.spines['bottom'].set_color('grey')  # Set bottom spine color to grey
    ax.spines['left'].set_color('grey')  # Set left spine color to grey


def plot_subclass_heatmap(ax, kde_group_dict, x_var_order, layer_info, layer_colors, layer_labels, color_dict, xoffsets, plot_scalebar=True, plot_subclass=True, mask_min_dict=None, colormap='gist_heat_r', ylabel='', xlim=None, ylim=None, color_seaad=False):

    ax.axhline(0, c="lightgrey",linewidth=0.50)
    for l in layer_info: ax.axhline(-layer_info[l], c=layer_colors[l],linewidth=0.50)
    for l in layer_labels: ax.text(-100,-layer_labels[l], "{}".format(l),verticalalignment='center', horizontalalignment='left',fontsize=6, color='lightgrey') 

    # plot
    for i, subclass in enumerate([x for x in x_var_order if x in kde_group_dict.keys()]):

        xoffset = xoffsets[subclass]

        this_xi = kde_group_dict[subclass]['xi']
        this_yi = kde_group_dict[subclass]['yi']
        this_zi = kde_group_dict[subclass]['zi']
        this_num_cells = kde_group_dict[subclass]['num_cells']

        this_xi = this_xi + xoffset
        this_zi = this_zi.reshape(this_xi.shape)

        #custom remove heatmap below x value
        if not mask_min_dict is None:
            if subclass in mask_min_dict.keys():
                x_min = mask_min_dict[subclass]
                x_mask = this_xi[:,0] >= x_min
                this_xi = this_xi[x_mask,:]
                this_yi = this_yi[x_mask,:]
                this_zi = this_zi[x_mask,:]

        if color_seaad:
            colormap = mcolors.LinearSegmentedColormap.from_list("custom_seaad", list(zip([0, 1], ["#FFFFFF", color_dict[subclass]])))

        ax.pcolormesh(this_xi, this_yi, this_zi, shading='auto', cmap=colormap)

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


def plot_subclass_heatmap_old(ax, kde_group_dict, x_var_order, layer_info, layer_colors, layer_labels, colormap, color_dict, plot_scalebar=True, plot_title=True, xbuffer=300):

    for l in layer_info: ax.axhline(-layer_info[l], c=layer_colors[l],linewidth=0.50)
    for l in layer_labels: ax.text(-100,-layer_labels[l], "{}".format(l),verticalalignment='center', horizontalalignment='left',fontsize=6, color='lightgrey') 
    
    # plot
    xoffset= 0
    for i, subclass in enumerate([x for x in x_var_order if x in kde_group_dict.keys()]):

        this_xi = kde_group_dict[subclass]['xi']
        this_yi = kde_group_dict[subclass]['yi']
        this_zi = kde_group_dict[subclass]['zi']
        this_num_cells = kde_group_dict[subclass]['num_cells']

        left_width = np.abs(np.min(this_xi))
        right_width = np.max(this_xi)

        xoffset += xbuffer + left_width
        this_xi = this_xi + xoffset

        print(subclass)
        print(xoffset)
        plt.pcolormesh(this_xi, this_yi, this_zi.reshape(this_xi.shape), shading='auto', cmap=colormap) #'gist_heat_r') #'magma'
        plt.gca().set_aspect('equal')

        ax.axis('off')
        ax.axhline(0, c="lightgrey",linewidth=0.50)

        #save label 
        if plot_title: 
            ax.text(xoffset,500,
                "{}\n(n={}) ".format(subclass,this_num_cells),
                horizontalalignment='center',fontsize=5, color=color_dict[subclass], rotation=45)

        if plot_scalebar & (i == 0):
            #add a scale bar 
            scalebar = ScaleBar(1, "um", location='lower right', frameon=False, fixed_value=500, color = 'grey')
            ax.add_artist(scalebar)
        
        xoffset += right_width




        