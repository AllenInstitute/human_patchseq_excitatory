import numpy as np
import pandas as pd
import argschema as ags
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import colorConverter
import matplotlib.transforms as transforms
import seaborn as sns
import mouse_met_figs.utils as utils
import json
import networkx as nx

import sys 
sys.path.append(r'../../utils') 
from utils import get_tx_order_linux

class FigMetMatrixParameters(ags.ArgSchema):
    tx_anno_file = ags.fields.InputFile(
        description="feather file with patch-seq transcriptomic annotations")
    # facs_anno_file = ags.fields.InputFile(
    #     description="feather file with reference FACS transcriptomic annotations")
    me_cluster_labels_file = ags.fields.InputFile(
        description="csv file with cluster labels")
    spec_results_file = ags.fields.InputFile(
        description="json file with spectral coclustering results")
    met_partition_file = ags.fields.InputFile(
        description="json file with met-type partitions"
    )
    met_partition_index = ags.fields.Integer(
        default=2,
        description="index of met partition information to use for display"
    )
    met_cell_assignments_file = ags.fields.InputFile(
        description="met-type cell assignments"
    )
    subsample_assignments_file = ags.fields.InputFile()
    tx_mapping_file = ags.fields.InputFile()
    # dendrogram_file = ags.fields.InputFile()
    ephys_corr_file = ags.fields.InputFile()
    morph_corr_file = ags.fields.InputFile()
    genes_corr_file = ags.fields.InputFile()
    met_cell_weight_factor = ags.fields.Float(default=2.0)
    min_weight = ags.fields.Float(default=0.05)
    required_tree_calls = ags.fields.List(ags.fields.String,
        default=["Core", "I1", "I2", "I3"],
        description="tree call categories to restrict data (optional)",
        allow_none=True,
        cli_as_single_argument=True)
    output_file = ags.fields.OutputFile(
        description="output file")



# MET_NAME_DICT = {
# 	0: "L2/3 IT",
# 	13: "L4 IT",
# 	6: "L4/L5 IT",
# 	11: "L5 IT-1",
# 	12: "L5 IT-2",
# 	16: "L5 IT-3 Pld5",
# 	8: "L6 IT-1",
# 	7: "L6 IT-2",
# 	15: "L6 IT-3",
# 	14: "L5/L6 IT Car3",
# 	10: "L5 ET-1 Chrna6",
# 	1: "L5 ET-2",
# 	2: "L5 ET-3",
# 	9: "L5 NP",
# 	3: "L6 CT-1",
# 	4: "L6 CT-2",
# 	5: "L6b",
# }

# MET_TYPE_COLORS = {
#     'L2/3 IT': '#7AE6AB',
#     'L4 IT': '#00979D',
#     'L4/L5 IT': '#00DDC5',
#     'L5 IT-1': '#00A809',
#     'L5 IT-2': '#00FF00',
#     'L5 IT-3 Pld5': '#26BF64',
#     'L6 IT-1': '#C2E32C',
#     'L6 IT-2': '#96E32C',
#     'L6 IT-3': '#A19922',
#     'L5/L6 IT Car3': '#5100FF',
#     'L5 ET-1 Chrna6': '#0000FF',
#     'L5 ET-2': '#22737F',
#     'L5 ET-3': '#29E043',
#     'L5 NP': '#73CA95',
#     'L6 CT-1': '#74CAFF',
#     'L6 CT-2': '#578EBF',
#     'L6b': '#2B7880',
# }

def met_tuple_to_str(name):
    return str(name[0]) + "|" + str(name[1])


def met_str_to_tuple(name):
    split_name = name.split("|")
    return (int(split_name[0]), split_name[1])

def plot_spectral_results_matrix(spec_results, me_vs_ttype, ax,
        type_color_dict, type_order_list, scatter_factor=4,
        subclasses=[]):

    subclass_order = {sc: i for i, sc in enumerate(subclasses)}
    ttypes_in_order = []
    me_clust_in_order = []
    me_clust_order_count = []
    spectral_results_to_use = {}
    for sc in subclasses:
        k_vals, nclust_vals, nonzero_off_vals, zero_in_vals = zip(
            *[(d["k"], d["n_biclust"], d["nonzeros_out_of_cluster"], d["zeros_in_cluster"])
            for d in spec_results[sc]["spectral_results"]])
        select_ind = np.argmin(np.array(nonzero_off_vals) + np.array(zero_in_vals))
        select_k = k_vals[select_ind]
        print(sc, "k", select_k, np.array(nonzero_off_vals)[select_ind], np.array(zero_in_vals)[select_ind])

        for d in spec_results[sc]["spectral_results"]:
            if d["k"] == select_k:
                break
        spectral_results_to_use[sc] = d
        ttypes = np.array(spec_results[sc]["ttypes"])
        row_labels = np.array(d["row_labels"])
        col_labels = np.array(d["column_labels"])

        type_order_dict = dict(zip(type_order_list, range(len(type_order_list))))
        row_tree_positions = np.array([type_order_dict[t] for t in ttypes])
        print(row_tree_positions.dtype)
        print(row_tree_positions[:10])
        print(row_labels[:10])

        biclust_avg_tree_position = {
            bic: np.median(row_tree_positions[row_labels == bic])
            for bic in np.unique(row_labels)
        }

        bicluster_ordering = sorted(np.unique(row_labels), key=lambda x: biclust_avg_tree_position[x])
        bic_order_dict = {v: i for i, v in enumerate(bicluster_ordering)}
        print(ttypes)
        rl_bic = [bic_order_dict[rl] for rl in row_labels]
        print(row_tree_positions)
        print(rl_bic)
        row_ordering = np.lexsort((row_tree_positions, rl_bic))
        print(row_ordering)
        ttypes_in_order += ttypes[row_ordering].tolist()
        me_clusts = np.array(spec_results[sc]["me_clusters"])
        for j in bicluster_ordering:
            to_add = me_clusts[col_labels == j]
            related_ttypes = ttypes[row_labels == j]
            for mec in to_add:
                    me_clust_in_order.append(mec)
                    me_clust_order_count.append(
                        me_vs_ttype.loc[related_ttypes, mec].values.sum()
                    )
        for j in np.unique(col_labels):
            if j not in row_labels:
                to_add = me_clusts[col_labels == j]
                for mec in to_add:
                    me_clust_in_order.append(mec)
                    me_clust_order_count.append(
                        me_vs_ttype.loc[ttypes, mec].values.sum()
                    )
    print("ttypes in order")
    print(ttypes_in_order)

    # Order the ME types
    me_clust_in_order = np.array(me_clust_in_order)
    me_clust_in_order_inds = np.arange(len(me_clust_in_order))
    me_clust_order_count = np.array(me_clust_order_count)
    kept_inds = []
    for mec in np.unique(me_clust_in_order):
        mask = me_clust_in_order == mec
        top = np.argmax(me_clust_order_count[mask])
        top_ind = me_clust_in_order_inds[mask][top]
        kept_inds.append(top_ind)
    me_clust_in_order = me_clust_in_order[np.sort(kept_inds)]

    # Create the data structure for plotting
    sub_df = me_vs_ttype.loc[ttypes_in_order, me_clust_in_order]
    melted = pd.melt(sub_df.reset_index(), id_vars=["cluster_label"])
    ttype_y = {t: i for i, t in enumerate(ttypes_in_order)}
    me_x = {c: i for i, c in enumerate(me_clust_in_order)}
    melted["y"] = [ttype_y[v] for v in melted["cluster_label"]]
    melted["x"] = [me_x[v] for v in melted["0"]]
    y_counter = 0
    y_labels = []

    # Plot the grid
    for t in ttypes_in_order:
        sub = melted.loc[melted["cluster_label"] == t, :]
        ax.scatter(sub["x"], [y_counter] * sub.shape[0], s=sub["value"] * scatter_factor,
            zorder=10, c=type_color_dict[t], edgecolors="white", linewidths=0.25)
        y_counter += 1
        if "PT" in t:
            t = t.replace("PT", "ET")
        y_labels.append(t)
    ax.set_ylim(-1, len(ttypes_in_order))
    ax.set_xlim(-1, len(me_clust_in_order))
    ax.invert_yaxis()

    # Label the grid
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=5)

    ax.set_xticks(np.arange(len(me_clust_in_order)))

    print(me_clust_in_order)
    # me_order_tick_labels = [m[9:] for m in me_clust_in_order]
    me_order_tick_labels = [f'ME_{m}' for m in me_clust_in_order]
    print(me_order_tick_labels)
    ax.set_xticklabels(me_order_tick_labels, rotation=90, fontsize=5)
    ax.xaxis.grid(True, color="#eeeeee", zorder=-1)
    ax.yaxis.grid(True, color="#eeeeee", zorder=-1)
    ax.set_aspect("equal")
    sns.despine(ax=ax, left=True, bottom=True)
    ax.tick_params(axis="both", length=0, width=0)


def plot_met_graph(ax, vertices, edges, t_type_color_dict, met_type_df,
        k_scale=10, node_scale=10, width_scale=4, label_offset=10,
        starting_buffer=300, buffer_factor=250):

    G = nx.Graph()
    G.add_nodes_from(vertices)
    G.add_edges_from(edges)

    print(f"Number of used nodes: {len(vertices)}")

    layoutG = G.copy()

    layout_node_size = node_scale
    pos = nx.nx_agraph.graphviz_layout(layoutG, args=f"-Goverlap=false -Nshape=circle -Nheight={layout_node_size} -Nwidth={layout_node_size} -Gsep=+8")

    to_me_t_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d["type"] == "me_t_mapping"]
    to_me_t_weights = np.array([d["weight"] for u, v, d in to_me_t_edges])

    cell_cell_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d["type"] == "cell_cell_corr"]
    cell_cell_weights = np.array([d["weight"] for u, v, d in cell_cell_edges])

    me_t_nodes = [n for n in G.nodes() if "|" in n]
    me_t_node_color = "#cccccc"
    me_t_node_edgecolors = [t_type_color_dict[n.split("|")[1]] for n in me_t_nodes]
    me_t_node_size = 8
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=me_t_nodes,
        node_size=me_t_node_size * node_scale,
        node_color=me_t_node_color,
        edgecolors=me_t_node_edgecolors,
        linewidths=0.5,
        ax=ax,
    )

    cell_nodes = [n for n in G.nodes() if "|" not in n]
    print(len(cell_nodes))
    # cell_node_color = [MET_TYPE_COLORS[MET_NAME_DICT[met_type_df.at[int(n), "met_type"]]] for n in cell_nodes]
    cell_node_size = 2
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=cell_nodes,
        node_size=cell_node_size * node_scale,
        # node_color=cell_node_color,
        linewidths=0,
        ax=ax,
    )

    # Draw cell-cell correlation edges
    nx.draw_networkx_edges(G, pos,
                           edgelist=cell_cell_edges,
                           edge_color="steelblue", width=cell_cell_weights * width_scale,
                           ax=ax)
    # Draw ME/T mapping edges
    nx.draw_networkx_edges(G, pos,
                           edgelist=to_me_t_edges,
                           edge_color="black", width=to_me_t_weights * width_scale,
                           ax=ax)

    ax.set_aspect("equal")
    sns.despine(left=True, bottom=True)

    # met_types_for_nodes = np.array([MET_NAME_DICT[met_type_df.at[int(n), "met_type"]] for n in cell_nodes])
    cell_node_name_arr = np.array(cell_nodes)
    met_centers = {}
    # for met in MET_TYPE_COLORS.keys():
    #     met_centers[met] = np.mean([
    #         [pos[n][0] for n in cell_node_name_arr[met_types_for_nodes == met]],
    #         [pos[n][1] for n in cell_node_name_arr[met_types_for_nodes == met]],
    #     ], axis=1)

    return met_centers


def define_vertices_and_edges(me_t_labels_df, used_t_types, met_type_df,
        tx_anno, tx_mapping_df, subsample_assignments_norm, avg_corr_df,
        min_weight, met_cell_weight_factor, corr_thresh):
    cell_verts = met_type_df.index.values.tolist()
    cell_verts = [str(i) for i in cell_verts]

    cell_to_me_t_edge_list = []
    used_me_t_type_verts = []
    for specimen_id in met_type_df.index.values:
        # find t mapping edges
        sample_id = tx_anno.at[specimen_id, "sample_id"]
        # cell_mapping_t = tx_mapping_df.loc[sample_id, :]
        cell_mapping_t = tx_mapping_df.loc[specimen_id, :]
        cell_mapping_me = subsample_assignments_norm.loc[specimen_id, :]

        t_nonzero_mask = cell_mapping_t > 0
        me_nonzero_mask = cell_mapping_me > 0
        for t in cell_mapping_t.index[t_nonzero_mask]:
            if t not in used_t_types:
                continue
            for me in cell_mapping_me.index[me_nonzero_mask]:
                me_t_vertex_name = met_tuple_to_str((me, t))
                combo_weight = cell_mapping_t[t] * cell_mapping_me[me]
                if combo_weight < min_weight:
                    continue
                used_me_t_type_verts.append(me_t_vertex_name)
                cell_to_me_t_edge_list.append((
                    str(specimen_id),
                    me_t_vertex_name,
                    {
                        "weight": combo_weight * met_cell_weight_factor,
                        "type": "me_t_mapping",
                    },
                ))

    print("edges between cells and me/t nodes: ", len(cell_to_me_t_edge_list))
    me_t_type_verts = np.unique(used_me_t_type_verts).tolist()

    # add correlation-based edges
    inds = np.tril_indices_from(avg_corr_df.values, k=-1)
    corr_mask = avg_corr_df.values[inds] > corr_thresh
    print(f"{corr_mask.sum()} between-cell edges")
    corr_edge_list = []
    for i, j in zip(inds[0][corr_mask], inds[1][corr_mask]):
        corr_edge_list.append((
            str(avg_corr_df.index[i]),
            str(avg_corr_df.index[j]),
            {
                "weight": avg_corr_df.iloc[i, j],
                "type": "cell_cell_corr",
            },
        ))

    return me_t_type_verts + cell_verts, cell_to_me_t_edge_list + corr_edge_list


def main(args):
    # Set up figure formatting
    sns.set(style="white", context="paper", font="Helvetica")
    matplotlib.rc('font', family='Helvetica')

    tx_anno_df = pd.read_feather(args["tx_anno_file"])
    tx_anno_df = tx_anno_df.loc[tx_anno_df["spec_id_label"] != "ZZ_Missing", :]
    tx_anno_df["spec_id_label"] = pd.to_numeric(tx_anno_df["spec_id_label"])
    tx_anno_df.set_index("spec_id_label", inplace=True)

    # facs_anno_df = pd.read_feather(args["facs_anno_file"])
    # facs_type_order_dict = utils.dict_from_facs(facs_anno_df,
    #     key_column="cluster_label", value_column="cluster_id")
    # type_subclass_dict = utils.dict_from_facs(tx_anno_df,
    #     key_column="cluster_label", value_column="subclass_label")
    ttype_order, subclass_order = get_tx_order_linux()
    type_color_dict = utils.cluster_colors(tx_anno_df)

    me_labels_df = pd.read_csv(args["me_cluster_labels_file"], index_col=0)

    subsample_assignments_df = pd.read_csv(args["subsample_assignments_file"], index_col=0)
    subsample_assignments_norm = subsample_assignments_df / subsample_assignments_df.sum(axis=1).values[:, np.newaxis]

    if args["tx_mapping_file"][-3:] == "csv":
        tx_mapping_df = pd.read_csv(args["tx_mapping_file"], index_col=0)
    else:
        tx_mapping_df = pd.read_feather(args["tx_mapping_file"]).set_index("sample_id")
    # with open(args["dendrogram_file"], "r") as f:
    #     dendrogram_info = json.load(f)
    # leaves = [d for d in dendrogram_info if dendrogram_info[d][0] is None]
    # tx_mapping_df = tx_mapping_df.loc[:, tx_mapping_df.columns.intersection(leaves)]

    ephys_corr_df = pd.read_csv(args["ephys_corr_file"], index_col=0)
    ephys_corr_df.columns = pd.to_numeric(ephys_corr_df.columns)
    ephys_corr_df = ephys_corr_df.loc[me_labels_df.index, me_labels_df.index]

    morph_corr_df = pd.read_csv(args["morph_corr_file"], index_col=0)
    morph_corr_df.columns = pd.to_numeric(morph_corr_df.columns)
    morph_corr_df = morph_corr_df.loc[me_labels_df.index, me_labels_df.index]

    genes_corr_df = pd.read_csv(args["genes_corr_file"], index_col=0)
    genes_corr_df.columns = pd.to_numeric(genes_corr_df.columns)
    genes_corr_df = genes_corr_df.loc[me_labels_df.index, me_labels_df.index]

    avg_corr_df = (ephys_corr_df +
                   morph_corr_df +
                   genes_corr_df) / 3

    me_t_labels_df = pd.merge(
        me_labels_df,
        tx_anno_df.loc[:, ["cluster_label"]],
        left_index=True, right_index=True)
    used_t_types = me_t_labels_df["cluster_label"].unique().tolist()

    met_type_df = pd.read_csv(args["met_cell_assignments_file"], index_col=0)

    with open(args["met_partition_file"], "r") as f:
        partition_info = json.load(f)
    selected_partition_info = partition_info[args["met_partition_index"]]
    corr_thresh = selected_partition_info["correlation_threshold"]

    vertices, edges = define_vertices_and_edges(me_t_labels_df, used_t_types, met_type_df,
        tx_anno_df, tx_mapping_df, subsample_assignments_norm, avg_corr_df,
        args["min_weight"], args["met_cell_weight_factor"], corr_thresh)

    # Plotting the ME / T matrix
    with open(args["spec_results_file"], "r") as f:
        spec_results = json.load(f)
    all_ttypes_from_spec_results = []
    for sc in spec_results:
        all_ttypes_from_spec_results += spec_results[sc]["ttypes"]

    fig = plt.figure(figsize=(5, 9))
    g = gridspec.GridSpec(5, 2,
        height_ratios=(0.05, 1.6, 0.05, 0.3, 0.9), width_ratios=[0.8, 0.1],
        hspace=0.15, wspace=0)

    scatter_factor = 4
    subclasses = ['L2/3 IT', 'L4', 'L5 IT', 'L6 IT', 'L5 PT', 'NP', 'L6 CT', 'L6b']

    # Full matrix with co-clusters
    ax = plt.subplot(g[1, 0])

    me_labels_df = me_labels_df.join(
        tx_anno_df[["cluster_label"]],
        how="left")
    me_labels_core_i1 = me_labels_df #.loc[me_labels_df["Tree_call_label"].isin(args["required_tree_calls"]), :]
    me_vs_ttype = pd.crosstab(index=me_labels_core_i1["cluster_label"], columns=me_labels_core_i1["0"])
    print("size of data set", me_vs_ttype.values.sum())
    print("size of data set with spectral results",
        me_labels_core_i1.loc[me_labels_core_i1["cluster_label"].isin(all_ttypes_from_spec_results), :].shape[0])

    plot_spectral_results_matrix(
        spec_results, me_vs_ttype, ax,
        scatter_factor=scatter_factor,
        subclasses=subclasses,
        type_color_dict=type_color_dict,
        type_order_list=ttype_order)

    # Plot matrix legend
    ax_leg = plt.subplot(g[1, 1])
    cell_nums = [1, 5, 10]
    ax_leg.scatter([0, 0, 0], [0, 3, 6], s=np.array(cell_nums) * scatter_factor, c="k",
        edgecolors="white", linewidths=0.25)
    ax_leg.set_ylim([-50, 20])
    ax_leg.set_yticks([0, 3, 6, 9])
    ax_leg.set_yticklabels(cell_nums + ["n cells:"], fontsize=6)
    ax_leg.yaxis.set_ticks_position("right")
    ax_leg.yaxis.set_tick_params(length=0, width=0)
    ax_leg.set_xlim(-2, 2)
    ax_leg.set_xticks([])
    sns.despine(ax=ax_leg, left=True, bottom=True)


    # Plot graph
    ax = plt.subplot(g[3:, 0])
    node_scale = 2
    width_scale = 0.25
    met_centers = plot_met_graph(ax, vertices, edges, type_color_dict, met_type_df, 
                                 node_scale=node_scale, 
                                 width_scale=width_scale, 
                                 k_scale=30, #10
                                 starting_buffer=500, #130,
                                 buffer_factor=130,
                                 )

    # # Plot MET labels
    # label_offsets = {
    #     'L2/3 IT': (-1000, 700),
    #     'L4 IT': (700, 400),
    #     'L4/L5 IT': (1400, -50),
    #     'L5 IT-1': (1000, 0),
    #     'L5 IT-2': (1300, 0),
    #     'L5 IT-3 Pld5': (0, -400),
    #     'L6 IT-1': (-900, -400),
    #     'L6 IT-2': (-1000, 400),
    #     'L6 IT-3': (-800, 0),
    #     'L5/L6 IT Car3': (-1300, 0),
    #     'L5 ET-1 Chrna6': (-1300, -50),
    #     'L5 ET-2': (1300, 0),
    #     'L5 ET-3': (-1400, 0),
    #     'L5 NP': (800, 0),
    #     'L6 CT-1': (0, 1100),
    #     'L6 CT-2': (1200, 200),
    #     'L6b': (-500, -700),
    # }

    # for l, coords in met_centers.items():
    #     if l in label_offsets:
    #         xoffset, yoffset = label_offsets[l]
    #     else:
    #         xoffset = yoffset = 0

    #     ax.text(
    #         coords[0] + xoffset,
    #         coords[1] + yoffset,
    #         l,
    #         fontsize=5,
    #         color=MET_TYPE_COLORS[l],
    #         ha="center",
    #         va="center"
    #     )





    # Graph legend
    ax = plt.subplot(g[4, 1])

    # node types
    ax.scatter([0, 0], [9, 6], s=np.array([8, 2]) * node_scale,
        linewidths=[0.25, 0], edgecolors="black", c=["#cccccc", "black"])
    ax.text(-0.11, 7.5, "type of\nnode", ha="right", va="center", fontsize=6)
    ax.text(0.1, 9, "ME-type/T-type combination\n(border color indicates T-type)", ha="left", va="center", fontsize=6)
    ax.text(0.1, 6, "individual cell\n(color indicates MET-type)", ha="left", va="center", fontsize=6)

    ax.plot([-0.05, 0.05], [3, 3], c="black", lw=1)
    ax.text(0.11, 3, "probability of mapping to\nME-type/T-type combination", ha="left", va="center", fontsize=6)

    ax.plot([-0.05, 0.05], [0, 0], c="steelblue", lw=1)
    ax.text(0.11, 0, "avg. triple-modality\ncell-cell correlation", ha="left", va="center", fontsize=6)

    ax.text(0.11, -3, "(thicker lines indicate\nhigher probability / correlation)",
        ha="left", va="center", fontsize=6, fontstyle="italic")

    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-5, 17)
    ax.set_yticks([])
    ax.set_xticks([])
    sns.despine(ax=ax, left=True, bottom=True)


    # Panel labels
    ax.text(
        g.get_grid_positions(fig)[2][0] - 0.04,
        g.get_grid_positions(fig)[1][1],
        "a", transform=fig.transFigure,
        fontsize=16, fontweight="bold", va="baseline", ha="right")
    ax.text(
        g.get_grid_positions(fig)[2][0] - 0.04,
        g.get_grid_positions(fig)[1][3] - 0.02,
        "b", transform=fig.transFigure,
        fontsize=16, fontweight="bold", va="baseline", ha="right")

    plt.savefig(args["output_file"], bbox_inches="tight", dpi=600)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=FigMetMatrixParameters)
    main(module.args)
