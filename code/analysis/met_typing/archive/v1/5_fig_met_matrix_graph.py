#!/usr/bin/env python

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
# import mouse_met_figs.constants as constants
import mouse_met_figs.simple_sankey as sankey
from mouse_met_figs.graph import sum_directed_edges, met_tuple_to_str, met_str_to_tuple
import json
import networkx as nx
import shapely.geometry as geometry
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
from descartes import PolygonPatch

class FigMetMatrixParameters(ags.ArgSchema):
    tx_anno_file = ags.fields.InputFile(
        default="/allen/programs/celltypes/workgroups/ivscc/agatab/data/status_reports/basal_ganglia/met_types/20240621_BT014-RSC-366_mouse_patchseq_star2.7_cpm_WB21_mapping.csv",
        description="csv file with patch-seq transcriptomic annotations")
    tx_color_file = ags.fields.InputFile(
        default="/allen/programs/celltypes/workgroups/rnaseqanalysis/shiny/Taxonomies/AIT21.0_mouse/WB_colorpal - clusters 230815.tsv",
    )
    me_cluster_labels_file = ags.fields.InputFile(
        default="../me_clust_output/bg_mouse_me_clustering/refined_text_labels.csv",
        description="csv file with cluster labels")
    spec_results_file = ags.fields.InputFile(
        default="spectral_coclust_results.json",
        description="json file with spectral coclustering results")
    met_partition_file = ags.fields.InputFile(
        default="bg_met_node_partitions.json",
        description="json file with met-type partitions"
    )
    me_map_pt_file = ags.fields.InputFile(
        default="bg_me_map_pt.csv",
        description="me-type mapping pivot table"
    )
    t_map_pt_file = ags.fields.InputFile(
        default="bg_t_map_pt.csv",
        description="me-type mapping pivot table"
    )
    met_cell_assignments_file = ags.fields.InputFile(
        default="bg_met_cell_assignments.csv",
        description="met-type cell assignments"
    )
    output_file = ags.fields.OutputFile(
        default="fig_met_matrix_graph.pdf",
        description="output file")


# MET_NAMES = {
#     0: "MSN-D2-1",
#     1: "MSN-D1-1",
#     2: "STN",
#     3: "MSN-D1-D2",
#     4: "MSN-D1-2",
#     5: "MSN-D1-3",
#     6: "MSN-D1-4",
#     7: "MSN-D2-2",
#     8: "GPe-Pvalb",
#     9: "GPi-SNr",
#     10: "STR-FS",
#     11: "MSN-D1-5",
#     12: "MSN-D1-6",
#     13: "GPe-Penk",
#     14: "STR-Chol",
#     15: "MSN-D2-3",
# }


def plot_spectral_results_matrix(spec_results, me_vs_ttype, ax,
        type_color_dict, type_order_dict, scatter_factor, subclasses):

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

        row_tree_positions = np.array([type_order_dict[t] for t in ttypes])
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
        y_labels.append(t)
    ax.set_ylim(-1, len(ttypes_in_order))
    ax.set_xlim(-1, len(me_clust_in_order))
    ax.invert_yaxis()

    # Label the grid
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=5)

    ax.set_xticks(np.arange(len(me_clust_in_order)))

    print(type(me_clust_in_order))
    print(me_clust_in_order)

    # me_order_tick_labels = [m[8:] for m in me_clust_in_order]
    me_order_tick_labels = [str(m) for m in me_clust_in_order] #SWB debugging

    ax.set_xticklabels(me_order_tick_labels, rotation=90, fontsize=5)
    ax.xaxis.grid(True, color="#eeeeee", zorder=-1)
    ax.yaxis.grid(True, color="#eeeeee", zorder=-1)
    ax.set_aspect("equal")
    sns.despine(ax=ax, left=True, bottom=True)
    ax.tick_params(axis="both", length=0, width=0)


def main(tx_anno_file, tx_color_file,
        me_cluster_labels_file, spec_results_file,
        met_partition_file,
        me_map_pt_file, t_map_pt_file, met_cell_assignments_file,
        output_file, **kwargs):
    
    # Set up figure formatting
    sns.set(style="white", context="paper", font="Helvetica")
    matplotlib.rc('font', family='Helvetica')

    tx_anno_df = pd.read_csv(tx_anno_file, index_col=0)

    tx_color_df = pd.read_csv(tx_color_file, sep="\t")

    #SWB addition: get MET_NAMES dynamically 
    met_assignment_df = pd.read_csv(met_cell_assignments_file)
    met_ids = sorted(met_assignment_df.met_type.unique())
    met_name = [f'exc_hMET_{x}' for x in met_ids]
    MET_NAMES = dict(zip(met_ids, met_name))
    ###

    facs_type_order_dict = utils.dict_from_facs(tx_anno_df,
        key_column="cluster_label", value_column="cluster_id")
    type_subclass_dict = utils.dict_from_facs(tx_anno_df,
        key_column="cluster_label", value_column="subclass_label")
    type_color_dict = dict(zip(tx_color_df.cluster_label, tx_color_df.cluster_color))

    # Plotting
    with open(spec_results_file, "r") as f:
        spec_results = json.load(f)
    all_ttypes_from_spec_results = []
    for sc in spec_results:
        all_ttypes_from_spec_results += spec_results[sc]["ttypes"]

    fig = plt.figure(figsize=(6, 9))
    g = gridspec.GridSpec(5, 2,
        height_ratios=(0.05, 1.6, 0.05, 0.1, 1.1), width_ratios=[1.5, 0.1],
        hspace=0.15, wspace=0.15)

    scatter_factor = 4
    # subclasses = ['L2/3 IT', 'L4 IT', 'L5 ET', 'L5 IT', 'L5/6 NP', 'L6 CT', 'L6b', 'L6 IT', 'L6 IT Car3']
    subclasses = ['L2/3 IT', 'L4 IT', 'L5 IT', 'L5/6 NP', 'L6 CT', 'L6b', 'L6 IT', 'L6 IT Car3']

    # Full matrix with co-clusters - VISp
    ax = plt.subplot(g[1, 0])

    me_labels_df = pd.read_csv(me_cluster_labels_file, index_col=0)
    me_labels_df = me_labels_df.join(
        tx_anno_df[["cluster_label"]],
        how="left")
    me_vs_ttype = pd.crosstab(index=me_labels_df["cluster_label"], columns=me_labels_df["0"])
    print("size of data set", me_vs_ttype.values.sum())
    print("size of data set with spectral results",
        me_labels_df.loc[me_labels_df["cluster_label"].isin(all_ttypes_from_spec_results), :].shape[0])

    print(me_vs_ttype)

    plot_spectral_results_matrix(
        spec_results, me_vs_ttype, ax,
        scatter_factor=scatter_factor,
        subclasses=subclasses,
        type_color_dict=type_color_dict,
        type_order_dict=facs_type_order_dict)

    # Plot matrix legend
    ax_leg = plt.subplot(g[0, 0])
    cell_nums = [1, 5, 10]
    ax_leg.scatter([0, 3, 6], [0, 0, 0], s=np.array(cell_nums) * scatter_factor, c="k",
        edgecolors="white", linewidths=0.25)
    ax_leg.set_xlim([-50, 20])
    ax_leg.set_xticks([-7, 0, 3, 6])
    ax_leg.set_xticklabels( ["n cells:"] + cell_nums, fontsize=6)
    ax_leg.xaxis.set_ticks_position("bottom")
    ax_leg.xaxis.set_tick_params(length=0, width=0)
    ax_leg.set_ylim(-2, 2)
    ax_leg.set_yticks([])
    sns.despine(ax=ax_leg, left=True, bottom=True)


    # Plot graph
    me_map_pt = pd.read_csv(me_map_pt_file, index_col=0)
    t_map_pt = pd.read_csv(t_map_pt_file, index_col=0)

    with open(met_partition_file, "r") as f:
        partition_info = json.load(f)

    ax = plt.subplot(g[3:, 0])
    node_scale = 10
    width_scale = 2
    consistency_type = 't' #'me' 't'
    met_centers = plot_met_graph(
        ax,
        me_map_pt,
        t_map_pt,
        partition_info["membership"],
        partition_info["counts"],
        facs_type_order_dict,
        type_color_dict,
        type_subclass_dict,
        k_scale=30, #10
        node_scale=node_scale,
        width_scale=width_scale,
        starting_buffer=500, #130,
        buffer_factor=130,
        consistency_type=consistency_type
    )


    # Plot MET labels
    label_offsets = {
    }

    for l, coords in met_centers.items():
        if MET_NAMES[l] in label_offsets:
            xoffset, yoffset = label_offsets[MET_NAMES[l]]
        else:
            xoffset = yoffset = 0

        ax.text(
            coords[0] + xoffset,
            coords[1] + yoffset,
            MET_NAMES[l],
            fontsize=5,
            color="gray",
            ha="center",
            va="center")

    print(ax.get_xlim(), ax.get_ylim())

    # Graph legend
    ax = plt.subplot(g[4, 1])

    # consistency
    ax.scatter([0], [10], c="#bbbbbb", s=20 * node_scale, linewidths=0)
    ax.scatter([0], [10], c="black", s=20 * node_scale * 0.3, linewidths=0)
    ax.plot([0, 0.1], [10, 10], c="black", lw=0.5)
    ax.text(0.11, 10, f"fraction {consistency_type}-type\nconsistent", ha="left", va="center", fontsize=6)

    ax.scatter([0, 0, 0], [8, 7, 6], s=np.array([10, 5, 1]) * node_scale, linewidths=0, c="black")
    ax.text(0.11, 7, "number\nof cells", ha="left", va="center", fontsize=6)
    ax.text(-0.1, 8, "10", ha="right", va="center", fontsize=6)
    ax.text(-0.1, 7, "5", ha="right", va="center", fontsize=6)
    ax.text(-0.1, 6, "1", ha="right", va="center", fontsize=6)

    ax.plot([-0.05, 0.05], [4.25, 4.25], c="tomato", lw=1.5)
    ax.plot([-0.05, 0.05], [2.25, 2.25], c="steelblue", lw=1.5)
    ax.text(0.11, 4.25, "same me-type,\ndifferent t-type", ha="left", va="center", fontsize=6)
    ax.text(0.11, 2.25, "same t-type,\ndifferent me-type", ha="left", va="center", fontsize=6)

    ax.plot([-0.05, 0.05], [0, 0], c="black", lw=(width_scale * 0.5) * 2)
    ax.plot([-0.05, 0.05], [-1, -1], c="black", lw=(width_scale * 0.1) * 2)
    ax.text(-0.1, 0, "0.5", ha="right", va="center", fontsize=6)
    ax.text(-0.1, -1, "0.1", ha="right", va="center", fontsize=6)
    ax.text(0.11, -0.5, "avg. probability\nof cross-mapping", ha="left", va="center", fontsize=6)


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

    plt.savefig(output_file, bbox_inches="tight", dpi=600)


def plot_met_graph(ax, me_map_pt, t_map_pt, partition, count_dict, order_dict, color_dict, subclass_dict,
        k_scale=10, node_scale=10, width_scale=4, label_offset=10,
        starting_buffer=300, buffer_factor=250, consistency_type = 'me'): #k_scale=10 - swb changed

    if not (consistency_type == 'me' or consistency_type == 't'): 
        consistency_type == 'me'
    me_consistency = {}
    for met in me_map_pt.index.tolist():
        print(met, met_tuple_to_str(tuple(met)))
        if consistency_type == 'me':
            me_consistency[met] = me_map_pt.at[met, met]
        else:
            me_consistency[met] = t_map_pt.at[met, met] # Changing to t-type consistency

    melted_t = pd.melt(t_map_pt.reset_index(), id_vars=["index"]).dropna()
    melted_t = melted_t[melted_t["value"] > 0]
    melted_t.columns= ["assigned_met", "other_met", "value"]
    print(melted_t.head())

    melted_me = pd.melt(me_map_pt.reset_index(), id_vars=["index"]).dropna()
    melted_me = melted_me[melted_me["value"] > 0]
    melted_me.columns= ["assigned_met", "other_met", "value"]

    nodes_in_graph = list(partition.keys())
    undir_edge_list = []
    t_edge_list = sum_directed_edges(melted_t)
    for i in range(len(t_edge_list)):
        node_a = t_edge_list[i][0]
        if node_a not in nodes_in_graph:
            continue
        node_b = t_edge_list[i][1]
        if node_b not in nodes_in_graph:
            continue
        weight = t_edge_list[i][2]["weight"]
        undir_edge_list.append((
            node_a,
            node_b,
            {
                "weight": weight,
                "len": (2 - weight) * k_scale + 0.000001, #swb uncommented for more spacing between nodes (0.01)
                "type": "t",
            }
        ))

    me_edge_list = sum_directed_edges(melted_me)
    for i in range(len(me_edge_list)):
        node_a = me_edge_list[i][0]
        if node_a not in nodes_in_graph:
            continue
        node_b = me_edge_list[i][1]
        if node_b not in nodes_in_graph:
            continue
        weight = me_edge_list[i][2]["weight"]
        undir_edge_list.append((
            node_a,
            node_b,
            {
                "weight": weight,
                "len": (2 - weight) * k_scale + 0.000001, #swb uncommented for more spacing between nodes (0.01)
                "type": "me",
            }
        ))

    G = nx.Graph()
    G.add_nodes_from(nodes_in_graph)
    G.add_edges_from(undir_edge_list)


    print(f"Number of used nodes: {len(nodes_in_graph)}")
    partition_nodes = np.array(nodes_in_graph)
    partition_labels = np.array([partition[n] for n in nodes_in_graph])
    print(f"Identified {len(np.unique(partition_labels))} partitions")

    # print('\n\npartition_nodes: {}'.format(partition_nodes))
    # print('\n\npartitionpartition_labels_nodes: {}'.format(partition_labels))

    layoutG = G.copy()

#     avg_len = np.mean([d for u, v, d in layoutG.edges(data="len")])
#     print("avg len", avg_len)
#     print("avg weight", np.mean([d for u, v, d in layoutG.edges(data="weight")]))

    # add edges to keep partitions together
    for l in np.unique(partition_labels):
        new_partition_edges = []
        nodes_for_partition = list(partition_nodes[partition_labels == l])
        centrality = nx.centrality.degree_centrality(G.subgraph(nodes_for_partition))
        nodes_for_partition = sorted(nodes_for_partition, key=lambda x: centrality[x], reverse=True)

        n_to_connect = 2
        for n0 in nodes_for_partition[:n_to_connect]:
            for n1 in nodes_for_partition[n_to_connect:]:
                if not layoutG.has_edge(n0, n1):
                    new_partition_edges.append((
                        n0,
                        n1,
                        {
                            "weight": 0.001,
#                             "len": avg_len * 0.3,
                            "type": "layout",
                        }
                    ))
        layoutG.add_edges_from(new_partition_edges)

    # Tether the isolates to their own subclass
    isolated_nodes = nx.isolates(layoutG)
    for n in isolated_nodes:
        n_ttype = n.split("|")[-1]
        for n_other in list(layoutG.nodes())[::-1]:
            if n == n_other:
                continue
            n_other_ttype = n_other.split("|")[-1]
            if subclass_dict[n_ttype] == subclass_dict[n_other_ttype]:
                layoutG.add_edge(
                    n,
                    n_other,
                    **{
                        "weight": 0.01,
#                         "len": k_scale * 3,
                        "type": "layout",
                    })
                break

    # Determine positions based on all edges
#     pos = nx.spring_layout(G, k=k_scale * 1/np.sqrt(len(G.nodes())), iterations=2000, weight="weight", seed=42)

    node_colors = [color_dict[n.split("|")[-1]] for n in G.nodes()]
    node_sizes = np.array([count_dict[n] for n in G.nodes()])
    node_me_fractions = np.array([me_consistency[n] for n in G.nodes()])
    node_me_labels = {n: str(n[0]) for n in G.nodes()}

    layout_node_size = np.sqrt(np.mean(node_sizes)) * 2
    pos = nx.nx_agraph.graphviz_layout(layoutG, args=f"-Goverlap=false -Nshape=circle -Nheight={layout_node_size} -Nwidth={layout_node_size} -Gsep=+8")

    me_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d["type"] == "me"]
    me_weights = np.array([d["weight"] for u, v, d in me_edges])

    t_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d["type"] == "t"]
    t_weights = np.array([d["weight"] for u, v, d in t_edges])

    # Draw gray at size n
    nx.draw_networkx_nodes(G,
                           pos,
                           node_size=node_sizes * node_scale,
                           node_color="#bbbbbb",
                           linewidths=0,
                           ax=ax)

    # Draw color at n * me_consistency
    min_fraction = 0.05
    plot_node_me_fractions = [n if n > min_fraction else min_fraction for n in node_me_fractions]
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes * plot_node_me_fractions * node_scale,
        node_color=node_colors,
        linewidths=0,
        ax=ax)

    # Draw ME and T edges
    nx.draw_networkx_edges(G, pos,
                           edgelist=me_edges,
                           edge_color="steelblue", width=me_weights * width_scale,
                           ax=ax)
    nx.draw_networkx_edges(G, pos,
                           edgelist=t_edges,
                           edge_color="tomato", width=t_weights * width_scale,
                           ax=ax)

    sns.despine(left=True, bottom=True)

    partition_polys = []
    points_in_polys = []

    met_centers = {}

    for l in np.unique(partition_labels):
        print('\nl:{}'.format(l))
        met_centers[l] = np.mean([
            [pos[n][0] for n in partition_nodes[partition_labels == l]],
            [pos[n][1] for n in partition_nodes[partition_labels == l]],
        ], axis=1)

        points = [geometry.Point(pos[n]) for n in partition_nodes[partition_labels == l]]
        points_in_polys.append(points)

        tight_border, edge_points = alpha_shape(points, alpha=0.001, base_scale=1000) #alpha=0.1 - swb changed
        looser_border = tight_border.buffer(starting_buffer)
        coords = looser_border.exterior.coords.xy
        partition_polys.append(looser_border)


#######
    # # check for intersections
    # for i, p1 in enumerate(partition_polys):
    #     p1_points = points_in_polys[i]
    #     for j, p2 in enumerate(partition_polys[i + 1:]):
    #         p2_points = points_in_polys[i + 1 + j]
    #         overlap = p1.intersection(p2)
    #         if not overlap.is_empty:
    #             overlap = overlap.buffer(buffer_factor)
    #             overlap_has_p1_points = poly_contains_any_points(overlap, p1_points)
    #             overlap_has_p2_points = poly_contains_any_points(overlap, p2_points)
    #             if overlap_has_p1_points and overlap_has_p2_points:
    #                 overlap = p1.intersection(p2).buffer(0)
    #                 overlap_has_p1_points = poly_contains_any_points(overlap, p1_points)
    #                 overlap_has_p2_points = poly_contains_any_points(overlap, p2_points)


    #             if not overlap_has_p1_points and overlap_has_p2_points:
    #                 new_p1 = p1.difference(overlap).buffer(-buffer_factor).buffer(buffer_factor)
    #                 if not poly_contains_all_points(new_p1, p1_points):
    #                     print("p1 no overlap: problem with buffered p1")
    #                 if new_p1.geom_type == "MultiPolygon":
    #                     for b in (buffer_factor / 2, buffer_factor / 4, 0):
    #                         new_p1 = p1.difference(overlap).buffer(-b).buffer(b)
    #                         if new_p1.geom_type != "MultiPolygon":
    #                             break
    #                 new_p2 = p2
    #             elif overlap_has_p1_points and not overlap_has_p2_points:
    #                 new_p1 = p1
    #                 new_p2 = p2.difference(overlap).buffer(-buffer_factor).buffer(buffer_factor)

    #                 if not poly_contains_all_points(new_p2, p2_points):
    #                     print("p2 no overlap: problem with buffered p2")
    #                 if new_p2.geom_type == "MultiPolygon":
    #                     for b in (buffer_factor / 2, buffer_factor / 4, 0):
    #                         new_p2 = p2.difference(overlap).buffer(-b).buffer(b)
    #                         if new_p2.geom_type != "MultiPolygon":
    #                             break


    #             elif not overlap_has_p1_points and not overlap_has_p2_points:
    #                 if not p1.difference(overlap).buffer(-buffer_factor).is_empty:
    #                     new_p1 = p1.difference(overlap).buffer(-buffer_factor).buffer(buffer_factor)
    #                     if not poly_contains_all_points(new_p1, p1_points):
    #                         print("both no overlap: problem with buffered p1")
    #                         new_p1 = p1
    #                 else:
    #                     new_p1 = p1

    #                 if not p2.difference(overlap).buffer(-buffer_factor).is_empty:
    #                     new_p2 = p2.difference(overlap).buffer(-buffer_factor).buffer(buffer_factor)
    #                     print(type(new_p1))

    #                     if not poly_contains_all_points(new_p2, p2_points):
    #                         print("both no overlap: problem with buffered p2")
    #                         new_p2 = p2
    #                 else:
    #                     new_p2 = p2
    #             else:
    #                 print("both regions having points in overlap")
    #                 poly = PolygonPatch(overlap, fc="firebrick", ec="none", lw=0.5, alpha=0.3, zorder=-2)
    #                 ax.add_patch(poly)


    #                 print(partition_nodes[partition_labels == i])
    #                 print(partition_nodes[partition_labels == i + j + 1])
    #                 new_p1 = p1
    #                 new_p2 = p2
    #             partition_polys[j + i + 1] = new_p2
    #             partition_polys[i] = new_p1
###########

    for p in partition_polys:
        if not p.is_empty:
            poly = PolygonPatch(p, fc="lightgray", ec="#aaaaaa", lw=0.5, alpha=0.3, zorder=-1)
            ax.add_patch(poly)
        else: 
            print('Polygon is empty')
            print('Polygon: {}'.format(p))

    ax.set_aspect("equal")
    return met_centers

def poly_contains_any_points(poly, points):
    return np.any([poly.contains(pt) for pt in points])


def poly_contains_all_points(poly, points):
    return np.all([poly.contains(pt) for pt in points])


def alpha_shape(points, alpha, base_scale=1.):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull, points

    coords = np.array([point.coords[0] for point in points])
    tri = Delaunay(coords)
    triangles = coords[tri.simplices]
    a = ((triangles[:,0,0] - triangles[:,1,0]) ** 2 + (triangles[:,0,1] - triangles[:,1,1]) ** 2) ** 0.5
    b = ((triangles[:,1,0] - triangles[:,2,0]) ** 2 + (triangles[:,1,1] - triangles[:,2,1]) ** 2) ** 0.5
    c = ((triangles[:,2,0] - triangles[:,0,0]) ** 2 + (triangles[:,2,1] - triangles[:,0,1]) ** 2) ** 0.5
    s = ( a + b + c ) / 2.0
    areas = (s*(s-a)*(s-b)*(s-c)) ** 0.5
    circums = a * b * c / (4.0 * areas)
    filtered = triangles[circums < (base_scale / alpha)]
    edge1 = filtered[:,(0,1)]
    edge2 = filtered[:,(1,2)]
    edge3 = filtered[:,(2,0)]
    edge_points = np.unique(np.concatenate((edge1,edge2,edge3)), axis = 0).tolist()
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return unary_union(triangles), edge_points


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=FigMetMatrixParameters)
    main(**module.args)
