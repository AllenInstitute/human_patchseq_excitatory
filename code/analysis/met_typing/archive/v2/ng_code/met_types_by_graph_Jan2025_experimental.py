import numpy as np
import pandas as pd
import argschema as ags
import json
import igraph as ig
import leidenalg as la
from mouse_met_figs.graph import sum_directed_edges, met_tuple_to_str, met_str_to_tuple


class MetTypesByGraphParameters(ags.ArgSchema):
    tx_anno_file = ags.fields.InputFile()
    cluster_labels_file = ags.fields.InputFile()
    subsample_assignments_file = ags.fields.InputFile()
    tx_mapping_file = ags.fields.InputFile()
    dendrogram_file = ags.fields.InputFile()
    ephys_corr_file = ags.fields.InputFile()
    morph_corr_file = ags.fields.InputFile()
    genes_corr_file = ags.fields.InputFile()
    met_cell_weight_factor = ags.fields.Float(default=2.0)
    resolution_parameter = ags.fields.Float(default=1.0)
    min_weight = ags.fields.Float(default=0.05)
    met_node_partition_file = ags.fields.OutputFile()
    met_assignment_file = ags.fields.OutputFile()
    required_tree_calls = ags.fields.List(ags.fields.String,
        cli_as_single_argument=True,
        default=["Core", "I1", "I2", "I3"])


def partition_met_graph(vertex_list, edge_list, weight_list, resolution_parameter,
    seeds=[1234, 4224, 1001, 4096, 8765, 3091, 981, 2920, 7738, 8898,
        5678, 9918, 3495, 5837, 5480, 8965, 34, 6936, 7292, 8740]):
    G = ig.Graph()
    G.add_vertices(vertex_list)
    G.add_edges(edge_list)
    G.es["weight"] = weight_list
    
#     isolated_vertices = G.vs.select(_degree=0)
#     vertices_to_remove = []
#     for v in isolated_vertices:
#         n = met_str_to_tuple(v["name"])
#         print(n)
#         if count_dict[n] < 2:
#             print("removing n of 1 isolated vertex ", v["name"])
#             vertices_to_remove.append(v["name"])
#     G.delete_vertices(vertices_to_remove)

    # CPMVertexPartition, ModularityVertexPartition, RBConfigurationVertexPartition
    memb_list = []
    for seed in seeds:
        partition = la.find_partition(G,
            la.ModularityVertexPartition,
            weights="weight",
            seed=seed)
#         print(partition)
        memb = dict(zip(
            [v["name"] for v in G.vs],
            partition.membership,
        ))
    
        memb_list.append({"seed": seed, "memb": memb})
    return memb_list
#     labels = [v if "-" in v else "" for v in G.vs["name"]]
#     ig.plot(partition,
#         layout=G.layout("kamada_kawai"),
#         vertex_size=10,
#         vertex_label=labels,
#         vertex_label_size=8,
#     target="test_met_graph_plot.pdf")
        


def main(args):
    tx_anno = pd.read_feather(args["tx_anno_file"])
    tx_anno = tx_anno.loc[tx_anno["spec_id_label"] != "ZZ_Missing", :]
    tx_anno["spec_id_label"] = pd.to_numeric(tx_anno["spec_id_label"])
    tx_anno.set_index("spec_id_label", inplace=True)

    me_labels_df = pd.read_csv(args["cluster_labels_file"], index_col=0)
    me_labels_df["me_num"] = [int(me.split("_")[-1]) - 1 for me in me_labels_df["0"]]

    subsample_assignments_df = pd.read_csv(args["subsample_assignments_file"], index_col=0)
    subsample_assignments_norm = subsample_assignments_df / subsample_assignments_df.sum(axis=1).values[:, np.newaxis]

    if args["tx_mapping_file"][-3:] == "csv":
        tx_mapping_df = pd.read_csv(args["tx_mapping_file"], index_col=0)
    else:
        tx_mapping_df = pd.read_feather(args["tx_mapping_file"]).set_index("sample_id")
    with open(args["dendrogram_file"], "r") as f:
        dendrogram_info = json.load(f)
    leaves = [d for d in dendrogram_info if dendrogram_info[d][0] is None]
    tx_mapping_df = tx_mapping_df.loc[:, tx_mapping_df.columns.intersection(leaves)]

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
    
    
#     median_corr = np.median(
#         np.vstack([
#             ephys_corr_df.values.flatten(),
#             morph_corr_df.values.flatten(),
#             genes_corr_df.values.flatten()]),
#         axis=0).reshape(ephys_corr_df.values.shape)
#     avg_corr_df = pd.DataFrame(median_corr, index=ephys_corr_df.index, columns=ephys_corr_df.columns)
#     print(avg_corr_df.values[:5, :5])

    me_t_labels_df = pd.merge(
        me_labels_df,
        tx_anno.loc[:, ["Tree_first_cl_label"]],
        left_index=True, right_index=True)

    unique_combos = me_t_labels_df.reset_index().loc[:, ["me_num", "Tree_first_cl_label"]].drop_duplicates().sort_values("me_num")
    
    # Vertices include t-types, me-types, and cells
    # Cells are connected to me or t types by their mapping probabilities
    
    t_type_verts = unique_combos["Tree_first_cl_label"].unique().tolist()
    me_type_verts = unique_combos["me_num"].unique().tolist()
    me_type_verts = [f"ME_{me}" for me in me_type_verts]

    # Trying with just the ones that are observed
#     me_t_type_verts = [met_tuple_to_str((r.me_num, r.Tree_first_cl_label))
#         for i, r in unique_combos.iterrows()]
    

    cell_verts = me_t_labels_df.index.values.tolist()
    cell_verts = [str(i) for i in cell_verts]
    
    cell_to_me_t_edge_list = []
    cell_to_me_t_weight_list = []
    used_me_t_type_verts = []
    counts_per_me_t_vert = {}
    for specimen_id in me_t_labels_df.index.values:
        # find t mapping edges
        sample_id = tx_anno.at[specimen_id, "sample_id"]
        cell_mapping_t = tx_mapping_df.loc[sample_id, :]
        cell_mapping_me = subsample_assignments_norm.loc[specimen_id, :]

        t_nonzero_mask = cell_mapping_t > 0
        me_nonzero_mask = cell_mapping_me > 0
        for t in cell_mapping_t.index[t_nonzero_mask]:
            if t not in t_type_verts:
                continue
            for me in cell_mapping_me.index[me_nonzero_mask]:
                me_t_vertex_name = met_tuple_to_str((me, t))
                if me_t_vertex_name in counts_per_me_t_vert:
                    counts_per_me_t_vert[me_t_vertex_name] += 1
                else:
                    counts_per_me_t_vert[me_t_vertex_name] = 1
                combo_weight = cell_mapping_t[t] * cell_mapping_me[me]
                if combo_weight < args["min_weight"]:
                    continue
                used_me_t_type_verts.append(me_t_vertex_name)
                cell_to_me_t_edge_list.append((str(specimen_id), me_t_vertex_name))
                cell_to_me_t_weight_list.append(combo_weight * args["met_cell_weight_factor"])

    print("edges between cells and me/t nodes: ", len(cell_to_me_t_edge_list))
    me_t_type_verts = np.unique(used_me_t_type_verts).tolist()
    
    # add correlation-based edges
    inds = np.tril_indices_from(avg_corr_df.values, k=-1)
    threshold_quantiles = [0.98, 0.985, 0.99]
    
    memb_results = []
    for tq in threshold_quantiles:
        corr_thresh = np.quantile(avg_corr_df.values[inds], tq)
        print("Corr. threshold quantile:", tq, corr_thresh)
        corr_mask = avg_corr_df.values[inds] > corr_thresh       
        print(f"{corr_mask.sum()} between-cell edges")
        
        edge_list = []
        weight_list = []
        for i, j in zip(inds[0][corr_mask], inds[1][corr_mask]):
            ttype_i = tx_anno.loc[avg_corr_df.index[i], "Tree_first_cl_label"]
            ttype_j = tx_anno.loc[avg_corr_df.index[j], "Tree_first_cl_label"]
            metype_i = me_labels_df.loc[avg_corr_df.index[i], "me_num"]
            metype_j = me_labels_df.loc[avg_corr_df.index[j], "me_num"]
            ttypes_to_check = []
            me_types_to_check = []
            if (ttype_i in ttypes_to_check) or (ttype_j in ttypes_to_check):
                print(ttype_i, ttype_j, avg_corr_df.iloc[i, j])
                print(metype_i, metype_j)
                print(ephys_corr_df.iloc[i, j], morph_corr_df.iloc[i, j], genes_corr_df.iloc[i, j])
            if (metype_i in me_types_to_check) or (ttype_j in me_types_to_check):
                print(ttype_i, ttype_j, avg_corr_df.iloc[i, j])
                print(metype_i, metype_j)
                print(ephys_corr_df.iloc[i, j], morph_corr_df.iloc[i, j], genes_corr_df.iloc[i, j])
            edge_list.append((str(avg_corr_df.index[i]), str(avg_corr_df.index[j])))
            weight_list.append(avg_corr_df.iloc[i, j])
            
        all_verts = me_t_type_verts + cell_verts
        edge_list += cell_to_me_t_edge_list
        weight_list += cell_to_me_t_weight_list
        memb_list = partition_met_graph(all_verts, edge_list, weight_list, args["resolution_parameter"])
        memb_results.append({
            "threshold_quantile": float(tq),
            "correlation_threshold": float(corr_thresh),
            "n_corr_edges": int(corr_mask.sum()),
            "memb_list": memb_list,
        })
    with open(args["met_node_partition_file"], "w") as f:
        json.dump(
            memb_results,
            f, indent=4
        )

    assignment_results = {}
    for tq_set in memb_results:
        tq = tq_set["threshold_quantile"]
        for r in tq_set["memb_list"]:
            seed = r["seed"]
            key = f"run_{tq:0.3f}_{seed}"
            memb = r["memb"]
            assignment_results[key] = [memb[str(sid)] for sid in me_labels_df.index]
            
    print("saving assignments")
    assign_df = pd.DataFrame(assignment_results, index=me_labels_df.index)
    assign_df.to_csv(args["met_assignment_file"])



if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=MetTypesByGraphParameters)
    main(module.args)
