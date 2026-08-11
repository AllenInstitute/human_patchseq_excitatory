import numpy as np
import pandas as pd
import argschema as ags
import json
import igraph as ig
import leidenalg as la
from mouse_met_figs.graph import sum_directed_edges

class MetTypesByGraphParameters(ags.ArgSchema):
    tx_anno_file = ags.fields.InputFile()
    cluster_labels_file = ags.fields.InputFile()
    subsample_assignments_file = ags.fields.InputFile()
    tx_mapping_file = ags.fields.InputFile()
    met_node_partition_file = ags.fields.OutputFile()
    met_assignment_file = ags.fields.OutputFile()
    me_map_pt_file = ags.fields.OutputFile()
    t_map_pt_file = ags.fields.OutputFile()


def met_tuple_to_str(name):
    return str(name[0]) + "|" + str(name[1])


def met_str_to_tuple(name):
    split_name = name.split("|")
    return (int(split_name[0]), split_name[1])


def partition_met_graph(me_map_pt, t_map_pt, count_dict):

    melted_t = pd.melt(t_map_pt.reset_index(), id_vars=["assigned_met"]).dropna()
    melted_t = melted_t[melted_t["value"] > 0]

    melted_me = pd.melt(me_map_pt.reset_index(), id_vars=["assigned_met"]).dropna()
    melted_me = melted_me[melted_me["value"] > 0]

    undir_edge_list = sum_directed_edges(melted_t)
    undir_edge_list += sum_directed_edges(melted_me)

    G = ig.Graph()
    vertex_list = [met_tuple_to_str(i) for i in me_map_pt.index.tolist()]
    G.add_vertices(vertex_list)
    edge_list_for_igraph = [(met_tuple_to_str(e[0]), met_tuple_to_str(e[1])) for e in undir_edge_list]
    G.add_edges(edge_list_for_igraph)
    G.es["weight"] = [e[2]["weight"] for e in undir_edge_list]

    isolated_vertices = G.vs.select(_degree=0)
    vertices_to_remove = []
    for v in isolated_vertices:
        n = met_str_to_tuple(v["name"])
        print(n)
        if count_dict[n] < 2:
            vertices_to_remove.append(v["name"])
    print("removing isolated vertices with only one cell")
    print(vertices_to_remove)
    G.delete_vertices(vertices_to_remove)

    partition = la.find_partition(G, la.ModularityVertexPartition, weights="weight")
    print(partition)
    memb = dict(zip(
        [v["name"] for v in G.vs],
        partition.membership,
    ))

    return memb


def main(tx_anno_file, cluster_labels_file, subsample_assignments_file,
        tx_mapping_file, met_node_partition_file, met_assignment_file, me_map_pt_file, t_map_pt_file,
        **kwargs):
    tx_anno = pd.read_csv(tx_anno_file).set_index("cell_id")

    me_labels_df = pd.read_csv(cluster_labels_file, index_col=0)

    subsample_assignments_df = pd.read_csv(subsample_assignments_file, index_col=0)
    subsample_assignments_norm = subsample_assignments_df / subsample_assignments_df.sum(axis=1).values[:, np.newaxis]

    tx_mapping_df = pd.read_csv(tx_mapping_file).set_index("cell_id")

    observed_met_combinations = []
    count_dict = {}
    for sid in me_labels_df.index:
        me = me_labels_df.at[sid, "0"]
#         me_num = int(me.split("_")[-1])
        me_num = int(me) + 1
        ttype = tx_anno.at[sid, "cluster_label"]

        if (me_num, ttype) not in count_dict:
            count_dict[(me_num, ttype)] = 1
        else:
            count_dict[(me_num, ttype)] += 1

        observed_met_combinations.append((me_num, ttype))

    observed_met_combinations = set(observed_met_combinations)

    me_map_results = []
    t_map_results = []
    for sid in me_labels_df.index:
        me = me_labels_df.at[sid, "0"]
#         me_num = int(me.split("_")[-1])
        me_num = int(me) + 1

        ttype = tx_anno.at[sid, "cluster_label"]

        map_values = tx_mapping_df.loc[sid, :]
        for t in map_values.index:
            if (me_num, t) in observed_met_combinations:
                t_map_results.append({
                    "assigned_met": (me_num, ttype),
                    "other_met": (me_num, t),
                    "value": map_values[t],
                })
        for j in range(subsample_assignments_norm.shape[1]):
            if (j + 1, ttype) in observed_met_combinations:
                me_map_results.append({
                    "assigned_met": (me_num, ttype),
                    "other_met": (j + 1, ttype),
                    "value": subsample_assignments_norm.at[sid, str(j)],
                })

    me_map_df = pd.DataFrame(me_map_results)
    t_map_df = pd.DataFrame(t_map_results)

    me_map_pt = me_map_df.pivot_table(values="value", index="assigned_met", columns="other_met")
    t_map_pt = t_map_df.pivot_table(values="value", index="assigned_met", columns="other_met")

    memb = partition_met_graph(me_map_pt, t_map_pt, count_dict)
    output_count_dict = {met_tuple_to_str(k): v for k, v in count_dict.items()}
    with open(met_node_partition_file, "w") as f:
        json.dump(
            {"counts": output_count_dict, "membership": memb},
            f, indent=4
        )

    assignments = []
    for sid in me_labels_df.index:
        me = me_labels_df.at[sid, "0"]
#         me_num = int(me.split("_")[-1])
        me_num = int(me) + 1
        ttype = tx_anno.at[sid, "cluster_label"]

        met_node = (me_num, ttype)
        met_node_str = met_tuple_to_str(met_node)
        if met_node_str in memb:
            assignments.append((sid, memb[met_node_str]))
        else:
            continue

    print("saving assignments")
    assign_df = pd.DataFrame(assignments, columns=["specimen_id", "met_type"]).set_index("specimen_id")
    assign_df.to_csv(met_assignment_file)

    print("saving pivot tables")

    me_map_pt.index = [met_tuple_to_str(met) for met in me_map_pt.index]
    me_map_pt.columns = [met_tuple_to_str(met) for met in me_map_pt.columns]
    me_map_pt.to_csv(me_map_pt_file)

    t_map_pt.index = [met_tuple_to_str(met) for met in t_map_pt.index]
    t_map_pt.columns = [met_tuple_to_str(met) for met in t_map_pt.columns]
    t_map_pt.to_csv(t_map_pt_file)

if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=MetTypesByGraphParameters)
    main(**module.args)
