import numpy as np
import pandas as pd
import json
import argschema as ags
from sklearn.cluster import SpectralCoclustering
import mouse_met_figs.utils as utils


class SpectralCoclustMeTtypeParameters(ags.ArgSchema):
    tx_anno_file = ags.fields.InputFile(
        default="/allen/programs/celltypes/workgroups/ivscc/agatab/data/status_reports/basal_ganglia/met_types/20240621_BT014-RSC-366_mouse_patchseq_star2.7_cpm_WB21_mapping.csv",
        description="csv file with patch-seq transcriptomic annotations")
    me_cluster_labels_file = ags.fields.InputFile(
        default="/allen/programs/celltypes/workgroups/ivscc/nathang/me_clust_output/bg_mouse_me_clustering/refined_text_labels.csv",
        description="csv file with cluster labels")
    output_file = ags.fields.OutputFile(
        default="spectral_coclust_results.json",
        description="output file")
    min_n_ttype = ags.fields.Integer(
        default=1,
        description="minimum n for t-type inclusion")


def cluster_data_set(me_vs_ttype, min_n_ttype, subclass=None, type_subclass_dict=None):
    all_ttypes = me_vs_ttype.index.values
    if subclass:
        print("sc", subclass)
        ttypes = [t for t in all_ttypes if type_subclass_dict[t] == subclass]
    else:
        print("all")
        ttypes = all_ttypes.tolist()

    counts = me_vs_ttype.loc[ttypes, :].sum(axis=1)
    ttypes = counts[counts >= min_n_ttype].index.values
    sub_df = me_vs_ttype.loc[ttypes, :]
    sub_df = sub_df.loc[:, sub_df.sum(axis=0) > 0]
    max_k = min(sub_df.shape)
    print(sub_df)
    data = sub_df.values

    print("data shape", data.shape)
    result = {
        "ttypes": ttypes.tolist(),
        "me_clusters": sub_df.columns.tolist(),
        "spectral_results": [],
    }

    zero_factors = sub_df.loc[ttypes, :].values.sum(axis=1) / sub_df.shape[1]
    weighted_zero_in_cluster = np.sum((sub_df.loc[ttypes, :].values == 0).sum(axis=1) * zero_factors)
    result["spectral_results"].append({
        "k": 1,
        "n_biclust": 1,
        "row_labels": np.zeros(len(ttypes)).tolist(),
        "column_labels":np.zeros(sub_df.shape[1]).tolist(),
        "nonzeros_out_of_cluster": 0,
        "zeros_in_cluster": int(np.sum(sub_df.loc[ttypes, :].values == 0)),
        "weighted_nonzeros_out_of_cluster": 0,
        "weighted_zeros_in_cluster": weighted_zero_in_cluster,
    })

    print(data)
    for k in range(2, max_k):
        print("k", k)
        model = SpectralCoclustering(n_clusters=k, n_init=500, init="random", svd_method="randomized")
        model.fit(data)

        # check if any row biclusters have zero columns
        # or if they are assigned to bicluster with no cells actually in it
#         problem_row_bicluster = False
#         for r, c in zip(model.rows_, model.columns_):
#             if np.any(r) and not np.any(c):
#                 print(len(r), len(c))
#                 print("found a bicluster with rows but no columns")
#                 print(sub_df.index.values[r])
# #                 print(sub_df.columns.values[c])
#                 problem_row_bicluster = True
#                 break
#             inds = np.flatnonzero(r)
#             for i in inds:
#                 n_cells_in_bicluster = sub_df.iloc[i, c].values.sum()
#                 if n_cells_in_bicluster == 0:
#                     print("found a bicluster with a row with no cells")
#                     print(sub_df.index.values[i])
#                     problem_row_bicluster = True
#                     break
#         if problem_row_bicluster:
#             continue


        n_row_biclusters = len(np.unique(model.row_labels_))
        print("n biclusters", n_row_biclusters)
        n_nonzero_out_of_cluster = 0
        n_zero_in_cluster = 0
        weighted_nonzero_out_of_cluster = 0
        weighted_zero_in_cluster = 0
        for l in np.unique(model.row_labels_):
            row_mask = model.row_labels_ == l
            col_mask = model.column_labels_ == l
            nonclust_cols = sub_df.columns[model.column_labels_ != l]
            n_nonzero_out_of_cluster += np.sum(sub_df.loc[row_mask, ~col_mask].values > 0)
            nonzero_mask = sub_df.loc[row_mask, ~col_mask].values > 0
            weighted_nonzero_out_of_cluster += sub_df.loc[row_mask, ~col_mask].values[nonzero_mask].sum()

            n_zero_in_cluster += np.sum(sub_df.loc[row_mask, col_mask].values == 0)
            zero_factors = sub_df.loc[row_mask, :].values.sum(axis=1) / sub_df.shape[1]
            weighted_zero_in_cluster += np.sum((sub_df.loc[row_mask, col_mask].values == 0).sum(axis=1) * zero_factors)

        print("nonzero out", n_nonzero_out_of_cluster)
        print("zero in", n_zero_in_cluster)
        print("weighted nonzero out", weighted_nonzero_out_of_cluster)
        print("weighted zero in", weighted_zero_in_cluster)
        result["spectral_results"].append({
            "k": k,
            "n_biclust": n_row_biclusters,
            "row_labels": model.row_labels_.tolist(),
            "column_labels": model.column_labels_.tolist(),
            "nonzeros_out_of_cluster": int(n_nonzero_out_of_cluster),
            "zeros_in_cluster": int(n_zero_in_cluster),
            "weighted_nonzeros_out_of_cluster": float(weighted_nonzero_out_of_cluster),
            "weighted_zeros_in_cluster": float(weighted_zero_in_cluster),
        })

    print(result["spectral_results"])
    return result

def main(tx_anno_file, me_cluster_labels_file, min_n_ttype, output_file, **kwargs):
    tx_anno_df = pd.read_csv(tx_anno_file, index_col=0)

    type_subclass_dict = dict(zip(tx_anno_df.cluster_label, tx_anno_df.subclass_label))

    me_labels_df = pd.read_csv(me_cluster_labels_file, index_col=0)
    me_labels_df = me_labels_df.join(
        tx_anno_df[["cluster_label", "subclass_label"]],
        how="left")

    print(me_labels_df.head())

    me_vs_ttype = pd.crosstab(index=me_labels_df["cluster_label"], columns=me_labels_df["0"])

    all_ttypes = me_vs_ttype.index.values
    subclasses = me_labels_df.subclass_label.unique()
    print(subclasses)

    results = {}
#     results["all"] = cluster_data_set(me_vs_ttype, min_n_ttype)

    for sc in subclasses:
        results[sc] = cluster_data_set(
            me_vs_ttype,
            min_n_ttype,
            subclass=sc,
            type_subclass_dict=type_subclass_dict)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=SpectralCoclustMeTtypeParameters)
    main(**module.args)
