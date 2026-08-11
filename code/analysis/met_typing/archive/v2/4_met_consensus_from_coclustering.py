import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from drcme.ephys_morph_clustering import coclust_rates, consensus_clusters
import argschema as ags

class MetConsensusFromCoclusteringParameters(ags.ArgSchema):
    met_type_runs_file = ags.fields.InputFile()
    consensus_type_file = ags.fields.OutputFile()

def main(args):
    met_type_runs_df = pd.read_csv(args["met_type_runs_file"], index_col=0)

    clust_labels, shared_norm, cc_rates = consensus_clusters(
        met_type_runs_df.values, min_clust_size=5)

    # find singletons that ended up assigned to a cluster
    isolated_ids_to_remove = []
    for i in range(shared_norm.shape[0]):
        self_mask = np.ones(shared_norm.shape[0]).astype(bool)
        self_mask[i] = False
        cl = clust_labels[i]
        cluster_mask = clust_labels == cl
        mask = cluster_mask & self_mask
        if shared_norm[i, mask].mean() <= 0.3:
            print("isolated", i, cl)
            isolated_ids_to_remove.append(met_type_runs_df.index.values[i])

    consensus_met_df = pd.DataFrame(
        {"met_type": clust_labels}, index=met_type_runs_df.index)
    consensus_met_df = consensus_met_df.loc[
        ~consensus_met_df.index.isin(isolated_ids_to_remove), :].copy()
    consensus_met_df.to_csv(args["consensus_type_file"])


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=MetConsensusFromCoclusteringParameters)
    main(module.args)
