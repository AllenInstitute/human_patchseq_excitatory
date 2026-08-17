# This module is from https://github.com/AllenInstitute/mouse-met-figures
import numpy as np
import pandas as pd
import sklearn.ensemble as ensemble
import sklearn.model_selection as model_selection
import argschema as ags


class RfSubsampleMeAssigmentsParameters(ags.ArgSchema):
    n_iter = ags.fields.Integer(default=100)
    n_folds = ags.fields.Integer(default=10)
    ephys_data_file = ags.fields.InputFile()
    morph_data_file = ags.fields.InputFile()
    cluster_labels_file = ags.fields.InputFile()
    assignments_file = ags.fields.OutputFile()


def main(n_iter, n_folds, ephys_data_file, morph_data_file, cluster_labels_file, assignments_file, **kwargs):
    clusts_df = pd.read_csv(cluster_labels_file, index_col=0)

    ephys_data = pd.read_csv(ephys_data_file, index_col=0)
    morph_data = pd.read_csv(morph_data_file, index_col=0)
    ephys_morph_ids = ephys_data.index.intersection(morph_data.index)


    morph_vals = morph_data.loc[ephys_morph_ids, :].values
    ephys_vals = ephys_data.loc[ephys_morph_ids, :].values
    me_data = np.hstack([morph_vals, ephys_vals])
    me_df = pd.DataFrame(me_data, index=ephys_morph_ids)
    me_df["cluster"] = clusts_df.loc[me_df.index, "0"].values.astype(str)

    rf = ensemble.RandomForestClassifier(n_estimators=200)
    labels_for_rf = np.array([int(l.split("_")[-1]) - 1 for l in me_df["cluster"]])

    n_clusters = len(me_df["cluster"].unique())
    subsample_assignments = np.zeros((me_data.shape[0], n_clusters))

    for i in range(n_iter):
        print(i)
        kf = model_selection.KFold(n_splits=n_folds, shuffle=True, random_state=i)
        for train_ind, test_ind in kf.split(labels_for_rf):
            rf.fit(me_data[train_ind, :], labels_for_rf[train_ind])
            y_pred = rf.predict(me_data[test_ind, :])
            subsample_assignments[test_ind, y_pred] += 1

    pd.DataFrame(subsample_assignments, index=ephys_morph_ids).to_csv(assignments_file)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=RfSubsampleMeAssigmentsParameters)
    main(**module.args)
