# This module is from https://github.com/AllenInstitute/mouse-met-figures

import json
import numpy as np
import pandas as pd

def dict_from_facs(facs_anno_df, key_column="cluster_label", value_column="subclass_label"):
    """ Create a dict of cluster/label values

    Parameters
    ----------
    facs_anno_df: DataFrame
        FACS annotation information

    key_column: str (optional, default "cluster_label")
        Name of column to be keys of dictionary

    value_column: str (optional, default "subclass_label")
        Name of column to be values of dictionary

    Returns
    -------
    dict

    """

    info = facs_anno_df.loc[:, [key_column, value_column]].drop_duplicates()
    result = dict(zip(info[key_column].tolist(), info[value_column].tolist()))
    return result


def intersect_ephys_morph(ephys_df, morph_df):
    morph_ids = morph_df.index.values

    joint_ids = morph_df.index.intersection(ephys_df.index)
    # Get ephys data for cells with morphologies
    ephys_df_joint = ephys_df.loc[joint_ids, :]
    morph_df_joint = morph_df.loc[joint_ids, :]

    # Only use morphs that have ephys
    mask = [s in ephys_df_joint.index.tolist() for s in morph_ids]

    elmo_data = np.hstack([morph_df_joint.values,  ephys_df_joint.values])
    return pd.DataFrame(elmo_data, index=joint_ids)


def get_mMET_annotation(mouse_anno_feather_file, keys="spec_id_label"):
    """Loads in mouse Patch-seq annotation data.

    Returns
    -------
        Pandas DataFrame
    """
    anno_df = pd.read_feather(mouse_anno_feather_file)
    anno_df.set_index(keys=keys, drop=True, inplace=True)
    if keys == "spec_id_label":
        anno_df.index = anno_df.index.values.astype(int)
    return anno_df


def get_facs_annotation(facs_anno_feather_file):
    """Loads in FACS annotation.


    Returns
    -------
        Pandas DataFrame

    """
    facs_anno_df = pd.read_feather(facs_anno_feather_file)
    return facs_anno_df

def make_mask(df, df_field, allowed_list):
    # Filter data if necessary
    if allowed_list is not None and len(allowed_list) > 0:
        mask = df[df_field].isin(allowed_list).values
    else:
        mask = np.ones(df.shape[0], dtype=bool)
    return mask

def get_tx_masks(tx_anno_df, specimens_file, required_tree_calls, required_subclasses):
    """Get various masks to limit transcriptomic data by specific inclusion criteria.
    
    Parameters
    ----------
        tx_anno_df (Pandas DataFrame): transcriptomic annotations
        specimens_file (str or None): txt file with specimens to use (optional)
        required_tree_calls (list of strings or None): tree call categories to restrict data (optional)
        required_subclasses (list of strings or None): transcriptomic subcalsses to restrict data (optional)
    """
    if required_tree_calls is not None and len(required_tree_calls) > 0:
        tree_mask = tx_anno_df["Tree_call_label"].isin(required_tree_calls).values
    else:
        tree_mask = np.ones(tx_anno_df.shape[0], dtype=bool)
    
    if required_subclasses is not None and len(required_subclasses) > 0:
        temp = tx_anno_df.copy()
        subclass_mask = temp["subclass"].isin(required_subclasses).values
    else:
        subclass_mask = np.ones(tx_anno_df.shape[0], dtype=bool)

    if specimens_file is not None:
        with open(specimens_file, 'r') as tf:
            str_ids = tf.readlines()
        specimen_ids = [int(s_id) for s_id in str_ids]
        specimens_mask = tx_anno_df["spec_id_label"].apply(int).isin(specimen_ids).values
    else:
        specimen_ids = None
        specimens_mask = np.ones(tx_anno_df.shape[0], dtype=bool)

    return tree_mask, subclass_mask, specimens_mask

def reduce_tx_dims(n_dims, data, scale_features=False):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = data.values
    
    if scale_features == True:
        X = StandardScaler().fit_transform(X)
        print("Standardizing featues with StandardScaler")
    
    print(F"Original number of features: {X.shape[1]}")
    pca = PCA(n_components=n_dims, random_state=43)
    X_pca = pca.fit_transform(X)

    print(F"Number of features after PCA: {X_pca.shape[1]}")
    print(F"Total explained variance for {n_dims} components: {sum(pca.explained_variance_ratio_)}")

    return X_pca, pca.explained_variance_ratio_, pca.components_

def get_subclass_ids(tx_anno_df, subclass_dict, subclass, required_tree_calls=None):
    temp = tx_anno_df.copy()
    mask = make_mask(tx_anno_df, "Tree_call_label", required_tree_calls)
    masked_temp = temp.loc[mask,:]
    subclasses = masked_temp["Tree_first_cl_label"].apply(lambda x: subclass_dict[x] if x in subclass_dict.keys() else "None")
    masked_temp["subclass"] = subclasses
    subclass_ids = masked_temp[masked_temp["subclass"]==subclass].index.values
    return subclass_ids

def get_ttype_ids(tx_anno_df, ttype, required_tree_calls=None):
    temp = tx_anno_df.copy()
    mask = make_mask(tx_anno_df, "Tree_call_label", required_tree_calls)
    masked_temp = temp.loc[mask,:]
    ttype_ids = masked_temp[masked_temp["Tree_first_cl_label"]==ttype].index.values
    return ttype_ids

def t_types_for_ids(specimen_ids, anno_df, t_type_key="Tree_first_cl_label"):
    """Returns t_types from anno_df for given specimen_ids

    Parameters
    ----------
        specimen_ids (list): list of specimen IDs
        anno_df (Pandas DataFrame): MET annotation
        t_type_key (str, optional): Which cluster label to use in anno_df. Defaults to "Tree_first_cl_label".

    Returns
    -------
        list: t types

    """
    ids_in_df = anno_df.index.tolist()
    t_types = [anno_df.at[spec_id, t_type_key] if spec_id in ids_in_df else "not found"
        for spec_id in specimen_ids]
    return t_types

def get_labels_used(labels, min_size=20):
    """Returns list of labels that satisfy a minimum sample size per label.

    Parameters
    ----------
        labels (NumPy array): a sequence of labels
        min_size (int, optional): Minimum sample size per label. Defaults to 20.

    Returns
    -------
        list: a unique sequence of labels

    """
    label_counts = pd.Series(labels).value_counts()
    labels_used = label_counts[label_counts >= min_size].index.tolist()
    return labels_used


def get_min_sample_mask(labels, min_size=20):
    """Returns mask describing which elements to keep to satisfy minimum sample size per label.

    Parameters
    ----------
        labels (NumPy array): a sequence of labels
        min_size (int, optional): Minimum sample size per label. Defaults to 20.

    Returns
    -------
        NumPy array: Booleans

    """
    labels_used = get_labels_used(labels, min_size)
    min_size_mask = np.array([i in labels_used for i in labels])
    return min_size_mask


def assign_structure_group(structure_label):
    """Group structures into VISp, higher_VIS, non_VIS and no_label.

    Parameters
    ----------
        structure_label (str): structure label from LIMS or Shiny

    Returns
    -------
        str: VISp, higher_VIS, non_VIS or no_label

    """
    import re
    pinning_pattern = r'^([A-Za-z-]*)([2][\/][3]|[1-6]{1}[a-b]?)*$'

    structure_mappings = dict([(r'^([A-Za-z-]*|ZZ_Missing)$','no_label'),
                               (r'^(VISp)([2][\/][3]|[1-6]{1}[a-b]?)+$','VISp'),
                               (r'^(VIS[a-oq-z]{1}[a-z]*|VISp[a-z]+)([2][\/][3]|[1-6]{1}[a-b]?)+$','higher_VIS'),
                               (r'^(?!VIS)([a-zA-Z-]*)([2][\/][3]|[1-6]{1}[a-b]?)+$','non_VIS')])
    match = None
    for i, (pattern, str_map) in enumerate(structure_mappings.items()):
            if re.match(pattern, structure_label):
                match = str_map
    return match


def component_column_lookup(component_info_file):
    with open(component_info_file, "r") as f:
        comp_info = json.load(f)

    col_lookup = {}
    counter = 0
    for ci in comp_info:
        col_lookup[ci["key"]] = [str(counter + i) for i, v in enumerate(ci["indices"])]
        counter += len(ci["indices"])
    return col_lookup


def cluster_colors(df, change_chodl=True,
        cluster_key="cluster_label", color_key="cluster_color"):
    color_dict = dict(zip(df[cluster_key], df[color_key]))

    if change_chodl:
        if "Sst Chodl" in color_dict:
            color_dict["Sst Chodl"] = "#b9bb67"

    return color_dict


def cluster_subclasses(df, change_serpinf1=True,
        cluster_key="cluster_label", subclass_key="subclass_label"):
    subclass_dict = dict(zip(df[cluster_key], df[subclass_key]))

    if change_serpinf1:
        for k in subclass_dict:
            if subclass_dict[k] == "Serpinf1":
                subclass_dict[k] = "Sncg"

    return subclass_dict
