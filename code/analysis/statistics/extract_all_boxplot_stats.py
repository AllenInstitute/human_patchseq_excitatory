#!/usr/bin/env python3
"""
Extract ALL boxplot statistics needed for the statistics table.

This script generates statistics for:
- bySubclass (already exists, but we'll verify)
- byLayer
- bySubclass_Layer6
- L5_IT_byTtype
- Layer_4_byTtype

For both morphology (dendrite) and electrophysiology features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import kruskal
from statsmodels.stats.multitest import fdrcorrection
from scikit_posthocs import posthoc_dunn
import warnings
warnings.filterwarnings('ignore')

# Paths
SCRATCH_DIR = Path('/scratch/20260807')
DATA_DIR = Path('/root/capsule/data')
DENDRITE_DIR = SCRATCH_DIR / 'dendrite'
EPHYS_DIR = SCRATCH_DIR / 'ephys'

# Create output directories
DENDRITE_DIR.mkdir(parents=True, exist_ok=True)
EPHYS_DIR.mkdir(parents=True, exist_ok=True)

# Subclass and layer orders (from notebooks)
subclass_order = [
    'L2/3 IT', 'L4 IT', 'L5 IT', 'L5 ET',
    'L5/6 NP', 'L6 IT', 'L6 IT Car3', 'L6 CT', 'L6b'
]

layer_order = [
    'Layer1', 'Layer2', 'Layer3', 'Layer4',
    'Layer5', 'Layer6'
]


def load_morphology_data():
    """Load morphology feature table and metadata."""
    print("Loading morphology data...")

    # Load metadata
    metadata_path = DATA_DIR / 'human' / 'metadata' / '20260807' / 'human_metadata.csv'
    metadata = pd.read_csv(metadata_path, encoding='latin1')
    metadata = metadata.rename(columns={
        "cell_id": "specimen_id",
        "subclass_label_Hierarchical": "subclass_label",
        "cluster_label_Hierarchical": "t_type",
        "cortex_layer": "layer"
    })
    metadata = metadata.dropna(subset=['specimen_id'])
    metadata['specimen_id'] = metadata['specimen_id'].astype('int')
    metadata['subclass_label'] = metadata['subclass_label'].str.replace('L5 ET_ME', 'L5 ET')
    metadata['subclass_label'] = metadata['subclass_label'].str.replace('L5 ET(ME)', 'L5 ET')
    metadata = metadata.set_index('specimen_id')

    # Load feature table
    feature_path = DATA_DIR / 'human' / 'morphology' / 'dendrite' / '20260712' / 'RawFeatureWide_imputed.csv'
    features_df = pd.read_csv(feature_path, index_col='specimen_id')

    # Filter and merge
    features_df = features_df[features_df.index.isin(metadata.index)]
    data = features_df.merge(
        metadata[['subclass_label', 'layer', 't_type']],
        left_index=True, right_index=True, how='left'
    )
    data = data.reset_index()

    print(f"Loaded {len(data)} specimens with morphology data")
    return data


def load_ephys_data():
    """Load ephys feature table and metadata."""
    print("Loading ephys data...")

    # Load metadata
    metadata_path = DATA_DIR / 'human' / 'metadata' / '20260807' / 'human_metadata.csv'
    metadata = pd.read_csv(metadata_path, encoding='latin1')
    metadata = metadata.rename(columns={
        "cell_id": "specimen_id",
        "subclass_label_Hierarchical": "subclass_label",
        "cluster_label_Hierarchical": "t_type",
        "cortex_layer": "layer"
    })
    metadata = metadata.dropna(subset=['specimen_id'])
    metadata['specimen_id'] = metadata['specimen_id'].astype('int')
    metadata['subclass_label'] = metadata['subclass_label'].str.replace('L5 ET_ME', 'L5 ET')
    metadata['subclass_label'] = metadata['subclass_label'].str.replace('L5 ET(ME)', 'L5 ET')
    metadata = metadata.set_index('specimen_id')

    # Load feature table - use correct path
    feature_path = DATA_DIR / 'human' / 'electrophysiology' / 'features' / '20260807' / 'ephys_features_imputed.csv'
    features_df = pd.read_csv(feature_path, index_col='specimen_id')

    # Get combined features only
    ephys_feature_cols = [c for c in features_df.columns
                          if c.startswith('combined_') and c not in ['combined_cell_id', 'combined_label']]
    features_df = features_df[ephys_feature_cols]

    # Filter and merge
    features_df = features_df[features_df.index.isin(metadata.index)]
    data = features_df.merge(
        metadata[['subclass_label', 'layer', 't_type']],
        left_index=True, right_index=True, how='left'
    )
    data = data.reset_index()

    print(f"Loaded {len(data)} specimens with ephys data")
    return data


def run_kw_dunn_tests(data, features, grouping_col, group_order=None):
    """
    Run Kruskal-Wallis and Dunn posthoc tests.

    Parameters:
    -----------
    data : pd.DataFrame
        Data with features and grouping column
    features : list
        List of feature column names
    grouping_col : str
        Column name to group by (e.g., 'subclass_label', 'layer', 't_type')
    group_order : list, optional
        Order of groups for consistent output

    Returns:
    --------
    kw_df : pd.DataFrame
        Kruskal-Wallis results
    dunn_df : pd.DataFrame
        Dunn posthoc results
    """
    print(f"  Running KW tests on {len(features)} features grouped by {grouping_col}...")

    # Get unique groups
    if group_order is None:
        groups_present = data[grouping_col].dropna().unique()
        group_order = sorted(groups_present)

    # Kruskal-Wallis
    kw_results = []
    kw_rejected = {}

    for feature in features:
        groups = []
        for group_name in group_order:
            group_data = data.loc[data[grouping_col] == group_name, feature].dropna()
            if len(group_data) >= 1:
                groups.append(group_data.values)

        if len(groups) < 2:
            continue

        try:
            hvalue, pvalue = kruskal(*groups, nan_policy='omit')
            kw_results.append({
                'feature': feature,
                'H_statistic': hvalue,
                'p_value': pvalue
            })
        except Exception as e:
            print(f"    KW error for {feature}: {e}")
            continue

    # FDR correction
    if len(kw_results) > 0:
        kw_df = pd.DataFrame(kw_results)
        _, kw_df['FDR_p_value'] = fdrcorrection(kw_df['p_value'].values, alpha=0.05)
        kw_df['FDR_significant'] = kw_df['FDR_p_value'] < 0.05

        for _, row in kw_df.iterrows():
            kw_rejected[row['feature']] = row['FDR_significant']
    else:
        kw_df = pd.DataFrame()

    # Dunn posthoc for significant features
    print(f"  Running Dunn posthoc on {sum(kw_rejected.values())} significant features...")
    dunn_results = []

    for feature, is_sig in kw_rejected.items():
        if not is_sig:
            continue

        test_data = data[[grouping_col, feature]].dropna()
        if len(test_data) < 2:
            continue

        try:
            dunn_result = posthoc_dunn(
                test_data,
                val_col=feature,
                group_col=grouping_col,
                p_adjust='fdr_bh',
                sort=True
            )

            # Extract upper triangle
            for i, group1 in enumerate(dunn_result.index):
                for j, group2 in enumerate(dunn_result.columns):
                    if i < j:
                        p_val = dunn_result.iloc[i, j]
                        dunn_results.append({
                            'feature': feature,
                            'group1': group1,
                            'group2': group2,
                            'p_value_fdr': p_val,
                            'significant': p_val < 0.05
                        })
        except Exception as e:
            print(f"    Dunn error for {feature}: {e}")
            continue

    dunn_df = pd.DataFrame(dunn_results)

    return kw_df, dunn_df


def extract_byLayer_stats(modality='morpho'):
    """Extract statistics grouped by Layer."""
    print(f"\n{'='*60}")
    print(f"Extracting {modality.upper()} byLayer statistics")
    print(f"{'='*60}")

    # Load data
    if modality == 'morpho':
        data = load_morphology_data()
        output_dir = DENDRITE_DIR
    else:
        data = load_ephys_data()
        output_dir = EPHYS_DIR

    # Filter to specimens with layer info
    data = data[data['layer'].notna()]

    # Get features
    exclude_cols = ['specimen_id', 'subclass_label', 'layer', 't_type']
    features = [col for col in data.columns if col not in exclude_cols]

    # Run tests
    kw_df, dunn_df = run_kw_dunn_tests(data, features, 'layer', layer_order)

    # Save
    kw_path = output_dir / f'{modality}_feature_box_plot_stats_byLayer_KW_Dunn_FDR.csv'
    dunn_path = output_dir / f'{modality}_feature_box_plot_stats_byLayer_Dunn_pairs.csv'

    kw_df.to_csv(kw_path, index=False)
    dunn_df.to_csv(dunn_path, index=False)

    print(f"  Saved KW: {kw_path} ({len(kw_df)} features)")
    print(f"  Saved Dunn: {dunn_path} ({len(dunn_df)} pairs)")


def extract_bySubclass_Layer6_stats(modality='morpho'):
    """Extract statistics for Layer 6 subclasses only."""
    print(f"\n{'='*60}")
    print(f"Extracting {modality.upper()} bySubclass_Layer6 statistics")
    print(f"{'='*60}")

    # Load data
    if modality == 'morpho':
        data = load_morphology_data()
        output_dir = DENDRITE_DIR
    else:
        data = load_ephys_data()
        output_dir = EPHYS_DIR

    # Filter to Layer 6 only
    data = data[data['layer'] == 'Layer6']

    # Get Layer 6 subclasses
    l6_subclasses = ['L6 IT', 'L6 IT Car3', 'L6 CT', 'L6b']
    data = data[data['subclass_label'].isin(l6_subclasses)]

    # Get features
    exclude_cols = ['specimen_id', 'subclass_label', 'layer', 't_type']
    features = [col for col in data.columns if col not in exclude_cols]

    # Run tests
    kw_df, dunn_df = run_kw_dunn_tests(data, features, 'subclass_label', l6_subclasses)

    # Save
    kw_path = output_dir / f'{modality}_feature_box_plot_stats_bySubclass_Layer6_KW_Dunn_FDR.csv'
    dunn_path = output_dir / f'{modality}_feature_box_plot_stats_bySubclass_Layer6_Dunn_pairs.csv'

    kw_df.to_csv(kw_path, index=False)
    dunn_df.to_csv(dunn_path, index=False)

    print(f"  Saved KW: {kw_path} ({len(kw_df)} features)")
    print(f"  Saved Dunn: {dunn_path} ({len(dunn_df)} pairs)")


def extract_L5_IT_byTtype_stats(modality='morpho'):
    """Extract statistics for L5 IT supertypes."""
    print(f"\n{'='*60}")
    print(f"Extracting {modality.upper()} L5_IT_byTtype statistics")
    print(f"{'='*60}")

    # Load data
    if modality == 'morpho':
        data = load_morphology_data()
        output_dir = DENDRITE_DIR
    else:
        data = load_ephys_data()
        output_dir = EPHYS_DIR

    # Filter to L5 IT only
    data = data[data['subclass_label'] == 'L5 IT']
    data = data[data['t_type'].notna()]

    # Get features
    exclude_cols = ['specimen_id', 'subclass_label', 'layer', 't_type']
    features = [col for col in data.columns if col not in exclude_cols]

    # Run tests
    kw_df, dunn_df = run_kw_dunn_tests(data, features, 't_type')

    # Save
    kw_path = output_dir / f'{modality}_feature_box_plot_stats_L5_IT_byTtype_KW_Dunn_FDR.csv'
    dunn_path = output_dir / f'{modality}_feature_box_plot_stats_L5_IT_byTtype_Dunn_pairs.csv'

    kw_df.to_csv(kw_path, index=False)
    dunn_df.to_csv(dunn_path, index=False)

    print(f"  Saved KW: {kw_path} ({len(kw_df)} features)")
    print(f"  Saved Dunn: {dunn_path} ({len(dunn_df)} pairs)")


def extract_Layer_4_byTtype_stats(modality='morpho'):
    """Extract statistics for Layer 4 supertypes."""
    print(f"\n{'='*60}")
    print(f"Extracting {modality.upper()} Layer_4_byTtype statistics")
    print(f"{'='*60}")

    # Load data
    if modality == 'morpho':
        data = load_morphology_data()
        output_dir = DENDRITE_DIR
    else:
        data = load_ephys_data()
        output_dir = EPHYS_DIR

    # Filter to Layer 4 only
    data = data[data['layer'] == 'Layer4']
    data = data[data['t_type'].notna()]

    # Get features
    exclude_cols = ['specimen_id', 'subclass_label', 'layer', 't_type']
    features = [col for col in data.columns if col not in exclude_cols]

    # Run tests
    kw_df, dunn_df = run_kw_dunn_tests(data, features, 't_type')

    # Save
    kw_path = output_dir / f'{modality}_feature_box_plot_stats_Layer_4_byTtype_KW_Dunn_FDR.csv'
    dunn_path = output_dir / f'{modality}_feature_box_plot_stats_Layer_4_byTtype_Dunn_pairs.csv'

    kw_df.to_csv(kw_path, index=False)
    dunn_df.to_csv(dunn_path, index=False)

    print(f"  Saved KW: {kw_path} ({len(kw_df)} features)")
    print(f"  Saved Dunn: {dunn_path} ({len(dunn_df)} pairs)")


def main():
    """Extract all missing boxplot statistics."""
    print("="*80)
    print("EXTRACTING ALL BOXPLOT STATISTICS FOR STATISTICS TABLE")
    print("="*80)

    # Extract for both morphology and ephys
    for modality in ['morpho', 'ephys']:
        extract_byLayer_stats(modality)
        extract_bySubclass_Layer6_stats(modality)
        extract_L5_IT_byTtype_stats(modality)
        extract_Layer_4_byTtype_stats(modality)

    print("\n" + "="*80)
    print("ALL BOXPLOT STATISTICS EXTRACTION COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  Morphology dendrite:")
    for f in sorted(DENDRITE_DIR.glob('*.csv')):
        if f.name.startswith('morpho_'):
            print(f"    - {f.name}")
    print("  Electrophysiology:")
    for f in sorted(EPHYS_DIR.glob('*.csv')):
        if f.name.startswith('ephys_'):
            print(f"    - {f.name}")


if __name__ == '__main__':
    main()
