#!/usr/bin/env python3
"""
Extract subclass morphology boxplot statistics for Tab 4.
Replicates the statistical tests from dendrite_statistics.ipynb.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import kruskal
from statsmodels.stats.multitest import fdrcorrection
from scikit_posthocs import posthoc_dunn

# Paths
SCRATCH_DIR = Path('/root/capsule/scratch/20260807')
DATA_DIR = Path('/root/capsule/data')
OUTPUT_DIR = SCRATCH_DIR / 'subclass_stats'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Subclass order (from notebooks)
subclass_order = [
    'L2/3 IT',
    'L4 IT',
    'L5 IT',
    'L5 ET',
    'L5/6 NP',
    'L6 IT',
    'L6 IT Car3',
    'L6 CT',
    'L6b'
]

def load_morphology_data():
    """Load morphology feature table and metadata."""
    print("Loading morphology data...")

    # Load metadata
    metadata_path = DATA_DIR / 'human' / 'metadata' / '20260807' / 'human_metadata.csv'
    metadata = pd.read_csv(metadata_path, encoding='latin1')
    metadata = metadata.rename(columns={
        "cell_id": "specimen_id",
        "subclass_label_Hierarchical": "subclass_label"
    })
    # Drop rows with NaN specimen_id
    metadata = metadata.dropna(subset=['specimen_id'])
    metadata['specimen_id'] = metadata['specimen_id'].astype('int')
    # Fix L5 ET label
    metadata['subclass_label'] = metadata['subclass_label'].str.replace('L5 ET_ME', 'L5 ET')
    metadata['subclass_label'] = metadata['subclass_label'].str.replace('L5 ET(ME)', 'L5 ET')
    metadata = metadata.set_index('specimen_id')

    # Load feature table
    feature_path = DATA_DIR / 'human' / 'morphology' / 'dendrite' / '20260712' / 'RawFeatureWide_imputed.csv'
    features_df = pd.read_csv(feature_path, index_col='specimen_id')

    # Filter features to those with metadata
    features_df = features_df[features_df.index.isin(metadata.index)]

    # Merge with metadata
    data = features_df.merge(metadata[['subclass_label']], left_index=True, right_index=True, how='left')
    data = data.reset_index()

    # Filter to specimens with subclass labels
    data = data[data['subclass_label'].isin(subclass_order)]

    print(f"Loaded {len(data)} specimens with morphology data")
    return data

def calculate_kruskal_wallis(data, features):
    """Calculate Kruskal-Wallis test for all morphology features."""
    print(f"Running Kruskal-Wallis tests on {len(features)} features...")

    results = []
    kw_hvals = {}
    kw_pvals = {}

    for f in features:
        # Create groups
        groups = []
        for subclass in subclass_order:
            group_data = data.loc[data['subclass_label'] == subclass, f].dropna()
            if len(group_data) >= 1:
                groups.append(group_data.values)

        if len(groups) < 2:
            print(f"Not enough groups with data for feature {f}")
            continue

        try:
            hvalue, pvalue = kruskal(*groups, nan_policy='omit')
            kw_hvals[f] = hvalue
            kw_pvals[f] = pvalue
        except ValueError as e:
            print(f'Kruskal-Wallis not calculated for {f} because of an error: {e}')
            continue

    # FDR correction
    kw_pvals_list = list(kw_pvals.values())
    kw_rejected_list, kw_pvals_fdr_list = fdrcorrection(kw_pvals_list, alpha=0.05)
    kw_pvals_fdr = dict(zip(list(kw_pvals.keys()), list(kw_pvals_fdr_list)))
    kw_rejected = dict(zip(list(kw_pvals.keys()), list(kw_rejected_list)))

    # Create results dataframe
    for feature in kw_hvals.keys():
        # Count specimens per subclass
        n_per_group = []
        for subclass in subclass_order:
            n = data.loc[data['subclass_label'] == subclass, feature].notna().sum()
            n_per_group.append(n)

        results.append({
            'feature': feature,
            'H_statistic': kw_hvals[feature],
            'p_value': kw_pvals[feature],
            'FDR_p_value': kw_pvals_fdr[feature],
            'FDR_significant': kw_rejected[feature],
            'n_total': sum(n_per_group),
            **{f'n_{subclass}': n for subclass, n in zip(subclass_order, n_per_group)}
        })

    print(f"Completed Kruskal-Wallis tests: {len(results)} features tested")
    return pd.DataFrame(results), kw_rejected

def calculate_dunn_posthoc(data, kw_rejected):
    """Calculate Dunn posthoc tests for significant features."""
    print(f"Running Dunn posthoc tests on significant features...")

    all_results = []

    for feature, is_significant in kw_rejected.items():
        if not is_significant:
            continue

        print(f"  Dunn test for {feature}...")

        # Prepare data for posthoc_dunn
        test_data = data[data['subclass_label'].isin(subclass_order)][['subclass_label', feature]].dropna()

        if len(test_data) < 2:
            continue

        try:
            # Run Dunn test with FDR correction
            dunn_result = posthoc_dunn(
                test_data,
                val_col=feature,
                group_col='subclass_label',
                p_adjust='fdr_bh',
                sort=True
            )

            # Convert matrix to long format
            for i, group1 in enumerate(dunn_result.index):
                for j, group2 in enumerate(dunn_result.columns):
                    if i < j:  # Upper triangle only (avoid duplicates)
                        p_val = dunn_result.iloc[i, j]
                        all_results.append({
                            'feature': feature,
                            'group1': group1,
                            'group2': group2,
                            'p_value_fdr': p_val,
                            'significant': p_val < 0.05
                        })
        except Exception as e:
            print(f"  Error in Dunn test for {feature}: {e}")
            continue

    print(f"Completed Dunn posthoc tests: {len(all_results)} pairwise comparisons")
    return pd.DataFrame(all_results)

def main():
    """Main extraction function."""
    print("=" * 60)
    print("Extracting Subclass Morphology Statistics (Tab 4)")
    print("=" * 60)

    # Load data
    data = load_morphology_data()

    # Get feature columns (exclude metadata columns)
    exclude_cols = ['specimen_id', 'subclass_label', 'supertype_label', 'layer',
                    'donor_id', 'species', 'sex', 'age', 'brain_region']
    features = [col for col in data.columns if col not in exclude_cols]

    print(f"Testing {len(features)} morphology features")

    # Calculate Kruskal-Wallis
    kw_results, kw_rejected = calculate_kruskal_wallis(data, features)

    # Save Kruskal-Wallis results
    kw_output = OUTPUT_DIR / 'subclass_morpho_kruskal_wallis.csv'
    kw_results.to_csv(kw_output, index=False)
    print(f"\nSaved Kruskal-Wallis results: {kw_output}")
    print(f"  Features tested: {len(kw_results)}")
    print(f"  Significant features (FDR<0.05): {kw_results['FDR_significant'].sum()}")

    # Calculate Dunn posthoc
    dunn_results = calculate_dunn_posthoc(data, kw_rejected)

    # Save Dunn posthoc results
    dunn_output = OUTPUT_DIR / 'subclass_morpho_dunn_posthoc.csv'
    dunn_results.to_csv(dunn_output, index=False)
    print(f"\nSaved Dunn posthoc results: {dunn_output}")
    print(f"  Pairwise comparisons: {len(dunn_results)}")
    print(f"  Significant pairs (FDR<0.05): {dunn_results['significant'].sum()}")

    print("\n" + "=" * 60)
    print("Morphology statistics extraction complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()
