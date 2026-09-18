"""
Extract Depth Interaction and Correlation Statistics

This script replicates analysis from me_interaction.ipynb to generate:
- Tab 9: DEPTH_INTERACTION_L2L3 - Two-way ANOVA depth × layer for L2 vs L3
- Tab 10: DEPTH_INTERACTION_L5_SUBCLASS - Two-way ANOVA depth × subclass for L5
- Tab 11: DEPTH_CORRELATION_WITHIN_GROUPS - Pearson correlations within groups

Based on cells 22, 24, 26 (correlations) and cells 30, 35 (ANOVAs) from me_interaction.ipynb
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import fdrcorrection
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_ROOT = Path('/root/capsule/data')
SCRATCH_DIR = Path('/root/capsule/scratch/20260807')
OUTPUT_DIR = SCRATCH_DIR / 'depth_statistics'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load taxonomy mappings
ttype_layer_file = DATA_ROOT / 'formatting_dicts' / 'ttype_layer_dict.json'
l23_ttype_rename_file = DATA_ROOT / 'formatting_dicts' / 'L23_ttype_rename.json'

with open(ttype_layer_file, "r") as f:
    ttype_to_layer_dict = json.load(f)
with open(l23_ttype_rename_file, "r") as f:
    L23_ttype_rename_dict = json.load(f)

print("Loading data...")

# Load metadata
metadata_path = DATA_ROOT / 'human' / 'metadata' / '20260807' / 'human_metadata.csv'
metadata = pd.read_csv(metadata_path, encoding='latin1')
metadata = metadata.dropna(subset=['cell_id'])
metadata['specimen_id'] = metadata['cell_id'].astype('int')
metadata = metadata.rename(columns={
    "cluster_label_Hierarchical": "t_type",
    "subclass_label_Hierarchical": "subclass_label"
})
metadata.subclass_label = metadata.subclass_label.str.replace('L5 ET_ME', 'L5 ET')
metadata.subclass_label = metadata.subclass_label.str.replace('L5 ET(ME)', 'L5 ET')
metadata.t_type = metadata.t_type.str.replace('L5 ET_ME', 'L5 ET(ME)')
metadata.t_type = metadata.t_type.apply(lambda x: L23_ttype_rename_dict.get(x, x))
metadata['layer'] = metadata["t_type"].map(ttype_to_layer_dict)
metadata = metadata.set_index('specimen_id')

# Load morphology features with depth
m_feature_path = DATA_ROOT / 'human' / 'morphology' / 'dendrite' / '20260712' / 'RawFeatureWide_imputed.csv'
morpho_data = pd.read_csv(m_feature_path, index_col='specimen_id')
morpho_data = morpho_data[morpho_data.index.isin(metadata.index)]
morpho_data = morpho_data.merge(metadata[['subclass_label', 't_type', 'layer']],
                                 left_index=True, right_index=True, how='inner')
# Add depth column
morpho_data['depth_from_pia_norm'] = morpho_data['soma_aligned_dist_from_pia']
m_features = [c for c in morpho_data.columns if c not in ['subclass_label', 't_type', 'layer', 'depth_from_pia_norm']]

# Load electrophysiology features
e_feature_path = DATA_ROOT / 'human' / 'electrophysiology' / 'features' / '20260807' / 'ephys_features_imputed.csv'
ephys_data = pd.read_csv(e_feature_path, index_col='specimen_id')
ephys_feature_cols = [c for c in ephys_data.columns
                      if c not in {"subclass_label_Hierarchical", 'cluster_label_Hierarchical', "depth_from_pia_norm"}]
ephys_data = ephys_data[[c for c in ephys_data.columns if c in ephys_feature_cols or c == 'depth_from_pia_norm']]
ephys_data = ephys_data[ephys_data.index.isin(metadata.index)]
ephys_data = ephys_data.merge(metadata[['subclass_label', 't_type', 'layer']],
                               left_index=True, right_index=True, how='inner')
# Exclude features with < 25% coverage
exclude_features = ['Chirp peak', 'ZAP (peak impedance)']
e_features = [f for f in ephys_feature_cols if f not in exclude_features]

print(f"Morphology: {len(morpho_data)} specimens, {len(m_features)} features")
print(f"Ephys: {len(ephys_data)} specimens, {len(e_features)} features")


def AOV_Two_mixed(flist, a, b, df):
    """
    Two-way ANOVA with one categorical (a) and one continuous (b) variable.
    From cell 5 of me_interaction.ipynb
    """
    all_results = []

    for f in flist:
        try:
            # Build formula
            formula = f'{f} ~ C({a}) + {b} + C({a}):{b}'

            # Fit model
            model = ols(formula, data=df).fit()
            aov_table = anova_lm(model, typ=2)

            # Extract results for each term
            for term in aov_table.index:
                if term == 'Residual':
                    continue

                all_results.append({
                    'feature': f,
                    'term': term,
                    'sum_sq': aov_table.loc[term, 'sum_sq'],
                    'df': aov_table.loc[term, 'df'],
                    'F': aov_table.loc[term, 'F'],
                    'PR(>F)': aov_table.loc[term, 'PR(>F)']
                })
        except Exception as e:
            print(f"  Warning: Could not fit model for {f}: {e}")
            continue

    return pd.DataFrame(all_results)


print("\n" + "="*80)
print("TAB 9: DEPTH INTERACTION L2/L3 (Two-way ANOVA)")
print("="*80)

# Cell 30: L2 vs L3 interaction
for modality in ['ephys', 'morpho']:
    print(f"\nProcessing {modality}...")

    if modality == 'morpho':
        feature_cols = m_features
        data = morpho_data.copy()
    else:
        feature_cols = e_features
        data = ephys_data.copy()

    # Filter to L2 and L3 only
    data = data[data['layer'].isin(['Layer 2', 'Layer 3'])]
    print(f"  Samples: {len(data)} specimens")

    # Run ANOVA
    results_df = AOV_Two_mixed(feature_cols, 'layer', 'depth_from_pia_norm', data)

    if not results_df.empty:
        # Add FDR correction per term
        for term in results_df['term'].unique():
            term_data = results_df[results_df['term'] == term]
            _, fdr_pvals = fdrcorrection(term_data['PR(>F)'].values, alpha=0.05)
            results_df.loc[results_df['term'] == term, 'FDR_p'] = fdr_pvals

        # Save
        output_file = OUTPUT_DIR / f'{modality}_Layer2_Layer3_multimodal_regression_with_interaction.csv'
        results_df.to_csv(output_file, index=False)
        print(f"  ✓ Saved {len(results_df)} rows to {output_file.name}")

        # Count significant interactions
        interaction_rows = results_df[results_df['term'].str.contains(':')]
        sig_interactions = interaction_rows[interaction_rows['FDR_p'] < 0.05]
        print(f"  Significant interactions (FDR<0.05): {len(sig_interactions)}")


print("\n" + "="*80)
print("TAB 10: DEPTH INTERACTION L5 SUBCLASSES (Two-way ANOVA)")
print("="*80)

# Cell 35: L5 subclass interaction
for modality in ['ephys', 'morpho']:
    print(f"\nProcessing {modality}...")

    if modality == 'morpho':
        feature_cols = m_features
        data = morpho_data.copy()
    else:
        feature_cols = e_features
        data = ephys_data.copy()

    # Filter to L5 subclasses only
    l5_subclasses = ['L5 IT', 'L5 ET', 'L5/6 NP']
    data = data[data['subclass_label'].isin(l5_subclasses)]
    print(f"  Samples: {len(data)} specimens")

    # Run ANOVA
    results_df = AOV_Two_mixed(feature_cols, 'subclass_label', 'depth_from_pia_norm', data)

    if not results_df.empty:
        # Add FDR correction per term
        for term in results_df['term'].unique():
            term_data = results_df[results_df['term'] == term]
            _, fdr_pvals = fdrcorrection(term_data['PR(>F)'].values, alpha=0.05)
            results_df.loc[results_df['term'] == term, 'FDR_p'] = fdr_pvals

        # Save
        output_file = OUTPUT_DIR / f'{modality}_L5_subclasses_multimodal_regression_with_interaction.csv'
        results_df.to_csv(output_file, index=False)
        print(f"  ✓ Saved {len(results_df)} rows to {output_file.name}")

        # Count significant interactions
        interaction_rows = results_df[results_df['term'].str.contains(':')]
        sig_interactions = interaction_rows[interaction_rows['FDR_p'] < 0.05]
        print(f"  Significant interactions (FDR<0.05): {len(sig_interactions)}")


print("\n" + "="*80)
print("TAB 11: DEPTH CORRELATIONS WITHIN GROUPS")
print("="*80)

# Cells 24, 26: Correlations within subclasses and layers
for modality in ['ephys', 'morpho']:
    print(f"\nProcessing {modality}...")

    if modality == 'morpho':
        feature_cols = m_features
        data = morpho_data.copy()
    else:
        feature_cols = e_features
        data = ephys_data.copy()

    # Subclass-level correlations (cell 24)
    subclass_order = ['L2/3 IT_superficial', 'L2/3 IT_deep', 'L4 IT', 'L5 IT', 'L5 ET', 'L5/6 NP', 'L6 IT', 'L6 IT Car3', 'L6 CT', 'L6b']
    subclass_order = [s for s in subclass_order if s in data['subclass_label'].unique()]

    results = []
    for subclass in subclass_order:
        subclass_data = data[data['subclass_label'] == subclass]

        for feature in feature_cols:
            if feature not in subclass_data.columns:
                continue

            df = subclass_data[['depth_from_pia_norm', feature]].dropna()
            if len(df) < 3 or df[feature].nunique() == 1:
                continue

            # Regression
            X = sm.add_constant(df['depth_from_pia_norm'])
            y = df[feature]
            try:
                model = sm.OLS(y, X).fit(cov_type='HC3')
                r_val, _ = pearsonr(df['depth_from_pia_norm'], df[feature])

                results.append({
                    'subclass': subclass,
                    'feature': feature,
                    'num_cells': len(df),
                    'r': r_val,
                    'r_squared': model.rsquared,
                    'pval': model.f_pvalue,
                    'intercept': model.params[0],
                    'slope': model.params[1]
                })
            except:
                continue

    if results:
        results_df = pd.DataFrame(results)
        _, fdr_pvals = fdrcorrection(results_df['pval'].values, alpha=0.05)
        results_df['FDR_p'] = fdr_pvals

        output_file = OUTPUT_DIR / f'depth_x_{modality}_feature_stats_subclass.csv'
        results_df.to_csv(output_file, index=False)
        print(f"  ✓ Subclass correlations: {len(results_df)} rows saved to {output_file.name}")

        sig_corr = results_df[results_df['FDR_p'] < 0.05]
        print(f"    Significant correlations (FDR<0.05): {len(sig_corr)}")

    # Layer-level correlations (cell 26)
    layer_order = ['Layer 2', 'Layer 3']
    results = []
    for layer in layer_order:
        layer_data = data[data['layer'] == layer]

        for feature in feature_cols:
            if feature not in layer_data.columns:
                continue

            df = layer_data[['depth_from_pia_norm', feature]].dropna()
            if len(df) < 3 or df[feature].nunique() == 1:
                continue

            # Regression
            X = sm.add_constant(df['depth_from_pia_norm'])
            y = df[feature]
            try:
                model = sm.OLS(y, X).fit(cov_type='HC3')
                r_val, _ = pearsonr(df['depth_from_pia_norm'], df[feature])

                results.append({
                    'layer': layer,
                    'feature': feature,
                    'num_cells': len(df),
                    'r': r_val,
                    'r_squared': model.rsquared,
                    'pval': model.f_pvalue,
                    'intercept': model.params[0],
                    'slope': model.params[1]
                })
            except:
                continue

    if results:
        results_df = pd.DataFrame(results)
        _, fdr_pvals = fdrcorrection(results_df['pval'].values, alpha=0.05)
        results_df['FDR_p'] = fdr_pvals

        output_file = OUTPUT_DIR / f'depth_x_{modality}_feature_stats_layer.csv'
        results_df.to_csv(output_file, index=False)
        print(f"  ✓ Layer correlations: {len(results_df)} rows saved to {output_file.name}")

        sig_corr = results_df[results_df['FDR_p'] < 0.05]
        print(f"    Significant correlations (FDR<0.05): {len(sig_corr)}")

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
print(f"All files saved to: {OUTPUT_DIR}")
