"""
Extract Effect Sizes (Eta-Squared) from Morphology and Electrophysiology Data

This script replicates the eta-squared calculation from the figures notebook
(cells 45 and 47) and saves the results to CSV files for the statistics table.

Formula: η² = (H - (k-1)) / (N - k)
where H is the Kruskal-Wallis H statistic, k is number of groups, N is total sample size
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_ROOT = Path('/root/capsule/data')
SCRATCH_DIR = Path('/root/capsule/scratch/20260807')
OUTPUT_DIR = SCRATCH_DIR / 'effect_sizes'
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
metadata = metadata.rename(columns={
    "cell_id": "specimen_id",
    "cluster_label_Hierarchical": "t_type",
    "subclass_label_Hierarchical": "subclass_label"
})
# Drop rows with NaN specimen_id
metadata = metadata.dropna(subset=['specimen_id'])
metadata['specimen_id'] = metadata['specimen_id'].astype('int')
metadata.subclass_label = metadata.subclass_label.str.replace('L5 ET_ME', 'L5 ET')
metadata.subclass_label = metadata.subclass_label.str.replace('L5 ET(ME)', 'L5 ET')
metadata.t_type = metadata.t_type.str.replace('L5 ET_ME', 'L5 ET(ME)')
metadata.t_type = metadata.t_type.apply(lambda x: L23_ttype_rename_dict.get(x, x))
metadata['layer'] = metadata["t_type"].map(ttype_to_layer_dict)
metadata = metadata.set_index('specimen_id')

# Load morphology features
m_feature_path = DATA_ROOT / 'human' / 'morphology' / 'dendrite' / '20260712' / 'RawFeatureWide_imputed.csv'
morpho_data = pd.read_csv(m_feature_path, index_col='specimen_id')
morpho_data = morpho_data[morpho_data.index.isin(metadata.index)]
morpho_data = morpho_data.merge(metadata[['subclass_label', 't_type', 'layer']],
                                 left_index=True, right_index=True, how='inner')

# Load electrophysiology features
e_feature_path = DATA_ROOT / 'human' / 'electrophysiology' / 'features' / '20260807' / 'ephys_features_imputed.csv'
ephys_data = pd.read_csv(e_feature_path, index_col='specimen_id')
ephys_feature_cols = [c for c in ephys_data.columns
                      if c not in {"subclass_label_Hierarchical", 'cluster_label_Hierarchical', "depth_from_pia_norm"}]
ephys_data = ephys_data[ephys_feature_cols]
ephys_data = ephys_data[ephys_data.index.isin(metadata.index)]
ephys_data = ephys_data.merge(metadata[['subclass_label', 't_type', 'layer']],
                               left_index=True, right_index=True, how='inner')

# Exclude ephys features with < 25% coverage
exclude_features = ['Chirp peak', 'ZAP (peak impedance)']
ephys_features = [f for f in ephys_feature_cols if f not in exclude_features]

print(f"Morphology: {len(morpho_data)} specimens, {len(morpho_data.columns)-3} features")
print(f"Ephys: {len(ephys_data)} specimens, {len(ephys_features)} features")


def calculate_eta_squared_by_subclass(data, features, modality_name):
    """
    Calculate eta-squared for each feature across subclasses.
    Replicates cells 45 (morpho) and 47 (ephys) from the figures notebook.
    """
    print(f"\nCalculating {modality_name} effect sizes...")

    # Get subclasses (order from taxonomy)
    subclass_order = [
        'L2/3 IT_superficial', 'L2/3 IT_deep', 'L4 IT',
        'L5 IT', 'L5 ET', 'L5/6 NP',
        'L6 IT', 'L6 IT Car3', 'L6 CT', 'L6b'
    ]
    subclass_order = [s for s in subclass_order if s in data['subclass_label'].unique()]

    kw_hvals = {}
    kw_pvals = {}

    # Iterate over features to calculate KW test
    for f in features:
        if f not in data.columns:
            continue

        # Get groups of data for each subclass
        s_groups = [data.loc[data['subclass_label'] == x][f].dropna() for x in subclass_order]

        # Only include non-empty groups
        s_groups = [g for g in s_groups if len(g) >= 1]

        if len(s_groups) < 2:
            continue

        try:
            hvalue, pvalue = stats.kruskal(*s_groups, nan_policy='omit')
            kw_hvals[f] = hvalue
            kw_pvals[f] = pvalue
        except Exception as e:
            print(f'  Warning: Could not calculate for {f}: {e}')
            continue

    # Calculate eta-squared
    k = len([g for g in s_groups if len(g) > 0])  # number of groups
    n = sum([len(g) for g in s_groups])  # total observations

    print(f"  k={k} groups, n={n} total specimens")

    eta_squared = {}
    for f, h in kw_hvals.items():
        esq = (h - k + 1) / (n - k)  # eta^2 formula for KW test
        eta_squared[f] = esq

    # Create sorted dataframe (highest eta-squared first)
    results = []
    for f in sorted(eta_squared.keys(), key=lambda x: eta_squared[x], reverse=True):
        results.append({
            'feature': f,
            'eta_squared': eta_squared[f],
            'kw_h_statistic': kw_hvals[f],
            'kw_p_value': kw_pvals[f],
            'num_groups': k,
            'sample_size': n
        })

    df = pd.DataFrame(results)

    # Report top features
    print(f"  Calculated eta-squared for {len(df)} features")
    print(f"  Top 5 features by effect size:")
    for idx, row in df.head(5).iterrows():
        print(f"    {row['feature']}: η²={row['eta_squared']:.4f}")

    return df


# Calculate effect sizes
print("\n" + "="*80)
print("MORPHOLOGY EFFECT SIZES (across subclasses)")
print("="*80)
morpho_features = [c for c in morpho_data.columns if c not in ['subclass_label', 't_type', 'layer']]
morpho_eta_df = calculate_eta_squared_by_subclass(morpho_data, morpho_features, "Morphology")

print("\n" + "="*80)
print("ELECTROPHYSIOLOGY EFFECT SIZES (across subclasses)")
print("="*80)
ephys_eta_df = calculate_eta_squared_by_subclass(ephys_data, ephys_features, "Electrophysiology")

# Save results
morpho_output = OUTPUT_DIR / 'morphology_subclass_effect_sizes.csv'
ephys_output = OUTPUT_DIR / 'electrophysiology_subclass_effect_sizes.csv'

morpho_eta_df.to_csv(morpho_output, index=False)
ephys_eta_df.to_csv(ephys_output, index=False)

print("\n" + "="*80)
print("SAVED")
print("="*80)
print(f"✓ Morphology: {morpho_output}")
print(f"✓ Electrophysiology: {ephys_output}")
print("\nVerifying manuscript claim...")
print(f"Rheobase eta-squared: {ephys_eta_df[ephys_eta_df['feature']=='Rheobase']['eta_squared'].values[0]:.4f}")
print(f"Rheobase rank: {list(ephys_eta_df['feature']).index('Rheobase') + 1} of {len(ephys_eta_df)}")
