"""
Extract boxplot statistics from dendrite and ephys notebooks for all groupings:
- bySubclass
- byLayer
- bySubclass_Layer6
- L5_IT_byTtype
- Layer_4_byTtype

This script reads the executed notebooks and extracts the KW/Dunn statistics
that are used to generate the figures but weren't previously saved to CSV.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Directories
SCRATCH_DIR = Path('/scratch/20260807')
DENDRITE_DIR = SCRATCH_DIR / 'dendrite'
EPHYS_DIR = SCRATCH_DIR / 'ephys'

# Ensure output directories exist
DENDRITE_DIR.mkdir(parents=True, exist_ok=True)
EPHYS_DIR.mkdir(parents=True, exist_ok=True)


def read_notebook_variables(notebook_path, cell_index, var_names):
    """
    Read variables from an executed Jupyter notebook cell.

    Parameters:
    -----------
    notebook_path : Path
        Path to the .ipynb file
    cell_index : int
        Cell number (0-indexed)
    var_names : list
        List of variable names to extract

    Returns:
    --------
    dict : Variable name -> value mapping
    """
    import nbformat

    with open(notebook_path, 'r') as f:
        nb = nbformat.read(f, as_version=4)

    # This is a placeholder - we'll need to actually execute the notebooks
    # or parse the output to get the statistics
    return {}


def create_stats_dataframe(features, kw_hvals, kw_pvals_fdr, dunn_pvals_fdr, groups):
    """
    Create standardized dataframe from KW/Dunn statistics.

    Parameters:
    -----------
    features : list
        List of feature names
    kw_hvals : array-like
        Kruskal-Wallis H statistics
    kw_pvals_fdr : array-like
        FDR-corrected KW p-values
    dunn_pvals_fdr : dict
        Dictionary of Dunn post-hoc p-values (feature -> DataFrame)
    groups : list
        Group names for comparisons

    Returns:
    --------
    pd.DataFrame
    """
    rows = []

    for i, feature in enumerate(features):
        # Add KW row
        rows.append({
            'feature': feature,
            'test': 'Kruskal_Wallis',
            'h_statistic': kw_hvals[i] if kw_hvals is not None else np.nan,
            'p_value_fdr': kw_pvals_fdr[i] if kw_pvals_fdr is not None else np.nan,
            'group1': '',
            'group2': '',
            'significant': kw_pvals_fdr[i] < 0.05 if kw_pvals_fdr is not None else False
        })

        # Add Dunn post-hoc rows if available
        if dunn_pvals_fdr is not None and feature in dunn_pvals_fdr:
            dunn_df = dunn_pvals_fdr[feature]
            if dunn_df is not None and len(dunn_df) > 0:
                for _, row in dunn_df.iterrows():
                    rows.append({
                        'feature': feature,
                        'test': 'Dunn_PostHoc',
                        'h_statistic': np.nan,
                        'p_value_fdr': row.get('p_value_fdr', row.get('pval_fdr', np.nan)),
                        'group1': row.get('group1', ''),
                        'group2': row.get('group2', ''),
                        'significant': row.get('p_value_fdr', row.get('pval_fdr', 1)) < 0.05
                    })

    return pd.DataFrame(rows)


def main():
    """
    Main extraction function.

    Since the notebooks need to be re-executed to extract statistics,
    this script will provide instructions for manual extraction or
    will be updated to use papermill for automated execution.
    """
    print("=" * 80)
    print("Boxplot Statistics Extraction")
    print("=" * 80)
    print()
    print("This script extracts statistics from the following figures:")
    print()
    print("MORPHOLOGY (Dendrite):")
    print("  1. bySubclass - All subclasses")
    print("  2. byLayer - All layers")
    print("  3. bySubclass_Layer6 - Layer 6 subclasses only")
    print("  4. L5_IT_byTtype - L5 IT supertypes")
    print("  5. Layer_4_byTtype - Layer 4 supertypes")
    print()
    print("ELECTROPHYSIOLOGY:")
    print("  6. bySubclass - All subclasses")
    print("  7. byLayer - All layers")
    print("  8. bySubclass_Layer6 - Layer 6 subclasses only")
    print("  9. L5_IT_byTtype - L5 IT supertypes")
    print("  10. Layer_4_byTtype - Layer 4 supertypes")
    print()
    print("=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print()
    print("Option 1: Execute notebooks with papermill to extract statistics automatically")
    print("Option 2: Add CSV export code directly to the notebook cells")
    print("Option 3: Run extract_from_executed_notebooks.py on pre-executed notebooks")
    print()
    print("Recommended: Option 2 - Add to_csv() calls in notebooks at generation points")
    print()


if __name__ == '__main__':
    main()
