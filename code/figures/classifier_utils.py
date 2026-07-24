"""
Classifier Utilities for Random Forest Cell Type Classification

This module contains shared utilities for training and evaluating random forest
classifiers across three modalities: morphology (M), electrophysiology (E), and
morphoelectric (ME).

Key functions:
- Data loading and preprocessing (load_modality_features, prepare_metadata)
- Filtering and validation (filter_data_by_min_cells)
- Train/test splitting (create_stratified_split)
- Model training (train_rf_classifier)
- Evaluation (evaluate_classifier)
- Result saving (save_classifier_results)
- Variant-specific workflows (run_standard_variant, run_acute_only_variant, etc.)

Configuration structures (MODALITY_CONFIGS) capture all modality-specific
differences to enable exact reproduction of legacy notebook results.

Dependencies:
- Standard library: os, json, pathlib
- Data: pandas, numpy
- ML: scikit-learn (RandomForestClassifier, train_test_split, etc.)
- Plotting: matplotlib, seaborn (via summary_stats_utils)
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Import shared utilities
from core_utils import get_seaad_colors, get_tx_order
from summary_stats_utils import plot_confusion_matrix


# =============================================================================
# CONFIGURATION STRUCTURES
# =============================================================================

# Shared grid search parameters (identical across all modalities)
RF_PARAM_GRID = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

RF_SCORING = {
    'accuracy': 'accuracy',
    'f1_weighted': 'f1_weighted'
}

# Plotting configuration (figure sizes and annotation offsets per label column)
PLOT_CONFIG = {
    'subclass_label': {
        'fig_size': (6, 6),
        'num_cell_xbuffer': 0.42,  # Morphology uses 0.42, Ephys uses 0.45
        'n_height': 0.015
    },
    'layer_subclass': {
        'fig_size': (6, 6),
        'num_cell_xbuffer': 0.3,
        'n_height': 0.015
    },
    'layer': {
        'fig_size': (6, 6),
        'num_cell_xbuffer': 0.3,
        'n_height': 0.015
    },
    't_type': {
        'fig_size': (9, 9),
        'num_cell_xbuffer': 0.5,
        'n_height': -0.06
    }
}

# Modality-specific configurations
def get_modality_configs(data_root):
    """
    Get modality configurations with all modality-specific parameters.

    Parameters
    ----------
    data_root : Path
        Root data directory path

    Returns
    -------
    dict
        Dictionary with keys 'morphology', 'electrophysiology', 'morphoelectric'
    """
    return {
        'morphology': {
            'name': 'Morphology',
            'short_name': 'M',
            'feature_files': {
                'primary': data_root / 'human' / 'morphology' / 'dendrite' / '20260712' / 'RawFeatureWide_imputed.csv'
            },
            'feature_index_col': 'specimen_id',
            'depth_column': 'soma_aligned_dist_from_pia',
            'exclude_columns': [],  # No exclusions beyond depth removal
            'train_size': 0.8,  # 80/20 train/test split
            'stratify_variants': ['simple'],  # Only simple stratification
            'has_paradigm': False,
            'merge_mode': None,  # Single modality
            'output_variants': ['standard'],  # Only standard variant
            'min_cells': 4,
            'output_dir': 'M_classifier',
            'pdf_basename': 'human_excitatory_morpho_classifier_rf_min_{min_cells}_cells_rawFeatures_{depth_type}.pdf'
        },

        'electrophysiology': {
            'name': 'Electrophysiology',
            'short_name': 'E',
            'feature_files': {
                'primary': data_root / 'human' / 'electrophysiology' / 'features' / '20260722' / 'ephys_features_imputed.csv'
            },
            'feature_index_col': 'specimen_id',
            'depth_column': 'depth_from_pia_norm',
            'exclude_columns': ['subclass_label_Hierarchical'],  # Always exclude
            'train_size': 0.8,  # 80/20 train/test split
            'stratify_variants': ['Subclass', 'SubclassParadigm'],  # 'Subclass' = stratify on subclass only
            'has_paradigm': True,
            'merge_mode': None,  # Single modality
            'output_variants': ['standard', 'acuteOnly', 'acuteTrain_cultureTest', 'balanceParadigm'],
            'min_cells': 4,
            'output_dir': 'E_classifier',
            'pdf_basename_standard': 'human_excitatory_ephys_classifier_rf_min_{min_cells}_cells_{depth_type}_stratify{stratify}.pdf',
            'pdf_basename_acute_only': 'human_excitatory_ephys_classifier_rf_min_{min_cells}_cells_{depth_type}_acuteOnly.pdf',
            'pdf_basename_acute_train_culture': 'human_excitatory_ephys_classifier_rf_min_{min_cells}_cells_{depth_type}_acuteTrain_cultureTest.pdf',
            'pdf_basename_balance': 'human_excitatory_ephys_classifier_rf_min_{min_cells}_cells_{depth_type}_balanceParadigm.pdf'
        },

        'morphoelectric': {
            'name': 'Morphoelectric',
            'short_name': 'ME',
            'feature_files': {
                'morphology': data_root / 'human' / 'morphology' / 'dendrite' / '20260712' / 'RawFeatureWide_imputed.csv',
                'electrophysiology': data_root / 'human' / 'electrophysiology' / 'features' / '20260722' / 'ephys_features_imputed.csv'
            },
            'feature_index_col': 'specimen_id',
            'depth_column': 'soma_aligned_dist_from_pia',  # Use morphology depth only
            'exclude_columns': ['subclass_label_Hierarchical', 'depth_from_pia_norm'],  # Exclude from ephys before merge
            'train_size': 0.75,  # CRITICAL: 75/25 split (different from others!)
            'stratify_variants': ['simple'],
            'has_paradigm': True,  # Retained but not used for stratification
            'merge_mode': 'inner',  # Only cells with both modalities
            'output_variants': ['standard'],
            'min_cells': 4,
            'output_dir': 'ME_classifier',
            'pdf_basename': 'human_excitatory_morphoelectric_classifier_rf_min_{min_cells}_cells_{depth_type}.pdf',
            'save_cell_list': True  # Save ME_classifier_cells.csv
        }
    }


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_modality_features(modality_config, modality_name):
    """
    Load and prepare feature data for a modality.

    Parameters
    ----------
    modality_config : dict
        Configuration dictionary from get_modality_configs()
    modality_name : str
        Name of modality ('morphology', 'electrophysiology', 'morphoelectric')

    Returns
    -------
    pd.DataFrame
        Features with specimen_id as index

    Notes
    -----
    For morphoelectric:
    - Loads M and E features separately
    - Applies column exclusions to E before merge
    - Performs inner join on specimen_id (only cells with both modalities)
    - Result has morphology depth column only

    For single modalities:
    - Simple load with index_col=specimen_id
    - Column exclusions applied after loading
    """
    if modality_config['merge_mode'] == 'inner':
        # Morphoelectric: merge M and E features
        m_path = modality_config['feature_files']['morphology']
        e_path = modality_config['feature_files']['electrophysiology']

        # Load morphology features
        m_features = pd.read_csv(m_path, index_col='specimen_id')

        # Load ephys features and exclude specified columns BEFORE merge
        e_features = pd.read_csv(e_path, index_col='specimen_id')
        e_exclude = modality_config['exclude_columns']
        e_feature_columns = [c for c in e_features.columns if c not in e_exclude]
        e_features = e_features[e_feature_columns]

        # Inner join (only cells with both modalities)
        features = m_features.merge(e_features, how='inner', left_index=True, right_index=True)

        print(f"Loaded {len(m_features)} morphology cells, {len(e_features)} ephys cells")
        print(f"Inner join result: {len(features)} cells with both modalities")

    else:
        # Single modality: simple load
        feature_path = modality_config['feature_files']['primary']
        features = pd.read_csv(feature_path, index_col=modality_config['feature_index_col'])

        # Apply column exclusions
        exclude_cols = modality_config['exclude_columns']
        if exclude_cols:
            feature_columns = [c for c in features.columns if c not in exclude_cols]
            features = features[feature_columns]
            print(f"Excluded columns: {exclude_cols}")

        print(f"Loaded {len(features)} {modality_config['name']} cells")

    return features


def prepare_metadata(metadata_path, feature_ids, ttype_layer_dict, L23_rename_dict,
                     include_paradigm=False):
    """
    Load and prepare metadata with label relabeling.

    Parameters
    ----------
    metadata_path : Path
        Path to human_metadata.csv
    feature_ids : pd.Index
        specimen_ids from features (for filtering)
    ttype_layer_dict : dict
        Mapping of t_type to layer
    L23_rename_dict : dict
        L2/3 type renaming dictionary
    include_paradigm : bool, optional
        Whether to retain paradigm column (default=False)

    Returns
    -------
    pd.DataFrame
        Metadata with specimen_id as index, columns:
        [t_type, subclass_label, layer, layer_subclass, paradigm (optional)]

    Notes
    -----
    Applies transformations:
    - Rename: cell_id → specimen_id
    - Rename: cluster_label_Hierarchical → t_type
    - Rename: subclass_label_Hierarchical → subclass_label
    - Replace: 'L5 ET_ME' → 'L5 ET' in subclass_label
    - Replace: 'L5 ET_ME' → 'L5 ET(ME)' in t_type
    - Apply L23_ttype_rename_dict to t_type
    - Create layer from t_type mapping
    - Create layer_subclass (L2 IT, L3 IT split from L2/3 IT)
    - Filter to feature_ids
    - Remove von Economo neuron (1263702688)
    """
    # Load metadata
    metadata = pd.read_csv(metadata_path)

    # Filter to cells in features
    metadata = metadata[metadata.cell_id.isin(feature_ids.tolist())]

    # Rename columns
    metadata = metadata.rename(columns={
        'cell_id': 'specimen_id',
        'cluster_label_Hierarchical': 't_type',
        'subclass_label_Hierarchical': 'subclass_label'
    })

    # L5 ET_ME relabeling
    metadata.subclass_label = metadata.subclass_label.str.replace('L5 ET_ME', 'L5 ET')
    metadata.t_type = metadata.t_type.str.replace('L5 ET_ME', 'L5 ET(ME)')

    # L2/3 IT renaming
    metadata.t_type = metadata.t_type.apply(lambda x: L23_rename_dict.get(x, x))

    # Create layer column
    metadata['layer'] = metadata['t_type'].map(ttype_layer_dict)

    # Create layer_subclass column (split L2 and L3 IT)
    metadata['layer_subclass'] = metadata.apply(
        lambda row: "L2 IT" if row["layer"] == 'Layer 2'
                    else ("L3 IT" if row["layer"] == 'Layer 3'
                    else row["subclass_label"]),
        axis=1
    )

    # Select columns
    keep_cols = ['specimen_id', 't_type', 'subclass_label', 'layer', 'layer_subclass']
    if include_paradigm:
        keep_cols.append('paradigm')

    metadata = metadata[keep_cols]

    # Set index
    metadata['specimen_id'] = metadata['specimen_id'].astype('int')
    metadata = metadata.set_index('specimen_id')

    # Remove von Economo neuron
    ven_spid = 1263702688
    metadata = metadata.drop(index=ven_spid, errors='ignore')

    print(f"Prepared metadata for {len(metadata)} cells")

    return metadata


# =============================================================================
# FILTERING AND VALIDATION FUNCTIONS
# =============================================================================

def filter_data_by_min_cells(features, metadata, label_column, label_order, min_cells):
    """
    Filter dataset to labels meeting minimum cell threshold.

    Parameters
    ----------
    features : pd.DataFrame
        Feature data with specimen_id index
    metadata : pd.DataFrame
        Metadata with specimen_id index
    label_column : str
        Column to classify on (e.g., 'subclass_label')
    label_order : list
        Canonical ordering of labels
    min_cells : int
        Minimum cells required per label

    Returns
    -------
    tuple
        (data, labels, label_counts)
        - data: pd.DataFrame, merged features + metadata
        - labels: list, labels meeting threshold in canonical order
        - label_counts: dict, {label: count}

    Notes
    -----
    - Inner join features and metadata
    - Drop NaN in label_column
    - Count cells per label
    - Filter to labels with ≥ min_cells
    - Order by label_order (canonical transcriptomic ordering)
    """
    # Merge features and metadata
    data = features.merge(metadata, how='inner', left_index=True, right_index=True)

    # Drop NaN in label column
    initial_count = len(data)
    data = data.dropna(subset=[label_column])
    dropped = initial_count - len(data)
    if dropped > 0:
        print(f"Dropped {dropped} cells with NaN in {label_column}")

    # Count cells per label
    label_counts_series = data[label_column].value_counts()
    label_counts = label_counts_series.to_dict()

    # Filter to labels with >= min_cells
    labels_meeting_threshold = sorted(label_counts_series[label_counts_series >= min_cells].index)

    # Order by canonical ordering
    labels = [l for l in label_order if l in labels_meeting_threshold]

    # Filter data
    data = data[data[label_column].isin(labels)]

    print(f"Label column: {label_column}")
    print(f"Labels meeting threshold (>={min_cells} cells): {len(labels)}")
    print(f"Total cells after filtering: {len(data)}")

    return data, labels, label_counts


# =============================================================================
# TRAIN/TEST SPLITTING FUNCTIONS
# =============================================================================

def create_stratified_split(data, feature_columns, label_column, train_size,
                            stratify_mode='simple', random_state=42):
    """
    Create stratified train/test split.

    Parameters
    ----------
    data : pd.DataFrame
        Merged features + metadata
    feature_columns : list
        Feature column names
    label_column : str
        Target label column
    train_size : float
        Training proportion (0.75 or 0.8)
    stratify_mode : {'simple', 'SubclassParadigm'}, optional
        Stratification strategy:
        - 'simple': stratify on label only
        - 'SubclassParadigm': stratify on label + paradigm combined
    random_state : int, optional
        Random seed for reproducibility (default=42)

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, id_train, id_test)

    Notes
    -----
    For SubclassParadigm mode:
    - Creates combined stratification key: f"{label}_{paradigm}"
    - Requires 'paradigm' column in data
    - Ensures balanced representation of paradigms in train/test
    """
    y = data[label_column]
    X = data[feature_columns]

    if stratify_mode in ['simple', 'Subclass']:
        # Stratify on label only
        stratify = y
    elif stratify_mode == 'SubclassParadigm':
        # Stratify on label + paradigm
        if 'paradigm' not in data.columns:
            raise ValueError("SubclassParadigm stratification requires 'paradigm' column")
        stratify = data[label_column].astype(str) + "_" + data["paradigm"].astype(str)
    else:
        raise ValueError(f"Unknown stratify_mode: {stratify_mode}")

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, data.index,
        stratify=stratify,
        train_size=train_size,
        random_state=random_state
    )

    print(f"Train/test split: {len(X_train)}/{len(X_test)} ({train_size*100:.0f}/{(1-train_size)*100:.0f})")
    print(f"Stratification mode: {stratify_mode}")

    return X_train, X_test, y_train, y_test, id_train, id_test


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_rf_classifier(X_train, y_train, param_grid, scoring, cv_splits=3,
                        random_state=42):
    """
    Train Random Forest with grid search cross-validation.

    Parameters
    ----------
    X_train : np.ndarray
        Normalized training features
    y_train : pd.Series
        Training labels
    param_grid : dict
        Grid search parameter space
    scoring : dict
        Scoring metrics ({'accuracy': 'accuracy', 'f1_weighted': 'f1_weighted'})
    cv_splits : int, optional
        Number of CV folds (default=3)
    random_state : int, optional
        Random seed (default=42)

    Returns
    -------
    tuple
        (best_model, cv_metrics)
        - best_model: trained RandomForestClassifier
        - cv_metrics: dict with cv_acc_mean, cv_acc_std, cv_f1_mean, cv_f1_std

    Notes
    -----
    - Uses StratifiedKFold(shuffle=True, random_state=42)
    - Refits on 'f1_weighted'
    - Uses class_weight='balanced'
    - Returns best estimator from grid search
    """
    # Stratified K-fold CV
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    # Random forest with balanced class weights
    rf_classifier = RandomForestClassifier(random_state=random_state, class_weight='balanced')

    # Grid search
    grid_search = GridSearchCV(
        rf_classifier,
        param_grid,
        cv=cv,
        scoring=scoring,
        refit='f1_weighted',  # Refit on F1 weighted
        verbose=0,
        n_jobs=-1,
        return_train_score=False
    )

    grid_search.fit(X_train, y_train)

    # Extract best model and CV metrics
    best_model = grid_search.best_estimator_
    cv_results = grid_search.cv_results_
    best_idx = grid_search.best_index_

    cv_metrics = {
        'cv_acc_mean': cv_results['mean_test_accuracy'][best_idx],
        'cv_acc_std': cv_results['std_test_accuracy'][best_idx],
        'cv_f1_mean': cv_results['mean_test_f1_weighted'][best_idx],
        'cv_f1_std': cv_results['std_test_f1_weighted'][best_idx],
        'best_params': grid_search.best_params_
    }

    print(f"Best RF params: {cv_metrics['best_params']}")
    print(f"CV ({cv_splits} folds) | Accuracy: {cv_metrics['cv_acc_mean']:.3f} ± {cv_metrics['cv_acc_std']:.3f} | "
          f"F1: {cv_metrics['cv_f1_mean']:.3f} ± {cv_metrics['cv_f1_std']:.3f}")

    return best_model, cv_metrics


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_classifier(model, X_test, y_test, labels):
    """
    Evaluate classifier and compute confusion matrix.

    Parameters
    ----------
    model : sklearn estimator
        Trained classifier
    X_test : np.ndarray
        Normalized test features
    y_test : pd.Series
        True test labels
    labels : list
        Ordered list of all possible labels

    Returns
    -------
    dict
        Dictionary containing:
        - 'y_pred': predictions
        - 'accuracy': accuracy score
        - 'f1_weighted': weighted F1 score
        - 'cm': raw confusion matrix
        - 'true_counts': counts per label (row sums)
        - 'normalized_cm': row-normalized confusion matrix (safe divide)
        - 'row_acc': per-class accuracy (diagonal of normalized CM)

    Notes
    -----
    Uses np.divide with where clause to avoid divide-by-zero:
    - Returns 0.0 for labels with no test samples
    - Prevents NaN/Inf in normalized matrix
    """
    # Generate predictions
    y_pred = model.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    true_counts = np.sum(cm, axis=1)

    # Safe normalization (avoid divide-by-zero)
    normalized_cm = np.divide(
        cm,
        true_counts[:, np.newaxis],
        out=np.zeros_like(cm, dtype=float),
        where=true_counts[:, np.newaxis] != 0
    )

    row_acc = np.diag(normalized_cm)

    print(f"Test set | Accuracy: {accuracy:.3f} | F1: {f1_weighted:.3f}")

    return {
        'y_pred': y_pred,
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'cm': cm,
        'true_counts': true_counts,
        'normalized_cm': normalized_cm,
        'row_acc': row_acc
    }


# =============================================================================
# RESULT SAVING FUNCTIONS
# =============================================================================

def save_classifier_results(output_dir, modality_name, label_column, depth_type,
                            stratify_type, variant_name, results, feature_columns,
                            labels, y_test, cv_metrics, model, subset_name=None):
    """
    Save classifier results to structured files.

    Parameters
    ----------
    output_dir : Path
        Base output directory
    modality_name : str
        'morphology', 'electrophysiology', or 'morphoelectric'
    label_column : str
        Label predicted (e.g., 'subclass_label')
    depth_type : str
        'withDepth' or 'withoutDepth'
    stratify_type : str
        'simple' or 'SubclassParadigm'
    variant_name : str
        'standard', 'acuteOnly', 'acuteTrain_cultureTest', 'balanceParadigm'
    results : dict
        Output from evaluate_classifier()
    feature_columns : list
        Feature names used
    labels : list
        Ordered label list
    y_test : pd.Series
        True test labels with specimen_id index
    cv_metrics : dict
        Cross-validation metrics
    model : sklearn estimator
        Trained model
    subset_name : str, optional
        Subset name for multi-output variants (e.g., 'acute', 'culture', 'all')

    Saves
    -----
    Files with prefix: {modality}_{label}_{depth}_{stratify}_{variant}[_{subset}]
    - _confusion_matrix_raw.csv: Raw counts
    - _confusion_matrix_normalized.csv: Row-normalized
    - _predictions.csv: Per-cell predictions with true labels
    - _metadata.json: All configuration and metrics
    - _label_order.json: Canonical label order
    - _features.json: Feature names used

    Notes
    -----
    Creates directory structure: {output_dir}/{modality[0].lower()}/
    """
    # Create output directory
    # Map modality name to short name for subdirectory
    modality_short = {'morphology': 'm', 'electrophysiology': 'e', 'morphoelectric': 'me'}
    modality_subdir = output_dir / modality_short.get(modality_name, modality_name[0].lower())
    os.makedirs(modality_subdir, exist_ok=True)

    # Construct filename prefix
    label_short = label_column.replace('_label', '').replace('_', '')
    parts = [modality_name, label_short, depth_type, stratify_type, variant_name]
    if subset_name:
        parts.append(subset_name)
    prefix = '_'.join(parts)

    # Save confusion matrices
    cm_raw_path = modality_subdir / f"{prefix}_confusion_matrix_raw.csv"
    cm_norm_path = modality_subdir / f"{prefix}_confusion_matrix_normalized.csv"

    pd.DataFrame(results['cm'], index=labels, columns=labels).to_csv(cm_raw_path)
    pd.DataFrame(results['normalized_cm'], index=labels, columns=labels).to_csv(cm_norm_path)

    # Save predictions
    pred_df = pd.DataFrame({
        'specimen_id': y_test.index,
        'true_label': y_test.values,
        'predicted_label': results['y_pred']
    })
    pred_path = modality_subdir / f"{prefix}_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    # Save metadata
    metadata = {
        'modality': modality_name,
        'label_column': label_column,
        'depth_type': depth_type,
        'stratify_type': stratify_type,
        'variant': variant_name,
        'subset': subset_name,
        'num_cells_test': len(y_test),
        'num_labels': len(labels),
        'cv_accuracy_mean': float(cv_metrics['cv_acc_mean']),
        'cv_accuracy_std': float(cv_metrics['cv_acc_std']),
        'cv_f1_mean': float(cv_metrics['cv_f1_mean']),
        'cv_f1_std': float(cv_metrics['cv_f1_std']),
        'test_accuracy': float(results['accuracy']),
        'test_f1_weighted': float(results['f1_weighted']),
        'best_rf_params': cv_metrics['best_params'],
        'random_state': 42
    }
    metadata_path = modality_subdir / f"{prefix}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save label order
    label_order_path = modality_subdir / f"{prefix}_label_order.json"
    with open(label_order_path, 'w') as f:
        json.dump(labels, f, indent=2)

    # Save feature names
    features_path = modality_subdir / f"{prefix}_features.json"
    with open(features_path, 'w') as f:
        json.dump(list(feature_columns), f, indent=2)

    print(f"Saved results to {modality_subdir} with prefix {prefix}")


# =============================================================================
# VARIANT-SPECIFIC WORKFLOW FUNCTIONS
# =============================================================================

def run_standard_variant(features, metadata, modality_config, label_column,
                         depth_type, stratify_type, label_order_dict, color_dict,
                         output_base_dir, pdf_handle):
    """
    Run standard classifier variant (all data, standard train/test split).

    This is used for:
    - Morphology: all variants
    - Electrophysiology: standard variants with different stratifications
    - Morphoelectric: all variants

    Parameters
    ----------
    features : pd.DataFrame
        Feature data
    metadata : pd.DataFrame
        Metadata with labels
    modality_config : dict
        Configuration from get_modality_configs()
    label_column : str
        Label to classify
    depth_type : str
        'withDepth' or 'withoutDepth'
    stratify_type : str
        Stratification mode
    label_order_dict : dict
        Dictionary mapping label_column to ordered label list
    color_dict : dict
        Color mapping for labels
    output_base_dir : Path
        Base output directory
    pdf_handle : PdfPages
        PDF file handle for saving plots

    Returns
    -------
    dict
        Results dictionary from evaluate_classifier()
    """
    print(f"\n{'='*60}")
    print(f"STANDARD VARIANT: {modality_config['name']}")
    print(f"Label: {label_column}, Depth: {depth_type}, Stratify: {stratify_type}")
    print(f"{'='*60}")

    # Prepare feature columns (remove depth if withoutDepth)
    feature_columns = features.columns.tolist()
    if depth_type == 'withoutDepth' and modality_config['depth_column'] in feature_columns:
        feature_columns.remove(modality_config['depth_column'])

    # Filter data by min cells
    data, labels, label_counts = filter_data_by_min_cells(
        features, metadata, label_column,
        label_order_dict[label_column],
        modality_config['min_cells']
    )

    # Train/test split
    X_train, X_test, y_train, y_test, id_train, id_test = create_stratified_split(
        data, feature_columns, label_column,
        modality_config['train_size'],
        stratify_type
    )

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train classifier
    best_model, cv_metrics = train_rf_classifier(
        X_train_scaled, y_train,
        RF_PARAM_GRID, RF_SCORING
    )

    # Evaluate
    results = evaluate_classifier(best_model, X_test_scaled, y_test, labels)

    # Save structured outputs
    save_classifier_results(
        output_base_dir / 'classifiers' / '20260722',
        modality_config['name'].lower(),
        label_column, depth_type, stratify_type, 'standard',
        results, feature_columns, labels, y_test, cv_metrics, best_model
    )

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=PLOT_CONFIG[label_column]['fig_size'])

    title = (f"{modality_config['name']}\n"
             f"Predict: {label_column} (n≥{modality_config['min_cells']}) ({len(data)} cells)\n"
             f"Accuracy: {results['accuracy']:.3f}\n"
             f"F1 score: {results['f1_weighted']:.3f}")

    plot_confusion_matrix(
        ax=ax,
        cm=results['cm'],
        labels=labels,
        color_dict=color_dict,
        title=title,
        num_cell_xbuffer=PLOT_CONFIG[label_column]['num_cell_xbuffer'],
        n_height=PLOT_CONFIG[label_column]['n_height']
    )

    plt.tight_layout()
    pdf_handle.savefig(fig, bbox_inches='tight')
    plt.close()

    return results


def run_acute_only_variant(features, metadata, modality_config, label_column,
                            depth_type, label_order_dict, color_dict,
                            output_base_dir, pdf_handle):
    """
    Train and test on acute cells only.

    Parameters
    ----------
    features : pd.DataFrame
        Feature data
    metadata : pd.DataFrame
        Metadata with labels and paradigm
    modality_config : dict
        Configuration from get_modality_configs()
    label_column : str
        Label to classify
    depth_type : str
        'withDepth' or 'withoutDepth'
    label_order_dict : dict
        Dictionary mapping label_column to ordered label list
    color_dict : dict
        Color mapping for labels
    output_base_dir : Path
        Base output directory
    pdf_handle : PdfPages
        PDF file handle for saving plots

    Returns
    -------
    dict
        Results dictionary from evaluate_classifier()
    """
    print(f"\n{'='*60}")
    print(f"ACUTE-ONLY VARIANT: {modality_config['name']}")
    print(f"Label: {label_column}, Depth: {depth_type}")
    print(f"{'='*60}")

    # Filter to acute cells only
    metadata_acute = metadata[metadata.paradigm == 'acute']

    # Prepare feature columns
    feature_columns = features.columns.tolist()
    if depth_type == 'withoutDepth' and modality_config['depth_column'] in feature_columns:
        feature_columns.remove(modality_config['depth_column'])

    # Filter data by min cells (using acute pool only)
    data, labels, label_counts = filter_data_by_min_cells(
        features, metadata_acute, label_column,
        label_order_dict[label_column],
        modality_config['min_cells']
    )

    # Train/test split
    X_train, X_test, y_train, y_test, id_train, id_test = create_stratified_split(
        data, feature_columns, label_column,
        modality_config['train_size'],
        'simple'  # Simple stratification for acute-only
    )

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train classifier
    best_model, cv_metrics = train_rf_classifier(
        X_train_scaled, y_train,
        RF_PARAM_GRID, RF_SCORING
    )

    # Evaluate
    results = evaluate_classifier(best_model, X_test_scaled, y_test, labels)

    # Save structured outputs
    save_classifier_results(
        output_base_dir / 'classifiers' / '20260722',
        modality_config['name'].lower(),
        label_column, depth_type, 'simple', 'acuteOnly',
        results, feature_columns, labels, y_test, cv_metrics, best_model
    )

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=PLOT_CONFIG[label_column]['fig_size'])

    title = (f"{modality_config['name']} {depth_type}\n"
             f"Predict: {label_column} (n≥{modality_config['min_cells']}, {len(data)} cells)\n"
             f"CV Accuracy: {cv_metrics['cv_acc_mean']:.3f}±{cv_metrics['cv_acc_std']:.3f}, "
             f"CV F1: {cv_metrics['cv_f1_mean']:.3f}±{cv_metrics['cv_f1_std']:.3f}\n"
             f"Test Accuracy: {results['accuracy']:.3f}, Test F1: {results['f1_weighted']:.3f}")

    plot_confusion_matrix(
        ax=ax,
        cm=results['cm'],
        labels=labels,
        color_dict=color_dict,
        title=title,
        num_cell_xbuffer=PLOT_CONFIG[label_column]['num_cell_xbuffer'],
        n_height=PLOT_CONFIG[label_column]['n_height'],
        fontsize_diagonal=10,
        fontsize_counts=10
    )

    plt.tight_layout()
    pdf_handle.savefig(fig, bbox_inches='tight')
    plt.close()

    return results


def run_acute_train_culture_test_variant(features, metadata, modality_config,
                                          label_column, depth_type, label_order_dict,
                                          color_dict, output_base_dir, pdf_handle):
    """
    Train on acute, test on culture.

    Parameters
    ----------
    features : pd.DataFrame
        Feature data
    metadata : pd.DataFrame
        Metadata with labels and paradigm
    modality_config : dict
        Configuration from get_modality_configs()
    label_column : str
        Label to classify
    depth_type : str
        'withDepth' or 'withoutDepth'
    label_order_dict : dict
        Dictionary mapping label_column to ordered label list
    color_dict : dict
        Color mapping for labels
    output_base_dir : Path
        Base output directory
    pdf_handle : PdfPages
        PDF file handle for saving plots

    Returns
    -------
    dict
        Results dictionary with acute_test and culture_test results
    """
    print(f"\n{'='*60}")
    print(f"ACUTE TRAIN / CULTURE TEST VARIANT: {modality_config['name']}")
    print(f"Label: {label_column}, Depth: {depth_type}")
    print(f"{'='*60}")

    # Prepare feature columns
    feature_columns = features.columns.tolist()
    if depth_type == 'withoutDepth' and modality_config['depth_column'] in feature_columns:
        feature_columns.remove(modality_config['depth_column'])

    # Split metadata by paradigm
    acute_metadata = metadata[metadata.paradigm == 'acute']
    culture_metadata = metadata[metadata.paradigm == 'culture']

    # Merge with features
    acute_data = features.merge(acute_metadata, how='inner', left_index=True, right_index=True)
    culture_data = features.merge(culture_metadata, how='inner', left_index=True, right_index=True)

    # Determine labels from ACUTE pool only (min_cells threshold)
    acute_data_filtered = acute_data.dropna(subset=[label_column])
    label_counts = acute_data_filtered[label_column].value_counts()
    labels = sorted(label_counts[label_counts >= modality_config['min_cells']].index)
    labels = [l for l in label_order_dict[label_column] if l in labels]

    # Filter both datasets to these labels
    acute_data = acute_data[acute_data[label_column].isin(labels)]
    culture_data = culture_data[culture_data[label_column].isin(labels)]

    print(f"Acute cells: {len(acute_data)}")
    print(f"Culture cells: {len(culture_data)}")
    print(f"Labels (from acute pool): {labels}")

    if len(acute_data) == 0 or len(culture_data) == 0:
        print("⚠ Skipping: insufficient acute or culture data")
        return None

    # 80/20 split of ACUTE data only
    y_acute = acute_data[label_column]
    X_acute = acute_data[feature_columns]

    X_train, X_test_acute, y_train, y_test_acute, id_train, id_test_acute = train_test_split(
        X_acute, y_acute, acute_data.index,
        stratify=y_acute,
        train_size=modality_config['train_size'],
        random_state=42
    )

    # Normalize: fit scaler on acute train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_acute_scaled = scaler.transform(X_test_acute)

    # Transform culture data with same scaler
    X_culture = culture_data[feature_columns]
    y_culture = culture_data[label_column]
    X_culture_scaled = scaler.transform(X_culture)

    # Train classifier on acute data
    best_model, cv_metrics = train_rf_classifier(
        X_train_scaled, y_train,
        RF_PARAM_GRID, RF_SCORING
    )

    # Evaluate on held-out acute test
    results_acute = evaluate_classifier(best_model, X_test_acute_scaled, y_test_acute, labels)

    # Evaluate on culture data
    results_culture = evaluate_classifier(best_model, X_culture_scaled, y_culture, labels)

    # Save results for both test sets
    save_classifier_results(
        output_base_dir / 'classifiers' / '20260722',
        modality_config['name'].lower(),
        label_column, depth_type, 'simple', 'acuteTrain_cultureTest',
        results_acute, feature_columns, labels, y_test_acute, cv_metrics, best_model,
        subset_name='acute'
    )

    save_classifier_results(
        output_base_dir / 'classifiers' / '20260722',
        modality_config['name'].lower(),
        label_column, depth_type, 'simple', 'acuteTrain_cultureTest',
        results_culture, feature_columns, labels, y_culture, cv_metrics, best_model,
        subset_name='culture'
    )

    # Plot culture confusion matrix
    fig, ax = plt.subplots(figsize=PLOT_CONFIG[label_column]['fig_size'])

    title = (f"{modality_config['name']} {depth_type}\n"
             f"Train: acute | Apply/test: culture\n"
             f"Predict: {label_column} (acute n={len(acute_data)}, culture n={len(culture_data)}, "
             f"min acute n≥{modality_config['min_cells']})\n"
             f"CV acute Accuracy: {cv_metrics['cv_acc_mean']:.3f}±{cv_metrics['cv_acc_std']:.3f}, "
             f"CV acute F1: {cv_metrics['cv_f1_mean']:.3f}±{cv_metrics['cv_f1_std']:.3f}\n"
             f"Held-out acute Accuracy: {results_acute['accuracy']:.3f}, F1: {results_acute['f1_weighted']:.3f}\n"
             f"Culture Accuracy: {results_culture['accuracy']:.3f}, F1: {results_culture['f1_weighted']:.3f}")

    plot_confusion_matrix(
        ax=ax,
        cm=results_culture['cm'],
        labels=labels,
        color_dict=color_dict,
        title=title,
        num_cell_xbuffer=PLOT_CONFIG[label_column]['num_cell_xbuffer'],
        n_height=PLOT_CONFIG[label_column]['n_height'],
        fontsize_diagonal=10,
        fontsize_counts=10
    )

    plt.tight_layout()
    pdf_handle.savefig(fig, bbox_inches='tight')
    plt.close()

    return {'acute': results_acute, 'culture': results_culture}


def run_balance_paradigm_variant(features, metadata, modality_config, label_column,
                                  depth_type, label_order_dict, color_dict,
                                  output_base_dir, pdf_handle):
    """
    Balance acute/culture in train and test, report separately.

    Generates 3 confusion matrices:
    1. All test cells combined
    2. Acute test cells only
    3. Culture test cells only

    Parameters
    ----------
    features : pd.DataFrame
        Feature data
    metadata : pd.DataFrame
        Metadata with labels and paradigm
    modality_config : dict
        Configuration from get_modality_configs()
    label_column : str
        Label to classify
    depth_type : str
        'withDepth' or 'withoutDepth'
    label_order_dict : dict
        Dictionary mapping label_column to ordered label list
    color_dict : dict
        Color mapping for labels
    output_base_dir : Path
        Base output directory
    pdf_handle : PdfPages
        PDF file handle for saving plots

    Returns
    -------
    dict
        Results dictionary with 'all', 'acute', 'culture' keys
    """
    print(f"\n{'='*60}")
    print(f"BALANCE PARADIGM VARIANT: {modality_config['name']}")
    print(f"Label: {label_column}, Depth: {depth_type}")
    print(f"{'='*60}")

    # Prepare feature columns
    feature_columns = features.columns.tolist()
    if depth_type == 'withoutDepth' and modality_config['depth_column'] in feature_columns:
        feature_columns.remove(modality_config['depth_column'])

    # Filter data by min cells
    data, labels, label_counts = filter_data_by_min_cells(
        features, metadata, label_column,
        label_order_dict[label_column],
        modality_config['min_cells']
    )

    # Train/test split stratified on label + paradigm
    X_train, X_test, y_train, y_test, id_train, id_test = create_stratified_split(
        data, feature_columns, label_column,
        modality_config['train_size'],
        'SubclassParadigm'  # Stratify on both label and paradigm
    )

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train classifier
    best_model, cv_metrics = train_rf_classifier(
        X_train_scaled, y_train,
        RF_PARAM_GRID, RF_SCORING
    )

    # Get paradigm info for test set
    test_paradigms = data.loc[id_test, 'paradigm']

    # Create test results dataframe
    y_pred = best_model.predict(X_test_scaled)
    test_results = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred,
        'paradigm': test_paradigms
    }, index=id_test)

    # Define subsets
    cm_sets = {
        'all': test_results,
        'acute': test_results[test_results['paradigm'] == 'acute'],
        'culture': test_results[test_results['paradigm'] == 'culture']
    }

    results_dict = {}

    for subset_name, subset_data in cm_sets.items():
        if len(subset_data) == 0:
            print(f"⚠ Skipping {subset_name}: no test cells")
            continue

        # Compute confusion matrix for this subset
        cm = confusion_matrix(subset_data['y_true'], subset_data['y_pred'], labels=labels)
        true_counts = np.sum(cm, axis=1)

        # Safe normalization
        normalized_cm = np.divide(
            cm,
            true_counts[:, np.newaxis],
            out=np.zeros_like(cm, dtype=float),
            where=true_counts[:, np.newaxis] != 0
        )
        row_acc = np.diag(normalized_cm)

        # Compute metrics
        accuracy = accuracy_score(subset_data['y_true'], subset_data['y_pred'])
        f1_weighted = f1_score(subset_data['y_true'], subset_data['y_pred'], average='weighted')

        results = {
            'y_pred': subset_data['y_pred'].values,
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'cm': cm,
            'true_counts': true_counts,
            'normalized_cm': normalized_cm,
            'row_acc': row_acc
        }

        results_dict[subset_name] = results

        # Save results
        save_classifier_results(
            output_base_dir / 'classifiers' / '20260722',
            modality_config['name'].lower(),
            label_column, depth_type, 'SubclassParadigm', 'balanceParadigm',
            results, feature_columns, labels,
            subset_data['y_true'], cv_metrics, best_model,
            subset_name=subset_name
        )

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=PLOT_CONFIG[label_column]['fig_size'])

        title = (f"{modality_config['name']} {depth_type}\n"
                 f"Predict: {label_column} ({subset_name} test cells, n={len(subset_data)})\n"
                 f"CV Accuracy: {cv_metrics['cv_acc_mean']:.3f}±{cv_metrics['cv_acc_std']:.3f}, "
                 f"CV F1: {cv_metrics['cv_f1_mean']:.3f}±{cv_metrics['cv_f1_std']:.3f}\n"
                 f"Test Accuracy: {accuracy:.3f}, Test F1: {f1_weighted:.3f}")

        plot_confusion_matrix(
            ax=ax,
            cm=cm,
            labels=labels,
            color_dict=color_dict,
            title=title,
            num_cell_xbuffer=PLOT_CONFIG[label_column]['num_cell_xbuffer'],
            n_height=PLOT_CONFIG[label_column]['n_height'],
            fontsize_diagonal=10,
            fontsize_counts=10
        )

        plt.tight_layout()
        pdf_handle.savefig(fig, bbox_inches='tight')
        plt.close()

    return results_dict
