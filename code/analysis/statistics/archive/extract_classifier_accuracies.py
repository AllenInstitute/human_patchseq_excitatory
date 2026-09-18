"""
Extract Classifier Accuracy Metrics from Notebooks

This script extracts test accuracy and F1 scores from classifier notebook outputs
and saves them to a CSV file for the statistics table.
"""

import json
import re
import csv
from pathlib import Path
from collections import defaultdict

def parse_classifier_notebook(nb_path, modality_name):
    """
    Parse a classifier notebook to extract accuracy metrics.

    Returns list of dicts with:
    - Modality
    - Grouping_Level (subclass, layer_subclass, t_type, layer, or within_layer)
    - Group_Name (for within-layer classifiers, e.g., "Layer 4")
    - Label_Column (what's being predicted)
    - Depth_Included (withDepth or withoutDepth)
    - Min_Cells
    - Test_Accuracy
    - Test_F1_Weighted
    - CV_Accuracy_Mean
    - CV_Accuracy_Std
    - CV_F1_Mean
    - CV_F1_Std
    - Sample_Size
    """

    with open(nb_path, 'r') as f:
        nb = json.load(f)

    results = []

    # Find classifier cells (contain RandomForestClassifier)
    for cell_idx, cell in enumerate(nb['cells']):
        source = ''.join(cell.get('source', []))
        if 'RandomForestClassifier' not in source or 'outputs' not in cell:
            continue

        # Determine if this is cell 10 (overall) or cell 14 (within-layer)
        is_within_layer = 'group_column' in source and 'for group in groups' in source

        # Extract configuration from source code
        min_cells_match = re.search(r'min_num_cells = (\d+)', source)
        min_cells = int(min_cells_match.group(1)) if min_cells_match else None

        # Parse outputs to get actual results
        stdout_text = ""
        for output in cell['outputs']:
            if output.get('name') == 'stdout' or (output.get('output_type') == 'stream' and output.get('name') == 'stdout'):
                text = output.get('text', [])
                stdout_text += ''.join(text) if isinstance(text, list) else text

        # Parse the stdout to extract metrics
        # Pattern: "Best Parameters: ..." followed by "CV (n=3 folds) | Accuracy: X.XXX ± Y.YYY | F1 (weighted): ..."
        # But we need test accuracy which isn't printed to stdout - it's only in plot titles

        # For now, collect CV metrics which we can extract
        cv_pattern = r'CV \(n=\d+ folds\) \| Accuracy: ([\d.]+) ± ([\d.]+) \| F1 \(weighted\): ([\d.]+) ± ([\d.]+)'
        cv_matches = list(re.finditer(cv_pattern, stdout_text))

        # Count configurations based on source code
        if not is_within_layer:
            # Cell 10: label_columns × depth_types
            label_columns = ['t_type', 'subclass_label', 'layer_subclass', 'layer']
            depth_types = ['withDepth', 'withoutDepth']

            for i, (label_col, depth_type) in enumerate(
                [(lc, dt) for dt in depth_types for lc in label_columns]
            ):
                if i < len(cv_matches):
                    match = cv_matches[i]
                    results.append({
                        'Modality': modality_name,
                        'Grouping_Level': 'overall',
                        'Group_Name': None,
                        'Label_Column': label_col,
                        'Depth_Included': depth_type,
                        'Min_Cells': min_cells,
                        'Test_Accuracy': None,  # Not in stdout
                        'Test_F1_Weighted': None,
                        'CV_Accuracy_Mean': float(match.group(1)),
                        'CV_Accuracy_Std': float(match.group(2)),
                        'CV_F1_Mean': float(match.group(3)),
                        'CV_F1_Std': float(match.group(4)),
                        'Sample_Size': None
                    })
        else:
            # Cell 14: groups × label_columns × depth_types (but some are skipped)
            # Just collect all CV results in order
            for match in cv_matches:
                results.append({
                    'Modality': modality_name,
                    'Grouping_Level': 'within_layer',
                    'Group_Name': None,  # Would need more parsing
                    'Label_Column': None,
                    'Depth_Included': None,
                    'Min_Cells': min_cells,
                    'Test_Accuracy': None,
                    'Test_F1_Weighted': None,
                    'CV_Accuracy_Mean': float(match.group(1)),
                    'CV_Accuracy_Std': float(match.group(2)),
                    'CV_F1_Mean': float(match.group(3)),
                    'CV_F1_Std': float(match.group(4)),
                    'Sample_Size': None
                })

    return results


def main():
    # Notebook paths
    notebooks = [
        ('/root/capsule/code/analysis/classifiers/ephys_classifiers.ipynb', 'Electrophysiology'),
        ('/root/capsule/code/analysis/classifiers/morpho_classifiers.ipynb', 'Morphology'),
        ('/root/capsule/code/analysis/classifiers/morphoelectric_classifiers.ipynb', 'Morphoelectric'),
    ]

    all_results = []

    for nb_path, modality in notebooks:
        print(f"Processing {modality}...")
        try:
            results = parse_classifier_notebook(nb_path, modality)
            all_results.extend(results)
            print(f"  Extracted {len(results)} metrics")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Save to CSV
    output_path = Path('/root/capsule/scratch/20260807/classifiers/classifier_accuracy_metrics.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if all_results:
        # Write CSV
        fieldnames = list(all_results[0].keys())
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        print(f"\n✓ Saved {len(all_results)} metrics to {output_path}")

        # Summary by modality
        by_modality = defaultdict(list)
        for r in all_results:
            by_modality[r['Modality']].append(r['CV_Accuracy_Mean'])

        print(f"\nSummary by modality:")
        for mod, accs in by_modality.items():
            print(f"  {mod}: {len(accs)} metrics, mean={sum(accs)/len(accs):.3f}, range=[{min(accs):.3f}, {max(accs):.3f}]")
    else:
        print("No results extracted")

    return all_results


if __name__ == '__main__':
    df = main()
