#!/usr/bin/env python3

import pandas as pd
import numpy as np
import glob
import re
import sys
import yaml
from pathlib import Path
from collections import defaultdict
from io import StringIO

# ============== DATA EXTRACTION AND PARSING FUNCTIONS ==============

def load_config_from_yml(yml_file):
    """
    Load training configuration from YAML file.

    Args:
        yml_file: Path to YAML configuration file

    Returns:
        dict: Configuration parameters or None if file doesn't exist/can't be parsed
    """
    try:
        with open(yml_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Warning: Could not load config from {yml_file}: {e}")
        return None

def extract_all_data_from_files(csv_files):
    """
    Extract all data (performance, confusion matrices, profiling) from CSV files in a single pass.
    Reads training parameters from accompanying YAML files.

    Returns:
        tuple: (performance_df, matrices_data, profiling_data, overall_scores_data)
    """
    performance_data = []
    matrices_data = []
    profiling_data = []
    overall_scores_data = []

    print(f"Extracting all data from {len(csv_files)} files...")

    for csv_file in csv_files:
        # Get corresponding YAML file
        yml_file = Path(csv_file).with_suffix('.yml')

        # Load configuration from YAML
        config = load_config_from_yml(yml_file)
        if not config:
            print(f"Skipping {csv_file} - no valid config file")
            continue

        # Extract prebin - if dict, create bin name from multi-letter keys
        prebin_raw = config.get('prebin')
        if isinstance(prebin_raw, dict):
            # Filter keys with len > 1, sort by length (descending) then alphabetically, join with dash
            multi_letter_keys = [k for k in prebin_raw.keys() if len(k) > 1]
            if multi_letter_keys:
                # Sort by length (descending), then alphabetically
                multi_letter_keys.sort(key=lambda k: (-len(k), k))
                prebin = '-'.join(multi_letter_keys)
            else:
                prebin = 'none'
        elif prebin_raw:
            prebin = str(prebin_raw)
        else:
            prebin = 'none'

        # Extract relevant parameters from config
        params = {
            'img_enc_name': config.get('img_enc_name'),
            'resize_strategy': config.get('resize_strategy'),
            'posenc': config.get('pos_vec_coeff', 0) > 0,  # posenc is True if pos_vec_coeff > 0
            'dropouts': config.get('dropouts'),
            'num_epochs': config.get('num_epochs'),
            'num_layers': config.get('num_layers'),
            'prebin': prebin
        }

        # Validate required parameters
        if not all([params['img_enc_name'], params['resize_strategy'],
                    params['dropouts'] is not None, params['num_epochs'], params['num_layers']]):
            print(f"Skipping {csv_file} - missing required parameters in config")
            continue

        try:
            # Read file once
            with open(csv_file, 'r') as f:
                content = f.read()
            lines = content.split('\n')

            # Find confusion matrix section
            matrix_start = -1
            for i, line in enumerate(lines):
                if 'Confusion Matrix' in line:
                    matrix_start = i
                    break

            # Extract performance data (before confusion matrix)
            if matrix_start > 0:
                perf_content = '\n'.join(lines[:matrix_start]).strip()
                if perf_content:
                    try:
                        perf_df = pd.read_csv(StringIO(perf_content))
                        for _, row in perf_df.iterrows():
                            if row['Label'] != 'Overall':
                                performance_data.append({
                                    'filename': Path(csv_file).name,
                                    'img_enc_name': params['img_enc_name'],
                                    'resize_strategy': params['resize_strategy'],
                                    'posenc': params['posenc'],
                                    'dropouts': params['dropouts'],
                                    'num_epochs': params['num_epochs'],
                                    'num_layers': params['num_layers'],
                                    'prebin': params['prebin'],
                                    'label': row['Label'],
                                    'accuracy': row['Accuracy'],
                                    'precision': row['Precision'],
                                    'recall': row['Recall'],
                                    'f1_score': row['F1-Score']
                                })
                            else:
                                overall_scores_data.append({
                                    'filename': Path(csv_file).name,
                                    'f1_score': row['F1-Score']
                                })
                    except Exception as e:
                        print(f"Error parsing performance data from {csv_file}: {e}")

            # Extract confusion matrix
            if matrix_start >= 0 and matrix_start + 1 < len(lines):
                header_line = lines[matrix_start + 1].strip()
                if header_line.startswith(' ') or header_line.startswith(','):
                    labels = [label.strip() for label in header_line.split(',')[1:]]
                    matrix_data = []

                    for line in lines[matrix_start + 2:]:
                        line = line.strip()
                        if not line or line.startswith('training-time') or line.startswith('Epoch'):
                            break

                        parts = line.split(',')
                        if len(parts) < 2 or not parts[0].strip():
                            break

                        try:
                            row_data = [int(x.strip()) for x in parts[1:]]
                            if len(row_data) == len(labels):
                                matrix_data.append(row_data)
                            else:
                                break
                        except ValueError:
                            break

                    if matrix_data:
                        matrices_data.append({
                            'filename': Path(csv_file).name,
                            'img_enc_name': params['img_enc_name'],
                            'resize_strategy': params['resize_strategy'],
                            'posenc': params['posenc'],
                            'prebin': params['prebin'],
                            'matrix': np.array(matrix_data),
                            'labels': labels
                        })

            # Extract profiling data
            vram_match = re.search(r'peak-vram-usage,(.+)', content)
            train_time_match = re.search(r'training-time,([0-9.]+)', content)
            valid_time_match = re.search(r'validation-time,([0-9.]+)', content)

            vram_mb = 0
            if vram_match:
                mb_values = re.findall(r'([0-9.]+)MB peak', vram_match.group(1))
                vram_mb = max(float(mb) for mb in mb_values) if mb_values else 0

            if train_time_match and vram_mb > 0:
                profiling_data.append({
                    'filename': Path(csv_file).name,
                    'img_enc_name': params['img_enc_name'],
                    'resize_strategy': params['resize_strategy'],
                    'posenc': params['posenc'],
                    'prebin': params['prebin'],
                    'vram_peak_mb': vram_mb,
                    'training_time_sec': float(train_time_match.group(1)),
                    'validation_time_sec': float(valid_time_match.group(1)) if valid_time_match else None
                })

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue

    print(f"Successfully extracted:")
    print(f"  - {len(performance_data)} performance records")
    print(f"  - {len(matrices_data)} confusion matrices")
    print(f"  - {len(profiling_data)} profiling records")

    performance_df = pd.DataFrame(performance_data) if performance_data else None
    return performance_df, matrices_data, profiling_data, overall_scores_data

# ============== ANALYSIS FUNCTIONS ==============

def analyze_confusion_patterns(matrix, labels):
    """Analyze confusion patterns in the matrix, excluding '+' rows/columns (totals)"""
    # Filter out '+' labels as they represent totals, not actual classes
    filtered_indices = [i for i, label in enumerate(labels) if label != '+']
    filtered_labels = [labels[i] for i in filtered_indices]

    # Extract submatrix excluding '+' rows and columns
    if len(filtered_indices) == 0:
        return {}, []

    filtered_matrix = matrix[np.ix_(filtered_indices, filtered_indices)]

    # Calculate per-class metrics
    n_classes = len(filtered_labels)
    class_metrics = {}

    for i, label in enumerate(filtered_labels):
        if i >= filtered_matrix.shape[0] or i >= filtered_matrix.shape[1]:
            continue

        tp = filtered_matrix[i, i]
        fp = np.sum(filtered_matrix[:, i]) - tp
        fn = np.sum(filtered_matrix[i, :]) - tp
        tn = np.sum(filtered_matrix) - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        class_metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'support': tp + fn
        }

    # Find most confused label pairs (excluding '+')
    confusion_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and i < len(filtered_labels) and j < len(filtered_labels):
                if i < filtered_matrix.shape[0] and j < filtered_matrix.shape[1]:
                    confusion_count = filtered_matrix[i, j]
                    if confusion_count > 0:
                        confusion_pairs.append({
                            'true_label': filtered_labels[i],
                            'predicted_label': filtered_labels[j],
                            'count': confusion_count,
                            'percentage': confusion_count / np.sum(filtered_matrix[i, :]) * 100 if np.sum(filtered_matrix[i, :]) > 0 else 0
                        })

    # Sort by confusion count
    confusion_pairs.sort(key=lambda x: x['count'], reverse=True)

    return class_metrics, confusion_pairs

def merge_performance_and_resources(df, profiling_data):
    """Merge grid search performance with resource usage data"""
    # Calculate average performance by parameter combination
    gs_avg = df.groupby(['img_enc_name', 'resize_strategy', 'posenc'])['f1_score'].mean().reset_index()
    gs_avg = gs_avg.rename(columns={'f1_score': 'avg_f1'})

    # Create profiling dataframe
    prof_df = pd.DataFrame(profiling_data)

    # Filter out zero VRAM (failed runs)
    prof_df = prof_df[prof_df['vram_peak_mb'] > 0].copy()

    if len(prof_df) == 0:
        return pd.DataFrame()

    # Merge datasets
    merged = prof_df.merge(gs_avg, on=['img_enc_name', 'resize_strategy', 'posenc'], how='inner')

    if len(merged) > 0:
        # Calculate efficiency metrics
        merged['training_time_hours'] = merged['training_time_sec'] / 3600
        merged['f1_per_hour'] = merged['avg_f1'] / merged['training_time_hours']
        merged['f1_per_gb_vram'] = merged['avg_f1'] / (merged['vram_peak_mb'] / 1024)

    return merged

def find_pareto_optimal(df):
    """Find Pareto-optimal configurations"""
    if len(df) == 0:
        return pd.DataFrame()

    pareto_candidates = []

    for _, row in df.iterrows():
        is_dominated = False
        for _, other in df.iterrows():
            if (other['avg_f1'] >= row['avg_f1'] and
                other['vram_peak_mb'] <= row['vram_peak_mb'] and
                other['training_time_hours'] <= row['training_time_hours'] and
                (other['avg_f1'] > row['avg_f1'] or
                 other['vram_peak_mb'] < row['vram_peak_mb'] or
                 other['training_time_hours'] < row['training_time_hours'])):
                is_dominated = True
                break
        if not is_dominated:
            pareto_candidates.append(row)

    if pareto_candidates:
        pareto_df = pd.DataFrame(pareto_candidates)
        pareto_df = pareto_df.sort_values('avg_f1', ascending=False)
        return pareto_df
    return pd.DataFrame()

# ============== REPORTING FUNCTIONS ==============

def _calculate_macro_averages(matrix, labels):
    """Calculate macro-averaged precision, recall, and F1 score."""
    class_metrics, _ = analyze_confusion_patterns(matrix, labels)
    
    if not class_metrics:
        return {'macro_precision': 0, 'macro_recall': 0, 'macro_f1': 0}

    precisions = [metrics['precision'] for metrics in class_metrics.values()]
    recalls = [metrics['recall'] for metrics in class_metrics.values()]
    f1s = [metrics['f1'] for metrics in class_metrics.values()]

    macro_precision = np.mean(precisions) if precisions else 0
    macro_recall = np.mean(recalls) if recalls else 0
    macro_f1 = np.mean(f1s) if f1s else 0
    
    return {'macro_precision': macro_precision, 'macro_recall': macro_recall, 'macro_f1': macro_f1}

def _render_matrix_as_markdown(matrix, labels):
    """Render a numpy confusion matrix as a Markdown table."""
    header = "| True \ Pred | " + " | ".join(labels) + " |\n"
    divider = "|---|" + "---|" * len(labels) + "\n"
    body = ""
    for i, label in enumerate(labels):
        row_values = " | ".join(map(str, matrix[i]))
        body += f"| **{label}** | {row_values} |\n"
    return header + divider + body

def _generate_executive_summary(df, best_params, matrices_data):
    """Generate executive summary section"""
    grid_search_labels = [l for l in df['label'].unique() if l != '!AVG']

    # Get unique parameter combinations
    params = ['img_enc_name', 'resize_strategy', 'posenc', 'prebin', 'dropouts', 'num_epochs', 'num_layers']
    param_combos = df.groupby(params).size().reset_index(name='count')

    # Create grid table
    grid_table = "\n### Parameter Grid Coverage\n\n"
    grid_table += "| Model | Resize Strategy | PosEnc | Prebin | Dropout | Epochs | Layers | Experiments |\n"
    grid_table += "|-------|----------------|--------|--------|---------|--------|--------|-------------|\n"

    for _, row in param_combos.iterrows():
        grid_table += f"| {row['img_enc_name']} | {row['resize_strategy']} | {row['posenc']} | {row['prebin']} | {row['dropouts']} | {row['num_epochs']} | {row['num_layers']} | {row['count']} |\n"

    # Summary statistics by parameter
    grid_table += "\n**Parameter Value Distribution:**\n"
    for param in params:
        unique_vals = df[param].unique()
        if param == 'posenc':
            unique_vals_str = ', '.join(map(str, sorted(list(set(map(bool, unique_vals))))))
            grid_table += f"- **{param}**: {unique_vals_str}\n"
        else:
            grid_table += f"- **{param}**: {len(unique_vals)} unique values ({', '.join(map(str, sorted(unique_vals)))})\n"

    return f"""# Comprehensive Grid Search and Confusion Matrix Analysis

## Executive Summary

This analysis examines {len(df['filename'].unique())} training runs across {len(param_combos)} parameter combinations, incorporating both performance metrics and detailed error pattern analysis from confusion matrices.

**Key Findings:**
- **Best Overall Configuration**: `{best_params['img_enc_name']}` with F1=`{best_params['mean']:.4f}`
- **Experiments with Confusion Matrices**: {len(matrices_data)}
- **Labels in Grid Search**: {len(grid_search_labels)} ({', '.join(sorted(grid_search_labels))})

{grid_table}

---
"""

def _generate_performance_section(df, best_params):
    """Generate grid search performance analysis section"""
    grid_search_labels = [l for l in df['label'].unique() if l != '!AVG']

    section = f"""## Grid Search Performance Analysis

### Best Overall Parameter Combination
**Average F1 Score: {best_params['mean']:.4f} (±{best_params['std']:.4f})**

| Parameter | Value |
|-----------|-------|
| Model | {best_params['img_enc_name']} |
| Epochs | {best_params['num_epochs']} |
| Layers | {best_params['num_layers']} |
| Position Encoding | {best_params['posenc']} |
| Resize Strategy | {best_params['resize_strategy']} |
| Prebin | {best_params['prebin']} |
| Dropout | {best_params['dropouts']} |

### Parameter Impact Rankings

**Model Architecture Performance:**
"""

    # Add model performance table
    model_impact = df.groupby('img_enc_name')['f1_score'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
    for model, stats in model_impact.iterrows():
        section += f"- **{model}**: {stats['mean']:.3f} (±{stats['std']:.3f})\n"

    section += f"""
**Other Parameter Impact:**
- **Resize Strategy**: {df.groupby('resize_strategy')['f1_score'].mean().sort_values(ascending=False).index[0]} performs best

### Label-Specific Performance

| Label | Avg F1 | Std | Min | Max | Best Configuration |
|-------|--------|-----|-----|-----|-------------------|
"""

    # Add label performance table
    for label in grid_search_labels:
        label_data = df[df['label'] == label]
        if len(label_data) > 0:
            stats = label_data['f1_score'].agg(['mean', 'std', 'min', 'max'])
            best_idx = label_data['f1_score'].idxmax()
            best_row = label_data.loc[best_idx]
            best_config = f"{best_row['img_enc_name']}, {best_row['resize_strategy']}"
            section += f"| {label} | {stats['mean']:.3f} | {stats['std']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} | {best_config} |\n"

    section += "\n---\n"
    return section

def _generate_confusion_matrix_section(df, matrices_data, overall_scores_data):
    """Generate confusion matrix analysis section for the best run of each prebin strategy."""
    section = """
## Confusion Matrix Analysis

### Methodology
For each `prebin` strategy, the single best-performing experiment was identified based on the highest **micro-average F1 score**. The following sections detail the confusion matrix, performance metrics, and most frequent errors for this top run. We also report where this experiment ranks in terms of its **macro-average F1 score** to provide a more complete picture of its performance.

**A Note on Averaging:** This report includes two types of averaged scores. **Micro-averaging** calculates metrics globally by counting the total true positives, false negatives, and false positives across all classes. It gives equal weight to each sample. **Macro-averaging** calculates metrics for each class independently and then takes the unweighted average, giving equal weight to each class, regardless of its size. For class-imbalanced datasets, macro-averages provide a better view of the model\'s performance on smaller classes.
"""
    prebin_schemes = sorted(df['prebin'].unique())

    # Create df with micro F1 scores and prebin info
    overall_scores_df = pd.DataFrame(overall_scores_data)
    prebin_info_df = df[['filename', 'prebin']].drop_duplicates()
    overall_scores_df = overall_scores_df.merge(prebin_info_df, on='filename', how='left')
    overall_scores_df.rename(columns={'f1_score': 'micro_f1'}, inplace=True)

    # Create df with macro F1 scores
    macro_scores_df = df.groupby(['filename', 'prebin'])['f1_score'].mean().reset_index()
    macro_scores_df.rename(columns={'f1_score': 'macro_f1'}, inplace=True)

    for prebin_scheme in prebin_schemes:
        section += f"""
---
### Prebin Strategy: `{prebin_scheme}`
"""

        # --- SELECTION (by Micro F1) ---
        scheme_micro_scores = overall_scores_df[overall_scores_df['prebin'] == prebin_scheme]
        if scheme_micro_scores.empty:
            section += "\nNo data available for this prebin strategy.\n"
            continue
        best_run_series = scheme_micro_scores.loc[scheme_micro_scores['micro_f1'].idxmax()]
        best_filename = best_run_series['filename']
        best_run_micro_f1 = best_run_series['micro_f1']

        # --- RANKING (by Macro F1) ---
        scheme_macro_scores = macro_scores_df[macro_scores_df['prebin'] == prebin_scheme].copy()
        scheme_macro_scores['rank'] = scheme_macro_scores['macro_f1'].rank(method='min', ascending=False).astype(int)
        total_runs = len(scheme_macro_scores)
        macro_rank_str = "N/A"
        if total_runs > 0:
            try:
                selected_run_macro_info = scheme_macro_scores[scheme_macro_scores['filename'] == best_filename].iloc[0]
                macro_rank = selected_run_macro_info['rank']
                macro_rank_str = f"{macro_rank} of {total_runs}"
            except IndexError:
                macro_rank_str = f"N/A of {total_runs}"

        # --- REPORTING ---
        matrix_data = next((m for m in matrices_data if m['filename'] == best_filename), None)
        if not matrix_data:
            section += f"\nCould not find confusion matrix for the best run file: `{best_filename}`.\n"
            continue

        # Display best run info
        best_run_params = df[df['filename'] == best_filename].iloc[0]
        section += f"""

**Best Run File (by Micro-F1)**: `{best_filename}` (Micro-F1: {best_run_micro_f1:.4f})

**Best Run Parameters:**
| Parameter | Value |
|-----------|-------|
| Model | {best_run_params['img_enc_name']} |
| Epochs | {best_run_params['num_epochs']} |
| Layers | {best_run_params['num_layers']} |
| Position Encoding | {best_run_params['posenc']} |
| Resize Strategy | {best_run_params['resize_strategy']} |
| Dropout | {best_run_params['dropouts']} |
"""

        # Render matrix
        matrix = matrix_data['matrix']
        labels = matrix_data['labels']
        section += """

**Confusion Matrix:**
"""
        section += _render_matrix_as_markdown(matrix, labels)

        # Overall metrics
        macro_averages = _calculate_macro_averages(matrix, labels)
        
        section += f"""

**Overall Performance Metrics:**
| Metric | Score |
|------------------------|-------|
| Micro-Average F1 | {best_run_micro_f1:.4f} |
| Macro-Average F1 | {macro_averages['macro_f1']:.4f} |
| Macro-Average F1 Rank | {macro_rank_str} |
| Macro-Average Precision| {macro_averages['macro_precision']:.4f} |
| Macro-Average Recall | {macro_averages['macro_recall']:.4f} |
"""

        # Top 10 confusions
        _, confusion_pairs = analyze_confusion_patterns(matrix, labels)
        section += """

**Top 10 Most Frequent Label Confusions:**

| Rank | True Label | Predicted Label | Count | Percentage |
|------|------------|-----------------|-------|------------|
"""
        for i, pair in enumerate(confusion_pairs[:10]):
            section += f"| {i+1} | {pair['true_label']} | {pair['predicted_label']} | {pair['count']} | {pair['percentage']:.1f}% |\n"

    section += "\n---\n"
    return section

def _generate_resource_section(resource_df):
    """Generate resource usage and efficiency analysis section"""
    section = """
## Resource Usage and Efficiency Analysis

### Computational Requirements

"""

    if len(resource_df) == 0:
        return section + "Resource usage data not available for this analysis.\n\n---\n"

    # VRAM analysis
    vram_by_model = resource_df.groupby('img_enc_name')['vram_peak_mb'].agg(['mean', 'min', 'max']).round(1)
    section += """
**VRAM Usage by Model:**

| Model | Mean (MB) | Min (MB) | Max (MB) |
|-------|-----------|----------|----------|
"""
    for model, stats in vram_by_model.sort_values('mean', ascending=False).iterrows():
        section += f"| {model} | {stats['mean']:.0f} | {stats['min']:.0f} | {stats['max']:.0f} |\n"

    # Training time analysis
    time_by_model = resource_df.groupby('img_enc_name')['training_time_hours'].agg(['mean', 'min', 'max']).round(2)
    section += """

**Training Time by Model:**

| Model | Mean (hours) | Min (hours) | Max (hours) |
|-------|--------------|-------------|-------------|
"""
    for model, stats in time_by_model.sort_values('mean', ascending=False).iterrows():
        section += f"| {model} | {stats['mean']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} |\n"

    # Efficiency analysis
    f1_per_hour = resource_df.groupby('img_enc_name')['f1_per_hour'].mean().round(4)
    f1_per_gb = resource_df.groupby('img_enc_name')['f1_per_gb_vram'].mean().round(4)

    section += """

### Efficiency Metrics

**Time Efficiency (F1 Score per Training Hour):**
"""
    for model, efficiency in f1_per_hour.sort_values(ascending=False).items():
        section += f"- **{model}**: {efficiency:.4f} F1/hour\n"

    section += """

**Memory Efficiency (F1 Score per GB VRAM):**
"""
    for model, efficiency in f1_per_gb.sort_values(ascending=False).items():
        section += f"- **{model}**: {efficiency:.4f} F1/GB\n"

    # Pareto optimal configurations
    pareto_df = find_pareto_optimal(resource_df)
    if len(pareto_df) > 0:
        section += f"""

### Pareto-Optimal Configurations

A configuration is **Pareto-optimal** if no other configuration is strictly better in all dimensions. In other words, a Pareto-optimal configuration cannot be improved in one metric (F1 score, memory usage, or training time) without sacrificing performance in at least one other metric. These configurations represent the best possible trade-offs along the efficiency frontier.

Found **{len(pareto_df)} Pareto-optimal configurations** out of {len(resource_df)} total experiments:

| Rank | F1 Score | VRAM (MB) | Time (hours) | Model | Config |
|------|----------|-----------|--------------|-------|--------|\n"""
        for i, (_, row) in enumerate(pareto_df.head(10).iterrows(), 1):
            section += f"| {i} | {row['avg_f1']:.3f} | {row['vram_peak_mb']:.0f} | {row['training_time_hours']:.2f} | {row['img_enc_name']} | {row['resize_strategy']}, posenc={row['posenc']} |\n"

        # Resource recommendations
        section += """

### Resource-Based Recommendations

**For High-Performance Requirements:**
"""
        best_performance = pareto_df.iloc[0]
        section += f"- Use **{best_performance['img_enc_name']}** with {best_performance['resize_strategy']} strategy\n"
        section += f"- Expected: F1={best_performance['avg_f1']:.3f}, {best_performance['vram_peak_mb']:.0f}MB VRAM, {best_performance['training_time_hours']:.1f}h training\n"

        # Most efficient models
        most_time_efficient = f1_per_hour.index[0]
        most_memory_efficient = f1_per_gb.index[0]

        section += f"""

**For Resource-Constrained Environments:**
- **Time-efficient**: {most_time_efficient} ({f1_per_hour[most_time_efficient]:.4f} F1/hour)
- **Memory-efficient**: {most_memory_efficient} ({f1_per_gb[most_memory_efficient]:.4f} F1/GB VRAM)

"""

    section += "---\n"
    return section

def _generate_recommendations_section(best_params):
    """Generate combined insights and recommendations section"""
    return f"""\n## Combined Insights and Recommendations

### Performance vs Error Pattern Correlation

Based on the integrated analysis of F1 scores and confusion matrices:

**High-Performing Labels** (F1 > 0.7):
- Generally show clean confusion patterns with errors concentrated in semantically similar labels
- Less susceptible to systematic misclassification

**Medium-Performing Labels** (0.3 < F1 < 0.7):
- Show more diverse error patterns
- May benefit from targeted data augmentation

**Low-Performing Labels** (F1 < 0.3):
- Often suffer from class imbalance and systematic confusion with dominant classes
- Require specialized attention in model design or data collection

### Recommendations

**For Production Deployment:**
1. Use the best configuration: {best_params['img_enc_name']} with identified optimal parameters
2. Implement post-processing rules to handle systematic confusions
3. Consider ensemble approaches for problematic label pairs

**For Model Improvement:**
1. **Data Augmentation**: Focus on underrepresented labels showing high confusion rates
2. **Architecture Modifications**: Consider label-specific heads for highly confused pairs
3. **Training Strategy**: Implement class balancing for labels with poor recall

**For Future Research:**
1. Investigate why certain label pairs are systematically confused
2. Explore domain-specific regularization techniques
3. Consider hierarchical classification for semantically related labels

---

## Files Generated
- `comprehensive_grid_search_analysis.md`: This comprehensive report
- Additional analysis files as referenced

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

def _generate_posenc_impact_section(df):
    """Generate a section analyzing the impact of posenc on labels in the 'none' prebin group."""
    section = """## Positional Encoding Impact Analysis (prebin='none')

This section analyzes the impact of the `posenc` parameter on the F1 score of each label, specifically for experiments where no pre-binning was used (`prebin='none'-`).

"""
    try:
        # 1. Filter for prebin == 'none'
        none_df = df[df['prebin'] == 'none'].copy()
        if none_df.empty:
            return section + "No data available for the `prebin='none'` group to analyze `posenc` impact.\n"

        # 2. Group by label and posenc, calculate mean F1
        posenc_impact_df = none_df.groupby(['label', 'posenc'])['f1_score'].mean().unstack()
        posenc_impact_df.rename(columns={True: 'f1_posenc_true', False: 'f1_posenc_false'}, inplace=True)

        # 3. Calculate impact
        posenc_impact_df.fillna(0, inplace=True) # Handle cases where a label only exists for one posenc value
        posenc_impact_df['abs_impact'] = posenc_impact_df['f1_posenc_true'] - posenc_impact_df['f1_posenc_false']
        posenc_impact_df['impact(%)'] = posenc_impact_df['abs_impact'] / (posenc_impact_df['f1_posenc_false'] + 1e-6)  # Avoid division by zero
        posenc_impact_df['impact(%)'] = posenc_impact_df['impact(%)'].round(4)
        ## then turn to percentage notation
        posenc_impact_df['impact(%)'] = (posenc_impact_df['impact(%)'] * 100).round(2)
        posenc_impact_df['abs_impact'] = posenc_impact_df['abs_impact'].abs()

        # 4. Sort by absolute impact
        sorted_impact = posenc_impact_df.sort_values('impact(%)', ascending=False)

        # 5. Present results
        section += "### Impact of posenc=True\n"
        section += "NOTE that in the current report generation, all values of positional encoding coefficient > 0 are treated as `posenc=True`, hence the numbers below show the _best_ impact of all posenc values.\n\n"
        section += sorted_impact.to_markdown() + "\n\n"

    except Exception as e:
        section += f"An error occurred during posenc impact analysis: {e}\n"
    
    section += "---\n"
    return section

def generate_markdown_report(df, best_params, matrices_data, resource_df, overall_scores_data):
    """
    Generate comprehensive markdown report by assembling modular sections.

    Args:
        df: Performance dataframe
        best_params: Best parameter configuration
        matrices_data: List of confusion matrix data
        resource_df: Resource usage dataframe
        overall_scores_data: List of overall F1 scores from CSVs

    Returns:
        str: Complete markdown report
    """
    sections = [
        _generate_executive_summary(df, best_params, matrices_data),
        _generate_performance_section(df, best_params),
        _generate_confusion_matrix_section(df, matrices_data, overall_scores_data),
        _generate_posenc_impact_section(df),
        _generate_resource_section(resource_df),
        _generate_recommendations_section(best_params)
    ]

    return ''.join(sections)

# ============== MAIN EXECUTION ==============

def main():
    """Main function to run the analysis"""
    # Configuration - expect path/prefix argument
    if len(sys.argv) < 2:
        print("Usage: python analyze_gridsearch.py <path/prefix>")
        print("Example: python analyze_gridsearch.py modeling/results-aristotle/20250925-1540-newbinning")
        sys.exit(1)

    path_prefix = sys.argv[1]

    # Extract just the filename prefix for output naming
    prefix_name = Path(path_prefix).name

    print(f"Analyzing results with path/prefix: {path_prefix}")

    # Get all individual result files matching the pattern
    csv_files = glob.glob(f'{path_prefix}*.csv')

    if not csv_files:
        print(f"No result files found matching pattern: {path_prefix}*.csv")
        sys.exit(1)

    print(f"Found {len(csv_files)} result files")

    # Extract all data from files in a single pass
    print("\n" + "="*50 + "\nEXTRACTING ALL DATA\n" + "="*50)
    df, matrices_data, profiling_data, overall_scores_data = extract_all_data_from_files(csv_files)

    if df is None or df.empty:
        print("Failed to extract performance data")
        sys.exit(1)

    # Calculate best parameters
    params = ['dropouts', 'img_enc_name', 'num_epochs', 'num_layers', 'posenc', 'resize_strategy', 'prebin']
    avg_scores = df.groupby(params)['f1_score'].agg(['mean', 'std', 'count']).reset_index()
    avg_scores = avg_scores.sort_values('mean', ascending=False)
    best_params = avg_scores.iloc[0]

    print(f"\nBest configuration: {best_params['img_enc_name']} (F1={best_params['mean']:.4f})")

    # --- Analysis ---
    print("\n" + "="*50 + "\nANALYZING DATA\n" + "="*50)

    # Merge performance with resource data
    resource_df = merge_performance_and_resources(df, profiling_data)

    # --- Reporting ---
    print("\n" + "="*50 + "\nGENERATING REPORT\n" + "="*50)
    markdown_report = generate_markdown_report(df, best_params, matrices_data, resource_df, overall_scores_data)
    output_filename = f'comprehensive_grid_search_analysis_{prefix_name}.md'
    with open(output_filename, 'w') as f:
        f.write(markdown_report)
    print(f"Comprehensive analysis report saved to: {output_filename}")


if __name__ == "__main__":
    main()
