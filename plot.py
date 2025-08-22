import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pandas as pd
import sys
import os


# Function to parse strings of the form "mean (lower, upper)"
def parse_value(s):
    match = re.match(r"([\d\.nan]+)\s*\(\s*([\d\.nan]+),\s*([\d\.nan]+)\s*\)", str(s))
    if match:
        mean, lower, upper = map(float, match.groups())
        return mean, lower, upper
    else:
        return float("nan"), float("nan"), float("nan")


# Function to filter dataframe for only AUC, specificity, and sensitivity metrics
def filter_metrics(df, target_metrics=['auc', 'specificity', 'sensitivity']):
    """
    Filter dataframe to keep only rows for specified metrics.
    Each metric should have 3 consecutive rows: [All, Male, Female]
    """
    filtered_rows = []
    
    # Process the dataframe in groups of 3 rows (All, Male, Female for each metric)
    num_rows = len(df)
    if num_rows % 3 != 0:
        raise ValueError("Input table must have rows in multiples of 3: [All, Male, Female] per metric")
    
    for i in range(0, num_rows, 3):
        # Get the metric name from the first row of the group
        metric_row = df.iloc[i]
        metric_name = str(metric_row.iloc[0]).lower()
        
        # Check if this metric contains any of our target metrics
        for target in target_metrics:
            if target in metric_name:
                # Add all 3 rows for this metric (All, Male, Female)
                filtered_rows.extend([i, i+1, i+2])
                break
    
    if not filtered_rows:
        raise ValueError(f"No metrics found matching: {target_metrics}")
    
    return df.iloc[filtered_rows].reset_index(drop=True)


# Function to create subplots per metric, x-axis = [all, male, female]; color = model
def plot_metrics_with_ci_groups(df):
    # Filter for only AUC, specificity, and sensitivity
    df_filtered = filter_metrics(df)
    
    # Expect rows repeating by [All, Male, Female] for each metric
    num_rows = len(df_filtered)
    if num_rows % 3 != 0:
        raise ValueError("Input table must have rows in multiples of 3: [All, Male, Female] per metric")

    models = df_filtered.columns[1:]
    num_models = len(models)
    group_names = ["all", "male", "female"]
    model_colors = sns.color_palette("Set2", n_colors=num_models)

    num_metrics = num_rows // 3

    sns.set_theme(style="whitegrid")

    fig, axs = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 6), sharey=True)
    if num_metrics == 1:
        axs = [axs]

    for metric_index in range(num_metrics):
        # Rows for this metric: All, Male, Female (in order)
        subset = df_filtered.iloc[metric_index * 3 : metric_index * 3 + 3]

        # Derive a clean metric title from the first column value
        raw_metric_label = str(subset.iloc[0, 0])
        metric_label = re.split(r"\s*[-–—]\s*", raw_metric_label)[0]

        # x-axis: groups (all, male, female)
        x = np.arange(len(group_names))
        width = 0.8 / max(num_models, 1)

        ax = axs[metric_index]

        # For each model, draw bars across groups, color by model
        for model_idx, model in enumerate(models):
            means, lowers, uppers = [], [], []
            for group_idx in range(len(group_names)):
                val = subset.iloc[group_idx][model]
                mean, lower, upper = parse_value(val)
                means.append(mean)
                lowers.append(mean - lower)
                uppers.append(upper - mean)

            ax.bar(
                x + model_idx * width - (num_models - 1) / 2.0 * width,
                means,
                width,
                color=model_colors[model_idx],
                yerr=[lowers, uppers],
                capsize=3,
                label=model,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(group_names, ha="center", fontsize=9)
        ax.set_ylabel("Metric Value")
        ax.set_title(metric_label)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Shared legend above plots (models)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 1.05),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    # Default file path
    default_file = "/home/sunx/data/aiiih/projects/sunx/projects/ICD_sex_diff/scmr_table.xlsx"
    
    # Check if a file path is provided as command line argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = default_file
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        if len(sys.argv) <= 1:
            print("Usage: python plot.py [path_to_excel_file]")
        sys.exit(1)
    
    try:
        # Load the table
        print(f"Loading data from: {file_path}")
        table = pd.read_excel(file_path)
        print(f"Data shape: {table.shape}")
        
        # Display metrics that will be plotted
        df_filtered = filter_metrics(table)
        print(f"Filtered to {len(df_filtered)} rows for AUC, Sensitivity, and Specificity metrics")
        
        # Create the plot
        plot_metrics_with_ci_groups(table)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
