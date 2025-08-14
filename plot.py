import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pandas as pd


# Function to parse strings of the form "mean (lower, upper)"
def parse_value(s):
    match = re.match(r"([\d\.nan]+)\s*\(\s*([\d\.nan]+),\s*([\d\.nan]+)\s*\)", str(s))
    if match:
        mean, lower, upper = map(float, match.groups())
        return mean, lower, upper
    else:
        return float("nan"), float("nan"), float("nan")


# Function to create subplots grouped by metric, each subplot shows All/Male/Female per model
def plot_metrics_with_ci_groups(df):
    # Expect rows repeating by [All, Male, Female] for each metric
    num_rows = len(df)
    if num_rows % 3 != 0:
        raise ValueError("Input table must have rows in multiples of 3: [All, Male, Female] per metric")

    models = df.columns[1:]
    num_models = len(models)
    group_names = ["All", "Male", "Female"]
    group_colors = sns.color_palette("Set2", n_colors=3)

    num_metrics = num_rows // 3

    sns.set_theme(style="whitegrid")

    fig, axs = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 6), sharey=True)
    if num_metrics == 1:
        axs = [axs]

    for metric_index in range(num_metrics):
        # Rows for this metric: All, Male, Female (in order)
        subset = df.iloc[metric_index * 3 : metric_index * 3 + 3]

        # Derive a clean metric title from the first column value
        raw_metric_label = str(subset.iloc[0, 0])
        metric_label = re.split(r"\s*[-–—]\s*", raw_metric_label)[0]

        x = np.arange(num_models)
        width = 0.8 / 3.0

        ax = axs[metric_index]

        for group_idx, group_name in enumerate(group_names):
            means, lowers, uppers = [], [], []
            for model in models:
                val = subset.iloc[group_idx][model]
                mean, lower, upper = parse_value(val)
                means.append(mean)
                lowers.append(mean - lower)
                uppers.append(upper - mean)

            ax.bar(
                x + group_idx * width - (len(group_names) - 1) / 2.0 * width,
                means,
                width,
                color=group_colors[group_idx],
                yerr=[lowers, uppers],
                capsize=3,
                label=group_name,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="center", fontsize=8)
        ax.set_ylabel("Metric Value")
        ax.set_title(metric_label)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Shared legend above plots (groups)
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
    # Load the table
    table = pd.read_excel(
        "/home/sunx/data/aiiih/projects/sunx/projects/ICD_sex_diff/scmr_table.xlsx"
    )
    plot_metrics_with_ci_groups(table)
