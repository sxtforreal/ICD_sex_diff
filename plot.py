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


# Function to plot a single group (All, Male, or Female)
def plot_group(ax, df, group_name, colors):
    metrics = df.iloc[:, 0].values  # First column: metric names
    models = df.columns[1:]  # Remaining columns: model names

    x = np.arange(len(metrics))  # X-axis positions for metrics
    width = 0.8 / len(models)  # Bar width, total width ~0.8

    for i, model in enumerate(models):
        means, lowers, uppers = [], [], []

        # Parse mean and confidence interval for each metric
        for val in df[model]:
            mean, lower, upper = parse_value(val)
            means.append(mean)
            lowers.append(mean - lower)  # Lower error length
            uppers.append(upper - mean)  # Upper error length

        # Plot bars with error bars
        ax.bar(
            x + i * width - (len(models) - 1) / 2 * width,
            means,
            width,
            color=colors[i],
            yerr=[lowers, uppers],
            capsize=3,
            label=model,
        )

    # Remove anything after dash-like symbols (handles "-", "–", "—")
    clean_labels = [re.split(r"\s*[-–—]\s*", label)[0] for label in metrics]
    ax.set_xticks(x)
    ax.set_xticklabels(clean_labels, rotation=45, ha="center", fontsize=8)

    ax.set_ylabel("Metric Value")
    ax.set_title(f"{group_name}")
    ax.grid(axis="y", linestyle="--", alpha=0.3)


# Function to create the 3-panel grouped bar plot
def plot_metrics_with_ci_groups(df):
    # Split into All, Male, Female (assuming repeating row order)
    all_df = df.iloc[[i for i in range(len(df)) if i % 3 == 0]]
    male_df = df.iloc[[i for i in range(len(df)) if i % 3 == 1]]
    female_df = df.iloc[[i for i in range(len(df)) if i % 3 == 2]]

    # Use seaborn theme and palette
    sns.set_theme(style="whitegrid")
    colors = sns.color_palette("Set2", n_colors=len(df.columns) - 1)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    plot_group(axs[0], all_df, "All", colors)
    plot_group(axs[1], male_df, "Male", colors)
    plot_group(axs[2], female_df, "Female", colors)

    # Add shared legend above plots
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
