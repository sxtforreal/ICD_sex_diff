import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
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


# Helpers to normalize model naming and styling
def _clean_base_model_name(model_label: str) -> str:
    """
    Remove qualifiers like "Sex-agnostic", "Sex-specific" and parenthetical notes
    like "(undersampled)" from a model label to get the base model name used for
    color mapping.
    """
    if model_label is None:
        return ""
    base = re.sub(r"\(.*?\)", "", str(model_label))  # drop parenthetical notes
    base = re.sub(r"\bsex[-\s]*agnostic\b", "", base, flags=re.IGNORECASE)
    base = re.sub(r"\bsex[-\s]*specific\b", "", base, flags=re.IGNORECASE)
    # Collapse multiple spaces and strip
    base = " ".join(base.split()).strip(" -–—")
    return base


def _is_sex_specific(model_label: str) -> bool:
    """Infer whether a model label denotes a sex-specific model."""
    if model_label is None:
        return False
    return bool(re.search(r"\bsex[-\s]*specific\b", str(model_label), flags=re.IGNORECASE))


# Function to create subplots per metric, x-axis = [all, male, female]; color = model
def plot_metrics_with_ci_groups(df):
    # Expect rows repeating by [All, Male, Female] for each metric
    num_rows = len(df)
    if num_rows % 3 != 0:
        raise ValueError(
            "Input table must have rows in multiples of 3: [All, Male, Female] per metric"
        )

    # Filter to keep only AUC, specificity, and sensitivity
    allowed_keywords = ["auc", "auroc", "specificity", "sensitivity"]
    keep_metric_indices = []
    total_metrics = num_rows // 3
    for metric_index in range(total_metrics):
        raw_metric_label = str(df.iloc[metric_index * 3, 0])
        metric_label = re.split(r"\s*[-–—]\s*", raw_metric_label)[0].lower()
        if any(keyword in metric_label for keyword in allowed_keywords):
            keep_metric_indices.append(metric_index)
    if not keep_metric_indices:
        raise ValueError(
            "No metrics matching AUC, specificity, or sensitivity were found."
        )
    df = pd.concat(
        [df.iloc[i * 3 : i * 3 + 3] for i in keep_metric_indices], ignore_index=True
    )
    num_rows = len(df)

    models = [str(c) for c in df.columns[1:]]
    num_models = len(models)
    group_names = ["ALL", "MALE", "FEMALE"]

    # Determine base models and assign consistent colors to each base model
    base_models = [_clean_base_model_name(m) for m in models]
    unique_base_models = list(dict.fromkeys(base_models))  # preserve order
    color_palette = sns.color_palette("Set2", n_colors=len(unique_base_models))
    base_to_color = {bm: color_palette[i] for i, bm in enumerate(unique_base_models)}

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

        # x-axis: groups (all, male, female)
        x = np.arange(len(group_names))
        width = 0.8 / max(num_models, 1)

        ax = axs[metric_index]

        # For each model, draw bars across groups
        for model_idx, model in enumerate(models):
            means, lowers, uppers = [], [], []
            for group_idx in range(len(group_names)):
                val = subset.iloc[group_idx][model]
                mean, lower, upper = parse_value(val)
                means.append(mean)
                lowers.append(mean - lower)
                uppers.append(upper - mean)

            base_name = base_models[model_idx]
            is_specific = _is_sex_specific(model)
            hatch_style = "//" if is_specific else None

            ax.bar(
                x + model_idx * width - (num_models - 1) / 2.0 * width,
                means,
                width,
                color=base_to_color.get(base_name, "C0"),
                yerr=[lowers, uppers],
                capsize=3,
                hatch=hatch_style,
                edgecolor="black",
                linewidth=0.6,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(group_names, ha="center", fontsize=12)

        ax.set_ylabel("Metric Value", fontsize=12)
        ax.set_title(metric_label, fontsize=14)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Dual legends: colors for models, hatch for type (sex-agnostic vs sex-specific)
    color_handles = [
        mpatches.Patch(facecolor=base_to_color[bm], edgecolor="black", label=bm)
        for bm in unique_base_models
    ]
    hatch_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", label="Sex-agnostic"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", label="Sex-specific"),
    ]

    # Place legends on the right side, stacked
    leg1 = fig.legend(
        handles=color_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        title="Model",
        fontsize=12,
        title_fontsize=12,
        frameon=False,
    )
    fig.add_artist(leg1)

    fig.legend(
        handles=hatch_handles,
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        title="Type",
        fontsize=12,
        title_fontsize=12,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.show()


def plot_single_metric_rows_as_models(df, metric_title="Metric"):
    """
    Plot a single metric from a table where:
    - Column 0 contains row names (treated as model labels)
    - Remaining columns are groups (e.g., ["all", "male", "female"]) and
      each cell is a string formatted as "mean (lower, upper)".
    """
    if df.shape[1] < 2:
        raise ValueError(
            "Expected at least two columns: row labels and one group column"
        )

    # Identify names
    model_labels = [str(v) for v in df.iloc[:, 0].tolist()]
    group_names = [str(c) for c in df.columns[1:]]

    num_models = len(model_labels)
    num_groups = len(group_names)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6))

    x = np.arange(num_groups)
    width = 0.8 / max(num_models, 1)
    # Determine base models and assign colors consistently per base model
    base_model_labels = [_clean_base_model_name(m) for m in model_labels]
    unique_base_models = list(dict.fromkeys(base_model_labels))
    color_palette = sns.color_palette("Set2", n_colors=len(unique_base_models))
    base_to_color = {bm: color_palette[i] for i, bm in enumerate(unique_base_models)}

    for model_index, model_label in enumerate(model_labels):
        row = df.iloc[model_index]

        means_for_groups = []
        lower_errors_for_groups = []
        upper_errors_for_groups = []

        for group_name in group_names:
            mean, lower, upper = parse_value(row[group_name])
            means_for_groups.append(mean)
            lower_errors_for_groups.append(mean - lower)
            upper_errors_for_groups.append(upper - mean)

        is_specific = _is_sex_specific(model_label)
        hatch_style = "//" if is_specific else None
        base_name = base_model_labels[model_index]

        ax.bar(
            x + model_index * width - (num_models - 1) / 2.0 * width,
            means_for_groups,
            width,
            color=base_to_color.get(base_name, "C0"),
            yerr=[lower_errors_for_groups, upper_errors_for_groups],
            capsize=3,
            hatch=hatch_style,
            edgecolor="black",
            linewidth=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in df.columns[1:]], ha="center", fontsize=15)

    ax.set_ylabel("Metric Value", fontsize=15)
    ax.set_title(metric_title, fontsize=18)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Dual legends: colors for base models; hatch indicates sex-specific vs agnostic
    color_handles = [
        mpatches.Patch(facecolor=base_to_color[bm], edgecolor="black", label=bm)
        for bm in unique_base_models
    ]
    hatch_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", label="Sex-agnostic"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", label="Sex-specific"),
    ]

    leg1 = fig.legend(
        handles=color_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        title="Model",
        fontsize=15,
        title_fontsize=15,
        frameon=False,
    )
    fig.add_artist(leg1)

    fig.legend(
        handles=hatch_handles,
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        title="Type",
        fontsize=15,
        title_fontsize=15,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.show()


if __name__ == "__main__":
    # Load the table
    table = pd.read_excel(
        "/home/sunx/data/aiiih/projects/sunx/projects/ICD_sex_diff/results_cox.xlsx"
    )
    # For DataFrames shaped like image1 (rows=models, columns=[all, male, female])
    # produce a single-subplot grouped bar chart.
    plot_single_metric_rows_as_models(table, metric_title="C-Index")
