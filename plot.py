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


# ---------- Styling helpers (for nicer aesthetics) ----------
def _apply_pretty_style(context: str = "talk", font_scale: float = 0.95) -> None:
    """Set a clean, publication-friendly seaborn/matplotlib style."""
    sns.set_theme(style="whitegrid", context=context)
    sns.set_context(context=context, font_scale=font_scale)
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.edgecolor": "0.5",
            "axes.linewidth": 1.0,
            "legend.frameon": False,
        }
    )


def _build_color_mapping(model_names):
    """Return (unique_base_models, base_to_color) with a colorblind-safe palette."""
    base_models = [_clean_base_model_name(m) for m in model_names]
    unique_base_models = list(dict.fromkeys(base_models))
    color_palette = sns.color_palette("colorblind", n_colors=len(unique_base_models))
    base_to_color = {bm: color_palette[i] for i, bm in enumerate(unique_base_models)}
    return base_models, unique_base_models, base_to_color


def _overall_y_bounds(df: pd.DataFrame) -> tuple[float, float]:
    """Compute overall [min(lower), max(upper)] across all value cells."""
    min_lower, max_upper = np.inf, -np.inf
    for _, row in df.iterrows():
        for col in df.columns[1:]:
            mean, lower, upper = parse_value(row[col])
            if np.isfinite(lower):
                min_lower = min(min_lower, lower)
            if np.isfinite(upper):
                max_upper = max(max_upper, upper)
    if not np.isfinite(min_lower) or not np.isfinite(max_upper):
        return 0.0, 1.0
    # Add a small visual headroom
    margin = max(0.02, (max_upper - min_lower) * 0.04)
    return min_lower - margin, max_upper + margin


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
    base_models, unique_base_models, base_to_color = _build_color_mapping(models)

    num_metrics = num_rows // 3

    _apply_pretty_style(context="talk", font_scale=0.95)

    fig, axs = plt.subplots(
        1,
        num_metrics,
        figsize=(5.6 * num_metrics + 1.5, 5.4),
        sharey=True,
    )
    if num_metrics == 1:
        axs = [axs]

    # Reserve room for legends on the right
    fig.subplots_adjust(right=0.80)

    # Consistent y-range across subplots
    y_min, y_max = _overall_y_bounds(df)

    for metric_index in range(num_metrics):
        # Rows for this metric: All, Male, Female (in order)
        subset = df.iloc[metric_index * 3 : metric_index * 3 + 3]

        # Derive a clean metric title from the first column value
        raw_metric_label = str(subset.iloc[0, 0])
        metric_label = re.split(r"\s*[-–—]\s*", raw_metric_label)[0]

        # x-axis: groups (all, male, female)
        x = np.arange(len(group_names))
        group_width = 0.78
        width = group_width / max(num_models, 1)

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
                error_kw={"ecolor": "0.3", "lw": 1.2, "capthick": 1.2, "capsize": 3},
                hatch=hatch_style,
                edgecolor="0.35",
                linewidth=0.8,
                alpha=0.95,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(group_names, ha="center", fontsize=13)

        ax.set_ylabel("Metric Value", fontsize=13)
        ax.set_title(metric_label, fontsize=18, weight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_ylim(y_min, y_max)

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
        bbox_to_anchor=(0.83, 1.0),
        title="Model",
        fontsize=12,
        title_fontsize=12,
        frameon=False,
    )
    fig.add_artist(leg1)

    fig.legend(
        handles=hatch_handles,
        loc="lower left",
        bbox_to_anchor=(0.83, 0.0),
        title="Type",
        fontsize=12,
        title_fontsize=12,
        frameon=False,
    )

    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_single_metric_rows_as_models(df, metric_title="Metric", save_path: str | None = None):
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

    _apply_pretty_style(context="talk", font_scale=1.0)
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.6))

    x = np.arange(num_groups)
    group_width = 0.78
    width = group_width / max(num_models, 1)
    # Determine base models and assign colors consistently per base model
    base_model_labels, unique_base_models, base_to_color = _build_color_mapping(
        model_labels
    )

    # Reserve room for legends on the right and set consistent y range
    fig.subplots_adjust(right=0.80)
    y_min, y_max = _overall_y_bounds(df)

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
            error_kw={"ecolor": "0.3", "lw": 1.2, "capthick": 1.2, "capsize": 3},
            hatch=hatch_style,
            edgecolor="0.35",
            linewidth=0.8,
            alpha=0.95,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in df.columns[1:]], ha="center", fontsize=14)

    ax.set_ylabel("Metric Value", fontsize=14)
    ax.set_title(metric_title, fontsize=20, weight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_ylim(y_min, y_max)

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
        bbox_to_anchor=(0.83, 1.0),
        title="Model",
        fontsize=15,
        title_fontsize=15,
        frameon=False,
    )
    fig.add_artist(leg1)

    fig.legend(
        handles=hatch_handles,
        loc="lower left",
        bbox_to_anchor=(0.83, 0.0),
        title="Type",
        fontsize=15,
        title_fontsize=15,
        frameon=False,
    )

    sns.despine()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Load the table
    table = pd.read_excel(
        "/home/sunx/data/aiiih/projects/sunx/projects/ICD_sex_diff/results_cox.xlsx"
    )
    # For DataFrames shaped like image1 (rows=models, columns=[all, male, female])
    # produce a single-subplot grouped bar chart.
    plot_single_metric_rows_as_models(table, metric_title="C-Index")
