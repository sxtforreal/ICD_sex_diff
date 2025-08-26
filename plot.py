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


def _standardize_group_columns(columns):
    """Return group column order matching [all, male, female] if present.

    The incoming DataFrame may have group columns in any case; we will
    map them case-insensitively and return the exact column names from the
    DataFrame in the standard order. Missing columns are ignored.
    """
    canonical_order = ["all", "male", "female"]
    lower_to_actual = {str(c).lower(): c for c in columns}
    ordered = [lower_to_actual[c] for c in canonical_order if c in lower_to_actual]
    if not ordered:
        raise ValueError("Expected group columns like 'all', 'male', 'female'.")
    return ordered


# Create a single subplot for one metric from a table like image1:
# rows = "<Model> - Sex-agnostic/Specific", columns = groups [all, male, female]
def plot_single_metric_from_rows_table(
    df: pd.DataFrame,
    metric_title: str = "AUC",
    variant_preference: str = "Sex-specific",
):
    # Identify first column as row-name column and following columns as groups
    row_name_col = df.columns[0]
    group_cols = _standardize_group_columns(df.columns[1:])

    # Prefer variant rows per user setting; fallback if not present
    preferred_mask = df[row_name_col].str.contains(variant_preference, case=False, na=False)
    selected = df[preferred_mask]
    if selected.empty:
        alt = "Sex-agnostic" if variant_preference.lower() == "sex-specific" else "Sex-specific"
        selected = df[df[row_name_col].str.contains(alt, case=False, na=False)]
    if selected.empty:
        raise ValueError("Could not find rows containing 'Sex-specific' or 'Sex-agnostic'.")

    # Clean model names and optional mapping to nicer labels
    def extract_model(label: str) -> str:
        model = re.split(r"\s*[-–—]\s*", str(label))[0].strip()
        mapping = {"Guideline": "Guideline", "Benchmark": "Standard CMR", "Proposed": "Advanced CMR"}
        return mapping.get(model, model)

    selected = selected.copy()
    selected["__model_label__"] = selected[row_name_col].apply(extract_model)

    models = list(selected["__model_label__"].values)
    num_models = len(models)
    group_names = [str(c).lower() for c in group_cols]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    x = np.arange(len(group_cols))
    width = 0.8 / max(num_models, 1)
    colors = sns.color_palette("Set2", n_colors=num_models)

    # Bars: color by model, grouped by groups
    for model_idx, (_, row) in enumerate(selected.iterrows()):
        means, lowers, uppers = [], [], []
        for gc in group_cols:
            mean, lower, upper = parse_value(row[gc])
            means.append(mean)
            lowers.append(mean - lower)
            uppers.append(upper - mean)

        ax.bar(
            x + model_idx * width - (num_models - 1) / 2.0 * width,
            means,
            width,
            color=colors[model_idx],
            yerr=[lowers, uppers],
            capsize=3,
            label=models[model_idx],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(group_names, ha="center", fontsize=9)
    ax.set_ylabel("Metric Value")
    ax.set_title(metric_title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
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
    plot_single_metric_from_rows_table(table, metric_title="AUC", variant_preference="Sex-specific")
