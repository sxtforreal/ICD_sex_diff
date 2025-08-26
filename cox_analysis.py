import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from scipy import stats
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# Data Loading (adapt paths as necessary)
# =============================================================================

# NOTE: This script assumes the same data preparation steps as `a.py`.
# If you have already executed `a.py` and have the following dataframes in memory,
# you can simply import them. Otherwise, copy the corresponding code blocks from
# `a.py` that create `df` (features) and `survival_df` (time-to-event data).

from a import df, survival_df  # reuse pre-processed dataframes

# Merge features with survival labels
MERGED = df.merge(
    survival_df[["MRN", "PE", "PE_Time", "Female"]], on="MRN", how="inner"
).dropna(subset=["PE", "PE_Time"])

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def create_undersampled_dataset(data: pd.DataFrame, random_state: int) -> pd.DataFrame:
    """Undersample majority sex to match minority sex size."""
    males = data[data["Female"] == 0]
    females = data[data["Female"] == 1]
    if len(males) > len(females):
        males_down = resample(
            males,
            replace=False,
            n_samples=len(females),
            random_state=random_state,
        )
        balanced = pd.concat([males_down, females])
    else:
        females_down = resample(
            females,
            replace=False,
            n_samples=len(males),
            random_state=random_state,
        )
        balanced = pd.concat([males, females_down])
    return balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)


def ci_ci(values: list, alpha: float = 0.95):
    """Return mean and (lower, upper) CI of a list."""
    mean = np.mean(values)
    sem = stats.sem(values)
    t = stats.t.ppf((1 + alpha) / 2, len(values) - 1)
    half = t * sem
    return mean, mean - half, mean + half


# -----------------------------------------------------------------------------
# Feature sets (identical to a.py)
# -----------------------------------------------------------------------------
FEATURE_SETS = {
    "guideline": ["NYHA Class", "LVEF"],
    "benchmark": [
        "Female",
        "Age by decade",
        "BMI",
        "AF",
        "Beta Blocker",
        "CrCl>45",
        "LVEF",
        "QTc",
        "NYHA>2",
        "CRT",
        "AAD",
        "Significant LGE",
    ],
    "proposed": [
        "Female",
        "Age by decade",
        "BMI",
        "AF",
        "Beta Blocker",
        "CrCl>45",
        "LVEF",
        "QTc",
        "NYHA>2",
        "CRT",
        "AAD",
        "Significant LGE",
        "DM",
        "HTN",
        "HLP",
        "LVEDVi",
        "LV Mass Index",
        "RVEDVi",
        "RVEF",
        "LA EF",
        "LAVi",
        "MRF (%)",
        "Sphericity Index",
        "Relative Wall Thickness",
        "MV Annular Diameter",
        "ACEi/ARB/ARNi",
        "Aldosterone Antagonist",
    ],
    "real_proposed": [
        "Female",
        "Age by decade",
        "BMI",
        "AF",
        "Beta Blocker",
        "CrCl>45",
        "LVEF",
        "QTc",
        "CRT",
        "AAD",
        "LGE Burden 5SD",
        "DM",
        "HTN",
        "HLP",
        "LVEDVi",
        "LV Mass Index",
        "RVEDVi",
        "RVEF",
        "LA EF",
        "LAVi",
        "MRF (%)",
        "Sphericity Index",
        "Relative Wall Thickness",
        "MV Annular Diameter",
        "ACEi/ARB/ARNi",
        "Aldosterone Antagonist",
        "NYHA Class",
    ],
}

MODEL_CONFIGS = [
    {"name": "Guideline", "features": "guideline", "type": "rule_based"},
    {"name": "Benchmark Sex-agnostic", "features": "benchmark", "type": "cox"},
    {"name": "Benchmark Sex-agnostic (undersampled)", "features": "benchmark", "type": "cox_us"},
    {"name": "Benchmark Male", "features": "benchmark", "type": "male"},
    {"name": "Benchmark Female", "features": "benchmark", "type": "female"},
    {"name": "Benchmark Sex-specific", "features": "benchmark", "type": "sex_specific"},
    {"name": "Proposed Sex-agnostic", "features": "proposed", "type": "cox"},
    {"name": "Proposed Sex-agnostic (undersampled)", "features": "proposed", "type": "cox_us"},
    {"name": "Proposed Male", "features": "proposed", "type": "male"},
    {"name": "Proposed Female", "features": "proposed", "type": "female"},
    {"name": "Proposed Sex-specific", "features": "proposed", "type": "sex_specific"},
    {"name": "Real Proposed Sex-agnostic", "features": "real_proposed", "type": "cox"},
    {"name": "Real Proposed Sex-agnostic (undersampled)", "features": "real_proposed", "type": "cox_us"},
    {"name": "Real Proposed Male", "features": "real_proposed", "type": "male"},
    {"name": "Real Proposed Female", "features": "real_proposed", "type": "female"},
    {"name": "Real Proposed Sex-specific", "features": "real_proposed", "type": "sex_specific"},
]

METRICS = ["c_index"]

# Results dict
RESULTS = {cfg["name"]: {m: [] for m in METRICS} for cfg in MODEL_CONFIGS}

# -----------------------------------------------------------------------------
# Main evaluation loop (50 random splits)
# -----------------------------------------------------------------------------
N_SPLITS = 50
LABEL_EVENT = "PE"
LABEL_TIME = "PE_Time"

for seed in range(N_SPLITS):
    print(f"=== Split {seed + 1}/{N_SPLITS} ===")
    train_df, val_df = train_test_split(
        MERGED, test_size=0.3, random_state=seed, stratify=MERGED[LABEL_EVENT]
    )

    # Pre-compute undersampled set once per split
    US_DF = create_undersampled_dataset(train_df, seed)

    for cfg in MODEL_CONFIGS:
        feats = FEATURE_SETS[cfg["features"]]
        cfg_type = cfg["type"]

        if cfg_type == "rule_based":
            # simple rule: NYHA>2 or LVEF<35% -> high risk (1) else 0
            val_score = (
                (val_df["NYHA>2"] == 1) | (val_df["LVEF"] < 35)
            ).astype(int)
            ci = concordance_index(
                val_df[LABEL_TIME], -val_score, val_df[LABEL_EVENT]
            )
            RESULTS[cfg["name"]]["c_index"].append(ci)
            continue

        # Select appropriate training data
        if cfg_type == "cox":
            tr = train_df
        elif cfg_type == "cox_us":
            tr = US_DF
        elif cfg_type == "male":
            tr = train_df[train_df["Female"] == 0]
        elif cfg_type == "female":
            tr = train_df[train_df["Female"] == 1]
        elif cfg_type == "sex_specific":
            # Train separate male and female models, evaluate combined
            cis = []
            for sex, grp in tr.groupby("Female"):
                if grp.empty:
                    continue
                model = CoxPHFitter()
                try:
                    model.fit(grp[[LABEL_TIME, LABEL_EVENT] + feats], duration_col=LABEL_TIME, event_col=LABEL_EVENT)
                    sub_val = val_df[val_df["Female"] == sex]
                    pred = model.predict_partial_hazard(sub_val[feats])
                    ci = concordance_index(
                        sub_val[LABEL_TIME], -pred, sub_val[LABEL_EVENT]
                    )
                    cis.append(ci)
                except Exception:
                    continue
            if cis:
                RESULTS[cfg["name"]]["c_index"].append(np.mean(cis))
            continue
        else:
            raise ValueError(cfg_type)

        # Fit overall Cox model
        model = CoxPHFitter()
        try:
            model.fit(
                tr[[LABEL_TIME, LABEL_EVENT] + feats],
                duration_col=LABEL_TIME,
                event_col=LABEL_EVENT,
            )
            # risk scores for validation
            risk = model.predict_partial_hazard(val_df[feats])
            ci = concordance_index(val_df[LABEL_TIME], -risk, val_df[LABEL_EVENT])
            RESULTS[cfg["name"]]["c_index"].append(ci)
        except Exception:
            # Fallback if model fails to converge
            RESULTS[cfg["name"]]["c_index"].append(np.nan)

# -----------------------------------------------------------------------------
# Aggregate and export
# -----------------------------------------------------------------------------
summary_rows = []
for model_name, metric_dict in RESULTS.items():
    vals = np.array(metric_dict["c_index"], dtype=float)
    vals = vals[~np.isnan(vals)]
    mean, lo, hi = ci_ci(vals)
    summary_rows.append({"Model": model_name, "Mean": mean, "CI Lower": lo, "CI Upper": hi})

df_summary = pd.DataFrame(summary_rows).set_index("Model")

# Write full results (each split) and summary to Excel
with pd.ExcelWriter("cox_results.xlsx", engine="xlsxwriter") as writer:
    # per-split sheet
    per_split = {
        m: pd.Series(v["c_index"], name=m) for m, v in RESULTS.items()
    }
    pd.DataFrame(per_split).to_excel(writer, sheet_name="per_split", index_label="Split")
    # summary sheet
    df_summary.to_excel(writer, sheet_name="summary")

print("Results saved to cox_results.xlsx")

# -----------------------------------------------------------------------------
# Simple feature importance plot example (sex-agnostic full data, first config)
# -----------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    example_cfg = MODEL_CONFIGS[1]  # first cox model
    feats = FEATURE_SETS[example_cfg["features"]]
    cph = CoxPHFitter()
    cph.fit(MERGED[[LABEL_TIME, LABEL_EVENT] + feats], LABEL_TIME, LABEL_EVENT)
    hr = np.exp(cph.params_)
    hr.sort_values(inplace=True)
    plt.figure(figsize=(6, 4))
    hr.plot(kind="barh")
    plt.axvline(1, color="k", ls="--")
    plt.title("Hazard Ratios – {}".format(example_cfg["name"]))
    plt.xlabel("Hazard Ratio")
    plt.tight_layout()
    plt.show()
except Exception:
    pass