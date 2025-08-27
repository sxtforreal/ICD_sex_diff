import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

# ==========================================
# Global store for Proposed selected features
# ==========================================
SELECTED_FEATURES_STORE: Dict[str, Any] = {
    "proposed_sex_agnostic": None,  # type: List[str] | None
    "proposed_sex_specific": {  # type: Dict[str, List[str]] | None
        "male": None,
        "female": None,
    },
}

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from lifelines.exceptions import ConvergenceError

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

try:
    from missingpy import MissForest

    _HAS_MISSFOREST = True
except Exception:
    _HAS_MISSFOREST = False

try:
    from tableone import TableOne

    _HAS_TABLEONE = True
except Exception:
    _HAS_TABLEONE = False


# ==========================================
# Utilities
# ==========================================


def create_undersampled_dataset(
    train_df: pd.DataFrame, label_col: str, random_state: int
) -> pd.DataFrame:
    sampled_parts = []
    for sex_val in (0, 1):
        grp = train_df[train_df["Female"] == sex_val]
        if grp.empty:
            continue
        pos = grp[grp[label_col] == 1]
        neg = grp[grp[label_col] == 0]
        P, N = len(pos), len(neg)
        if P == 0 or N == 0:
            sampled_parts.append(grp.copy())
            continue
        # 1:1 undersample within sex
        target_pos = min(P, N)
        target_neg = min(N, P)
        samp_pos = pos.sample(n=target_pos, replace=False, random_state=random_state)
        samp_neg = neg.sample(n=target_neg, replace=False, random_state=random_state)
        sampled_parts.append(pd.concat([samp_pos, samp_neg], axis=0))
    if len(sampled_parts) == 0:
        return train_df.copy()
    return (
        pd.concat(sampled_parts, axis=0)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )


def plot_cox_coefficients(
    model: CoxPHFitter,
    title: str,
    gray_features: List[str] = None,
    red_features: List[str] = None,
    reference_df: pd.DataFrame = None,
    effect_scale: str = "raw",  # one of {"raw", "per_sd", "per_iqr"}
) -> None:
    """Plot Cox coefficients with optional standardized scaling.

    effect_scale controls comparability between binary and continuous features:
      - "raw": plot raw log HR (model coefficients)
      - "per_sd": plot log HR per one standard deviation (binary kept as 1 vs 0)
      - "per_iqr": plot log HR per IQR (Q75-Q25) (binary kept as 1 vs 0)
    """
    base_coefs = model.params_.copy()

    # Compute scaling factors per feature if requested
    scales: Dict[str, float] = {}
    if effect_scale in ("per_sd", "per_iqr") and reference_df is not None:
        for col in base_coefs.index.tolist():
            if col not in reference_df.columns:
                scales[col] = 1.0
                continue
            series = pd.to_numeric(reference_df[col], errors="coerce")
            # Detect binary {0,1}
            vals = series.dropna().unique()
            try:
                is_binary = len(vals) <= 2 and set(np.unique(vals)).issubset({0, 1})
            except Exception:
                is_binary = False
            if is_binary:
                scales[col] = 1.0  # interpret as 1 vs 0
            else:
                if effect_scale == "per_sd":
                    s = float(np.nanstd(series.values, ddof=1))
                    scales[col] = 1.0 if not np.isfinite(s) or s == 0.0 else s
                else:  # per_iqr
                    try:
                        q75 = float(np.nanpercentile(series.values, 75))
                        q25 = float(np.nanpercentile(series.values, 25))
                        iqr = q75 - q25
                    except Exception:
                        iqr = np.nan
                    scales[col] = (
                        1.0 if not np.isfinite(iqr) or iqr == 0.0 else float(iqr)
                    )
    else:
        scales = {f: 1.0 for f in base_coefs.index.tolist()}

    # Apply scaling to coefficients (log HR per unit/SD/IQR)
    scaled_values = []
    order = []
    for f in base_coefs.index.tolist():
        order.append(f)
        scaled_values.append(float(base_coefs.loc[f]) * float(scales.get(f, 1.0)))
    coef_series = pd.Series(data=scaled_values, index=order).sort_values(
        ascending=False
    )
    feats = coef_series.index.tolist()

    # Build category sets from FEATURE_SETS
    try:
        guideline_set = set(FEATURE_SETS.get("Guideline", [])) | {"NYHA>2"}
        benchmark_set = set(FEATURE_SETS.get("Benchmark", []))
        proposed_set = set(FEATURE_SETS.get("Proposed", []))
    except Exception:
        guideline_set, benchmark_set, proposed_set = set(), set(), set()

    benchmark_only = benchmark_set - guideline_set
    proposed_only = proposed_set - (benchmark_set | guideline_set)

    # Map features to colors by category
    colors = []
    for f in feats:
        if f in guideline_set:
            colors.append("gray")
        elif f in benchmark_only:
            colors.append("green")
        elif f in proposed_only:
            colors.append("orange")
        else:
            colors.append("blue")

    # Choose ylabel according to effect_scale
    if effect_scale == "per_sd":
        ylabel = "Cox coefficient (log HR per SD)"
    elif effect_scale == "per_iqr":
        ylabel = "Cox coefficient (log HR per IQR)"
    else:
        ylabel = "Cox coefficient (log HR)"

    plt.figure(figsize=(9, 4))
    plt.bar(range(len(feats)), coef_series.values, color=colors)
    plt.xticks(range(len(feats)), feats, rotation=45, ha="right", fontsize=12)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_km_two_subplots_by_gender(merged_df: pd.DataFrame) -> None:
    """Plot KM curves in two subplots (Male left, Female right) using Primary Endpoint (PE).
    Each subplot contains two curves: Low risk (pred0) and High risk (pred1), with within-sex log-rank test.
    """
    if merged_df.empty:
        return
    if not {"PE_Time", "PE"}.issubset(merged_df.columns):
        return

    ep_time_col, ep_event_col = "PE_Time", "PE"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax_idx, (sex_val, sex_name) in enumerate([(0, "Male"), (1, "Female")]):
        ax = axes[ax_idx]
        subset = merged_df[merged_df["Female"] == sex_val]
        if subset.empty:
            ax.set_title(f"{sex_name.upper()} (no data)")
            ax.axis("off")
            continue

        kmf = KaplanMeierFitter()
        curves = [
            (0, "Low risk", "blue"),
            (1, "High risk", "red"),
        ]
        for pred_val, label_base, color in curves:
            grp = subset[subset["pred_label"] == pred_val]
            if grp.empty:
                continue
            n_samples = int(len(grp))
            events = int(grp[ep_event_col].sum())
            label = f"{label_base} (n={n_samples}, events={events})"
            kmf.fit(
                durations=grp[ep_time_col],
                event_observed=grp[ep_event_col],
                label=label,
            )
            kmf.plot(ax=ax, color=color)

        # Log-rank test between low/high risk within sex
        low = subset[subset["pred_label"] == 0]
        high = subset[subset["pred_label"] == 1]
        if not low.empty and not high.empty:
            lr = logrank_test(
                low[ep_time_col],
                high[ep_time_col],
                low[ep_event_col],
                high[ep_event_col],
            )
            ax.text(
                0.95,
                0.05,
                f"Log-rank p = {lr.p_value:.4f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=12,
            )

        ax.set_title(sex_name.upper())
        ax.set_xlabel("Time (days)")
        if ax_idx == 0:
            ax.set_ylabel("Survival Probability")
        ax.grid(alpha=0.3)
        # Legend with larger font size per subplot (only if labels exist)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="best", fontsize=12)

    plt.suptitle("Primary Endpoint - Survival by Gender and Risk Group", y=1.02)
    plt.tight_layout()
    plt.show()


# ==========================================
# TableOne generation (Sex x ICD -> 4 groups)
# ==========================================


def generate_tableone_by_sex_icd(
    df: pd.DataFrame, output_excel_path: str = None
) -> None:
    """Generate a TableOne after loading df, splitting into 4 groups by sex and ICD.

    Groups: Male-ICD, Male-No ICD, Female-ICD, Female-No ICD.
    Compares all available features and labels (exclude MRN and the group column itself).
    """
    if df is None or df.empty:
        return
    if not {"Female", "ICD"}.issubset(df.columns):
        return

    df_local = df.copy()

    # Ensure numeric coding for conditions
    female_num = pd.to_numeric(df_local["Female"], errors="coerce")
    icd_num = pd.to_numeric(df_local["ICD"], errors="coerce")

    conditions = [
        (female_num == 0) & (icd_num == 1),
        (female_num == 0) & (icd_num == 0),
        (female_num == 1) & (icd_num == 1),
        (female_num == 1) & (icd_num == 0),
    ]
    choices = ["Male-ICD", "Male-No ICD", "Female-ICD", "Female-No ICD"]
    df_local["Group"] = np.select(conditions, choices, default=np.nan)

    group_order = ["Male-ICD", "Male-No ICD", "Female-ICD", "Female-No ICD"]
    # Ensure ordered categorical for predictable column order in TableOne
    try:
        df_local["Group"] = pd.Categorical(
            df_local["Group"], categories=group_order, ordered=True
        )
    except Exception:
        pass

    # Use all columns except identifiers and helper columns
    exclude_cols = {"MRN", "Group"}
    variables = [c for c in df_local.columns if c not in exclude_cols]

    # Known categorical candidates; keep only those present
    known_cats = [
        "Female",
        "DM",
        "HTN",
        "HLP",
        "AF",
        "NYHA Class",
        "NYHA>2",
        "Beta Blocker",
        "ACEi/ARB/ARNi",
        "Aldosterone Antagonist",
        "VT/VF/SCD",
        "AAD",
        "CRT",
        "ICD",
        "CrCl>45",
        "Significant LGE",
    ]
    categorical_cols = [c for c in known_cats if c in variables]

    # Prefer TableOne if available
    if _HAS_TABLEONE:
        try:
            tab1 = TableOne(
                df_local,
                columns=variables,
                categorical=categorical_cols,
                groupby="Group",
                groupby_order=group_order,
                pval=True,
                overall=True,
                missing=True,
                label_suffix=True,
            )
            print("==== TableOne (Sex x ICD) ====")
            print(tab1)

            table_df = getattr(tab1, "tableone", None)
            if table_df is None:
                try:
                    table_df = tab1.to_dataframe()  # type: ignore[attr-defined]
                except Exception:
                    table_df = None
            # Reorder columns to: Missing | Overall | [groups...] | P-Value (if present)
            if table_df is not None and isinstance(table_df.columns, pd.MultiIndex):
                cols_level0 = list(table_df.columns.get_level_values(0))
                cols_unique = []
                for c in cols_level0:
                    if c not in cols_unique:
                        cols_unique.append(c)
                ordered_first = [c for c in ["Missing", "Overall"] if c in cols_unique]
                ordered_groups = [c for c in group_order if c in cols_unique]
                ordered_tail = [
                    c for c in ["p-value", "P-Value", "pval"] if c in cols_unique
                ]
                desired_order = ordered_first + ordered_groups + ordered_tail
                # Only reorder if we computed a non-empty desired order
                if desired_order:
                    # Build new column MultiIndex order
                    new_cols = []
                    for top in desired_order:
                        matches = [c for c in table_df.columns if c[0] == top]
                        new_cols.extend(matches)
                    # Append any remaining columns in their existing order
                    remaining = [c for c in table_df.columns if c not in new_cols]
                    new_cols.extend(remaining)
                    table_df = table_df[new_cols]

            if output_excel_path and table_df is not None:
                os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)
                # Write a more Excel-friendly sheet
                try:
                    table_df.to_excel(output_excel_path)
                except Exception:
                    # If writing MultiIndex fails due to engine quirks, flatten columns
                    if isinstance(table_df.columns, pd.MultiIndex):
                        flat_cols = [
                            (str(a) if a is not None else "")
                            + (f" | {b}" if b not in (None, "") else "")
                            for a, b in table_df.columns
                        ]
                        tmp_df = table_df.copy()
                        tmp_df.columns = flat_cols
                        tmp_df.to_excel(output_excel_path)
                    else:
                        table_df.to_excel(output_excel_path)
                print(f"Saved TableOne to: {output_excel_path}")
            return
        except Exception as e:
            warnings.warn(f"[TableOne] Failed to generate TableOne: {e}")

    # Fallback summary if TableOne is not available
    try:
        # Identify numeric vs categorical
        numeric_cols = [
            c
            for c in variables
            if pd.api.types.is_numeric_dtype(df_local[c]) and c not in categorical_cols
        ]
        cat_cols = [c for c in categorical_cols if c in variables]

        # Build a wide table with groups as columns to mirror TableOne orientation
        # Columns order: Missing | Overall | Male-ICD | Male-No ICD | Female-ICD | Female-No ICD
        col_order = ["Missing", "Overall"] + group_order

        # Collect rows as a dict keyed by a two-level index: (variable, sublabel)
        summary_rows: Dict[Tuple[str, str], Dict[str, object]] = {}

        # Numeric variables: mean (SD)
        for var in numeric_cols:
            row: Dict[str, object] = {}
            # Missing count overall
            row["Missing"] = int(
                pd.to_numeric(df_local[var], errors="coerce").isna().sum()
            )
            # Overall summary as "mean (sd)"
            overall_series = pd.to_numeric(df_local[var], errors="coerce")
            overall_mean = overall_series.mean()
            overall_std = overall_series.std()
            row["Overall"] = (
                ""
                if pd.isna(overall_mean)
                else f"{overall_mean:.1f} ({overall_std:.1f})"
            )
            # Per-group summaries
            for grp_name in group_order:
                grp_series = pd.to_numeric(
                    df_local.loc[df_local["Group"] == grp_name, var], errors="coerce"
                )
                grp_series = grp_series.dropna()
                if grp_series.empty:
                    row[grp_name] = ""
                else:
                    m = grp_series.mean()
                    s = grp_series.std()
                    row[grp_name] = f"{m:.1f} ({s:.1f})"

            summary_rows[(var, "mean (SD)")] = row

        # Categorical variables: n (%) per level
        for var in cat_cols:
            series = df_local[var]
            # Determine level order: prefer categorical order if available, else sorted unique
            try:
                levels = [
                    lvl for lvl in pd.Categorical(series).categories if pd.notna(lvl)
                ]
            except Exception:
                levels = [lvl for lvl in series.dropna().unique().tolist()]
                try:
                    levels = sorted(levels)
                except Exception:
                    pass

            if len(levels) == 0:
                continue

            overall_non_missing = series.notna().sum()
            overall_missing = int(series.isna().sum())

            for level_index, level_value in enumerate(levels):
                level_label = str(level_value)
                row: Dict[str, object] = {}
                # Missing shown only on the first level row for the variable
                row["Missing"] = overall_missing if level_index == 0 else ""

                # Overall n (%)
                if overall_non_missing > 0:
                    overall_count = int((series == level_value).sum())
                    overall_pct = overall_count / overall_non_missing * 100.0
                    row["Overall"] = f"{overall_count} ({overall_pct:.1f}%)"
                else:
                    row["Overall"] = ""

                # Group n (%)
                for grp_name in group_order:
                    grp_series = df_local.loc[df_local["Group"] == grp_name, var]
                    non_missing = grp_series.notna().sum()
                    if non_missing == 0:
                        row[grp_name] = ""
                    else:
                        count = int((grp_series == level_value).sum())
                        pct = count / non_missing * 100.0
                        row[grp_name] = f"{count} ({pct:.1f}%)"

                summary_rows[(f"{var}, n (%)", level_label)] = row

        if not summary_rows:
            return

        fallback_df = pd.DataFrame.from_dict(summary_rows, orient="index")
        # Ensure consistent column order
        fallback_df = fallback_df.reindex(columns=col_order)

        print("==== Summary by group (fallback) ====")
        print(fallback_df.head())

        if output_excel_path:
            os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)
            fallback_df.to_excel(output_excel_path)
            print(f"Saved fallback summary to: {output_excel_path}")
    except Exception as e:
        warnings.warn(f"[TableOne Fallback] Failed to generate summary: {e}")


# Drop constant/duplicate/highly correlated columns to mitigate singular matrices
def _sanitize_cox_features_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
    corr_threshold: float = 0.995,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    original_features = list(feature_cols)
    X = df[feature_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Drop columns with all missing or only one unique non-nan value
    nunique = X.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols and verbose:
        print(f"[Cox] 删除常量/无信息列: {constant_cols}")
    X = X.drop(columns=constant_cols, errors="ignore")

    if X.shape[1] == 0:
        return X, []

    # Remove exactly duplicated columns
    X_filled = X.fillna(0.0)
    duplicated_mask = X_filled.T.duplicated(keep="first")
    if duplicated_mask.any():
        dup_cols = X.columns[duplicated_mask.values].tolist()
        if verbose:
            print(f"[Cox] 删除重复列: {dup_cols}")
        X = X.loc[:, ~duplicated_mask.values]

    if X.shape[1] <= 1:
        kept = list(X.columns)
        if verbose:
            removed = [c for c in original_features if c not in kept]
            if removed:
                print(f"[Cox] 净化后仅保留 {kept}，删除: {removed}")
        return X, kept

    # Remove highly correlated columns (keep the first in order)
    corr = X.fillna(0.0).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if (upper[col] >= corr_threshold).any()]
    if to_drop and verbose:
        print(f"[Cox] 删除高相关列(|r|>={corr_threshold}): {to_drop}")
    X = X.drop(columns=to_drop, errors="ignore")

    kept = list(X.columns)
    if verbose:
        removed = [c for c in original_features if c not in kept]
        if removed:
            print(f"[Cox] 特征保留 {len(kept)}/{len(original_features)}: {kept}")
            print(f"[Cox] 总计删除: {removed}")
    return X, kept


# ==========================================
# CoxPH training/inference blocks
# ==========================================


def fit_cox_model(
    train_df: pd.DataFrame, feature_cols: List[str], time_col: str, event_col: str
) -> CoxPHFitter:
    X_sanitized, kept_features = _sanitize_cox_features_matrix(
        train_df, feature_cols, corr_threshold=0.995
    )
    if len(kept_features) == 0:
        candidates = [c for c in feature_cols if train_df[c].nunique(dropna=False) > 1]
        if not candidates:
            raise ValueError("No usable features for CoxPH model after sanitization.")
        X_sanitized = train_df[[candidates[0]]].copy()
        kept_features = [candidates[0]]

    df_fit = pd.concat(
        [
            train_df[[time_col, event_col]].reset_index(drop=True),
            X_sanitized.reset_index(drop=True),
        ],
        axis=1,
    )

    cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            cph.fit(df_fit, duration_col=time_col, event_col=event_col, robust=True)
        except ConvergenceError:
            # Fallback: stronger regularization and stricter collinearity removal
            X_sanitized2, _ = _sanitize_cox_features_matrix(
                train_df, kept_features, corr_threshold=0.95
            )
            df_fit2 = pd.concat(
                [
                    train_df[[time_col, event_col]].reset_index(drop=True),
                    X_sanitized2.reset_index(drop=True),
                ],
                axis=1,
            )
            cph = CoxPHFitter(penalizer=1.0, l1_ratio=0.0)
            cph.fit(df_fit2, duration_col=time_col, event_col=event_col, robust=True)
    return cph


def predict_risk(
    model: CoxPHFitter, df: pd.DataFrame, feature_cols: List[str]
) -> np.ndarray:
    # Use the model's trained features to avoid mismatch/singularity issues
    model_features = list(model.params_.index)
    X = df.copy()
    missing = [c for c in model_features if c not in X.columns]
    for c in missing:
        X[c] = 0.0
    X = X[model_features]
    risk = model.predict_partial_hazard(X).values.reshape(-1)
    return risk


def threshold_by_top_quantile(risk_scores: np.ndarray, quantile: float = 0.5) -> float:
    q = np.nanquantile(risk_scores, quantile)
    return float(q)


def select_features_max_cindex_forward(
    train_df: pd.DataFrame,
    candidate_features: List[str],
    time_col: str,
    event_col: str,
    random_state: int = 0,
    max_features: int = None,
    verbose: bool = False,
) -> List[str]:
    """Greedy forward selection to maximize validation C-index.

    - Uses an inner 70/30 split of train_df for selection.
    - Applies the same sanitization as model fitting to avoid degenerate cols.
    - Returns a subset (possibly empty -> will be handled by callers).
    """
    df_local = train_df.dropna(subset=[time_col, event_col]).copy()
    if df_local.empty:
        return []

    # Initial candidate pool after basic sanitization
    try:
        X_sanitized, kept = _sanitize_cox_features_matrix(
            df_local, candidate_features, corr_threshold=0.995, verbose=False
        )
        pool: List[str] = list(kept)
    except Exception:
        pool = [f for f in candidate_features if f in df_local.columns]

    if len(pool) <= 1:
        return list(pool)

    try:
        inner_tr, inner_val = train_test_split(
            df_local,
            test_size=0.3,
            random_state=random_state,
            stratify=df_local[event_col] if df_local[event_col].nunique() > 1 else None,
        )
    except Exception:
        inner_tr, inner_val = train_test_split(
            df_local, test_size=0.3, random_state=random_state
        )

    selected: List[str] = []
    best_cidx: float = -np.inf
    remaining = list(pool)

    # Limit the number of selected features if requested
    max_iters = (
        len(remaining)
        if max_features is None
        else max(0, min(len(remaining), max_features))
    )

    for step_idx in range(max_iters):
        best_feat = None
        best_feat_cidx = best_cidx
        if verbose:
            try:
                print(
                    f"[FS][Forward] seed={random_state} step={step_idx+1}/{max_iters} remaining={len(remaining)}"
                )
            except Exception:
                pass
        for feat in remaining:
            trial_feats = selected + [feat]
            try:
                cph = fit_cox_model(inner_tr, trial_feats, time_col, event_col)
                risk_val = predict_risk(cph, inner_val, trial_feats)
                cidx = concordance_index(
                    inner_val[time_col], -risk_val, inner_val[event_col]
                )
            except Exception:
                cidx = np.nan
            if np.isfinite(cidx) and (cidx > best_feat_cidx + 1e-10):
                best_feat_cidx = float(cidx)
                best_feat = feat

        if best_feat is None:
            break
        selected.append(best_feat)
        remaining.remove(best_feat)
        best_cidx = best_feat_cidx
        if verbose:
            try:
                print(f"[FS][Forward] + {best_feat} -> val c-index={best_cidx:.4f}")
            except Exception:
                pass

    return selected


def stability_select_features(
    df: pd.DataFrame,
    candidate_features: List[str],
    time_col: str,
    event_col: str,
    seeds: List[int],
    max_features: int = None,
    threshold: float = 0.5,
    min_features: int = None,
    verbose: bool = False,
) -> List[str]:
    """Run forward selection across multiple seeds and keep features that
    are repeatedly selected (frequency >= threshold).

    This stabilizes the feature set so downstream multi-split averages are
    computed for a consistent model specification.
    """
    if df is None or df.empty:
        return []

    # Initial sanitized pool
    try:
        _, pool = _sanitize_cox_features_matrix(
            df, candidate_features, corr_threshold=0.995, verbose=False
        )
    except Exception:
        pool = [f for f in candidate_features if f in df.columns]

    if len(pool) == 0:
        return []

    from collections import Counter

    counter: Counter = Counter()
    total_runs = 0
    total_seeds = len(seeds)
    for idx, s in enumerate(seeds, start=1):
        try:
            if verbose:
                try:
                    print(f"[FS][Stability] seed {idx}/{total_seeds} -> start")
                except Exception:
                    pass
            sel = select_features_max_cindex_forward(
                train_df=df,
                candidate_features=list(pool),
                time_col=time_col,
                event_col=event_col,
                random_state=s,
                max_features=max_features,
                verbose=verbose,
            )
            if sel:
                counter.update(sel)
            total_runs += 1
        except Exception:
            continue

    if total_runs == 0 or not counter:
        return list(pool)

    # Keep features meeting frequency threshold
    ranked = list(counter.most_common())
    kept: List[str] = []
    for feat, count in ranked:
        freq = count / total_runs
        if freq >= threshold:
            kept.append(feat)

    # No minimum backfilling when min_features is None (allow any size)

    if verbose:
        try:
            print(
                f"[FS][Stability] kept {len(kept)} features (threshold={threshold}) out of pool {len(pool)}"
            )
        except Exception:
            pass

    # Final sanitization to avoid collinearity
    if kept:
        X_final, kept_final = _sanitize_cox_features_matrix(
            df, kept, corr_threshold=0.995, verbose=False
        )
        return list(kept_final)
    return []


def sex_specific_inference(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    gray_features: List[str] = None,
    red_features: List[str] = None,
) -> pd.DataFrame:
    """Sex-specific CoxPH inference without external survival_df.
    Assumes survival labels/time are in the input dataframes (uses PE_Time and VT/VF/SCD).
    Female is excluded from features for single-sex models.
    """
    used_features = [f for f in features if f != "Female"]

    train_m = (
        train_df[train_df["Female"] == 0].dropna(subset=["PE_Time", "VT/VF/SCD"]).copy()
    )
    train_f = (
        train_df[train_df["Female"] == 1].dropna(subset=["PE_Time", "VT/VF/SCD"]).copy()
    )
    test_m = test_df[test_df["Female"] == 0].copy()
    test_f = test_df[test_df["Female"] == 1].copy()

    models = {}
    thresholds = {}

    if not train_m.empty:
        cph_m = fit_cox_model(train_m, used_features, "PE_Time", "VT/VF/SCD")
        tr_risk_m = predict_risk(cph_m, train_m, used_features)
        thresholds["male"] = float(np.nanmedian(tr_risk_m))
        models["male"] = cph_m
        plot_cox_coefficients(
            cph_m,
            "Male Cox Coefficients (log HR)",
            gray_features,
            red_features,
            reference_df=train_m,
            effect_scale="per_sd",
        )
    if not train_f.empty:
        cph_f = fit_cox_model(train_f, used_features, "PE_Time", "VT/VF/SCD")
        tr_risk_f = predict_risk(cph_f, train_f, used_features)
        thresholds["female"] = float(np.nanmedian(tr_risk_f))
        models["female"] = cph_f
        plot_cox_coefficients(
            cph_f,
            "Female Cox Coefficients (log HR)",
            gray_features,
            red_features,
            reference_df=train_f,
            effect_scale="per_sd",
        )

    df_out = test_df.copy()
    if "male" in models and not test_m.empty:
        risk_m = predict_risk(models["male"], test_m, used_features)
        df_out.loc[df_out["Female"] == 0, "pred_prob"] = risk_m
        df_out.loc[df_out["Female"] == 0, "pred_label"] = (
            risk_m >= thresholds["male"]
        ).astype(int)
    if "female" in models and not test_f.empty:
        risk_f = predict_risk(models["female"], test_f, used_features)
        df_out.loc[df_out["Female"] == 1, "pred_prob"] = risk_f
        df_out.loc[df_out["Female"] == 1, "pred_label"] = (
            risk_f >= thresholds["female"]
        ).astype(int)

    merged_df = (
        df_out[["MRN", "Female", "pred_label", "PE_Time", "VT/VF/SCD"]]
        .dropna(subset=["PE_Time", "VT/VF/SCD"])
        .drop_duplicates(subset=["MRN"])
        .rename(columns={"VT/VF/SCD": "PE"})
    )
    plot_km_two_subplots_by_gender(merged_df)
    return merged_df


def sex_specific_full_inference(
    df: pd.DataFrame,
    features: List[str],
    gray_features: List[str] = None,
    red_features: List[str] = None,
) -> pd.DataFrame:
    """Train sex-specific CoxPH models on all data (per sex) and analyze on all data.

    - Uses PE_Time and VT/VF/SCD
    - Excludes Female from submodel features
    - Dichotomizes by sex-specific median risks
    """
    used_features = [f for f in features if f != "Female"]

    data_m = df[df["Female"] == 0].dropna(subset=["PE_Time", "VT/VF/SCD"]).copy()
    data_f = df[df["Female"] == 1].dropna(subset=["PE_Time", "VT/VF/SCD"]).copy()

    models = {}
    thresholds = {}

    if not data_m.empty:
        used_features_m = list(used_features)
        # Feature selection only if this is the Proposed set
        try:
            if set(features) == set(FEATURE_SETS.get("Proposed", [])):
                selected_m = SELECTED_FEATURES_STORE.get(
                    "proposed_sex_specific", {}
                ).get("male")
                if not selected_m:
                    selected_m = select_features_max_cindex_forward(
                        data_m,
                        list(used_features),
                        "PE_Time",
                        "VT/VF/SCD",
                        random_state=42,
                        verbose=True,
                    )
                    if selected_m:
                        try:
                            SELECTED_FEATURES_STORE["proposed_sex_specific"]["male"] = (
                                list(selected_m)
                            )
                            print(
                                f"[FS][Store] Proposed male-specific features stored: {len(selected_m)}"
                            )
                        except Exception:
                            pass
                if selected_m:
                    used_features_m = list(selected_m)
        except Exception:
            pass
        cph_m = fit_cox_model(data_m, used_features_m, "PE_Time", "VT/VF/SCD")
        r_m = predict_risk(cph_m, data_m, used_features_m)
        thresholds["male"] = float(np.nanmedian(r_m))
        models["male"] = cph_m
        plot_cox_coefficients(
            cph_m,
            "Male Cox Coefficients (log HR)",
            gray_features,
            red_features,
            reference_df=data_m,
            effect_scale="per_sd",
        )
    if not data_f.empty:
        used_features_f = list(used_features)
        # Feature selection only if this is the Proposed set
        try:
            if set(features) == set(FEATURE_SETS.get("Proposed", [])):
                selected_f = SELECTED_FEATURES_STORE.get(
                    "proposed_sex_specific", {}
                ).get("female")
                if not selected_f:
                    selected_f = select_features_max_cindex_forward(
                        data_f,
                        list(used_features),
                        "PE_Time",
                        "VT/VF/SCD",
                        random_state=42,
                        verbose=True,
                    )
                    if selected_f:
                        try:
                            SELECTED_FEATURES_STORE["proposed_sex_specific"][
                                "female"
                            ] = list(selected_f)
                            print(
                                f"[FS][Store] Proposed female-specific features stored: {len(selected_f)}"
                            )
                        except Exception:
                            pass
                if selected_f:
                    used_features_f = list(selected_f)
        except Exception:
            pass
        cph_f = fit_cox_model(data_f, used_features_f, "PE_Time", "VT/VF/SCD")
        r_f = predict_risk(cph_f, data_f, used_features_f)
        thresholds["female"] = float(np.nanmedian(r_f))
        models["female"] = cph_f
        plot_cox_coefficients(
            cph_f,
            "Female Cox Coefficients (log HR)",
            gray_features,
            red_features,
            reference_df=data_f,
            effect_scale="per_sd",
        )

    out = df.copy()
    if "male" in models and not out[out["Female"] == 0].empty:
        te_m = out[out["Female"] == 0]
        # If selection happened, the model params capture the selected set, so we just pass any list
        risk_m = predict_risk(models["male"], te_m, list(models["male"].params_.index))
        out.loc[out["Female"] == 0, "pred_prob"] = risk_m
        out.loc[out["Female"] == 0, "pred_label"] = (
            risk_m >= thresholds["male"]
        ).astype(int)
    if "female" in models and not out[out["Female"] == 1].empty:
        te_f = out[out["Female"] == 1]
        risk_f = predict_risk(
            models["female"], te_f, list(models["female"].params_.index)
        )
        out.loc[out["Female"] == 1, "pred_prob"] = risk_f
        out.loc[out["Female"] == 1, "pred_label"] = (
            risk_f >= thresholds["female"]
        ).astype(int)

    merged_df = (
        out[["MRN", "Female", "pred_label", "PE_Time", "VT/VF/SCD"]]
        .dropna(subset=["PE_Time", "VT/VF/SCD"])
        .drop_duplicates(subset=["MRN"])
        .rename(columns={"VT/VF/SCD": "PE"})
    )
    plot_km_two_subplots_by_gender(merged_df)
    return merged_df


def sex_agnostic_inference(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    label_col: str = "VT/VF/SCD",
    use_undersampling: bool = True,
    gray_features: List[str] = None,
    red_features: List[str] = None,
) -> pd.DataFrame:
    """Sex-agnostic CoxPH inference without external survival_df.
    Assumes survival labels/time are in the input dataframes (uses PE_Time and label_col).
    Female is INCLUDED in features for agnostic model.
    """
    used_features = features
    tr_base = (
        create_undersampled_dataset(train_df, label_col, 42)
        if use_undersampling
        else train_df
    )
    tr = tr_base.dropna(subset=["PE_Time", label_col]).copy()
    if tr.empty:
        return test_df.copy()

    cph = fit_cox_model(tr, used_features, "PE_Time", label_col)
    plot_cox_coefficients(
        cph,
        "Sex-Agnostic Cox Coefficients (log HR)",
        gray_features,
        red_features,
        reference_df=tr,
        effect_scale="per_sd",
    )
    tr_risk = predict_risk(cph, tr, used_features)
    thr = float(np.nanmedian(tr_risk))

    te = test_df.copy()
    te_risk = predict_risk(cph, te, used_features)
    te["pred_prob"] = te_risk
    te["pred_label"] = (te_risk >= thr).astype(int)

    merged_df = (
        te[["MRN", "Female", "pred_label", "PE_Time", label_col]]
        .dropna(subset=["PE_Time", label_col])
        .drop_duplicates(subset=["MRN"])
        .rename(columns={label_col: "PE"})
    )
    plot_km_two_subplots_by_gender(merged_df)
    return merged_df


def sex_agnostic_full_inference(
    df: pd.DataFrame,
    features: List[str],
    label_col: str = "VT/VF/SCD",
    use_undersampling: bool = True,
    gray_features: List[str] = None,
    red_features: List[str] = None,
) -> pd.DataFrame:
    """Train a sex-agnostic CoxPH model on all data and analyze on all data.

    - Uses PE_Time and label_col in the provided dataframe
    - Includes Female in features
    - Dichotomizes risk by overall median
    """
    df_base = (
        create_undersampled_dataset(df, label_col, 42) if use_undersampling else df
    )
    used_features = features
    # If features correspond to Proposed, perform feature selection to maximize c-index
    try:
        if set(features) == set(FEATURE_SETS.get("Proposed", [])):
            # Reuse from store if available; otherwise select once and store
            selected = SELECTED_FEATURES_STORE.get("proposed_sex_agnostic")
            if not selected:
                selected = select_features_max_cindex_forward(
                    df_base,
                    list(features),
                    "PE_Time",
                    label_col,
                    random_state=42,
                    verbose=True,
                )
                if selected:
                    SELECTED_FEATURES_STORE["proposed_sex_agnostic"] = list(selected)
                    print(
                        f"[FS][Store] Proposed sex-agnostic features stored: {len(selected)}"
                    )
            if selected:
                used_features = list(selected)
    except Exception:
        pass
    data = df_base.dropna(subset=["PE_Time", label_col]).copy()
    if data.empty:
        return pd.DataFrame()

    cph = fit_cox_model(data, used_features, "PE_Time", label_col)
    plot_cox_coefficients(
        cph,
        "Sex-Agnostic Cox Coefficients (log HR)",
        gray_features,
        red_features,
        reference_df=data,
        effect_scale="per_sd",
    )
    risk_all = predict_risk(cph, data, used_features)
    thr = float(np.nanmedian(risk_all))

    data["pred_prob"] = risk_all
    data["pred_label"] = (risk_all >= thr).astype(int)

    merged_df = (
        data[["MRN", "Female", "pred_label", "PE_Time", label_col]]
        .dropna(subset=["PE_Time", label_col])
        .drop_duplicates(subset=["MRN"])
        .rename(columns={label_col: "PE"})
    )
    plot_km_two_subplots_by_gender(merged_df)
    return merged_df


# ==========================================
# Experiment loops (sex-agnostic and sex-specific)
# ==========================================


def evaluate_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    time_col: str,
    event_col: str,
    mode: str,
    seed: int,
    use_undersampling: bool = False,
    disable_within_split_feature_selection: bool = False,
    sex_specific_feature_override: Dict[str, List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Train CoxPH and return (pred_label, risk_scores, metrics dict).

    mode in {"sex_agnostic", "male_only", "female_only", "sex_specific"}.
    Sex-agnostic removes Female from features if present.
    """
    if mode == "sex_agnostic":
        # In agnostic model, Female is included
        used_features = feature_cols
        tr = (
            create_undersampled_dataset(train_df, event_col, seed)
            if use_undersampling
            else train_df
        )
        # Feature selection only for Proposed set
        if not disable_within_split_feature_selection:
            try:
                if set(feature_cols) == set(FEATURE_SETS.get("Proposed", [])):
                    selected = select_features_max_cindex_forward(
                        tr, list(feature_cols), time_col, event_col, random_state=seed
                    )
                    if selected:
                        used_features = selected
                        print(
                            f"[FS] Sex-agnostic: selected {len(selected)} features for Proposed: {selected}"
                        )
            except Exception:
                pass
        cph = fit_cox_model(tr, used_features, time_col, event_col)
        risk_scores = predict_risk(cph, test_df, used_features)
        thr = threshold_by_top_quantile(risk_scores, 0.5)
        pred = (risk_scores >= thr).astype(int)
        cidx = concordance_index(test_df[time_col], -risk_scores, test_df[event_col])
        return pred, risk_scores, {"c_index": cidx}

    if mode == "male_only":
        # For single-sex model, exclude Female from features
        used_features = [f for f in feature_cols if f != "Female"]
        tr_m = train_df[train_df["Female"] == 0]
        te_m = test_df[test_df["Female"] == 0]
        if tr_m.empty or te_m.empty:
            return (
                np.zeros(len(test_df), dtype=int),
                np.zeros(len(test_df)),
                {"c_index": np.nan},
            )
        cph = fit_cox_model(tr_m, used_features, time_col, event_col)
        risk_m = predict_risk(cph, te_m, used_features)
        thr = threshold_by_top_quantile(risk_m, 0.5)
        pred_m = (risk_m >= thr).astype(int)
        pred = np.zeros(len(test_df), dtype=int)
        risk_scores = np.zeros(len(test_df))
        mask_m = test_df["Female"].values == 0
        pred[mask_m] = pred_m
        risk_scores[mask_m] = risk_m
        cidx = concordance_index(te_m[time_col], -risk_m, te_m[event_col])
        return pred, risk_scores, {"c_index": cidx}

    if mode == "female_only":
        # For single-sex model, exclude Female from features
        used_features = [f for f in feature_cols if f != "Female"]
        tr_f = train_df[train_df["Female"] == 1]
        te_f = test_df[test_df["Female"] == 1]
        if tr_f.empty or te_f.empty:
            return (
                np.zeros(len(test_df), dtype=int),
                np.zeros(len(test_df)),
                {"c_index": np.nan},
            )
        cph = fit_cox_model(tr_f, used_features, time_col, event_col)
        risk_f = predict_risk(cph, te_f, used_features)
        thr = threshold_by_top_quantile(risk_f, 0.5)
        pred_f = (risk_f >= thr).astype(int)
        pred = np.zeros(len(test_df), dtype=int)
        risk_scores = np.zeros(len(test_df))
        mask_f = test_df["Female"].values == 1
        pred[mask_f] = pred_f
        risk_scores[mask_f] = risk_f
        cidx = concordance_index(te_f[time_col], -risk_f, te_f[event_col])
        return pred, risk_scores, {"c_index": cidx}

    if mode == "sex_specific":
        # For single-sex submodels inside sex_specific, exclude Female from features
        used_features = [f for f in feature_cols if f != "Female"]
        # male branch
        tr_m = train_df[train_df["Female"] == 0]
        te_m = test_df[test_df["Female"] == 0]
        pred = np.zeros(len(test_df), dtype=int)
        risk_scores = np.zeros(len(test_df))
        if not tr_m.empty and not te_m.empty:
            # If override is provided, use it; otherwise optionally select per split
            if (
                sex_specific_feature_override
                and "male" in sex_specific_feature_override
            ):
                used_features_m = [
                    f for f in sex_specific_feature_override["male"] if f != "Female"
                ]
            else:
                used_features_m = used_features
                if not disable_within_split_feature_selection:
                    # Feature selection for Proposed per sex (male)
                    try:
                        if set(feature_cols) == set(FEATURE_SETS.get("Proposed", [])):
                            selected_m = select_features_max_cindex_forward(
                                tr_m,
                                list(used_features),
                                time_col,
                                event_col,
                                random_state=seed,
                            )
                            if selected_m:
                                used_features_m = selected_m
                                print(
                                    f"[FS] Sex-specific (male): selected {len(selected_m)} features for Proposed: {selected_m}"
                                )
                    except Exception:
                        pass
            cph_m = fit_cox_model(tr_m, used_features_m, time_col, event_col)
            risk_m = predict_risk(cph_m, te_m, used_features_m)
            thr_m = threshold_by_top_quantile(risk_m, 0.5)
            pred_m = (risk_m >= thr_m).astype(int)
            mask_m = test_df["Female"].values == 0
            pred[mask_m] = pred_m
            risk_scores[mask_m] = risk_m
        # female branch
        tr_f = train_df[train_df["Female"] == 1]
        te_f = test_df[test_df["Female"] == 1]
        if not tr_f.empty and not te_f.empty:
            if (
                sex_specific_feature_override
                and "female" in sex_specific_feature_override
            ):
                used_features_f = [
                    f for f in sex_specific_feature_override["female"] if f != "Female"
                ]
            else:
                used_features_f = used_features
                if not disable_within_split_feature_selection:
                    # Feature selection for Proposed per sex (female)
                    try:
                        if set(feature_cols) == set(FEATURE_SETS.get("Proposed", [])):
                            selected_f = select_features_max_cindex_forward(
                                tr_f,
                                list(used_features),
                                time_col,
                                event_col,
                                random_state=seed,
                            )
                            if selected_f:
                                used_features_f = selected_f
                                print(
                                    f"[FS] Sex-specific (female): selected {len(selected_f)} features for Proposed: {selected_f}"
                                )
                    except Exception:
                        pass
            cph_f = fit_cox_model(tr_f, used_features_f, time_col, event_col)
            risk_f = predict_risk(cph_f, te_f, used_features_f)
            thr_f = threshold_by_top_quantile(risk_f, 0.5)
            pred_f = (risk_f >= thr_f).astype(int)
            mask_f = test_df["Female"].values == 1
            pred[mask_f] = pred_f
            risk_scores[mask_f] = risk_f
        # get c-index on full test using risk_scores (where available)
        cidx = concordance_index(test_df[time_col], -risk_scores, test_df[event_col])
        return pred, risk_scores, {"c_index": cidx}

    raise ValueError(f"Unknown mode: {mode}")


# ==========================================
# Data preparation (mirroring a.py)
# ==========================================


def CG_equation(
    age: float, weight: float, female: int, serum_creatinine: float
) -> float:
    constant = 0.85 if female else 1.0
    return ((140 - age) * weight * constant) / (72 * serum_creatinine)


def impute_misforest(X: pd.DataFrame, random_seed: int) -> pd.DataFrame:
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    if not _HAS_MISSFOREST:
        # Fallback: simple mean imputation to keep script runnable
        X_imputed_scaled = X_scaled.fillna(X_scaled.mean())
    else:
        imputer = MissForest(random_state=random_seed)
        X_imputed_scaled = pd.DataFrame(
            imputer.fit_transform(X_scaled), columns=X.columns, index=X.index
        )
    X_imputed_unscaled = pd.DataFrame(
        scaler.inverse_transform(X_imputed_scaled), columns=X.columns, index=X.index
    )
    return X_imputed_unscaled


def conversion_and_imputation(
    df: pd.DataFrame, features: List[str], labels: List[str]
) -> pd.DataFrame:
    df = df.copy()
    df = df[features + labels]

    # Encode ordinal NYHA Class if present
    ordinal = "NYHA Class"
    if ordinal in df.columns:
        le = LabelEncoder()
        df[ordinal] = le.fit_transform(df[ordinal].astype(str))

    # Convert binary columns to numeric
    binary_cols = [
        "Female",
        "DM",
        "HTN",
        "HLP",
        "AF",
        "Beta Blocker",
        "ACEi/ARB/ARNi",
        "Aldosterone Antagonist",
        "VT/VF/SCD",
        "AAD",
        "CRT",
        "ICD",
    ]
    exist_bin = [c for c in binary_cols if c in df.columns]
    for c in exist_bin:
        if df[c].dtype == "object":
            df[c] = df[c].replace(
                {"Yes": 1, "No": 0, "Y": 1, "N": 0, "True": 1, "False": 0}
            )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Imputation on feature matrix
    X = df[features].copy()
    missing_cols = X.columns[X.isnull().any()].tolist()
    if missing_cols:
        imputed_part = impute_misforest(X[missing_cols], 0)
        imputed_X = X.copy()
        imputed_X[missing_cols] = imputed_part
    else:
        imputed_X = X.copy()

    imputed_X.index = df.index

    # Bring back key columns
    for col in ["MRN", "Female", "VT/VF/SCD", "ICD", "PE_Time"]:
        if col in df.columns:
            imputed_X[col] = df[col].values

    # Map to 0/1 by 0.5 threshold for binary cols
    for c in exist_bin:
        if c in imputed_X.columns:
            imputed_X[c] = (imputed_X[c] >= 0.5).astype(float)

    # Round NYHA Class
    if "NYHA Class" in imputed_X.columns:
        imputed_X["NYHA Class"] = imputed_X["NYHA Class"].round().astype("Int64")

    return imputed_X


def load_dataframes() -> pd.DataFrame:
    base = "/home/sunx/data/aiiih/projects/sunx/projects/ICD_sex_diff"
    icd = pd.read_excel(os.path.join(base, "NICM.xlsx"), sheet_name="ICD")
    noicd = pd.read_excel(os.path.join(base, "NICM.xlsx"), sheet_name="No_ICD")

    # Cockcroft-Gault
    noicd["Cockcroft-Gault Creatinine Clearance (mL/min)"] = noicd.apply(
        lambda row: (
            row["Cockcroft-Gault Creatinine Clearance (mL/min)"]
            if pd.notna(row["Cockcroft-Gault Creatinine Clearance (mL/min)"])
            else CG_equation(
                row["Age at CMR"],
                row["Weight (Kg)"],
                row["Female"],
                row["Serum creatinine (within 3 months of MRI)"],
            )
        ),
        axis=1,
    )

    # Stack dfs
    common_cols = icd.columns.intersection(noicd.columns)
    icd_common = icd[common_cols].copy()
    noicd_common = noicd[common_cols].copy()
    icd_common["ICD"] = 1
    noicd_common["ICD"] = 0
    nicm = pd.concat([icd_common, noicd_common], ignore_index=True)

    # PE time
    nicm["MRI Date"] = pd.to_datetime(nicm["MRI Date"])
    nicm["Date VT/VF/SCD"] = pd.to_datetime(nicm["Date VT/VF/SCD"])
    nicm["End follow-up date"] = pd.to_datetime(nicm["End follow-up date"])
    nicm["PE_Time"] = nicm.apply(
        lambda row: (
            (row["Date VT/VF/SCD"] - row["MRI Date"]).days
            if row["VT/VF/SCD"] == 1
            and pd.notna(row["Date VT/VF/SCD"])
            and pd.notna(row["MRI Date"])
            else (
                (row["End follow-up date"] - row["MRI Date"]).days
                if pd.notna(row["End follow-up date"]) and pd.notna(row["MRI Date"])
                else None
            )
        ),
        axis=1,
    )

    # Drop features
    nicm.drop(
        ["MRI Date", "Date VT/VF/SCD", "End follow-up date", "CRT Date"],
        axis=1,
        inplace=True,
        errors="ignore",
    )

    # Variables
    categorical = [
        "Female",
        "DM",
        "HTN",
        "HLP",
        "AF",
        "NYHA Class",
        "Beta Blocker",
        "ACEi/ARB/ARNi",
        "Aldosterone Antagonist",
        "VT/VF/SCD",
        "AAD",
        "CRT",
        "ICD",
    ]
    nicm[categorical] = nicm[categorical].astype("object")
    var = nicm.columns.tolist()
    labels = ["MRN", "VT/VF/SCD", "ICD", "PE_Time"]
    features = [v for v in var if v not in labels]

    # Imputation
    clean_df = conversion_and_imputation(nicm, features, labels)

    # Additional engineered features
    if "Age at CMR" in clean_df.columns:
        clean_df["Age by decade"] = (clean_df["Age at CMR"] // 10).values
    if "Cockcroft-Gault Creatinine Clearance (mL/min)" in clean_df.columns:
        clean_df["CrCl>45"] = (
            clean_df["Cockcroft-Gault Creatinine Clearance (mL/min)"] > 45
        ).astype(int)
    if "NYHA Class" in clean_df.columns:
        clean_df["NYHA>2"] = (clean_df["NYHA Class"] > 2).astype(int)
    if "LGE Burden 5SD" in clean_df.columns:
        clean_df["Significant LGE"] = (clean_df["LGE Burden 5SD"] > 2).astype(int)

    return clean_df


# ==========================================
# Main multi-split runner
# ==========================================


def run_cox_experiments(
    df: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    N: int = 50,
    time_col: str = "PE_Time",
    event_col: str = "VT/VF/SCD",
    export_excel_path: str = None,
) -> Tuple[Dict[str, Dict[str, List[float]]], pd.DataFrame]:
    """Run 50 random 70/30 splits across configurations, compute performance and export summary.

    Modes per feature set:
      - sex-agnostic (undersampled)
      - sex-specific
    """
    model_configs = [
        {"name": "Sex-agnostic (Cox, undersampled)", "mode": "sex_agnostic"},
        {"name": "Sex-specific (Cox)", "mode": "sex_specific"},
    ]
    metrics = ["c_index_all", "c_index_male", "c_index_female"]

    results: Dict[str, Dict[str, List[float]]] = {}
    for featset_name in feature_sets:
        for cfg in model_configs:
            results[f"{featset_name} - {cfg['name']}"] = {m: [] for m in metrics}

    # Precompute stable feature sets to ensure consistent specification across splits
    seeds_for_stability = list(range(min(N, 20)))
    stable_agnostic_features: Dict[str, List[str]] = {}
    stable_sex_specific_features: Dict[str, Dict[str, List[str]]] = {}

    for featset_name, feature_cols in feature_sets.items():
        # Default to original sets when no selection logic is intended
        stable_agnostic_features[featset_name] = list(feature_cols)
        stable_sex_specific_features[featset_name] = {
            "male": [f for f in feature_cols if f != "Female"],
            "female": [f for f in feature_cols if f != "Female"],
        }

        # Only Proposed set uses selection; stabilize it once globally
        try:
            if set(feature_cols) == set(FEATURE_SETS.get("Proposed", [])):
                # Sex-agnostic stabilization
                sel_agn = stability_select_features(
                    df.dropna(subset=[time_col, event_col]).copy(),
                    candidate_features=list(feature_cols),
                    time_col=time_col,
                    event_col=event_col,
                    seeds=seeds_for_stability,
                    max_features=None,
                    threshold=0.4,
                    min_features=None,
                    verbose=True,
                )
                if sel_agn:
                    stable_agnostic_features[featset_name] = sel_agn
                    try:
                        if featset_name == "Proposed":
                            SELECTED_FEATURES_STORE["proposed_sex_agnostic"] = list(
                                sel_agn
                            )
                            print(
                                f"[FS][Store] Proposed sex-agnostic features: {len(sel_agn)} selected"
                            )
                    except Exception:
                        pass

                # Sex-specific stabilization per sex
                df_m = df[df["Female"] == 0].dropna(subset=[time_col, event_col]).copy()
                df_f = df[df["Female"] == 1].dropna(subset=[time_col, event_col]).copy()
                base_feats = [f for f in feature_cols if f != "Female"]

                sel_m = (
                    stability_select_features(
                        df=df_m,
                        candidate_features=list(base_feats),
                        time_col=time_col,
                        event_col=event_col,
                        seeds=seeds_for_stability,
                        max_features=None,
                        threshold=0.4,
                        min_features=None,
                        verbose=True,
                    )
                    if not df_m.empty
                    else []
                )
                sel_f = (
                    stability_select_features(
                        df=df_f,
                        candidate_features=list(base_feats),
                        time_col=time_col,
                        event_col=event_col,
                        seeds=seeds_for_stability,
                        max_features=None,
                        threshold=0.4,
                        min_features=None,
                        verbose=True,
                    )
                    if not df_f.empty
                    else []
                )

                if sel_m:
                    stable_sex_specific_features[featset_name]["male"] = sel_m
                    try:
                        if featset_name == "Proposed":
                            SELECTED_FEATURES_STORE["proposed_sex_specific"]["male"] = (
                                list(sel_m)
                            )
                            print(
                                f"[FS][Store] Proposed male-specific features: {len(sel_m)} selected"
                            )
                    except Exception:
                        pass
                if sel_f:
                    stable_sex_specific_features[featset_name]["female"] = sel_f
                    try:
                        if featset_name == "Proposed":
                            SELECTED_FEATURES_STORE["proposed_sex_specific"][
                                "female"
                            ] = list(sel_f)
                            print(
                                f"[FS][Store] Proposed female-specific features: {len(sel_f)} selected"
                            )
                    except Exception:
                        pass
        except Exception:
            pass

    for seed in range(N):
        print(seed)
        tr, te = train_test_split(
            df, test_size=0.3, random_state=seed, stratify=df[event_col]
        )
        tr = tr.dropna(subset=[time_col, event_col])
        te = te.dropna(subset=[time_col, event_col])

        for featset_name, feature_cols in feature_sets.items():
            for cfg in model_configs:
                name = f"{featset_name} - {cfg['name']}"
                use_undersampling = cfg["mode"] == "sex_agnostic"
                # Use stabilized features and disable per-split selection
                if cfg["mode"] == "sex_agnostic":
                    frozen_feats = stable_agnostic_features.get(
                        featset_name, feature_cols
                    )
                    if not frozen_feats:
                        frozen_feats = feature_cols
                    # Ensure Proposed uses the stored selection if available
                    if featset_name == "Proposed":
                        stored = SELECTED_FEATURES_STORE.get("proposed_sex_agnostic")
                        if stored:
                            frozen_feats = list(stored)
                    pred, risk, met = evaluate_split(
                        tr,
                        te,
                        frozen_feats,
                        time_col,
                        event_col,
                        mode=cfg["mode"],
                        seed=seed,
                        use_undersampling=use_undersampling,
                        disable_within_split_feature_selection=True,
                    )
                else:
                    overrides = stable_sex_specific_features.get(featset_name, None)
                    # Ensure Proposed uses the stored sex-specific selections if available
                    if featset_name == "Proposed":
                        try:
                            stored_m = SELECTED_FEATURES_STORE.get(
                                "proposed_sex_specific", {}
                            ).get("male")
                            stored_f = SELECTED_FEATURES_STORE.get(
                                "proposed_sex_specific", {}
                            ).get("female")
                            if overrides is None:
                                overrides = {}
                            if stored_m:
                                overrides["male"] = list(stored_m)
                            if stored_f:
                                overrides["female"] = list(stored_f)
                        except Exception:
                            pass
                    pred, risk, met = evaluate_split(
                        tr,
                        te,
                        feature_cols,
                        time_col,
                        event_col,
                        mode=cfg["mode"],
                        seed=seed,
                        use_undersampling=use_undersampling,
                        disable_within_split_feature_selection=True,
                        sex_specific_feature_override=overrides,
                    )
                # overall, male, female C-index based on risk
                cidx_all = met.get("c_index", np.nan)
                try:
                    mask_m = te["Female"].values == 0
                    mask_f = te["Female"].values == 1

                    # Safety: ensure finite and sufficient variation
                    def _safe_cidx(t, e, r):
                        try:
                            if len(t) < 2:
                                return np.nan
                            if np.all(~np.isfinite(r)) or np.allclose(r, r[0]):
                                return np.nan
                            return concordance_index(t, -r, e)
                        except Exception:
                            return np.nan

                    cidx_m = _safe_cidx(
                        te.loc[mask_m, time_col].values,
                        te.loc[mask_m, event_col].values,
                        np.asarray(risk)[mask_m],
                    )
                    cidx_f = _safe_cidx(
                        te.loc[mask_f, time_col].values,
                        te.loc[mask_f, event_col].values,
                        np.asarray(risk)[mask_f],
                    )
                except Exception:
                    cidx_m, cidx_f = np.nan, np.nan

                results[name]["c_index_all"].append(cidx_all)
                results[name]["c_index_male"].append(cidx_m)
                results[name]["c_index_female"].append(cidx_f)

    # Summarize
    summary = {}
    for model, mvals in results.items():
        summary[model] = {}
        for metric, values in mvals.items():
            arr = np.array(values, dtype=float)
            mu = np.nanmean(arr)
            se = np.nanstd(arr, ddof=1) / np.sqrt(np.sum(~np.isnan(arr)))
            ci = 1.96 * se
            summary[model][metric] = (mu, mu - ci, mu + ci)

    summary_df = pd.concat(
        {
            model: pd.DataFrame.from_dict(
                mvals, orient="index", columns=["mean", "ci_lower", "ci_upper"]
            )
            for model, mvals in summary.items()
        },
        axis=0,
    )

    formatted = summary_df.apply(
        lambda row: f"{row['mean']:.3f} ({row['ci_lower']:.3f}, {row['ci_upper']:.3f})",
        axis=1,
    )
    summary_table = formatted.unstack(level=1)
    # Rename columns to desired group names
    summary_table = summary_table.rename(
        columns={
            "c_index_all": "all",
            "c_index_male": "male",
            "c_index_female": "female",
        }
    )

    if export_excel_path is not None:
        os.makedirs(os.path.dirname(export_excel_path), exist_ok=True)
        summary_table.to_excel(export_excel_path, index=True, index_label="RowName")

    return results, summary_table


FEATURE_SETS = {
    "Guideline": ["NYHA Class", "LVEF"],
    "Benchmark": [
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
    "Proposed": [
        "Female",
        "Age at CMR",
        "BMI",
        "DM",
        "HTN",
        "HLP",
        "AF",
        "NYHA Class",
        "LVEDVi",
        "LVEF",
        "LV Mass Index",
        "RVEDVi",
        "RVEF",
        "LA EF",
        "LAVi",
        "LGE Burden 5SD",
        "MRF (%)",
        "Sphericity Index",
        "Relative Wall Thickness",
        "MV Annular Diameter",
        "QRS",
        "QTc",
        "CrCl>45",
    ],
}


if __name__ == "__main__":
    # Load and prepare data
    df = load_dataframes()

    # Generate TableOne grouped by Sex x ICD (four groups)
    try:
        generate_tableone_by_sex_icd(
            df,
            output_excel_path="/home/sunx/data/aiiih/projects/sunx/projects/ICD_sex_diff/results_tableone_sex_icd.xlsx",
        )
    except Exception as e:
        warnings.warn(f"[Main] TableOne generation skipped due to error: {e}")

    # Run experiments (50 random splits; PE as event; PE_Time as duration)
    export_path = (
        "/home/sunx/data/aiiih/projects/sunx/projects/ICD_sex_diff/results_cox.xlsx"
    )
    _, summary = run_cox_experiments(
        df=df,
        feature_sets=FEATURE_SETS,
        N=100,
        time_col="PE_Time",
        event_col="VT/VF/SCD",
        export_excel_path=export_path,
    )
    print("Saved Excel:", export_path)

    # Full-data inference and analysis - Guideline
    features = FEATURE_SETS["Guideline"]
    print("Running sex-agnostic full-data inference (includes Female)...")
    _ = sex_agnostic_full_inference(df, features, use_undersampling=False)

    print("Running sex-specific full-data inference (excludes Female in submodels)...")
    _ = sex_specific_full_inference(df, features)

    # Full-data inference and analysis - Benchmark
    features = FEATURE_SETS["Benchmark"]
    print("Running sex-agnostic full-data inference (includes Female)...")
    _ = sex_agnostic_full_inference(df, features, use_undersampling=False)

    print("Running sex-specific full-data inference (excludes Female in submodels)...")
    _ = sex_specific_full_inference(df, features)

    # Full-data inference and analysis - Proposed
    features = FEATURE_SETS["Proposed"]
    print("Running sex-agnostic full-data inference (includes Female)...")
    _ = sex_agnostic_full_inference(df, features, use_undersampling=False)

    print("Running sex-specific full-data inference (excludes Female in submodels)...")
    _ = sex_specific_full_inference(df, features)
