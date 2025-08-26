import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

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
) -> None:
    coef_series = model.params_.sort_values(ascending=False)
    feats = coef_series.index.tolist()
    colors = []
    gray_set = set(gray_features) if gray_features is not None else set()
    red_set = set(red_features) if red_features is not None else set()
    for f in feats:
        if f in red_set:
            colors.append("red")
        elif f in gray_set:
            colors.append("gray")
        else:
            colors.append("blue")
    plt.figure(figsize=(9, 4))
    plt.bar(range(len(feats)), coef_series.values, color=colors)
    plt.xticks(range(len(feats)), feats, rotation=90)
    plt.ylabel("Cox coefficient (log HR)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_km_curves_four_groups(merged_df: pd.DataFrame) -> None:
    """Deprecated: kept for backward compatibility. Redirects to two-subplot KM plot."""
    plot_km_by_sex_two_subplots(merged_df)


def plot_km_by_sex_two_subplots(merged_df: pd.DataFrame) -> None:
    """Plot KM curves in two subplots: left=male, right=female. Each subplot contains
    two groups: pred1 (High risk) and pred0 (Low risk). Also perform within-sex logrank tests.
    """
    if merged_df.empty:
        return
    if not {"PE_Time", "PE"}.issubset(merged_df.columns):
        return

    ep_time_col, ep_event_col = "PE_Time", "PE"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    sex_panels = [(0, "Male", axes[0]), (1, "Female", axes[1])]
    for sex_val, sex_name, ax in sex_panels:
        sub = merged_df[merged_df["Female"] == sex_val]
        if sub.empty:
            ax.set_title(f"{sex_name} (no data)")
            ax.axis("off")
            continue

        kmf = KaplanMeierFitter()
        groups = [
            (1, "high risk", "red"),
            (0, "low risk", "blue"),
        ]
        for pred_val, label_name, color in groups:
            grp = sub[sub["pred_label"] == pred_val]
            if grp.empty:
                continue
            n_samples = len(grp)
            events = grp[ep_event_col].sum()
            series_label = f"{label_name} (n={n_samples}, events={events})"
            kmf.fit(
                durations=grp[ep_time_col],
                event_observed=grp[ep_event_col],
                label=series_label,
            )
            kmf.plot(ax=ax, color=color)

        # Logrank within sex: High vs Low
        grp_low = sub[sub["pred_label"] == 0]
        grp_high = sub[sub["pred_label"] == 1]
        if not grp_low.empty and not grp_high.empty:
            lr = logrank_test(
                grp_low[ep_time_col],
                grp_high[ep_time_col],
                grp_low[ep_event_col],
                grp_high[ep_event_col],
            )
            ax.text(0.02, 0.02, f"logrank p={lr.p_value:.4f}", transform=ax.transAxes)

        ax.set_title(f"{sex_name}")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Survival Probability")
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Primary Endpoint - Survival by Sex and Risk Group")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


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


def sex_specific_inference(
    df: pd.DataFrame,
    features: List[str],
    gray_features: List[str] = None,
    red_features: List[str] = None,
) -> pd.DataFrame:
    """Sex-specific CoxPH inference trained and evaluated on all available data.
    Uses PE_Time and VT/VF/SCD from the provided dataframe. Female is excluded from
    the feature list for the sex-specific submodels.
    """
    used_features = [f for f in features if f != "Female"]

    train_m = df[df["Female"] == 0].dropna(subset=["PE_Time", "VT/VF/SCD"]).copy()
    train_f = df[df["Female"] == 1].dropna(subset=["PE_Time", "VT/VF/SCD"]).copy()

    models = {}
    thresholds = {}

    if not train_m.empty:
        cph_m = fit_cox_model(train_m, used_features, "PE_Time", "VT/VF/SCD")
        tr_risk_m = predict_risk(cph_m, train_m, used_features)
        thresholds["male"] = float(np.nanmedian(tr_risk_m))
        models["male"] = cph_m
        plot_cox_coefficients(
            cph_m, "Male Cox Coefficients (log HR)", gray_features, red_features
        )
    if not train_f.empty:
        cph_f = fit_cox_model(train_f, used_features, "PE_Time", "VT/VF/SCD")
        tr_risk_f = predict_risk(cph_f, train_f, used_features)
        thresholds["female"] = float(np.nanmedian(tr_risk_f))
        models["female"] = cph_f
        plot_cox_coefficients(
            cph_f, "Female Cox Coefficients (log HR)", gray_features, red_features
        )

    df_out = df.copy()
    if "male" in models:
        sub_m = df_out[df_out["Female"] == 0]
        if not sub_m.empty:
            risk_m = predict_risk(models["male"], sub_m, used_features)
            df_out.loc[df_out["Female"] == 0, "pred_prob"] = risk_m
            df_out.loc[df_out["Female"] == 0, "pred_label"] = (
                risk_m >= thresholds["male"]
            ).astype(int)
    if "female" in models:
        sub_f = df_out[df_out["Female"] == 1]
        if not sub_f.empty:
            risk_f = predict_risk(models["female"], sub_f, used_features)
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
    plot_km_by_sex_two_subplots(merged_df)
    return merged_df


def sex_agnostic_inference(
    df: pd.DataFrame,
    features: List[str],
    label_col: str = "VT/VF/SCD",
    use_undersampling: bool = True,
    gray_features: List[str] = None,
    red_features: List[str] = None,
) -> pd.DataFrame:
    """Sex-agnostic CoxPH inference trained and evaluated on all available data.
    Assumes survival labels/time are in the dataframe (uses PE_Time and label_col).
    Female is INCLUDED in features for the agnostic model.
    """
    used_features = features
    tr_base = (
        create_undersampled_dataset(df, label_col, 42)
        if use_undersampling
        else df
    )
    tr = tr_base.dropna(subset=["PE_Time", label_col]).copy()
    if tr.empty:
        return df.copy()

    cph = fit_cox_model(tr, used_features, "PE_Time", label_col)
    plot_cox_coefficients(
        cph, "Sex-Agnostic Cox Coefficients (log HR)", gray_features, red_features
    )
    tr_risk = predict_risk(cph, tr, used_features)
    thr = float(np.nanmedian(tr_risk))

    out = df.copy()
    out_risk = predict_risk(cph, out, used_features)
    out["pred_prob"] = out_risk
    out["pred_label"] = (out_risk >= thr).astype(int)

    merged_df = (
        out[["MRN", "Female", "pred_label", "PE_Time", label_col]]
        .dropna(subset=["PE_Time", label_col])
        .drop_duplicates(subset=["MRN"])
        .rename(columns={label_col: "PE"})
    )
    plot_km_by_sex_two_subplots(merged_df)
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
        cph = fit_cox_model(tr, used_features, time_col, event_col)
        risk_scores = predict_risk(cph, test_df, used_features)
        thr = threshold_by_top_quantile(risk_scores, 0.5)
        pred = (risk_scores >= thr).astype(int)
        cidx_all = concordance_index(test_df[time_col], -risk_scores, test_df[event_col])
        # subgroup c-index
        male_mask = test_df["Female"].values == 0
        female_mask = test_df["Female"].values == 1
        cidx_m = np.nan
        cidx_f = np.nan
        if male_mask.any():
            cidx_m = concordance_index(
                test_df.loc[male_mask, time_col],
                -risk_scores[male_mask],
                test_df.loc[male_mask, event_col],
            )
        if female_mask.any():
            cidx_f = concordance_index(
                test_df.loc[female_mask, time_col],
                -risk_scores[female_mask],
                test_df.loc[female_mask, event_col],
            )
        return pred, risk_scores, {"c_index_all": cidx_all, "c_index_male": cidx_m, "c_index_female": cidx_f}

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
        return pred, risk_scores, {"c_index_all": cidx, "c_index_male": cidx, "c_index_female": np.nan}

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
        return pred, risk_scores, {"c_index_all": cidx, "c_index_male": np.nan, "c_index_female": cidx}

    if mode == "sex_specific":
        # For single-sex submodels inside sex_specific, exclude Female from features
        used_features = [f for f in feature_cols if f != "Female"]
        # male branch
        tr_m = train_df[train_df["Female"] == 0]
        te_m = test_df[test_df["Female"] == 0]
        pred = np.zeros(len(test_df), dtype=int)
        risk_scores = np.zeros(len(test_df))
        if not tr_m.empty and not te_m.empty:
            cph_m = fit_cox_model(tr_m, used_features, time_col, event_col)
            risk_m = predict_risk(cph_m, te_m, used_features)
            thr_m = threshold_by_top_quantile(risk_m, 0.5)
            pred_m = (risk_m >= thr_m).astype(int)
            mask_m = test_df["Female"].values == 0
            pred[mask_m] = pred_m
            risk_scores[mask_m] = risk_m
        # female branch
        tr_f = train_df[train_df["Female"] == 1]
        te_f = test_df[test_df["Female"] == 1]
        if not tr_f.empty and not te_f.empty:
            cph_f = fit_cox_model(tr_f, used_features, time_col, event_col)
            risk_f = predict_risk(cph_f, te_f, used_features)
            thr_f = threshold_by_top_quantile(risk_f, 0.5)
            pred_f = (risk_f >= thr_f).astype(int)
            mask_f = test_df["Female"].values == 1
            pred[mask_f] = pred_f
            risk_scores[mask_f] = risk_f
        # c-indexes
        cidx_all = concordance_index(test_df[time_col], -risk_scores, test_df[event_col])
        male_mask = test_df["Female"].values == 0
        female_mask = test_df["Female"].values == 1
        cidx_m = np.nan
        cidx_f = np.nan
        if male_mask.any():
            cidx_m = concordance_index(
                test_df.loc[male_mask, time_col],
                -risk_scores[male_mask],
                test_df.loc[male_mask, event_col],
            )
        if female_mask.any():
            cidx_f = concordance_index(
                test_df.loc[female_mask, time_col],
                -risk_scores[female_mask],
                test_df.loc[female_mask, event_col],
            )
        return pred, risk_scores, {"c_index_all": cidx_all, "c_index_male": cidx_m, "c_index_female": cidx_f}

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
                pred, risk, met = evaluate_split(
                    tr,
                    te,
                    feature_cols,
                    time_col,
                    event_col,
                    mode=cfg["mode"],
                    seed=seed,
                    use_undersampling=use_undersampling,
                )
                # collect metrics for all/male/female
                for m in metrics:
                    results[name][m].append(met.get(m, np.nan))

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
    # Rename metric columns to all/male/female for readability
    col_rename = {
        "c_index_all": "all",
        "c_index_male": "male",
        "c_index_female": "female",
    }
    summary_table = summary_table.rename(columns={k: v for k, v in col_rename.items() if k in summary_table.columns})

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
        "Age by decade",
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
        "Beta Blocker",
        "ACEi/ARB/ARNi",
        "Aldosterone Antagonist",
        "AAD",
        "CRT",
        "QRS",
        "QTc",
        "CrCl>45",
    ],
}


if __name__ == "__main__":
    # Load and prepare data
    df = load_dataframes()

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

    # Run inference and analysis on the full dataset
    features = FEATURE_SETS["Benchmark"]

    print("Running sex-agnostic inference on all data (includes Female; undersampling=True)...")
    _ = sex_agnostic_inference(df, features, use_undersampling=True)

    print("Running sex-specific inference on all data (excludes Female in submodels)...")
    _ = sex_specific_inference(df, features)
