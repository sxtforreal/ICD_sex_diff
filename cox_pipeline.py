import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

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

def standardize_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    scaler = StandardScaler()
    X = df[feature_cols].astype(float).copy()
    X.loc[:, feature_cols] = scaler.fit_transform(X[feature_cols])
    return X


def create_undersampled_dataset(train_df: pd.DataFrame, label_col: str, random_state: int) -> pd.DataFrame:
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
    return pd.concat(sampled_parts, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def compute_incidence_rate(df: pd.DataFrame, pred_col: str, label_col: str) -> Tuple[float, float]:
    def rate(sub: pd.DataFrame) -> float:
        n_pred = (sub[pred_col] == 1).sum()
        n_true = (sub[label_col] == 1).sum()
        return n_true / n_pred if n_pred > 0 else np.nan
    male_rate = rate(df[df["Female"] == 0])
    female_rate = rate(df[df["Female"] == 1])
    return male_rate, female_rate


def plot_cox_coefficients(model: CoxPHFitter, title: str, gray_features: List[str] = None, red_features: List[str] = None) -> None:
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


# ==========================================
# CoxPH training/inference blocks
# ==========================================

def fit_cox_model(train_df: pd.DataFrame, feature_cols: List[str], time_col: str, event_col: str) -> CoxPHFitter:
    df_fit = train_df[[time_col, event_col] + feature_cols].copy()
    cph = CoxPHFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(df_fit, duration_col=time_col, event_col=event_col, robust=True)
    return cph


def predict_risk(model: CoxPHFitter, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    X = df[feature_cols].copy()
    risk = model.predict_partial_hazard(X).values.reshape(-1)
    return risk


def threshold_by_top_quantile(risk_scores: np.ndarray, quantile: float = 0.5) -> float:
    q = np.nanquantile(risk_scores, quantile)
    return float(q)


# ==========================================
# Experiment loops (sex-agnostic and sex-specific)
# ==========================================

def evaluate_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
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
        used_features = [f for f in feature_cols if f != "Female"]
        tr = create_undersampled_dataset(train_df, label_col, seed) if use_undersampling else train_df
        cph = fit_cox_model(tr, used_features, time_col, event_col)
        risk_scores = predict_risk(cph, test_df, used_features)
        thr = threshold_by_top_quantile(risk_scores, 0.5)
        pred = (risk_scores >= thr).astype(int)
        cidx = concordance_index(test_df[time_col], -risk_scores, test_df[event_col])
        return pred, risk_scores, {"c_index": cidx}

    if mode == "male_only":
        used_features = feature_cols
        tr_m = train_df[train_df["Female"] == 0]
        te_m = test_df[test_df["Female"] == 0]
        if tr_m.empty or te_m.empty:
            return np.zeros(len(test_df), dtype=int), np.zeros(len(test_df)), {"c_index": np.nan}
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
        used_features = feature_cols
        tr_f = train_df[train_df["Female"] == 1]
        te_f = test_df[test_df["Female"] == 1]
        if tr_f.empty or te_f.empty:
            return np.zeros(len(test_df), dtype=int), np.zeros(len(test_df)), {"c_index": np.nan}
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
        used_features = feature_cols
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
        # get c-index on full test using risk_scores (where available)
        cidx = concordance_index(test_df[time_col], -risk_scores, test_df[event_col])
        return pred, risk_scores, {"c_index": cidx}

    raise ValueError(f"Unknown mode: {mode}")


# ==========================================
# Data preparation (mirroring a.py)
# ==========================================

def CG_equation(age: float, weight: float, female: int, serum_creatinine: float) -> float:
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


def conversion_and_imputation(df: pd.DataFrame, features: List[str], labels: List[str]) -> pd.DataFrame:
    df = df.copy()
    df = df[features + labels]

    # Encode ordinal NYHA Class if present
    ordinal = "NYHA Class"
    if ordinal in df.columns:
        le = LabelEncoder()
        df[ordinal] = le.fit_transform(df[ordinal].astype(str))

    # Convert binary columns to numeric
    binary_cols = [
        "Female", "DM", "HTN", "HLP", "AF", "Beta Blocker", "ACEi/ARB/ARNi",
        "Aldosterone Antagonist", "VT/VF/SCD", "AAD", "CRT", "ICD",
    ]
    exist_bin = [c for c in binary_cols if c in df.columns]
    for c in exist_bin:
        if df[c].dtype == "object":
            df[c] = df[c].replace({
                "Yes": 1, "No": 0, "Y": 1, "N": 0, "True": 1, "False": 0
            })
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Imputation on feature matrix
    X = df[features].copy()
    imputed_X = impute_misforest(X, 0)
    imputed_X.index = df.index
    # Bring back key columns
    for col in ["MRN", "Female", "VT/VF/SCD", "ICD"]:
        if col in df.columns:
            imputed_X[col] = df[col].values

    # Map to 0/1 by 0.5 threshold for binary cols
    for c in exist_bin:
        if c in imputed_X.columns:
            imputed_X[c] = (imputed_X[c] >= 0.5).astype(float)

    return imputed_X


def load_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = "/home/sunx/data/aiiih/projects/sunx/projects/ICD_sex_diff"

    # Survival: ICD
    icd_survival = pd.read_excel(os.path.join(base, "icd_survival.xlsx"))
    icd_survival["PE_Time"] = icd_survival.apply(
        lambda row: (
            row["Time from ICD Implant to Primary Endpoint (in days)"]
            if row["Was Primary Endpoint Reached? (Appropriate ICD Therapy)"] == 1
            else row["Time from ICD Implant to Last Cardiology Encounter (in days)"]
        ), axis=1,
    )
    icd_survival["SE_Time"] = icd_survival.apply(
        lambda row: (
            row["Time from ICD Implant to Secondary Endpoint (in days)"]
            if row["Was Secondary Endpoint Reached?"] == 1
            else row["Time from ICD Implant to Last Cardiology Encounter (in days)"]
        ), axis=1,
    )
    icd_survival = icd_survival[[
        "MRN",
        "Was Primary Endpoint Reached? (Appropriate ICD Therapy)",
        "PE_Time",
        "Was Secondary Endpoint Reached?",
        "SE_Time",
    ]].rename(columns={
        "Was Primary Endpoint Reached? (Appropriate ICD Therapy)": "PE",
        "Was Secondary Endpoint Reached?": "SE",
    })

    # Survival: No ICD
    no_icd_survival = pd.read_csv(os.path.join(base, "no_icd_survival.csv"))
    no_icd_survival["PE_Time"] = no_icd_survival.apply(
        lambda row: (
            row["days_MRI_to_VTVFSCD"] if row["VT/VF/SCD"] == 1 else row["days_MRI_to_followup"]
        ), axis=1,
    )
    no_icd_survival["SE_Time"] = no_icd_survival.apply(
        lambda row: (
            row["days_MRI_to_death"] if row["Death"] == 1 else row["days_MRI_to_followup"]
        ), axis=1,
    )
    no_icd_survival = no_icd_survival[[
        "MRN", "VT/VF/SCD", "PE_Time", "Death", "SE_Time"
    ]].rename(columns={"VT/VF/SCD": "PE", "Death": "SE"})

    survival_df = pd.concat([icd_survival, no_icd_survival], ignore_index=True)

    # Features: ICD and No ICD sheets
    nicm_path = os.path.join(base, "NICM.xlsx")
    with_icd = pd.read_excel(nicm_path, sheet_name="ICD")
    with_icd["ICD"] = 1
    without_icd = pd.read_excel(nicm_path, sheet_name="No_ICD")
    without_icd["ICD"] = 0
    # Cockcroft-Gault for no ICD
    without_icd["Cockcroft-Gault Creatinine Clearance (mL/min)"] = without_icd.apply(
        lambda row: CG_equation(
            row["Age at CMR"], row["Weight (Kg)"], row["Female"],
            row["Serum creatinine (within 3 months of MRI)"]
        ), axis=1,
    )

    common_cols = with_icd.columns.intersection(without_icd.columns)
    df = pd.concat([with_icd[common_cols], without_icd[common_cols]], ignore_index=True)
    df.drop([
        "Date VT/VF/SCD", "End follow-up date", "CRT Date", "QRS"
    ], axis=1, inplace=True, errors="ignore")

    # Variables
    var = df.columns.tolist()
    categorical = [
        "Female", "DM", "HTN", "HLP", "AF", "NYHA Class", "Beta Blocker",
        "ACEi/ARB/ARNi", "Aldosterone Antagonist", "VT/VF/SCD", "AAD", "CRT", "ICD",
    ]
    numerical = list(set(var) - set(categorical))
    df[categorical] = df[categorical].astype("object")

    labels = ["MRN", "Female", "VT/VF/SCD", "ICD"]
    features = [c for c in var if c not in labels]

    clean_df = conversion_and_imputation(df, features, labels)

    # Additional engineered features
    if "Age at CMR" in df.columns:
        clean_df["Age by decade"] = (df["Age at CMR"] // 10).values
    if "Cockcroft-Gault Creatinine Clearance (mL/min)" in clean_df.columns:
        clean_df["CrCl>45"] = (clean_df["Cockcroft-Gault Creatinine Clearance (mL/min)"] > 45).astype(int)
    if "NYHA Class" in clean_df.columns:
        clean_df["NYHA>2"] = (clean_df["NYHA Class"] > 2).astype(int)
    if "LGE Burden 5SD" in clean_df.columns:
        clean_df["Significant LGE"] = (clean_df["LGE Burden 5SD"] > 2).astype(int)

    return clean_df, survival_df


# ==========================================
# Main multi-split runner
# ==========================================

def run_cox_experiments(
    df: pd.DataFrame,
    survival_df: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    N: int = 50,
    label: str = "VT/VF/SCD",
    time_col: str = "PE_Time",
    event_col: str = "PE",
    random_state: int = 42,
    export_excel_path: str = None,
) -> Tuple[Dict[str, Dict[str, List[float]]], pd.DataFrame]:
    """Run 50 random 70/30 splits across configurations, compute performance and export summary.

    Modes per feature set:
      - sex-agnostic (undersampled)
      - sex-specific
      - male-only
      - female-only
    """
    model_configs = [
        {"name": "Sex-agnostic (Cox, undersampled)", "mode": "sex_agnostic"},
        {"name": "Sex-specific (Cox)", "mode": "sex_specific"},
        {"name": "Male (Cox)", "mode": "male_only"},
        {"name": "Female (Cox)", "mode": "female_only"},
    ]

    metrics = [
        "c_index",
        "male_rate", "female_rate",
    ]

    results: Dict[str, Dict[str, List[float]]] = {}
    for featset_name in feature_sets:
        for cfg in model_configs:
            results[f"{featset_name} - {cfg['name']}"] = {m: [] for m in metrics}

    for seed in range(N):
        train_df, test_df = train_test_split(
            df, test_size=0.3, random_state=seed, stratify=df[label]
        )
        # Join survival for time/event columns
        tr = train_df.merge(survival_df[["MRN", time_col, event_col]], on="MRN", how="left")
        te = test_df.merge(survival_df[["MRN", time_col, event_col]], on="MRN", how="left")
        tr = tr.dropna(subset=[time_col, event_col])
        te = te.dropna(subset=[time_col, event_col])

        for featset_name, feature_cols in feature_sets.items():
            for cfg in model_configs:
                name = f"{featset_name} - {cfg['name']}"
                use_undersampling = (cfg["mode"] == "sex_agnostic")
                pred, risk, met = evaluate_split(
                    tr, te, feature_cols, label, time_col, event_col,
                    mode=cfg["mode"], seed=seed, use_undersampling=use_undersampling
                )
                eval_df = te[[label, "Female"]].reset_index(drop=True).copy()
                eval_df["pred"] = pred
                m_rate, f_rate = compute_incidence_rate(eval_df, "pred", label)
                results[name]["c_index"].append(met["c_index"]) 
                results[name]["male_rate"].append(m_rate)
                results[name]["female_rate"].append(f_rate)

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
        }, axis=0,
    )

    formatted = summary_df.apply(
        lambda row: f"{row['mean']:.3f} ({row['ci_lower']:.3f}, {row['ci_upper']:.3f})",
        axis=1,
    )
    summary_table = formatted.unstack(level=1)

    if export_excel_path is not None:
        os.makedirs(os.path.dirname(export_excel_path), exist_ok=True)
        summary_table.to_excel(export_excel_path, index=True, index_label="RowName")

    return results, summary_table


# Feature sets (mirroring a.py)
FEATURE_SETS = {
    'guideline': ["NYHA Class", "LVEF"],
    'benchmark': [
        "Female", "Age by decade", "BMI", "AF", "Beta Blocker", "CrCl>45",
        "LVEF", "QTc", "NYHA>2", "CRT", "AAD", "Significant LGE",
    ],
    'proposed': [
        "Female", "Age by decade", "BMI", "AF", "Beta Blocker", "CrCl>45",
        "LVEF", "QTc", "NYHA>2", "CRT", "AAD", "Significant LGE", "DM", "HTN",
        "HLP", "LVEDVi", "LV Mass Index", "RVEDVi", "RVEF", "LA EF", "LAVi",
        "MRF (%)", "Sphericity Index", "Relative Wall Thickness",
        "MV Annular Diameter", "ACEi/ARB/ARNi", "Aldosterone Antagonist",
    ],
    'real_proposed': [
        "Female", "Age by decade", "BMI", "AF", "Beta Blocker", "CrCl>45",
        "LVEF", "QTc", "CRT", "AAD", "LGE Burden 5SD", "DM", "HTN",
        "HLP", "LVEDVi", "LV Mass Index", "RVEDVi", "RVEF", "LA EF", "LAVi",
        "MRF (%)", "Sphericity Index", "Relative Wall Thickness",
        "MV Annular Diameter", "ACEi/ARB/ARNi", "Aldosterone Antagonist", "NYHA Class"
    ],
}


if __name__ == "__main__":
    # Load and prepare data
    df, survival_df = load_dataframes()

    # Run experiments (50 random splits; PE as event; PE_Time as duration)
    export_path = "/home/sunx/data/aiiih/projects/sunx/projects/ICD_sex_diff/results_cox.xlsx"
    _, summary = run_cox_experiments(
        df=df,
        survival_df=survival_df,
        feature_sets=FEATURE_SETS,
        N=50,
        label="VT/VF/SCD",
        time_col="PE_Time",
        event_col="PE",
        export_excel_path=export_path,
    )
    print("Saved Excel:", export_path)