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
    """Plot KM curves for 4 groups: Male-Pred0, Male-Pred1, Female-Pred0, Female-Pred1, with log-rank tests."""
    if merged_df.empty:
        return
    kmf = KaplanMeierFitter()
    groups = [
        (0, 0, "Male-Pred0", "blue"),
        (0, 1, "Male-Pred1", "red"),
        (1, 0, "Female-Pred0", "lightblue"),
        (1, 1, "Female-Pred1", "pink"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for ax, (ep_name, ep_time_col, ep_event_col) in zip(
        axes,
        [
            ("Primary Endpoint", "PE_Time", "PE"),
            ("Secondary Endpoint", "SE_Time", "SE"),
        ],
    ):
        for gender_val, pred_val, group_name, color in groups:
            mask = (merged_df["Female"] == gender_val) & (
                merged_df["pred_label"] == pred_val
            )
            group_data = merged_df[mask]
            if group_data.empty:
                continue
            n_samples = len(group_data)
            events = group_data[ep_event_col].sum()
            label = f"{group_name} (n={n_samples}, events={events})"
            kmf.fit(
                durations=group_data[ep_time_col],
                event_observed=group_data[ep_event_col],
                label=label,
            )
            kmf.plot(ax=ax, color=color)
        # Pairwise within-sex logrank
        male_pred0 = merged_df[
            (merged_df["Female"] == 0) & (merged_df["pred_label"] == 0)
        ]
        male_pred1 = merged_df[
            (merged_df["Female"] == 0) & (merged_df["pred_label"] == 1)
        ]
        female_pred0 = merged_df[
            (merged_df["Female"] == 1) & (merged_df["pred_label"] == 0)
        ]
        female_pred1 = merged_df[
            (merged_df["Female"] == 1) & (merged_df["pred_label"] == 1)
        ]
        if not male_pred0.empty and not male_pred1.empty:
            lr_male = logrank_test(
                male_pred0[ep_time_col],
                male_pred1[ep_time_col],
                male_pred0[ep_event_col],
                male_pred1[ep_event_col],
            )
            ax.text(0.02, 0.02, f"Male p={lr_male.p_value:.4f}", transform=ax.transAxes)
        if not female_pred0.empty and not female_pred1.empty:
            lr_female = logrank_test(
                female_pred0[ep_time_col],
                female_pred1[ep_time_col],
                female_pred0[ep_event_col],
                female_pred1[ep_event_col],
            )
            ax.text(
                0.02, 0.08, f"Female p={lr_female.p_value:.4f}", transform=ax.transAxes
            )
        ax.set_title(f"{ep_name} - Survival by Gender and Prediction")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Survival Probability")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==========================================
# CoxPH training/inference blocks
# ==========================================


def fit_cox_model(
    train_df: pd.DataFrame, feature_cols: List[str], time_col: str, event_col: str
) -> CoxPHFitter:
    df_fit = train_df[[time_col, event_col] + feature_cols].copy()
    
    # Check for minimum sample size and events
    if len(df_fit) < 10 or df_fit[event_col].sum() < 5:
        raise ValueError(f"Insufficient data: {len(df_fit)} samples, {df_fit[event_col].sum()} events")
    
    # Remove features with no variance or all missing
    valid_features = []
    for col in feature_cols:
        if col in df_fit.columns:
            if df_fit[col].var() > 1e-8 and not df_fit[col].isna().all():
                valid_features.append(col)
    
    if not valid_features:
        raise ValueError("No valid features for Cox model")
    
    # Standardize features to improve convergence
    scaler = StandardScaler()
    df_fit_scaled = df_fit.copy()
    df_fit_scaled[valid_features] = scaler.fit_transform(df_fit[valid_features].fillna(0))
    
    # Use penalization to improve stability
    cph = CoxPHFitter(penalizer=0.01)
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph.fit(
                df_fit_scaled[[time_col, event_col] + valid_features], 
                duration_col=time_col, 
                event_col=event_col, 
                robust=True,
                step_size=0.5  # Smaller step size for better convergence
            )
        # Store scaler and valid features for prediction
        cph._scaler = scaler
        cph._valid_features = valid_features
        return cph
        
    except Exception as e:
        # Fallback with higher penalization
        print(f"Cox model failed with error: {e}")
        print("Trying with higher penalization...")
        try:
            cph_backup = CoxPHFitter(penalizer=0.1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph_backup.fit(
                    df_fit_scaled[[time_col, event_col] + valid_features], 
                    duration_col=time_col, 
                    event_col=event_col, 
                    robust=True,
                    step_size=0.1
                )
            cph_backup._scaler = scaler
            cph_backup._valid_features = valid_features
            return cph_backup
        except Exception as e2:
            print(f"Backup Cox model also failed: {e2}")
            raise e2


def predict_risk(
    model: CoxPHFitter, df: pd.DataFrame, feature_cols: List[str]
) -> np.ndarray:
    # Use the valid features that were used in training
    if hasattr(model, '_valid_features'):
        valid_features = model._valid_features
        scaler = model._scaler
    else:
        # Fallback for models without stored features
        valid_features = feature_cols
        scaler = None
    
    X = df[valid_features].copy().fillna(0)
    
    # Apply the same scaling as training
    if scaler is not None:
        X_scaled = pd.DataFrame(
            scaler.transform(X), 
            columns=valid_features, 
            index=X.index
        )
        risk = model.predict_partial_hazard(X_scaled).values.reshape(-1)
    else:
        risk = model.predict_partial_hazard(X).values.reshape(-1)
    
    return risk


def threshold_by_top_quantile(risk_scores: np.ndarray, quantile: float = 0.5) -> float:
    q = np.nanquantile(risk_scores, quantile)
    return float(q)


def sex_specific_inference(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    survival_df: pd.DataFrame,
    gray_features: List[str] = None,
    red_features: List[str] = None,
) -> pd.DataFrame:
    """Sex-specific CoxPH inference. Fits separate Cox models by sex on training survival data, predicts risk on test,
    dichotomizes by median training risk within sex, plots KM and Cox coefficients, and returns merged_df.
    """
    used_features = [f for f in features if f != "Female"]
    train_m = (
        train_df[train_df["Female"] == 0]
        .merge(
            survival_df[["MRN", "PE_Time", "PE", "SE_Time", "SE"]],
            on="MRN",
            how="inner",
        )
        .dropna(subset=["PE_Time", "PE"])
    )
    train_f = (
        train_df[train_df["Female"] == 1]
        .merge(
            survival_df[["MRN", "PE_Time", "PE", "SE_Time", "SE"]],
            on="MRN",
            how="inner",
        )
        .dropna(subset=["PE_Time", "PE"])
    )
    test_m = test_df[test_df["Female"] == 0].copy()
    test_f = test_df[test_df["Female"] == 1].copy()

    models = {}
    thresholds = {}

    if not train_m.empty:
        cph_m = fit_cox_model(train_m, used_features, "PE_Time", "PE")
        tr_risk_m = predict_risk(cph_m, train_m, used_features)
        thresholds["male"] = float(np.nanmedian(tr_risk_m))
        models["male"] = cph_m
        plot_cox_coefficients(
            cph_m, "Male Cox Coefficients (log HR)", gray_features, red_features
        )
    if not train_f.empty:
        cph_f = fit_cox_model(train_f, used_features, "PE_Time", "PE")
        tr_risk_f = predict_risk(cph_f, train_f, used_features)
        thresholds["female"] = float(np.nanmedian(tr_risk_f))
        models["female"] = cph_f
        plot_cox_coefficients(
            cph_f, "Female Cox Coefficients (log HR)", gray_features, red_features
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

    pred_labels = df_out[["MRN", "pred_label", "Female"]].drop_duplicates()
    merged_df = survival_df.merge(pred_labels, on="MRN", how="inner").drop_duplicates(
        subset=["MRN"]
    )
    plot_km_curves_four_groups(merged_df)
    return merged_df


def sex_agnostic_inference(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    survival_df: pd.DataFrame,
    label_col: str = "VT/VF/SCD",
    use_undersampling: bool = True,
    gray_features: List[str] = None,
    red_features: List[str] = None,
) -> pd.DataFrame:
    """Sex-agnostic CoxPH inference. Removes Female from features, optionally undersamples, fits Cox on training survival,
    predicts test risk, dichotomizes by training median risk, plots KM and Cox coefficients, returns merged_df.
    """
    used_features = [f for f in features if f != "Female"]
    tr_base = (
        create_undersampled_dataset(train_df, label_col, 42)
        if use_undersampling
        else train_df
    )
    tr = tr_base.merge(
        survival_df[["MRN", "PE_Time", "PE", "SE_Time", "SE"]], on="MRN", how="inner"
    ).dropna(subset=["PE_Time", "PE"])
    if tr.empty:
        return test_df.copy()

    cph = fit_cox_model(tr, used_features, "PE_Time", "PE")
    plot_cox_coefficients(
        cph, "Sex-Agnostic Cox Coefficients (log HR)", gray_features, red_features
    )
    tr_risk = predict_risk(cph, tr, used_features)
    thr = float(np.nanmedian(tr_risk))

    te = test_df.copy()
    te_risk = predict_risk(cph, te, used_features)
    te["pred_prob"] = te_risk
    te["pred_label"] = (te_risk >= thr).astype(int)

    pred_labels = te[["MRN", "pred_label", "Female"]].drop_duplicates()
    merged_df = survival_df.merge(pred_labels, on="MRN", how="inner").drop_duplicates(
        subset=["MRN"]
    )
    plot_km_curves_four_groups(merged_df)
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
        used_features = [f for f in feature_cols if f != "Female"]
        tr = (
            create_undersampled_dataset(train_df, event_col, seed)
            if use_undersampling
            else train_df
        )
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
        used_features = feature_cols
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

    # Data validation and cleaning
    print(f"Dataset shape before validation: {clean_df.shape}")
    
    # Remove rows with missing critical survival data
    critical_cols = ["PE_Time", "VT/VF/SCD"]
    initial_rows = len(clean_df)
    clean_df = clean_df.dropna(subset=critical_cols)
    dropped_survival = initial_rows - len(clean_df)
    if dropped_survival > 0:
        print(f"Dropped {dropped_survival} rows with missing survival data")
    
    # Remove rows with invalid survival times
    clean_df = clean_df[clean_df["PE_Time"] > 0]
    dropped_time = len(clean_df) - (initial_rows - dropped_survival)
    if dropped_time < 0:  # means we dropped more
        print(f"Dropped {abs(dropped_time)} rows with invalid survival times")
    
    # Check for minimum events
    n_events = clean_df["VT/VF/SCD"].sum()
    print(f"Total events: {n_events} out of {len(clean_df)} samples ({n_events/len(clean_df)*100:.1f}%)")
    
    if n_events < 10:
        print("WARNING: Very few events (<10) may cause convergence issues")
    
    # Check for extreme outliers in continuous features and cap them
    numeric_cols = clean_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ["MRN", "Female", "VT/VF/SCD", "ICD", "PE_Time"] and col in clean_df.columns:
            Q1 = clean_df[col].quantile(0.01)
            Q99 = clean_df[col].quantile(0.99)
            outliers = ((clean_df[col] < Q1) | (clean_df[col] > Q99)).sum()
            if outliers > 0:
                clean_df[col] = clean_df[col].clip(Q1, Q99)
                print(f"Capped {outliers} outliers in {col}")
    
    print(f"Final dataset shape: {clean_df.shape}")
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
    metrics = ["c_index"]

    results: Dict[str, Dict[str, List[float]]] = {}
    for featset_name in feature_sets:
        for cfg in model_configs:
            results[f"{featset_name} - {cfg['name']}"] = {m: [] for m in metrics}

    for seed in range(N):
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
                eval_df = te[[event_col, "Female"]].reset_index(drop=True).copy()
                eval_df["pred"] = pred
                results[name]["c_index"].append(met["c_index"])

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
        N=2,
        time_col="PE_Time",
        event_col="VT/VF/SCD",
        export_excel_path=export_path,
    )
    print("Saved Excel:", export_path)
    
    # Run inference using the loaded data
    print("\n" + "="*50)
    print("Running Inference Examples")
    print("="*50)
    
    # Split data for inference demonstration
    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["VT/VF/SCD"]
    )
    
    # Create a survival dataframe for inference functions
    survival_df = df[["MRN", "PE_Time", "VT/VF/SCD", "Female"]].copy()
    survival_df = survival_df.rename(columns={"VT/VF/SCD": "PE"})
    # Add secondary endpoint (using same as primary for demo)
    survival_df["SE_Time"] = survival_df["PE_Time"]
    survival_df["SE"] = survival_df["PE"]
    
    # Example 1: Sex-specific inference
    print("\n1. Sex-specific inference with Benchmark features:")
    try:
        benchmark_features = FEATURE_SETS["Benchmark"]
        merged_df_sex_specific = sex_specific_inference(
            train_df=train_df,
            test_df=test_df, 
            features=benchmark_features,
            survival_df=survival_df,
            gray_features=["Female", "Age by decade"],
            red_features=["LVEF", "QTc"]
        )
        print(f"Sex-specific inference completed. Merged dataframe shape: {merged_df_sex_specific.shape}")
    except Exception as e:
        print(f"Sex-specific inference failed: {e}")
    
    # Example 2: Sex-agnostic inference  
    print("\n2. Sex-agnostic inference with Proposed features:")
    try:
        proposed_features = FEATURE_SETS["Proposed"] 
        merged_df_sex_agnostic = sex_agnostic_inference(
            train_df=train_df,
            test_df=test_df,
            features=proposed_features,
            survival_df=survival_df,
            label_col="VT/VF/SCD",
            use_undersampling=True,
            gray_features=["Female", "Age by decade", "BMI"],
            red_features=["LVEF", "QTc", "LGE Burden 5SD"]
        )
        print(f"Sex-agnostic inference completed. Merged dataframe shape: {merged_df_sex_agnostic.shape}")
    except Exception as e:
        print(f"Sex-agnostic inference failed: {e}")
        
    print("\nInference examples completed!")
