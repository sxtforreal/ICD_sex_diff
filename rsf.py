from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import argparse
import os
import warnings

import numpy as np
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)

from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv


# =============================
# Minimal progress logging utils
# =============================

PROGRESS = True


def _log(msg: str) -> None:
    if PROGRESS:
        print(f"[Progress] {msg}")


# =============================
# Feature groups (edit if needed)
# =============================

# These names match the original dataset columns. You can edit them if necessary.
GLOBAL_FEATURES: List[str] = [
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
    "LGE_LGE Burden 5SD",
]

LOCAL_FEATURES: List[str] = [
    "LGE_Circumural",
    "LGE_Ring-Like",
    "LGE_Basal anterior  (0; No, 1; yes)",
    "LGE_Basal anterior septum",
    "LGE_Basal inferoseptum",
    "LGE_Basal inferio",
    "LGE_Basal inferolateral",
    "LGE_Basal anterolateral",
    "LGE_mid anterior",
    "LGE_mid anterior septum",
    "LGE_mid inferoseptum",
    "LGE_mid inferior",
    "LGE_mid inferolateral",
    "LGE_mid anterolateral",
    "LGE_apical anterior",
    "LGE_apical septum",
    "LGE_apical inferior",
    "LGE_apical lateral",
    "LGE_Apical cap",
    "LGE_RV insertion site (1 superior, 2 inferior. 3 both)",
]

GLOBAL_FEATURES = [str(c).strip() for c in GLOBAL_FEATURES]
LOCAL_FEATURES = [str(c).strip() for c in LOCAL_FEATURES]


# =============================
# 数据清洗（沿用原实现）
# =============================


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
        "LGE_Circumural",
        "LGE_Ring-Like",
        "ICD",
    ]
    exist_bin = [c for c in binary_cols if c in df.columns]
    for c in exist_bin:
        if df[c].dtype == "object":
            _tmp = df[c].replace(
                {"Yes": 1, "No": 0, "Y": 1, "N": 0, "True": 1, "False": 0}
            )
            try:
                df[c] = _tmp.infer_objects(copy=False)
            except Exception:
                df[c] = _tmp
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Imputation on feature matrix (simple: median for numeric)
    X = df[features].copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

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


def load_dataframes(
    data_file: Optional[str] = None,
    sheet_icd: str = "ICD",
    sheet_no_icd: str = "No_ICD",
) -> pd.DataFrame:
    """
    读取原始 Excel, 合并 ICD/No_ICD, 计算随访时间, 丢弃无用列, 并做基本类型规范化与简单插补。

    注意：默认路径保持与旧脚本一致，可通过 --data-file 覆盖。
    """
    if data_file is None:
        base = "/home/sunx/data/aiiih/projects/sunx/projects/ICD"
        data_file = os.path.join(base, "LGE granularity.xlsx")

    icd = pd.read_excel(data_file, sheet_name=sheet_icd)
    noicd = pd.read_excel(data_file, sheet_name=sheet_no_icd)

    # Stack dfs
    common_cols = icd.columns.intersection(noicd.columns)
    icd_common = icd[common_cols].copy()
    noicd_common = noicd[common_cols].copy()
    icd_common["ICD"] = 1
    noicd_common["ICD"] = 0
    nicm = pd.concat([icd_common, noicd_common], ignore_index=True)

    # Normalize column names to avoid trailing spaces mismatch
    nicm.columns = nicm.columns.str.strip()

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

    # Ensure non-negative follow-up times
    nicm["PE_Time"] = pd.to_numeric(nicm["PE_Time"], errors="coerce").clip(lower=0)

    # Drop features (non-modelling columns)
    nicm.drop(
        [
            "MRI Date",
            "Date VT/VF/SCD",
            "End follow-up date",
            "CRT Date",
            "LGE Burden 5SD",
            "LGE_Unnamed: 1",
            "LGE_Notes",
            "LGE_RV insertion sites (0 No, 1 yes)",
            "LGE_Score",
            "LGE_Unnamed: 27",
            "LGE_Unnamed: 28",
            "Cockcroft-Gault Creatinine Clearance (mL/min)",
        ],
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
        "LGE_Extent (1; subendocardial, 2; mid mural, 3; epicardial, 4; transmural; 5 circumural)",
        "LGE_Circumural",
        "LGE_Ring-Like",
        "LGE_Basal anterior  (0; No, 1; yes)",
        "LGE_Basal anterior septum",
        "LGE_Basal inferoseptum",
        "LGE_Basal inferio",
        "LGE_Basal inferolateral",
        "LGE_Basal anterolateral",
        "LGE_mid anterior",
        "LGE_mid anterior septum",
        "LGE_mid inferoseptum",
        "LGE_mid inferior",
        "LGE_mid inferolateral",
        "LGE_mid anterolateral",
        "LGE_apical anterior",
        "LGE_apical septum",
        "LGE_apical inferior",
        "LGE_apical lateral",
        "LGE_Apical cap",
        "LGE_RV insertion site (1 superior, 2 inferior. 3 both)",
        "ICD",
    ]
    nicm[categorical] = nicm[categorical].astype("object")
    var = nicm.columns.tolist()
    labels = ["MRN", "VT/VF/SCD", "ICD", "PE_Time", "Female"]
    granularity = [
        "LGE_LGE Burden 5SD",
        "LGE_Extent (1; subendocardial, 2; mid mural, 3; epicardial, 4; transmural; 5 circumural)",
        "LGE_Circumural",
        "LGE_Ring-Like",
        "LGE_Basal anterior  (0; No, 1; yes)",
        "LGE_Basal anterior septum",
        "LGE_Basal inferoseptum",
        "LGE_Basal inferio",
        "LGE_Basal inferolateral",
        "LGE_Basal anterolateral",
        "LGE_mid anterior",
        "LGE_mid anterior septum",
        "LGE_mid inferoseptum",
        "LGE_mid inferior",
        "LGE_mid inferolateral",
        "LGE_mid anterolateral",
        "LGE_apical anterior",
        "LGE_apical septum",
        "LGE_apical inferior",
        "LGE_apical lateral",
        "LGE_Apical cap",
        "LGE_RV insertion site (1 superior, 2 inferior. 3 both)",
    ]
    nicm = nicm.dropna(subset=granularity)
    features = [v for v in var if v not in labels]
    # Explicitly treat Female and ICD as labels (remove from feature list)
    for lab in ["Female", "ICD"]:
        if lab in features:
            features.remove(lab)

    # Imputation
    clean_df = conversion_and_imputation(nicm, features, labels)
    clean_df["NYHA Class"] = clean_df["NYHA Class"].replace({5: 4, 0: 1})
    return clean_df


# =============================
# Survival utilities
# =============================


def _prepare_survival_xy(
    clean_df: pd.DataFrame, drop_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    if drop_cols is None:
        drop_cols = ["MRN", "VT/VF/SCD", "ICD", "PE_Time", "Female"]

    df = clean_df.copy()
    df = df.dropna(subset=["PE_Time"]).copy()
    if not pd.api.types.is_numeric_dtype(df["PE_Time"]):
        df["PE_Time"] = pd.to_numeric(df["PE_Time"], errors="coerce")
        df = df.dropna(subset=["PE_Time"]).copy()
    df["PE_Time"] = df["PE_Time"].astype(float).clip(lower=0.0)
    df["VT/VF/SCD"] = df["VT/VF/SCD"].fillna(0).astype(int).astype(bool)

    X = df.drop(columns=drop_cols, errors="ignore")
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
        if X[c].isnull().any():
            X[c] = X[c].fillna(X[c].median())
    X = X.astype(float)
    feature_names = X.columns.tolist()
    y = Surv.from_dataframe(event="VT/VF/SCD", time="PE_Time", data=df)
    return X, y, feature_names


def _event_by_horizon(y: np.ndarray, t0: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (label_at_t0, known_mask).

    label_at_t0 is 1 if event happened by t0, else 0. If censored before t0, mark unknown.
    """
    names = getattr(y.dtype, "names", ("event", "time"))
    e = y[names[0]].astype(bool)
    t = y[names[1]].astype(float)
    # unknown if censored before t0
    unknown = (~e) & (t < t0)
    known = ~unknown
    label = (e & (t <= t0)).astype(float)
    label[~known] = np.nan
    return label, known


def _risk_at_time(
    model: CoxPHSurvivalAnalysis, X: pd.DataFrame, t0: float
) -> np.ndarray:
    if model is None or X is None or len(X) == 0:
        return np.zeros(0, dtype=float)
    try:
        survf = model.predict_survival_function(X)
        s_at = np.array([sf(t0) for sf in survf], dtype=float)
        s_at = np.clip(s_at, 0.0, 1.0)
        risk = 1.0 - s_at
        return np.clip(risk, 0.0, 1.0)
    except Exception:
        try:
            scores = np.asarray(model.predict(X), dtype=float).ravel()
            finite = np.isfinite(scores)
            if finite.any():
                s_min = float(np.nanmin(scores[finite]))
                s_max = float(np.nanmax(scores[finite]))
                if s_max > s_min:
                    return (scores - s_min) / (s_max - s_min)
        except Exception:
            pass
    return np.zeros(len(X), dtype=float)


def _fit_cox(X: pd.DataFrame, y: np.ndarray) -> Optional[CoxPHSurvivalAnalysis]:
    if X is None or X.shape[0] == 0 or X.shape[1] == 0:
        return None
    # Defensive: drop columns with zero variance
    nunique = X.nunique(dropna=True)
    keep = nunique[nunique > 1].index.tolist()
    X2 = X[keep].copy()
    try:
        model = CoxPHSurvivalAnalysis(alpha=0.0)
        model.fit(X2, y)
        return model
    except Exception as ex:
        warnings.warn(f"Cox fit failed: {ex}")
        return None


# =============================
# Step 1: K 折 OOF 三模型损失
# =============================


def compute_oof_losses(
    clean_df: pd.DataFrame,
    global_cols: List[str],
    local_cols: List[str],
    horizon_days: float,
    k_folds: int = 5,
    repeats: int = 1,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    X_all, y_all, feat_names = _prepare_survival_xy(clean_df)
    all_cols = [c for c in feat_names]
    global_cols = [c for c in global_cols if c in all_cols]
    local_cols = [c for c in local_cols if c in all_cols]
    if len(global_cols) == 0 or len(local_cols) == 0:
        raise ValueError("GLOBAL_FEATURES 或 LOCAL_FEATURES 为空或与数据不匹配。")

    n = len(X_all)
    # Collect R repetitions
    losses_local_rep: List[np.ndarray] = []
    losses_global_rep: List[np.ndarray] = []
    losses_all_rep: List[np.ndarray] = []
    risks_local_rep: List[np.ndarray] = []
    risks_global_rep: List[np.ndarray] = []
    risks_all_rep: List[np.ndarray] = []

    labels_t0, known_mask = _event_by_horizon(y_all, horizon_days)

    for r in range(max(1, int(repeats))):
        rng = int(random_state + r)
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=rng)
        L_lo = np.full(n, np.nan, dtype=float)
        L_gl = np.full(n, np.nan, dtype=float)
        L_all = np.full(n, np.nan, dtype=float)
        R_lo = np.full(n, np.nan, dtype=float)
        R_gl = np.full(n, np.nan, dtype=float)
        R_all = np.full(n, np.nan, dtype=float)

        for tr_idx, va_idx in kf.split(X_all):
            X_tr, X_va = X_all.iloc[tr_idx], X_all.iloc[va_idx]
            y_tr, y_va = y_all[tr_idx], y_all[va_idx]

            # Fit three Cox models
            mdl_lo = _fit_cox(X_tr[local_cols], y_tr)
            mdl_gl = _fit_cox(X_tr[global_cols], y_tr)
            mdl_all = _fit_cox(X_tr[all_cols], y_tr)

            # Predict risk@t0
            risk_lo = (
                _risk_at_time(mdl_lo, X_va[local_cols], horizon_days)
                if mdl_lo
                else np.zeros(len(va_idx))
            )
            risk_gl = (
                _risk_at_time(mdl_gl, X_va[global_cols], horizon_days)
                if mdl_gl
                else np.zeros(len(va_idx))
            )
            risk_all = (
                _risk_at_time(mdl_all, X_va[all_cols], horizon_days)
                if mdl_all
                else np.zeros(len(va_idx))
            )

            # Per-sample Brier-like loss
            lab_va = labels_t0[va_idx]
            known_va = known_mask[va_idx]
            L_lo[va_idx] = (lab_va - risk_lo) ** 2
            L_gl[va_idx] = (lab_va - risk_gl) ** 2
            L_all[va_idx] = (lab_va - risk_all) ** 2
            # Mask unknown samples
            L_lo[va_idx[~known_va]] = np.nan
            L_gl[va_idx[~known_va]] = np.nan
            L_all[va_idx[~known_va]] = np.nan

            R_lo[va_idx] = risk_lo
            R_gl[va_idx] = risk_gl
            R_all[va_idx] = risk_all

        losses_local_rep.append(L_lo)
        losses_global_rep.append(L_gl)
        losses_all_rep.append(L_all)
        risks_local_rep.append(R_lo)
        risks_global_rep.append(R_gl)
        risks_all_rep.append(R_all)

    # Average across repeats when available
    def _nanmean_stack(arrs: List[np.ndarray]) -> np.ndarray:
        if len(arrs) == 1:
            return arrs[0]
        A = np.stack(arrs, axis=0)
        return np.nanmean(A, axis=0)

    return {
        "loss_local": _nanmean_stack(losses_local_rep),
        "loss_global": _nanmean_stack(losses_global_rep),
        "loss_all": _nanmean_stack(losses_all_rep),
        "risk_local": _nanmean_stack(risks_local_rep),
        "risk_global": _nanmean_stack(risks_global_rep),
        "risk_all": _nanmean_stack(risks_all_rep),
        "known_mask": known_mask,
        "labels_t0": labels_t0,
        "X": X_all,
        "y": y_all,
        "global_cols": global_cols,
        "local_cols": local_cols,
        "all_cols": all_cols,
    }


# =============================
# Step 2 & 3: 硬阈值选择 + meta-label 生成
# =============================


def apply_hard_threshold_rule(
    loss_local: np.ndarray,
    loss_global: np.ndarray,
    loss_all: np.ndarray,
    alpha_abs: float = 0.005,
    alpha_rel: float = 0.0,
    lambda_penalty: float = 0.0,
    complexity: Optional[Dict[str, float]] = None,
    prefer_between_local_global: bool = True,
) -> Dict[str, np.ndarray]:
    """
    返回基于三元 OOF 损失的严格标签 z_i ∈ {local, global, all} 以及 margin。

    - alpha_abs: 绝对最小改进阈值 (L_second - L_best >= alpha_abs)
    - alpha_rel: 相对改进阈值 ((L_second - L_best)/L_second >= alpha_rel)
    - lambda_penalty: 复杂度惩罚系数；对每个模型加 λ*C_m
    - complexity: {"local":C_l, "global":C_g, "all":C_a}
    - prefer_between_local_global: 当改进不足时，只在 local/global 中二选一
    """
    if complexity is None:
        complexity = {"local": 1.0, "global": 1.0, "all": 1.2}

    L_lo = loss_local.copy()
    L_gl = loss_global.copy()
    L_all = loss_all.copy()

    # Complexity penalty
    L_lo = L_lo + lambda_penalty * complexity.get("local", 1.0)
    L_gl = L_gl + lambda_penalty * complexity.get("global", 1.0)
    L_all = L_all + lambda_penalty * complexity.get("all", 1.0)

    n = len(L_lo)
    labels = np.full(n, fill_value=-1, dtype=int)
    margins = np.full(n, fill_value=np.nan, dtype=float)

    stack = np.vstack([L_lo, L_gl, L_all])  # rows: 0 local, 1 global, 2 all
    for i in range(n):
        li = stack[:, i]
        if np.any(~np.isfinite(li)):
            # If any nan among three, skip this sample
            if np.sum(np.isfinite(li)) < 2:
                continue
        order = np.argsort(li)
        best = order[0]
        second = order[1]
        Lb = li[best]
        L2 = li[second]
        margin = float(L2 - Lb)
        margins[i] = margin

        cond_abs = margin >= alpha_abs
        cond_rel = (margin / max(L2, 1e-12)) >= alpha_rel
        if cond_abs and cond_rel:
            labels[i] = int(best)
        else:
            if prefer_between_local_global:
                pair = np.nanargmin([li[0], li[1]])
                labels[i] = int(pair)
            else:
                labels[i] = int(best)

    # 0->local,1->global,2->all
    return {"labels": labels, "margins": margins}


# =============================
# Step 4: 学 gating（离散三分类）
# =============================


def train_gating_classifier(
    X: pd.DataFrame,
    labels: np.ndarray,
    k_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, object]:
    mask = labels >= 0
    Xy = X[mask]
    y = labels[mask]

    if Xy.shape[0] == 0:
        raise ValueError("没有可用于训练 gating 的样本标签。")

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    multi_class="auto",
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )

    # OOF evaluation of gating itself
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    oof_pred = np.full(len(Xy), fill_value=-1, dtype=int)
    for tr, va in kf.split(Xy):
        pipe.fit(Xy.iloc[tr], y[tr])
        oof_pred[va] = pipe.predict(Xy.iloc[va])

    acc = accuracy_score(y, oof_pred)
    f1m = f1_score(y, oof_pred, average="macro")
    rep = classification_report(y, oof_pred, digits=4)

    # Fit on all labelled data for downstream use
    pipe.fit(Xy, y)

    return {
        "model": pipe,
        "oof_acc": float(acc),
        "oof_f1_macro": float(f1m),
        "report": rep,
        "mask": mask,
    }


# =============================
# Step 5: 最终评估（独立测试集）
# =============================


def evaluate_on_holdout(
    clean_df: pd.DataFrame,
    global_cols: List[str],
    local_cols: List[str],
    horizon_days: float,
    test_size: float = 0.25,
    random_state: int = 42,
    k_folds: int = 5,
    repeats: int = 1,
    alpha_abs: float = 0.005,
    alpha_rel: float = 0.0,
    lambda_penalty: float = 0.0,
) -> Dict[str, object]:
    X_all, y_all, feat_names = _prepare_survival_xy(clean_df)
    all_cols = [c for c in feat_names]
    global_cols = [c for c in global_cols if c in all_cols]
    local_cols = [c for c in local_cols if c in all_cols]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=test_size, random_state=random_state
    )

    # Build a DataFrame back to reuse OOF pipeline conveniently
    df_tr = X_tr.copy()
    # Robustly detect field names of structured array y_tr/y_te
    e_field_tr = "event" if "event" in y_tr.dtype.names else y_tr.dtype.names[0]
    t_field_tr = "time" if "time" in y_tr.dtype.names else y_tr.dtype.names[1]
    e_field_te = "event" if "event" in y_te.dtype.names else y_te.dtype.names[0]
    t_field_te = "time" if "time" in y_te.dtype.names else y_te.dtype.names[1]
    df_tr["VT/VF/SCD"] = y_tr[e_field_tr].astype(int)
    df_tr["PE_Time"] = y_tr[t_field_tr].astype(float)
    df_te = X_te.copy()
    df_te["VT/VF/SCD"] = y_te[e_field_te].astype(int)
    df_te["PE_Time"] = y_te[t_field_te].astype(float)

    _log("Computing OOF losses on training set for meta-labels ...")
    oof_tr = compute_oof_losses(
        clean_df=df_tr,
        global_cols=global_cols,
        local_cols=local_cols,
        horizon_days=horizon_days,
        k_folds=k_folds,
        repeats=repeats,
        random_state=random_state,
    )

    _log("Generating strict meta-labels by hard-threshold rule ...")
    lab_res = apply_hard_threshold_rule(
        oof_tr["loss_local"],
        oof_tr["loss_global"],
        oof_tr["loss_all"],
        alpha_abs=alpha_abs,
        alpha_rel=alpha_rel,
        lambda_penalty=lambda_penalty,
    )

    _log("Training gating classifier on training set ...")
    gating = train_gating_classifier(
        oof_tr["X"], lab_res["labels"], k_folds=k_folds, random_state=random_state
    )

    # Fit base models on full training set
    _log("Fitting base models on the full training set ...")
    mdl_local = _fit_cox(X_tr[local_cols], y_tr)
    mdl_global = _fit_cox(X_tr[global_cols], y_tr)
    mdl_all = _fit_cox(X_tr[all_cols], y_tr)

    # Predict on test set via learned gating
    _log("Applying gating to test set and computing metrics ...")
    gate_mask = gating["mask"]
    gate_model: Pipeline = gating["model"]  # type: ignore

    # For test, gating uses all features
    gate_pred = gate_model.predict(
        df_te.drop(columns=["VT/VF/SCD", "PE_Time"], errors="ignore")
    )

    risk_lo_te = (
        _risk_at_time(mdl_local, X_te[local_cols], horizon_days)
        if mdl_local
        else np.zeros(len(X_te))
    )
    risk_gl_te = (
        _risk_at_time(mdl_global, X_te[global_cols], horizon_days)
        if mdl_global
        else np.zeros(len(X_te))
    )
    risk_all_te = (
        _risk_at_time(mdl_all, X_te[all_cols], horizon_days)
        if mdl_all
        else np.zeros(len(X_te))
    )

    # Compose mixture by gating decision: 0 local, 1 global, 2 all
    mix_risk = np.zeros(len(X_te), dtype=float)
    for i in range(len(X_te)):
        z = int(gate_pred[i])
        if z == 0:
            mix_risk[i] = risk_lo_te[i]
        elif z == 1:
            mix_risk[i] = risk_gl_te[i]
        else:
            mix_risk[i] = risk_all_te[i]

    # Metrics on test
    e_field = "event" if "event" in y_te.dtype.names else y_te.dtype.names[0]
    t_field = "time" if "time" in y_te.dtype.names else y_te.dtype.names[1]
    c_index = float(
        concordance_index_censored(
            y_te[e_field].astype(bool), y_te[t_field].astype(float), mix_risk
        )[0]
    )
    # Brier-like at t0 (without IPCW)
    lab_t0, known = _event_by_horizon(y_te, horizon_days)
    brier_mix = float(np.nanmean((lab_t0 - mix_risk) ** 2))

    # For reference: all-only model
    c_index_all = float(
        concordance_index_censored(
            y_te[e_field].astype(bool), y_te[t_field].astype(float), risk_all_te
        )[0]
    )
    brier_all = float(np.nanmean((lab_t0 - risk_all_te) ** 2))

    return {
        "gating_oof_acc": gating["oof_acc"],
        "gating_oof_f1_macro": gating["oof_f1_macro"],
        "gating_report": gating["report"],
        "test_c_index_mix": c_index,
        "test_brier_mix": brier_mix,
        "test_c_index_all": c_index_all,
        "test_brier_all": brier_all,
    }


# =============================
# CLI
# =============================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "三模型+gating 框架：\n"
            "1) K 折 OOF 生成 per-sample 三元损失；\n"
            "2) 最小改进幅度 α 的硬阈值规则产出严格 meta-label；\n"
            "3) 训练 gating 三分类；\n"
            "4) 在独立测试集上评估混合模型 (C-index/Brier)。"
        )
    )
    parser.add_argument(
        "--data-file", type=str, default=None, help="Excel 路径，默认与旧脚本一致"
    )
    parser.add_argument(
        "--horizon-days", type=float, default=1825.0, help="评估时点 t0 (天)"
    )
    parser.add_argument("--kfold", type=int, default=5, help="K 折数")
    parser.add_argument("--repeats", type=int, default=1, help="重复 CV 次数")
    parser.add_argument("--test-size", type=float, default=0.25, help="独立测试集占比")
    parser.add_argument(
        "--alpha-abs", type=float, default=0.005, help="最小绝对改进阈值"
    )
    parser.add_argument("--alpha-rel", type=float, default=0.0, help="最小相对改进阈值")
    parser.add_argument(
        "--lambda-penalty", type=float, default=0.0, help="复杂度惩罚系数 λ"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no-progress", action="store_true", help="关闭进度输出")
    args = parser.parse_args()

    global PROGRESS
    if args.no_progress:
        PROGRESS = False

    _log("Loading and cleaning data ...")
    clean_df = load_dataframes(args.data_file)

    # Feature groups from constants; intersect with data to be safe
    global_cols = [c for c in GLOBAL_FEATURES if c in clean_df.columns]
    local_cols = [c for c in LOCAL_FEATURES if c in clean_df.columns]
    if len(global_cols) == 0 or len(local_cols) == 0:
        raise ValueError("GLOBAL_FEATURES/LOCAL_FEATURES 与数据列不匹配，请调整名称。")

    _log("Running end-to-end training and evaluation ...")
    results = evaluate_on_holdout(
        clean_df=clean_df,
        global_cols=global_cols,
        local_cols=local_cols,
        horizon_days=args.horizon_days,
        test_size=args.test_size,
        random_state=args.seed,
        k_folds=args.kfold,
        repeats=args.repeats,
        alpha_abs=args.alpha_abs,
        alpha_rel=args.alpha_rel,
        lambda_penalty=args.lambda_penalty,
    )

    print("\n=== Gating OOF (train) ===")
    print(f"Accuracy: {results['gating_oof_acc']:.4f}")
    print(f"F1-macro: {results['gating_oof_f1_macro']:.4f}")
    print(results["gating_report"])  # already nicely formatted by sklearn

    print("\n=== Test (independent holdout) ===")
    print(f"C-index (mixture by gating): {results['test_c_index_mix']:.4f}")
    print(f"Brier@t0 (mixture): {results['test_brier_mix']:.4f}")
    print(f"C-index (all-only baseline): {results['test_c_index_all']:.4f}")
    print(f"Brier@t0 (all-only baseline): {results['test_brier_all']:.4f}")


if __name__ == "__main__":
    main()
