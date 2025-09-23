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
from sklearn.exceptions import ConvergenceWarning

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sksurv.nonparametric import kaplan_meier_estimator


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

    # Imputation on feature matrix: allow optional MissForest via env flag; fallback to median
    X = df[features].copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")
    use_missforest = os.environ.get("RSF_USE_MISSFOREST", "0") in ("1", "true", "True")
    if use_missforest:
        try:
            from missingpy import MissForest  # type: ignore
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X), columns=X.columns, index=X.index
            )
            imputer = MissForest(random_state=0)
            X_imp_scaled = pd.DataFrame(
                imputer.fit_transform(X_scaled), columns=X.columns, index=X.index
            )
            X = pd.DataFrame(
                scaler.inverse_transform(X_imp_scaled),
                columns=X.columns,
                index=X.index,
            )
        except Exception:
            for col in X.columns:
                if X[col].isnull().any():
                    X[col] = X[col].fillna(X[col].median())
    else:
        for col in X.columns:
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
            # retain LGE Burden 5SD and CG CrCl for engineered features
            "LGE_Unnamed: 1",
            "LGE_Notes",
            "LGE_RV insertion sites (0 No, 1 yes)",
            "LGE_Score",
            "LGE_Unnamed: 27",
            "LGE_Unnamed: 28",
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

    # Engineered features to mirror cox.py
    if "Age at CMR" in clean_df.columns:
        try:
            clean_df["Age by decade"] = (pd.to_numeric(clean_df["Age at CMR"], errors="coerce") // 10).astype(float)
        except Exception:
            pass
    if "Cockcroft-Gault Creatinine Clearance (mL/min)" in clean_df.columns:
        try:
            crcl = pd.to_numeric(clean_df["Cockcroft-Gault Creatinine Clearance (mL/min)"], errors="coerce")
            clean_df["CrCl>45"] = (crcl > 45).astype(float)
        except Exception:
            pass
    if "NYHA Class" in clean_df.columns:
        try:
            nyha_num = pd.to_numeric(clean_df["NYHA Class"], errors="coerce")
            clean_df["NYHA>2"] = (nyha_num > 2).astype(float)
        except Exception:
            pass
    if "LGE Burden 5SD" in clean_df.columns:
        try:
            lge = pd.to_numeric(clean_df["LGE Burden 5SD"], errors="coerce")
            clean_df["Significant LGE"] = (lge > 2).astype(float)
        except Exception:
            pass
    return clean_df


# =============================
# Survival utilities
# =============================


def _prepare_survival_xy(
    clean_df: pd.DataFrame,
    drop_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    if drop_cols is None:
        # By default, keep Female as a potential feature (do not drop here)
        drop_cols = ["MRN", "VT/VF/SCD", "ICD", "PE_Time"]

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


class CoxPHWithScaler:
    """A thin wrapper adding feature standardization around CoxPHSurvivalAnalysis.

    Provides predict_survival_function and predict consistent with scikit-survival's API.
    """

    def __init__(self, alpha: float = 0.1, clip_value: float = float("inf")):
        self.alpha = float(alpha)
        self.clip_value = float(clip_value)
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[CoxPHSurvivalAnalysis] = None
        self.columns: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "CoxPHWithScaler":
        self.columns = list(X.columns)
        # drop zero-variance columns for stability
        nunique = X.nunique(dropna=True)
        keep_cols = nunique[nunique > 1].index.tolist()
        X_use = X[keep_cols].astype(float).copy()
        self.columns = keep_cols
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_use), columns=self.columns, index=X_use.index
        )
        # Clip scaled features to avoid extremely large linear predictors
        if np.isfinite(self.clip_value) and self.clip_value > 0:
            X_scaled = X_scaled.clip(lower=-self.clip_value, upper=self.clip_value)

        # Try multiple constructor signatures for version compatibility
        def _build_model(alpha: float) -> CoxPHSurvivalAnalysis:
            for kwargs in (
                {"alpha": alpha, "ties": "efron", "n_iter": 1024},
                {"alpha": alpha, "tie_method": "efron", "n_iter": 1024},
                {"alpha": alpha, "ties": "efron"},
                {"alpha": alpha, "tie_method": "efron"},
                {"alpha": alpha},
            ):
                try:
                    return CoxPHSurvivalAnalysis(**kwargs)  # type: ignore[arg-type]
                except TypeError:
                    continue
            return CoxPHSurvivalAnalysis(alpha=alpha)

        self.model = _build_model(self.alpha)

        # Attempt fit; if convergence warning occurs, refit with stronger regularization
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                self.model.fit(X_scaled, y)
            except ConvergenceWarning:
                # Increase regularization and try again
                stronger_alpha = max(self.alpha * 10.0, self.alpha + 0.9)
                self.model = _build_model(stronger_alpha)
                self.model.fit(X_scaled, y)
        return self

    def _ensure_ready(self) -> None:
        if self.model is None or self.scaler is None or self.columns is None:
            raise RuntimeError("Model not fitted.")

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._ensure_ready()
        # align and select the same columns
        X_sel = X.reindex(columns=self.columns, copy=False)
        X_sel = X_sel.astype(float)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_sel), columns=self.columns, index=X_sel.index
        )
        return X_scaled

    def predict_survival_function(self, X: pd.DataFrame):
        self._ensure_ready()
        X_scaled = self._transform(X)
        return self.model.predict_survival_function(X_scaled)  # type: ignore[union-attr]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._ensure_ready()
        X_scaled = self._transform(X)
        return np.asarray(self.model.predict(X_scaled), dtype=float)  # type: ignore[union-attr]


def _fit_cox(
    X: pd.DataFrame, y: np.ndarray, alpha: float = 0.1, clip_value: float = 8.0
) -> Optional[CoxPHWithScaler]:
    if X is None or X.shape[0] == 0 or X.shape[1] == 0:
        return None
    try:
        wrapper = CoxPHWithScaler(alpha=alpha, clip_value=clip_value)
        wrapper.fit(X, y)
        return wrapper
    except Exception as ex:
        warnings.warn(f"Cox fit failed: {ex}")
        return None


def _kaplan_meier_Ghat_at(times: np.ndarray, surv: np.ndarray, t: float) -> float:
    """Right-continuous step function evaluation for KM survival at time t."""
    if len(times) == 0:
        return 1.0
    idx = np.searchsorted(times, t, side="right") - 1
    if idx < 0:
        return 1.0
    return float(surv[idx])


def _ipcw_brier_per_sample(
    y_train: np.ndarray, y_valid: np.ndarray, s_at: np.ndarray, t0: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-sample IPCW Brier contributions at time t0 and a known mask.

    - y_train: used to estimate censoring distribution G(t)
    - y_valid: evaluation set structured array
    - s_at: survival probability S(t0) for each valid sample
    Returns (losses, known_mask)
    """
    # KM of censoring distribution G using 1-event indicator
    names = getattr(y_train.dtype, "names", ("event", "time"))
    e_tr = y_train[names[0]].astype(bool)
    t_tr = y_train[names[1]].astype(float)
    # censoring occurs when event==False
    times_g, surv_g = kaplan_meier_estimator(~e_tr, t_tr)

    names_v = getattr(y_valid.dtype, "names", ("event", "time"))
    e_va = y_valid[names_v[0]].astype(bool)
    t_va = y_valid[names_v[1]].astype(float)

    n = len(y_valid)
    losses = np.full(n, np.nan, dtype=float)
    known = np.zeros(n, dtype=bool)
    eps = 1e-12
    G_t0 = max(_kaplan_meier_Ghat_at(times_g, surv_g, t0), eps)

    for i in range(n):
        if e_va[i] and t_va[i] <= t0:
            G_ti = max(_kaplan_meier_Ghat_at(times_g, surv_g, float(t_va[i])), eps)
            losses[i] = ((1.0 - float(s_at[i])) ** 2.0) / G_ti
            known[i] = True
        elif t_va[i] > t0:
            losses[i] = (float(s_at[i]) ** 2.0) / G_t0
            known[i] = True
        else:
            # censored before t0: not usable for BS(t0)
            losses[i] = np.nan
            known[i] = False
    return losses, known


# =============================
# Step 1: K 折 OOF 双模型损失（global vs global+local）
# =============================


def compute_oof_losses(
    clean_df: pd.DataFrame,
    global_cols: List[str],
    local_cols: List[str],
    horizon_days: float,
    k_folds: int = 5,
    repeats: int = 1,
    random_state: int = 42,
    cox_alpha: float = 0.1,
    include_female: bool = False,
    cox_clip_value: float = 8.0,
) -> Dict[str, np.ndarray]:
    # Control whether Female is part of features
    drop_cols = ["MRN", "VT/VF/SCD", "ICD", "PE_Time"]
    if not include_female:
        drop_cols = drop_cols + ["Female"]
    X_all, y_all, feat_names = _prepare_survival_xy(clean_df, drop_cols=drop_cols)
    all_cols = [c for c in feat_names]
    global_cols = [c for c in global_cols if c in all_cols]
    local_cols = [c for c in local_cols if c in all_cols]
    if len(global_cols) == 0 or len(local_cols) == 0:
        raise ValueError("GLOBAL_FEATURES 或 LOCAL_FEATURES 为空或与数据不匹配。")

    n = len(X_all)
    # Collect R repetitions
    losses_global_rep: List[np.ndarray] = []
    losses_all_rep: List[np.ndarray] = []

    # For IPCW Brier, we will compute per-fold using KM G from training fold

    for r in range(max(1, int(repeats))):
        rng = int(random_state + r)
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=rng)
        L_gl = np.full(n, np.nan, dtype=float)
        L_all = np.full(n, np.nan, dtype=float)

        for tr_idx, va_idx in kf.split(X_all):
            X_tr, X_va = X_all.iloc[tr_idx], X_all.iloc[va_idx]
            y_tr, y_va = y_all[tr_idx], y_all[va_idx]

            # Fit two Cox models: global-only, and all (global+local)
            mdl_gl = _fit_cox(X_tr[global_cols], y_tr, alpha=cox_alpha, clip_value=cox_clip_value)
            mdl_all = _fit_cox(X_tr[all_cols], y_tr, alpha=cox_alpha, clip_value=cox_clip_value)

            # Predict S(t0) for IPCW-Brier (used for meta-labels). C-index will use linear predictors.
            if mdl_gl:
                survf_gl = mdl_gl.predict_survival_function(X_va[global_cols])
                s_gl = np.array([sf(horizon_days) for sf in survf_gl], dtype=float)
            else:
                s_gl = np.ones(len(va_idx), dtype=float)

            if mdl_all:
                survf_all = mdl_all.predict_survival_function(X_va[all_cols])
                s_all = np.array([sf(horizon_days) for sf in survf_all], dtype=float)
            else:
                s_all = np.ones(len(va_idx), dtype=float)

            # Per-sample IPCW Brier loss using KM from training fold
            L_gl_fold, known_gl = _ipcw_brier_per_sample(y_tr, y_va, s_gl, horizon_days)
            L_all_fold, known_all = _ipcw_brier_per_sample(y_tr, y_va, s_all, horizon_days)

            L_gl[va_idx] = L_gl_fold
            L_all[va_idx] = L_all_fold

            # Fold-level metrics
            e_field = y_va.dtype.names[0]
            t_field = y_va.dtype.names[1]
            # Use Cox linear predictors (risk scores) for C-index
            if mdl_gl:
                lin_gl = np.asarray(mdl_gl.predict(X_va[global_cols]), dtype=float).ravel()
            else:
                lin_gl = np.zeros(len(va_idx), dtype=float)
            if mdl_all:
                lin_all = np.asarray(mdl_all.predict(X_va[all_cols]), dtype=float).ravel()
            else:
                lin_all = np.zeros(len(va_idx), dtype=float)

            c_gl = concordance_index_censored(
                y_va[e_field].astype(bool), y_va[t_field].astype(float), lin_gl
            )[0]
            c_all = concordance_index_censored(
                y_va[e_field].astype(bool), y_va[t_field].astype(float), lin_all
            )[0]
            # Mean IPCW Brier over valid samples
            b_gl = float(np.nanmean(L_gl_fold))
            b_all = float(np.nanmean(L_all_fold))

            # init lists on first usage
            if r == 0 and 'c_gl_list' not in locals():
                c_gl_list = []
                c_all_list = []
                b_gl_list = []
                b_all_list = []
            c_gl_list.append(float(c_gl))
            c_all_list.append(float(c_all))
            b_gl_list.append(b_gl)
            b_all_list.append(b_all)

        losses_global_rep.append(L_gl)
        losses_all_rep.append(L_all)

    # Average across repeats when available
    def _nanmean_stack(arrs: List[np.ndarray]) -> np.ndarray:
        if len(arrs) == 1:
            return arrs[0]
        A = np.stack(arrs, axis=0)
        return np.nanmean(A, axis=0)

    # Aggregate CV stats across all folds and repeats
    def _mean_std(vals: List[float]) -> Tuple[float, float]:
        if len(vals) == 0:
            return float('nan'), float('nan')
        return float(np.nanmean(vals)), float(np.nanstd(vals, ddof=1) if len(vals) > 1 else 0.0)

    cv_cindex_global_mean, cv_cindex_global_std = _mean_std(c_gl_list if 'c_gl_list' in locals() else [])
    cv_cindex_all_mean, cv_cindex_all_std = _mean_std(c_all_list if 'c_all_list' in locals() else [])
    cv_brier_global_mean, cv_brier_global_std = _mean_std(b_gl_list if 'b_gl_list' in locals() else [])
    cv_brier_all_mean, cv_brier_all_std = _mean_std(b_all_list if 'b_all_list' in locals() else [])

    # known mask: any model has known loss (not NaN)
    known_mask = np.isfinite(np.nanmin(np.vstack([losses_global_rep[0], losses_all_rep[0]]), axis=0)) if len(losses_global_rep) > 0 else np.zeros(n, dtype=bool)

    return {
        "loss_global": _nanmean_stack(losses_global_rep),
        "loss_all": _nanmean_stack(losses_all_rep),
        "known_mask": known_mask,
        "X": X_all,
        "y": y_all,
        "global_cols": global_cols,
        "local_cols": local_cols,
        "all_cols": all_cols,
        "cv_cindex_global_mean": cv_cindex_global_mean,
        "cv_cindex_global_std": cv_cindex_global_std,
        "cv_cindex_all_mean": cv_cindex_all_mean,
        "cv_cindex_all_std": cv_cindex_all_std,
        "cv_brier_global_mean": cv_brier_global_mean,
        "cv_brier_global_std": cv_brier_global_std,
        "cv_brier_all_mean": cv_brier_all_mean,
        "cv_brier_all_std": cv_brier_all_std,
    }


# =============================
# Step 2 & 3: 硬阈值选择 + meta-label 生成
# =============================


def apply_hard_threshold_rule_binary(
    loss_global: np.ndarray,
    loss_all: np.ndarray,
    alpha_abs: float = 0.005,
    alpha_rel: float = 0.0,
    lambda_penalty: float = 0.0,
    complexity: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """
    基于双模型（global vs global+local）的 OOF 损失生成二元严格标签：
    z_i ∈ {0: global, 1: all}，并返回 margin = L_second - L_best。
    """
    if complexity is None:
        complexity = {"global": 1.0, "all": 1.2}

    L_gl = loss_global.copy() + lambda_penalty * complexity.get("global", 1.0)
    L_all = loss_all.copy() + lambda_penalty * complexity.get("all", 1.0)

    n = len(L_gl)
    labels = np.full(n, fill_value=-1, dtype=int)
    margins = np.full(n, fill_value=np.nan, dtype=float)

    stack = np.vstack([L_gl, L_all])  # 0: global, 1: all
    for i in range(n):
        li = stack[:, i]
        if np.sum(np.isfinite(li)) < 2:
            continue
        order = np.argsort(li)
        best = order[0]
        second = order[1]
        Lb = float(li[best])
        L2 = float(li[second])
        margin = float(L2 - Lb)
        margins[i] = margin

        cond_abs = margin >= alpha_abs
        cond_rel = (margin / max(L2, 1e-12)) >= alpha_rel
        if cond_abs and cond_rel:
            labels[i] = int(best)  # 0: global, 1: all
        else:
            labels[i] = int(np.nanargmin([li[0], li[1]]))

    return {"labels": labels, "margins": margins}


# =============================
# Step 4: 学 gating（离散三分类）
# =============================


def train_gating_classifier(
    X: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: List[str],
    k_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, object]:
    mask = labels >= 0
    Xy = X[mask]
    y = labels[mask]
    # Use only specified features (global features only for gating)
    Xy = Xy[feature_cols]

    if Xy.shape[0] == 0:
        raise ValueError("没有可用于训练 gating 的样本标签。")

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
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
        "features": feature_cols,
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
    cox_alpha: float = 0.1,
    topk_importance: int = 20,
    include_female: bool = False,
    cox_clip_value: float = 8.0,
) -> Dict[str, object]:
    # Control whether Female is included as a feature throughout
    drop_cols = ["MRN", "VT/VF/SCD", "ICD", "PE_Time"]
    if not include_female:
        drop_cols = drop_cols + ["Female"]
    X_all, y_all, feat_names = _prepare_survival_xy(clean_df, drop_cols=drop_cols)
    all_cols = [c for c in feat_names]
    global_cols = [c for c in global_cols if c in all_cols]
    # Optionally include Female in the global feature set if requested and available
    if include_female and "Female" in all_cols and "Female" not in global_cols:
        global_cols = list(global_cols) + ["Female"]
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
        cox_alpha=cox_alpha,
        include_female=include_female,
        cox_clip_value=cox_clip_value,
    )

    _log("Generating strict meta-labels by hard-threshold rule (binary) ...")
    lab_res = apply_hard_threshold_rule_binary(
        oof_tr["loss_global"],
        oof_tr["loss_all"],
        alpha_abs=alpha_abs,
        alpha_rel=alpha_rel,
        lambda_penalty=lambda_penalty,
    )

    _log("Training gating classifier on training set (global-only features) ...")
    gating = train_gating_classifier(
        oof_tr["X"], lab_res["labels"], feature_cols=global_cols, k_folds=k_folds, random_state=random_state
    )

    # Fit base models on full training set
    _log("Fitting base models on the full training set ...")
    mdl_global = _fit_cox(X_tr[global_cols], y_tr, alpha=cox_alpha, clip_value=cox_clip_value)
    mdl_all = _fit_cox(X_tr[all_cols], y_tr, alpha=cox_alpha, clip_value=cox_clip_value)

    # Predict on test set via learned gating
    _log("Applying gating to test set and computing metrics ...")
    gate_mask = gating["mask"]
    gate_model: Pipeline = gating["model"]  # type: ignore

    # For test, gating uses only global features
    gate_pred = gate_model.predict(X_te[global_cols])

    # Predict Cox linear predictors for test set
    if mdl_global:
        lin_gl_te = np.asarray(mdl_global.predict(X_te[global_cols]), dtype=float).ravel()
    else:
        lin_gl_te = np.zeros(len(X_te), dtype=float)
    if mdl_all:
        lin_all_te = np.asarray(mdl_all.predict(X_te[all_cols]), dtype=float).ravel()
    else:
        lin_all_te = np.zeros(len(X_te), dtype=float)

    # Compose mixture by gating decision: 0->global, 1->all
    mix_score = np.zeros(len(X_te), dtype=float)
    for i in range(len(X_te)):
        z = int(gate_pred[i])
        mix_score[i] = lin_gl_te[i] if z == 0 else lin_all_te[i]

    # Metrics on test
    e_field = "event" if "event" in y_te.dtype.names else y_te.dtype.names[0]
    t_field = "time" if "time" in y_te.dtype.names else y_te.dtype.names[1]
    c_index = float(
        concordance_index_censored(
            y_te[e_field].astype(bool), y_te[t_field].astype(float), mix_score
        )[0]
    )

    # For reference: all-only model
    c_index_all = float(
        concordance_index_censored(
            y_te[e_field].astype(bool), y_te[t_field].astype(float), lin_all_te
        )[0]
    )

    # Gating feature importance: coef norms across classes
    clf = None
    try:
        clf = gate_model.named_steps.get("clf", None)
    except Exception:
        clf = None
    feat_names = list(gating.get("features", [])) or list(oof_tr["X"].columns)
    gating_importance: List[Tuple[str, float]] = []
    if hasattr(clf, "coef_"):
        coef = np.asarray(getattr(clf, "coef_"), dtype=float)
        if coef.ndim == 1:
            coef = coef[None, :]
        imp = np.linalg.norm(coef, ord=2, axis=0)
        for name, val in zip(feat_names, imp):
            gating_importance.append((str(name), float(val)))
        gating_importance.sort(key=lambda x: x[1], reverse=True)
        gating_importance = gating_importance[: max(1, int(topk_importance))]

    return {
        "gating_oof_acc": gating["oof_acc"],
        "gating_oof_f1_macro": gating["oof_f1_macro"],
        "gating_report": gating["report"],
        "test_c_index_mix": c_index,
        "test_brier_mix": float('nan'),
        "test_c_index_all": c_index_all,
        "test_brier_all": float('nan'),
        "cv_cindex_local_mean": oof_tr.get("cv_cindex_local_mean", float("nan")),
        "cv_cindex_local_std": oof_tr.get("cv_cindex_local_std", float("nan")),
        "cv_cindex_global_mean": oof_tr.get("cv_cindex_global_mean", float("nan")),
        "cv_cindex_global_std": oof_tr.get("cv_cindex_global_std", float("nan")),
        "cv_cindex_all_mean": oof_tr.get("cv_cindex_all_mean", float("nan")),
        "cv_cindex_all_std": oof_tr.get("cv_cindex_all_std", float("nan")),
        "cv_brier_local_mean": oof_tr.get("cv_brier_local_mean", float("nan")),
        "cv_brier_local_std": oof_tr.get("cv_brier_local_std", float("nan")),
        "cv_brier_global_mean": oof_tr.get("cv_brier_global_mean", float("nan")),
        "cv_brier_global_std": oof_tr.get("cv_brier_global_std", float("nan")),
        "cv_brier_all_mean": oof_tr.get("cv_brier_all_mean", float("nan")),
        "cv_brier_all_std": oof_tr.get("cv_brier_all_std", float("nan")),
        "gating_feature_importance": gating_importance,
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
    parser.add_argument("--cox-alpha", type=float, default=0.1, help="Cox L2 正则强度 alpha")
    parser.add_argument("--topk-importance", type=int, default=20, help="打印 gating 特征重要性 Top-K")
    parser.add_argument("--include-female", action="store_true", help="是否将 Female 作为特征纳入 Cox 与 gating")
    parser.add_argument(
        "--cox-clip-value",
        type=float,
        default=8.0,
        help="标准化后特征的裁剪阈值(避免线性预测过大导致溢出)",
    )
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
        cox_alpha=args.cox_alpha,
        topk_importance=args.topk_importance,
        include_female=args.include_female,
        cox_clip_value=args.cox_clip_value,
    )

    print("\n=== Gating OOF (train) ===")
    print(f"Accuracy: {results['gating_oof_acc']:.4f}")
    print(f"F1-macro: {results['gating_oof_f1_macro']:.4f}")
    print(results["gating_report"])  # already nicely formatted by sklearn

    print("\n=== Test (independent holdout) ===")
    print(f"C-index (mixture by gating): {results['test_c_index_mix']:.4f}")
    print(f"C-index (all-only baseline): {results['test_c_index_all']:.4f}")

    # CV stats printout (C-index only)
    print("\n=== CV (train) Cox models ===")
    print(
        f"Global: C-index {results['cv_cindex_global_mean']:.4f} ± {results['cv_cindex_global_std']:.4f}"
    )
    print(
        f"All:    C-index {results['cv_cindex_all_mean']:.4f} ± {results['cv_cindex_all_std']:.4f}"
    )

    # Gating feature importance
    if len(results.get("gating_feature_importance", [])) > 0:
        print("\n=== Gating feature importance (coef L2 norm, top-K) ===")
        for name, val in results["gating_feature_importance"]:
            print(f"{name}: {val:.6f}")


if __name__ == "__main__":
    main()
