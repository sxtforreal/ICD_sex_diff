from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

try:
    from missingpy import MissForest

    _HAS_MISSFOREST = True
except Exception:
    _HAS_MISSFOREST = False


def impute_misforest(X, random_seed):
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


def conversion_and_imputation(df, features, labels):
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
    base = "/home/sunx/data/aiiih/projects/sunx/projects/ICD"
    icd = pd.read_excel(os.path.join(base, "LGE granularity.xlsx"), sheet_name="ICD")
    noicd = pd.read_excel(
        os.path.join(base, "LGE granularity.xlsx"), sheet_name="No_ICD"
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
        "LGE_Basal anterolateral ",
        "LGE_mid anterior",
        "LGE_mid anterior septum",
        "LGE_mid inferoseptum",
        "LGE_mid inferior",
        "LGE_mid inferolateral ",
        "LGE_mid anterolateral ",
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
    labels = ["MRN", "VT/VF/SCD", "ICD", "PE_Time"]
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
        "LGE_Basal anterolateral ",
        "LGE_mid anterior",
        "LGE_mid anterior septum",
        "LGE_mid inferoseptum",
        "LGE_mid inferior",
        "LGE_mid inferolateral ",
        "LGE_mid anterolateral ",
        "LGE_apical anterior",
        "LGE_apical septum",
        "LGE_apical inferior",
        "LGE_apical lateral",
        "LGE_Apical cap",
        "LGE_RV insertion site (1 superior, 2 inferior. 3 both)",
    ]
    nicm = nicm.dropna(subset=granularity)
    features = [v for v in var if v not in labels]

    # Imputation
    clean_df = conversion_and_imputation(nicm, features, labels)
    clean_df["NYHA Class"] = clean_df["NYHA Class"].replace({5: 4, 0: 1})

    return clean_df


def _prepare_survival_xy(clean_df: pd.DataFrame,
                         drop_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Prepare X and y for survival modeling from cleaned data.

    Expects columns: "VT/VF/SCD" (event, 0/1) and "PE_Time" (time in days).
    """
    if drop_cols is None:
        drop_cols = ["MRN", "VT/VF/SCD", "ICD", "PE_Time"]

    df = clean_df.copy()
    df = df.dropna(subset=["PE_Time"])  # ensure valid times
    if not np.issubdtype(df["PE_Time"].dtype, np.number):
        df["PE_Time"] = pd.to_numeric(df["PE_Time"], errors="coerce")
        df = df.dropna(subset=["PE_Time"])

    df["VT/VF/SCD"] = df["VT/VF/SCD"].fillna(0).astype(int).astype(bool)

    X = df.drop(columns=drop_cols, errors="ignore")
    # Defensive: coerce any non-numeric leftovers
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X[non_numeric] = X[non_numeric].apply(pd.to_numeric, errors="coerce")
        for col in non_numeric:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median())

    feature_names = X.columns.tolist()
    y = Surv.from_dataframe(event="VT/VF/SCD", time="PE_Time", data=df)
    return X, y, feature_names


def train_random_survival_forest(
    clean_df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
    n_estimators: int = 500,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    max_features: Optional[str] = "sqrt",
) -> Tuple[RandomSurvivalForest, Dict[str, float], pd.Series]:
    """
    Train RSF on cleaned data and evaluate with concordance index on a hold-out test set.

    Returns: (fitted model, metrics dict, feature_importances series)
    """
    X, y, feature_names = _prepare_survival_xy(clean_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=-1,
        random_state=random_state,
    )
    rsf.fit(X_train, y_train)

    test_c_index = rsf.score(X_test, y_test)
    # scikit-survival's RandomSurvivalForest does not implement feature_importances_.
    # Use permutation importance on the hold-out set as a model-agnostic alternative.
    perm_result = permutation_importance(
        rsf,
        X_test,
        y_test,
        n_repeats=20,
        random_state=random_state,
        n_jobs=-1,
    )
    feat_imp = pd.Series(perm_result.importances_mean, index=feature_names).sort_values(
        ascending=False
    )
    metrics = {"test_c_index": float(test_c_index)}
    return rsf, metrics, feat_imp


def _find_feature_groups(feature_names: List[str]) -> Tuple[List[str], List[str], Optional[str]]:
    """
    Heuristically split features into global vs local (bull's-eye 17 segments) groups
    and pick a gating feature (overall scar burden) if available.

    Returns: (global_cols, local_cols, gating_feature_name_or_None)
    """
    names = list(feature_names)
    lower = [n.lower() for n in names]

    def has(substr: str) -> List[str]:
        s = substr.lower()
        return [names[i] for i, n in enumerate(lower) if s in n]

    # Local/bull's-eye patterns (17 segments + apical cap + RV insertion)
    local_patterns = [
        "lge_basal",
        "lge_mid ",
        "lge_mid",
        "lge_apical",
        "lge_apical cap",
        "rv insertion",
        "lge_rv insertion",
        "anterolateral",
        "inferolateral",
        "inferosept",
        "anterior sept",
        "inferior",
        "septum",
        "lateral",
    ]
    local_cols_set = set()
    for p in local_patterns:
        for c in has(p):
            local_cols_set.add(c)
    # Global patterns
    global_cols_set = set()
    for p in [
        "lge_lge burden 5sd",
        "lge burden 5sd",
        "lge_extent",
        "extent (1;",
        "circumural",
        "ring-like",
        "lvef",
        "nyha class",
    ]:
        for c in has(p):
            global_cols_set.add(c)

    # Avoid overlap
    local_cols = [c for c in names if c in local_cols_set]
    global_cols = [c for c in names if c in global_cols_set and c not in local_cols_set]

    # Choose gating feature: prefer explicit scar burden
    priorities = [
        "LGE_LGE Burden 5SD",
        "LGE Burden 5SD",
        "LGE_Extent (1; subendocardial, 2; mid mural, 3; epicardial, 4; transmural; 5 circumural)",
    ]
    gating = None
    for p in priorities:
        if p in names:
            gating = p
            break
    if gating is None:
        # Try relaxed search
        for p in ["burden", "extent", "circumural", "ring-like"]:
            hits = has(p)
            if hits:
                gating = hits[0]
                break

    return global_cols, local_cols, gating


def _rsf_risk_at_time(model: RandomSurvivalForest, X: pd.DataFrame, t: float) -> np.ndarray:
    """Risk score at time t as 1 - S(t)."""
    if X.empty:
        return np.zeros(0, dtype=float)
    try:
        surv = model.predict_survival_function(X)
        s_at = np.array([sf(t) for sf in surv], dtype=float)
    except Exception:
        # Fallback: approximate using nearest available event time grid
        times = getattr(model, "event_times_", None)
        if times is None or len(times) == 0:
            # As a last resort, use model.score ordering via leaf depth proxy (not ideal)
            # Return zeros to avoid crashing
            return np.zeros(len(X), dtype=float)
        idx = int(np.argmin(np.abs(times - float(t))))
        surv = model.predict_survival_function(X)
        s_at = np.array([sf(times[idx]) for sf in surv], dtype=float)
    s_at = np.clip(s_at, 0.0, 1.0)
    return 1.0 - s_at


def evaluate_two_stage_strategy(
    clean_df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
    q_low: float = 0.40,
    q_high: float = 0.75,
) -> Dict[str, float]:
    """
    Validate a two-stage decision strategy:
    1) Inspect global features first with a gating feature and quantile thresholds.
    2) For mid-range cases, defer to local (17-segment) features.

    Returns a metrics dict including C-index for baselines and the two-stage approach.
    """
    # Prepare dataset
    X_all, y_all, feature_names = _prepare_survival_xy(clean_df)
    global_cols, local_cols, gating = _find_feature_groups(feature_names)

    # Safety checks
    have_global = len(global_cols) > 0
    have_local = len(local_cols) > 0

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=random_state
    )

    # Time horizon = 75th percentile of observed times in training
    t_hor = float(np.percentile(y_train["time"], 75)) if len(y_train) else 365.0
    if not np.isfinite(t_hor) or t_hor <= 0:
        t_hor = 365.0

    # Train RSF models
    def _fit(X, y) -> Optional[RandomSurvivalForest]:
        if X.shape[1] == 0:
            return None
        rsf = RandomSurvivalForest(
            n_estimators=500,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        )
        rsf.fit(X, y)
        return rsf

    model_all = _fit(X_train, y_train)
    model_global = _fit(X_train[global_cols], y_train) if have_global else None
    model_local = _fit(X_train[local_cols], y_train) if have_local else None

    # Baseline risks at horizon
    risk_all = (
        _rsf_risk_at_time(model_all, X_test, t_hor) if model_all is not None else np.zeros(len(X_test))
    )
    risk_glob = (
        _rsf_risk_at_time(model_global, X_test[global_cols], t_hor)
        if model_global is not None
        else np.zeros(len(X_test))
    )
    risk_loc = (
        _rsf_risk_at_time(model_local, X_test[local_cols], t_hor)
        if model_local is not None
        else np.zeros(len(X_test))
    )

    # Two-stage gating setup
    # Obtain gating values (prefer explicit gating feature, else derive from local sum)
    if gating is not None and gating in X_all.columns:
        gate_train_vals = X_train[gating].astype(float).values
        gate_test_vals = X_test[gating].astype(float).values
    else:
        # Fallback: sum of local signals as a proxy for overall burden
        if have_local:
            gate_train_vals = X_train[local_cols].astype(float).sum(axis=1).values
            gate_test_vals = X_test[local_cols].astype(float).sum(axis=1).values
            gating = "(sum of local segments)"
        else:
            # No gating available -> degenerate to global risk
            gate_train_vals = np.zeros(len(X_train), dtype=float)
            gate_test_vals = np.zeros(len(X_test), dtype=float)
            gating = None

    if gating is not None:
        thr_low = float(np.nanquantile(gate_train_vals, q_low))
        thr_high = float(np.nanquantile(gate_train_vals, q_high))
        if not np.isfinite(thr_low):
            thr_low = float(np.nanmedian(gate_train_vals))
        if not np.isfinite(thr_high):
            thr_high = float(np.nanmedian(gate_train_vals))
        if thr_low >= thr_high:
            # Enforce separation
            thr_low, thr_high = float(np.nanpercentile(gate_train_vals, 40)), float(
                np.nanpercentile(gate_train_vals, 75)
            )

        # Combine risks per zone
        zone_high = gate_test_vals >= thr_high
        zone_low = gate_test_vals < thr_low
        zone_mid = ~(zone_high | zone_low)

        risk_two_stage = np.zeros(len(X_test), dtype=float)
        # High burden: rely on global risk (must-implant logic)
        risk_two_stage[zone_high] = risk_glob[zone_high]
        # Low burden: rely on global risk (generally safe)
        risk_two_stage[zone_low] = risk_glob[zone_low]
        # Mid zone: defer to local risk to avoid unnecessary ICD where possible
        risk_two_stage[zone_mid] = risk_loc[zone_mid]
    else:
        # No gating -> fall back to global risk
        thr_low = thr_high = np.nan
        zone_high = zone_low = zone_mid = np.zeros(len(X_test), dtype=bool)
        risk_two_stage = risk_glob.copy()

    # Evaluate C-index
    def _c_index(y, risk):
        evt = y["event"].astype(bool)
        tm = y["time"].astype(float)
        c = concordance_index_censored(evt, tm, risk)[0]
        return float(c)

    metrics = {
        "c_index_all": _c_index(y_test, risk_all) if model_all is not None else np.nan,
        "c_index_global_only": _c_index(y_test, risk_glob) if model_global is not None else np.nan,
        "c_index_local_only": _c_index(y_test, risk_loc) if model_local is not None else np.nan,
        "c_index_two_stage": _c_index(y_test, risk_two_stage),
        "time_horizon_days": t_hor,
        "gating_feature": gating if gating is not None else "<none>",
        "thr_low": thr_low,
        "thr_high": thr_high,
        "n_zone_low": int(zone_low.sum()) if gating is not None else 0,
        "n_zone_mid": int(zone_mid.sum()) if gating is not None else 0,
        "n_zone_high": int(zone_high.sum()) if gating is not None else 0,
        "n_test": int(len(X_test)),
    }

    # Also return top permutation importances (optional, compact)
    try:
        if model_global is not None:
            perm_glob = permutation_importance(
                model_global, X_test[global_cols], y_test, n_repeats=10, random_state=random_state, n_jobs=-1
            )
            fi_glob = pd.Series(perm_glob.importances_mean, index=global_cols).sort_values(ascending=False)
            topg = fi_glob.head(8)
            print("Global features (top 8 by permutation importance):")
            print(topg)
        if model_local is not None:
            perm_loc = permutation_importance(
                model_local, X_test[local_cols], y_test, n_repeats=10, random_state=random_state, n_jobs=-1
            )
            fi_loc = pd.Series(perm_loc.importances_mean, index=local_cols).sort_values(ascending=False)
            topl = fi_loc.head(8)
            print("Local features (top 8 by permutation importance):")
            print(topl)
    except Exception:
        pass

    print("\nTwo-stage strategy evaluation:")
    print(f"- Gating feature: {metrics['gating_feature']}")
    print(f"- Thresholds: low={metrics['thr_low']:.4g}, high={metrics['thr_high']:.4g}")
    print(f"- Time horizon (days): {metrics['time_horizon_days']:.1f}")
    if gating is not None:
        print(
            f"- Zone counts (low/mid/high): {metrics['n_zone_low']}/{metrics['n_zone_mid']}/{metrics['n_zone_high']} of {metrics['n_test']}"
        )
    print("- C-index baselines and two-stage (test):")
    print(f"  * All-features RSF:     {metrics['c_index_all']:.4f}")
    print(f"  * Global-only RSF:      {metrics['c_index_global_only']:.4f}")
    print(f"  * Local-only RSF:       {metrics['c_index_local_only']:.4f}")
    print(f"  * Two-stage (g->l mid): {metrics['c_index_two_stage']:.4f}")

    return metrics


def main():
    clean_df = load_dataframes()
    model, metrics, feat_imp = train_random_survival_forest(clean_df)
    print(f"Test C-index: {metrics['test_c_index']:.4f}")
    print("Top feature importances:")
    topn = 15 if len(feat_imp) >= 15 else len(feat_imp)
    print(feat_imp.head(topn))

    # Validate the global-first, local-then hypothesis
    _ = evaluate_two_stage_strategy(clean_df)


if __name__ == "__main__":
    main()

