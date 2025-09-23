from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.exceptions import ConvergenceWarning

# Global control for showing progress logs
PROGRESS = True

def _maybe_tqdm(iterable, total=None, desc=None, leave=False):
    """Lightweight progress wrapper that prints textual progress instead of tqdm bars."""
    if not PROGRESS:
        return iterable
    # Try to infer total if not provided and iterable is sized
    try:
        inferred_total = len(iterable) if total is None and hasattr(iterable, "__len__") else total
    except Exception:
        inferred_total = total

    def _generator():
        count = 0
        for item in iterable:
            count += 1
            if desc:
                if inferred_total is not None:
                    _log_progress(f"{desc}: {count}/{inferred_total}", True)
                else:
                    _log_progress(f"{desc}: {count}", True)
            yield item
        if desc and leave:
            _log_progress(f"{desc}: done", True)

    return _generator()

# Global max iterations for lifelines CoxPHFitter (None = library default)
MAX_LIFELINES_STEPS: Optional[int] = None

def set_max_iterations(n: Optional[int]) -> None:
    """Set global max Newton–Raphson steps for lifelines Cox fitting."""
    global MAX_LIFELINES_STEPS
    try:
        MAX_LIFELINES_STEPS = int(n) if n is not None and int(n) > 0 else None
    except Exception:
        MAX_LIFELINES_STEPS = None

try:
    from missingpy import MissForest

    _HAS_MISSFOREST = True
except Exception:
    _HAS_MISSFOREST = False


# ========================= User-configurable feature groups =========================
# Explicitly define global and local feature name collections; their union is all features.
# Usage:
# 1) Edit the lists below directly; or
# 2) Call set_feature_groups(["featA", ...], ["featB", ...]) at runtime
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


def set_feature_groups(global_features: List[str], local_features: List[str]) -> None:
    """Set the global/local feature name lists explicitly (unknown names are ignored)."""
    global GLOBAL_FEATURES, LOCAL_FEATURES
    GLOBAL_FEATURES = list(global_features)
    LOCAL_FEATURES = list(local_features)


# Figure output directory (can be set via set_figures_dir, CLI --figs-dir, or env FIGURES_DIR)
FIGURES_DIR: Optional[str] = None


def set_figures_dir(path: Optional[str]) -> None:
    global FIGURES_DIR
    FIGURES_DIR = path


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


def _ensure_dir(path: Optional[str]) -> None:
    if path is None:
        return
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _save_fig(fig: plt.Figure, output_dir: Optional[str], filename: str) -> None:
    if output_dir:
        _ensure_dir(output_dir)
        try:
            fig.savefig(
                os.path.join(output_dir, filename), dpi=150, bbox_inches="tight"
            )
            plt.close(fig)
            return
        except Exception:
            pass
    # Fallback to on-screen display
    try:
        fig.tight_layout()
    except Exception:
        pass
    plt.show()


def _log_progress(message: str, enabled: bool = True) -> None:
    if enabled:
        try:
            print(f"[Progress] {message}")
        except Exception:
            pass


def _plot_series_barh(
    series: pd.Series,
    topn: int,
    title: str,
    xlabel: str,
    output_dir: Optional[str],
    filename: str,
    color: str = "#1f77b4",
) -> None:
    ser = series.dropna()
    if len(ser) == 0:
        return
    ser = ser.sort_values(ascending=True)
    if topn is not None and topn > 0 and len(ser) > topn:
        ser = ser.iloc[-topn:]
    fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.35 * len(ser))))
    ax.barh(range(len(ser)), ser.values, color=color)
    ax.set_yticks(range(len(ser)))
    ax.set_yticklabels(ser.index)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    _save_fig(fig, output_dir, filename)


 


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
            _tmp = df[c].replace(
                {"Yes": 1, "No": 0, "Y": 1, "N": 0, "True": 1, "False": 0}
            )
            try:
                df[c] = _tmp.infer_objects(copy=False)
            except Exception:
                df[c] = _tmp
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


def search_best_gating_feature_by_cv(
    clean_df: pd.DataFrame,
    q_low_grid: Optional[List[float]] = None,
    q_high_grid: Optional[List[float]] = None,
    min_gap: float = 0.10,
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
    output_dir: Optional[str] = None,
    topk_report: int = 10,
) -> Dict[str, object]:
    """Deprecated: gating functionality removed. Returns a marker dict."""
    print("[INFO] Gating feature search has been removed and is disabled.")
    return {"removed": True}
    # Prepare X/y and feature names
    X_all, y_all, feature_names = _prepare_survival_xy(clean_df)
    global_cols, local_cols, _ = _find_feature_groups(feature_names)
    have_global = len(global_cols) > 0
    have_local = len(local_cols) > 0
    if not (have_global and have_local):
        print("Gating search skipped: missing global or local feature groups.")
        return {"have_global": have_global, "have_local": have_local}

    # Build modeling DataFrame for OOF assignment
    df_model = pd.concat(
        [
            clean_df[["PE_Time", "VT/VF/SCD"]].reset_index(drop=True),
            X_all.reset_index(drop=True),
        ],
        axis=1,
    )

    # OOF risks for assignment on the full data
    risk_gl_oof, risk_lo_oof, risk_all_oof = _compute_oof_three_risks_lifelines(
        df_model,
        global_cols,
        local_cols,
        time_col="PE_Time",
        event_col="VT/VF/SCD",
        n_splits=n_splits,
        random_state=random_state,
    )
    evt = df_model["VT/VF/SCD"].values.astype(int)
    risks_stack = np.vstack([risk_gl_oof, risk_lo_oof, risk_all_oof])
    best_idx = np.zeros(len(df_model), dtype=int)
    best_idx[evt == 1] = np.argmax(risks_stack[:, evt == 1], axis=0)
    best_idx[evt == 0] = np.argmin(risks_stack[:, evt == 0], axis=0)

    # Candidate features are all numeric columns in X_all
    candidate_features = list(X_all.columns)
    n = len(df_model)
    valid_label_mask = (best_idx >= 0) & (best_idx <= 2)

    # Threshold grids
    if q_low_grid is None:
        q_low_grid = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    if q_high_grid is None:
        q_high_grid = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    # CV loop over features
    rng_seeds = [int(random_state + r) for r in range(max(1, int(n_repeats)))]
    feat_results: List[Dict[str, object]] = []

    for f in candidate_features:
        x = X_all[f].astype(float).values
        finite = np.isfinite(x) & valid_label_mask
        if finite.sum() < max(30, n_splits):
            continue

        best_score = -np.inf
        best_q = (np.nan, np.nan)

        # Repeated KFold
        for seed in rng_seeds:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for tr_idx, va_idx in kf.split(x):
                tr_mask = finite & np.isin(np.arange(n), tr_idx)
                va_mask = finite & np.isin(np.arange(n), va_idx)
                if tr_mask.sum() == 0 or va_mask.sum() == 0:
                    continue
                x_tr, y_tr = x[tr_mask], best_idx[tr_mask]
                x_va, y_va = x[va_mask], best_idx[va_mask]

                # Evaluate grid of quantiles on training fold
                for ql in q_low_grid:
                    for qh in q_high_grid:
                        if qh - ql < min_gap:
                            continue
                        try:
                            thr_l = float(np.nanquantile(x_tr, ql))
                            thr_h = float(np.nanquantile(x_tr, qh))
                        except Exception:
                            continue
                        if (
                            not np.isfinite(thr_l)
                            or not np.isfinite(thr_h)
                            or thr_l >= thr_h
                        ):
                            continue

                        # Zones on training
                        z_low_tr = x_tr < thr_l
                        z_high_tr = x_tr >= thr_h
                        z_mid_tr = ~(z_low_tr | z_high_tr)

                        # Majority mapping per zone
                        z2m: Dict[str, int] = {}
                        acc_parts = []
                        for name, m in (
                            ("low", z_low_tr),
                            ("mid", z_mid_tr),
                            ("high", z_high_tr),
                        ):
                            if m.sum() == 0:
                                continue
                            labels = y_tr[m]
                            vals, counts = np.unique(labels, return_counts=True)
                            maj = int(vals[np.argmax(counts)])
                            z2m[name] = maj
                            acc_parts.append((labels == maj).mean())
                        if len(acc_parts) == 0:
                            continue

                        # Validation accuracy
                        z_low_va = x_va < thr_l
                        z_high_va = x_va >= thr_h
                        z_mid_va = ~(z_low_va | z_high_va)
                        y_pred = np.full_like(y_va, fill_value=0)
                        if "low" in z2m:
                            y_pred[z_low_va] = z2m["low"]
                        if "mid" in z2m:
                            y_pred[z_mid_va] = z2m["mid"]
                        if "high" in z2m:
                            y_pred[z_high_va] = z2m["high"]
                        acc = float((y_pred == y_va).mean())

                        if acc > best_score + 1e-12:
                            best_score = acc
                            best_q = (ql, qh)

        if (
            np.isfinite(best_score)
            and best_score > -np.inf
            and np.isfinite(best_q[0])
            and np.isfinite(best_q[1])
        ):
            feat_results.append(
                {
                    "feature": f,
                    "cv_acc": float(best_score),
                    "q_low": float(best_q[0]),
                    "q_high": float(best_q[1]),
                    "n_finite": int(finite.sum()),
                }
            )

    if len(feat_results) == 0:
        print("No valid features for gating search.")
        return {
            "have_global": have_global,
            "have_local": have_local,
            "n_features": int(len(candidate_features)),
            "n_valid_features": 0,
        }

    # Select best feature by CV accuracy
    feat_results.sort(key=lambda d: d.get("cv_acc", -np.inf), reverse=True)
    best = feat_results[0]
    best_feature: str = str(best["feature"])
    ql_sel = float(best["q_low"])  # type: ignore
    qh_sel = float(best["q_high"])  # type: ignore

    # Compute final thresholds on all data
    x_full = X_all[best_feature].astype(float).values
    thr_low = float(np.nanquantile(x_full, ql_sel))
    thr_high = float(np.nanquantile(x_full, qh_sel))

    # Majority mapping per zone on all finite samples
    finite_full = np.isfinite(x_full) & valid_label_mask
    z_low = x_full < thr_low
    z_high = x_full >= thr_high
    z_mid = ~(z_low | z_high)

    mapping: Dict[str, int] = {}
    comp_counts: Dict[str, Dict[str, int]] = {"low": {}, "mid": {}, "high": {}}
    for name, m in ("low", z_low), ("mid", z_mid), ("high", z_high):
        mm = m & finite_full
        if mm.sum() == 0:
            continue
        labels = best_idx[mm]
        vals, counts = np.unique(labels, return_counts=True)
        maj = int(vals[np.argmax(counts)])
        mapping[name] = maj
        # counts per class for stacked bar
        cc: Dict[str, int] = {}
        for v, c in zip(vals, counts):
            cc[{0: "Global", 1: "Local", 2: "All"}.get(int(v), str(int(v)))] = int(c)
        comp_counts[name] = cc

    # Predicted labels on all samples by the rule
    y_pred_rule = np.zeros(n, dtype=int)
    y_pred_rule[z_low] = int(mapping.get("low", 0))
    y_pred_rule[z_mid] = int(mapping.get("mid", 0))
    y_pred_rule[z_high] = int(mapping.get("high", 0))

    # Overall agreement with OOF best labels (not a test score, just descriptive)
    overall_acc = (
        float((y_pred_rule[finite_full] == best_idx[finite_full]).mean())
        if finite_full.any()
        else float("nan")
    )

    # Visualizations (deprecated; kept for compatibility but will no-op if called)
    try:
        if output_dir:
            _ensure_dir(output_dir)
            # Save top-k feature summary
            topk = min(topk_report, len(feat_results))
            pd.DataFrame(feat_results[:topk]).to_csv(
                os.path.join(output_dir, "gating_search_topk.csv"), index=False
            )
    except Exception:
        pass

    # (Deprecated visuals removed)

    # Package outputs (unreachable in current deprecated flow)
    human_mapping = {k: ["Global", "Local", "All"][int(v)] for k, v in mapping.items()}
    result: Dict[str, object] = {
        "selected_feature": best_feature,
        "q_low": ql_sel,
        "q_high": qh_sel,
        "thr_low": float(thr_low),
        "thr_high": float(thr_high),
        "zone_to_model": human_mapping,
        "overall_agreement": overall_acc,
        "n_samples": int(n),
        "counts": {
            "Global": int((best_idx == 0).sum()),
            "Local": int((best_idx == 1).sum()),
            "All": int((best_idx == 2).sum()),
        },
        "topk": feat_results[: min(topk_report, len(feat_results))],
        "best_idx": [int(v) for v in best_idx.tolist()],
        "pred_labels_by_rule": [int(v) for v in y_pred_rule.tolist()],
    }

    print("\n=== Gating feature search (three-group separation) ===")
    print(f"- Selected feature: {result['selected_feature']}")
    print(
        f"- Quantiles (q_low, q_high): ({result['q_low']:.2f}, {result['q_high']:.2f})"
    )
    print(
        f"- Thresholds (low, high): ({result['thr_low']:.5g}, {result['thr_high']:.5g})"
    )
    print(
        f"- Zone -> Model mapping: low->{human_mapping.get('low','?')}, mid->{human_mapping.get('mid','?')}, high->{human_mapping.get('high','?')}"
    )
    print(f"- Agreement with OOF best labels (all data): {overall_acc:.4f}")

    return result


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

    # Ensure non-negative follow-up times (defensive against date inconsistencies)
    nicm["PE_Time"] = pd.to_numeric(nicm["PE_Time"], errors="coerce")
    nicm["PE_Time"] = nicm["PE_Time"].clip(lower=0)

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
    # Explicitly treat Female and ICD as labels (remove from feature list)
    for lab in ["Female", "ICD"]:
        if lab in features:
            features.remove(lab)

    # Imputation
    clean_df = conversion_and_imputation(nicm, features, labels)
    clean_df["NYHA Class"] = clean_df["NYHA Class"].replace({5: 4, 0: 1})

    return clean_df


def _prepare_survival_xy(
    clean_df: pd.DataFrame, drop_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Prepare X and y for survival modeling from cleaned data.

    Expects columns: "VT/VF/SCD" (event, 0/1) and "PE_Time" (time in days).
    """
    if drop_cols is None:
        # Treat Female and ICD as labels; also drop identifiers and time/event columns
        drop_cols = ["MRN", "VT/VF/SCD", "ICD", "PE_Time", "Female"]

    df = clean_df.copy()
    df = df.dropna(subset=["PE_Time"])  # ensure valid times
    if not pd.api.types.is_numeric_dtype(df["PE_Time"]):
        df["PE_Time"] = pd.to_numeric(df["PE_Time"], errors="coerce")
        df = df.dropna(subset=["PE_Time"])

    # Clip negative durations to zero to satisfy scikit-survival requirements
    df["PE_Time"] = df["PE_Time"].astype(float).clip(lower=0.0)

    df["VT/VF/SCD"] = df["VT/VF/SCD"].fillna(0).astype(int).astype(bool)

    X = df.drop(columns=drop_cols, errors="ignore")
    # Defensive: coerce any non-numeric leftovers
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        X[non_numeric] = X[non_numeric].apply(pd.to_numeric, errors="coerce")
        for col in non_numeric:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median())

    # Ensure all features are plain float64 to avoid pandas extension dtypes
    try:
        X = X.astype(float)
    except Exception:
        # Fallback: convert columns individually
        for c in X.columns:
            try:
                X[c] = X[c].astype(float)
            except Exception:
                X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
                if X[c].isnull().any():
                    X[c] = X[c].fillna(X[c].median())
    feature_names = X.columns.tolist()
    y = Surv.from_dataframe(event="VT/VF/SCD", time="PE_Time", data=df)
    return X, y, feature_names


def train_coxph_model(
    clean_df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
    output_dir: Optional[str] = None,
) -> Tuple[CoxPHSurvivalAnalysis, Dict[str, float], pd.Series]:
    """
    Train CoxPH on cleaned data and evaluate with concordance index on a hold-out test set.

    Returns: (fitted model, metrics dict, feature_importances series)
    """
    X, y, feature_names = _prepare_survival_xy(clean_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    _log_progress("Training Cox model on train split", True)
    model = _fit_coxph_clean(X_train, y_train)

    # Use model's risk scores for C-index evaluation
    try:
        if model is not None:
            X_test_use = X_test
            try:
                feature_names_in = getattr(model, "feature_names_in_", None)
                if feature_names_in is not None:
                    X_test_use = X_test.loc[:, list(feature_names_in)]
            except Exception:
                X_test_use = X_test
            risk_scores = model.predict(X_test_use)
        else:
            risk_scores = np.zeros(len(X_test), dtype=float)
    except Exception:
        risk_scores = np.zeros(len(X_test), dtype=float)
    e_field, t_field = _surv_field_names(y_test)
    test_c_index = float(
        concordance_index_censored(
            y_test[e_field].astype(bool), y_test[t_field].astype(float), risk_scores
        )[0]
    )
    # Use permutation importance on the hold-out set as a model-agnostic alternative.
    if model is not None:
        try:
            names_in = _model_feature_names(model)
            if names_in is None:
                names_in = list(X_train.columns)
            X_pi = _align_X_to_model(model, X_test)
            _log_progress("Computing permutation importance on test split", True)
            perm_result = permutation_importance(
                model,
                X_pi,
                y_test,
                n_repeats=20,
                random_state=random_state,
                n_jobs=-1,
            )
            feat_imp = pd.Series(
                perm_result.importances_mean, index=names_in
            ).sort_values(ascending=False)
        except Exception:
            feat_imp = pd.Series(dtype=float)
    else:
        feat_imp = pd.Series(dtype=float)
    metrics = {"test_c_index": float(test_c_index)}
    # Visualization: feature importance
    try:
        _plot_series_barh(
            feat_imp,
            topn=min(20, len(feat_imp)),
            title="Permutation importance (test)",
            xlabel="Importance (mean)",
            output_dir=output_dir,
            filename="feature_importance.png",
            color="#2ca02c",
        )
    except Exception:
        pass
    return model, metrics, feat_imp


def _find_feature_groups(
    feature_names: List[str],
) -> Tuple[List[str], List[str], Optional[str]]:
    """
    Return (global_cols, local_cols, gating_feature); gating is always None.

    Priority:
    - Use user-configured GLOBAL_FEATURES and LOCAL_FEATURES (intersection with existing columns).
    - Otherwise, fall back to simple name-heuristics (still no gating).
    """
    names = list(feature_names)

    # 1) User-configured lists first (intersect with existing columns)
    if GLOBAL_FEATURES or LOCAL_FEATURES:
        g = [c for c in GLOBAL_FEATURES if c in names]
        l = [c for c in LOCAL_FEATURES if c in names]
        # Deduplicate and avoid overlap
        l = [c for c in l if c not in g]
        return g, l, None

    # 2) Heuristic fallback (no gating)
    lower = [n.lower() for n in names]

    def has(substr: str) -> List[str]:
        s = substr.lower()
        return [names[i] for i, n in enumerate(lower) if s in n]

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

    local_cols = [c for c in names if c in local_cols_set]
    global_cols = [c for c in names if c in global_cols_set and c not in local_cols_set]
    return global_cols, local_cols, None


def _risk_at_time(model: object, X: pd.DataFrame, t: float) -> np.ndarray:
    """Risk score at time t as 1 - S(t) when available; otherwise, scaled risk score.

    For models supporting predict_survival_function (e.g., CoxPHSurvivalAnalysis), use 1 - S(t).
    Fallback: use model.predict(X) and min-max scale to [0, 1] for comparability.
    """
    if X.empty:
        return np.zeros(0, dtype=float)
    # Align to training features if available
    X_use = _align_X_to_model(model, X)
    # Preferred path: survival function available
    try:
        if hasattr(model, "predict_survival_function"):
            surv = model.predict_survival_function(X_use)
            s_at = np.array([sf(t) for sf in surv], dtype=float)
            s_at = np.clip(s_at, 0.0, 1.0)
            return 1.0 - s_at
    except Exception:
        pass
    # Fallback: use risk scores and scale to [0,1]
    try:
        scores = np.asarray(model.predict(X_use), dtype=float).ravel()
        finite = np.isfinite(scores)
        if not finite.any():
            return np.zeros(len(X), dtype=float)
        s_min = float(np.nanmin(scores[finite]))
        s_max = float(np.nanmax(scores[finite]))
        if s_max > s_min:
            scaled = (scores - s_min) / (s_max - s_min)
        else:
            scaled = np.zeros_like(scores)
        scaled[~finite] = 0.0
        return np.clip(scaled, 0.0, 1.0)
    except Exception:
        return np.zeros(len(X), dtype=float)


def _surv_field_names(y_arr) -> Tuple[str, str]:
    names = getattr(y_arr.dtype, "names", None)
    if not names or len(names) < 2:
        return "event", "time"
    event_field = "event" if "event" in names else names[0]
    time_candidates = [n for n in names if n != event_field]
    time_field = "time" if "time" in names else time_candidates[0]
    return event_field, time_field


def _model_feature_names(model) -> Optional[List[str]]:
    """Best-effort to extract trained feature names from estimator or pipeline."""
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        try:
            return list(names)
        except Exception:
            pass
    try:
        steps = getattr(model, "named_steps", None)
        if isinstance(steps, dict):
            # Common last-step name
            last = (
                steps.get("coxph")
                or steps.get("final")
                or steps.get(list(steps.keys())[-1])
            )
            if last is not None:
                last_names = getattr(last, "feature_names_in_", None)
                if last_names is not None:
                    return list(last_names)
    except Exception:
        pass
    return None


def _align_X_to_model(model, X: pd.DataFrame) -> pd.DataFrame:
    """Align columns and order of X to match the model's trained features.

    - Adds any missing trained columns as zeros
    - Drops extra columns
    - Orders columns exactly as trained
    """
    names = _model_feature_names(model)
    if not names:
        return X
    Z = X.copy()
    for c in names:
        if c not in Z.columns:
            Z[c] = 0.0
    try:
        Z = Z.loc[:, names]
    except Exception:
        # Fallback: keep intersection only, in the learned order
        keep = [c for c in names if c in Z.columns]
        Z = Z[keep]
    return Z


def _sanitize_cox_features_matrix(
    X: pd.DataFrame, corr_threshold: float = 0.995, verbose: bool = False
) -> pd.DataFrame:
    """Drop constant, duplicate, and highly correlated columns to stabilize Cox fitting."""
    Xs = X.copy()
    for c in Xs.columns:
        if not pd.api.types.is_numeric_dtype(Xs[c]):
            Xs[c] = pd.to_numeric(Xs[c], errors="coerce")

    # Drop columns with all missing or only one unique non-nan value
    nunique = Xs.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols and verbose:
        print(f"[Cox] drop constant/no-info cols: {constant_cols}")
    Xs = Xs.drop(columns=constant_cols, errors="ignore")

    if Xs.shape[1] == 0:
        return Xs

    # Remove exactly duplicated columns
    try:
        X_filled = Xs.fillna(0.0)
        duplicated_mask = X_filled.T.duplicated(keep="first")
        if duplicated_mask.any():
            dup_cols = Xs.columns[duplicated_mask.values].tolist()
            if verbose:
                print(f"[Cox] drop duplicated cols: {dup_cols}")
            Xs = Xs.loc[:, ~duplicated_mask.values]
    except Exception:
        pass

    if Xs.shape[1] <= 1:
        return Xs

    # Remove highly correlated columns (keep the first in order)
    try:
        corr = Xs.fillna(0.0).corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if (upper[col] >= corr_threshold).any()]
        if to_drop and verbose:
            print(f"[Cox] drop high-corr cols (|r|>={corr_threshold}): {to_drop}")
        Xs = Xs.drop(columns=to_drop, errors="ignore")
    except Exception:
        pass

    return Xs


def _fit_cox_lifelines(
    train_df: pd.DataFrame, feature_cols: List[str], time_col: str, event_col: str
) -> Optional[CoxPHFitter]:
    """Fit CoxPH via lifelines with sanitization and automatic retries on convergence issues."""
    if train_df is None or len(train_df) == 0 or not feature_cols:
        return None

    # Try a sequence of (corr_threshold, penalizer) settings; treat convergence warnings as errors
    corr_thresholds = [0.995, 0.98, 0.95, 0.90]
    penalizers = [0.1, 0.5, 1.0, 5.0, 10.0]

    for corr_thr in corr_thresholds:
        try:
            X_sanitized = _sanitize_cox_features_matrix(
                train_df[feature_cols], corr_threshold=corr_thr, verbose=False
            )
        except Exception:
            X_sanitized = train_df[feature_cols].copy()

        kept_features = list(X_sanitized.columns)
        if len(kept_features) == 0:
            continue

        df_fit = pd.concat(
            [
                train_df[[time_col, event_col]].reset_index(drop=True),
                X_sanitized.reset_index(drop=True),
            ],
            axis=1,
        )

        for pen in penalizers:
            cph = CoxPHFitter(penalizer=pen, l1_ratio=0.0)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    cph.fit(
                        df_fit,
                        duration_col=time_col,
                        event_col=event_col,
                        robust=True,
                        max_steps=MAX_LIFELINES_STEPS,
                    )
                return cph
            except Exception:
                continue

    return None


def _predict_risk_lifelines(model: CoxPHFitter, df: pd.DataFrame) -> np.ndarray:
    # Ensure output length matches input rows to avoid shape mismatches downstream
    if df is None or len(df) == 0:
        return np.zeros(0, dtype=float)
    if model is None:
        return np.zeros(len(df), dtype=float)
    try:
        model_features = list(model.params_.index)
        X = df.copy()
        for c in model_features:
            if c not in X.columns:
                X[c] = 0.0
        X = X[model_features]
        risk = model.predict_partial_hazard(X).values.reshape(-1)
        risk = np.asarray(risk, dtype=float)
        risk[~np.isfinite(risk)] = 0.0
        return risk
    except Exception:
        return np.zeros(len(df), dtype=float)


def _forward_select_features_lifelines(
    df: pd.DataFrame,
    candidate_features: List[str],
    time_col: str,
    event_col: str,
    random_state: int = 42,
    max_features: Optional[int] = None,
    verbose: bool = False,
) -> List[str]:
    """Greedy forward selection to maximize validation C-index (lifelines-based).

    - Uses an inner 70/30 split for selection.
    - Applies sanitization to avoid degenerate columns.
    - Returns a subset (possibly empty).
    """
    df_local = df.dropna(subset=[time_col, event_col]).copy()
    if df_local.empty:
        return []

    # Initial candidate pool after basic sanitization
    try:
        X_sanitized = _sanitize_cox_features_matrix(
            df_local[candidate_features], corr_threshold=0.995, verbose=False
        )
        pool: List[str] = list(X_sanitized.columns)
    except Exception:
        pool = [f for f in candidate_features if f in df_local.columns]

    if len(pool) <= 1:
        return list(pool)

    # Inner split
    try:
        tr_df, va_df = train_test_split(
            df_local,
            test_size=0.3,
            random_state=random_state,
            stratify=df_local[event_col] if df_local[event_col].nunique() > 1 else None,
        )
    except Exception:
        tr_df, va_df = train_test_split(df_local, test_size=0.3, random_state=random_state)

    selected: List[str] = []
    best_val_cidx: float = -np.inf
    remaining = list(pool)
    max_iters = (
        len(remaining) if max_features is None else max(0, min(len(remaining), max_features))
    )

    for step_idx in _maybe_tqdm(range(max_iters), total=max_iters, desc=f"FS-Forward(seed={random_state})", leave=False):
        best_feat = None
        best_feat_cidx = best_val_cidx
        if verbose:
            try:
                _log_progress(
                    f"FS-Forward seed={random_state} step {step_idx+1}/{max_iters}, remaining {len(remaining)}",
                    True,
                )
            except Exception:
                pass
        for feat in list(remaining):
            trial_feats = selected + [feat]
            try:
                cph = _fit_cox_lifelines(tr_df, trial_feats, time_col, event_col)
                if cph is None:
                    cidx = np.nan
                else:
                    risk_val = _predict_risk_lifelines(cph, va_df)
                    cidx = concordance_index(
                        va_df[time_col].values, -risk_val, va_df[event_col].values
                    )
            except Exception:
                cidx = np.nan
            if np.isfinite(cidx) and (cidx > best_feat_cidx + 1e-12):
                best_feat_cidx = float(cidx)
                best_feat = feat

        if best_feat is None:
            break
        selected.append(best_feat)
        remaining.remove(best_feat)
        best_val_cidx = best_feat_cidx
        if verbose:
            try:
                _log_progress(
                    f"FS-Forward add {best_feat} -> val c-index {best_val_cidx:.4f}",
                    True,
                )
            except Exception:
                pass

    return selected


def _stability_select_features_lifelines(
    df: pd.DataFrame,
    candidate_features: List[str],
    time_col: str,
    event_col: str,
    seeds: List[int],
    threshold: float = 0.5,
    max_features: Optional[int] = None,
    verbose: bool = False,
) -> List[str]:
    """Run forward selection across multiple seeds and keep features
    with selection frequency >= threshold.
    """
    if df is None or df.empty:
        return []

    # Initial sanitized pool
    try:
        X_pool = _sanitize_cox_features_matrix(
            df[candidate_features], corr_threshold=0.995, verbose=False
        )
        pool = list(X_pool.columns)
    except Exception:
        pool = [f for f in candidate_features if f in df.columns]

    if len(pool) == 0:
        return []

    from collections import Counter

    counter: Counter = Counter()
    total_runs = 0
    for idx, s in _maybe_tqdm(list(enumerate(seeds, start=1)), total=len(seeds), desc="FS-Stability", leave=False):
        try:
            if verbose:
                try:
                    _log_progress(f"FS-Stability {idx}/{len(seeds)} starting", True)
                except Exception:
                    pass
            sel = _forward_select_features_lifelines(
                df=df,
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

    ranked = list(counter.most_common())
    kept: List[str] = []
    for feat, count in ranked:
        freq = count / total_runs
        if freq >= threshold:
            kept.append(feat)

    # Final sanitization
    if kept:
        try:
            X_final = _sanitize_cox_features_matrix(
                df[kept], corr_threshold=0.995, verbose=False
            )
            return list(X_final.columns)
        except Exception:
            return kept
    return []

def _compute_oof_three_risks_lifelines(
    df: pd.DataFrame,
    global_cols: List[str],
    local_cols: List[str],
    time_col: str,
    event_col: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Out-of-fold risk scores for Global/Local/All CoxPH (lifelines)."""
    n = len(df)
    risk_glob = np.full(n, np.nan, dtype=float)
    risk_loc = np.full(n, np.nan, dtype=float)
    risk_all = np.full(n, np.nan, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_feature_cols = [
        c for c in df.columns if c not in [time_col, event_col, "MRN", "ICD"]
    ]

    for fold_idx, (tr_idx, va_idx) in _maybe_tqdm(list(enumerate(kf.split(df), start=1)), total=n_splits, desc="OOF Folds", leave=False):
        _log_progress(f"OOF fold {fold_idx}/{n_splits} fitting models", True)
        tr = df.iloc[tr_idx]
        va = df.iloc[va_idx]

        # Fit three models on fold-train
        model_gl = (
            _fit_cox_lifelines(
                tr, [c for c in global_cols if c in tr.columns], time_col, event_col
            )
            if len(global_cols) > 0
            else None
        )
        model_lo = (
            _fit_cox_lifelines(
                tr, [c for c in local_cols if c in tr.columns], time_col, event_col
            )
            if len(local_cols) > 0
            else None
        )
        model_all = _fit_cox_lifelines(
            tr, [c for c in all_feature_cols if c in tr.columns], time_col, event_col
        )

        # Predict risks on fold-val
        risk_gl = (
            _predict_risk_lifelines(model_gl, va)
            if model_gl is not None
            else np.zeros(len(va))
        )
        risk_lo = (
            _predict_risk_lifelines(model_lo, va)
            if model_lo is not None
            else np.zeros(len(va))
        )
        risk_al = (
            _predict_risk_lifelines(model_all, va)
            if model_all is not None
            else np.zeros(len(va))
        )

        risk_glob[va_idx] = risk_gl
        risk_loc[va_idx] = risk_lo
        risk_all[va_idx] = risk_al
        _log_progress(f"OOF fold {fold_idx}/{n_splits} done", True)

    # Replace any residual NaNs with 0
    for arr in (risk_glob, risk_loc, risk_all):
        mask = ~np.isfinite(arr)
        if mask.any():
            arr[mask] = 0.0
    return risk_glob, risk_loc, risk_all


def evaluate_three_model_assignment_and_classifier(
    clean_df: pd.DataFrame,
    test_size: float = 0.30,
    random_state: int = 42,
    output_dir: Optional[str] = None,
) -> Dict[str, object]:
    """Lifelines CoxPH for three feature sets (Global/Local/All),
    assign per-patient best model without fixed horizon, and train a classifier to predict the assignment.

    Assignment rule per patient i:
      - if event_i == 1: choose model with highest predicted risk
      - if event_i == 0: choose model with lowest predicted risk
    """
    df = clean_df.copy()
    # Ensure basic dtypes
    df["VT/VF/SCD"] = (
        pd.to_numeric(df["VT/VF/SCD"], errors="coerce").fillna(0).astype(int)
    )
    df["PE_Time"] = (
        pd.to_numeric(df["PE_Time"], errors="coerce").fillna(0).astype(float)
    )
    df = df.dropna(subset=["PE_Time"]).copy()

    # Feature groups
    X_all, y_all, feature_names = _prepare_survival_xy(df)
    global_cols, local_cols, _ = _find_feature_groups(feature_names)
    have_global = len(global_cols) > 0
    have_local = len(local_cols) > 0
    if not (have_global and have_local):
        print("Three-model assignment skipped: missing global or local feature groups.")
        return {"have_global": have_global, "have_local": have_local}

    # Build a modeling DataFrame for lifelines containing all features
    feat_all_cols = [c for c in feature_names]
    # Keep MRN (if present) for identification in outputs; models ignore MRN
    left_cols = ["PE_Time", "VT/VF/SCD"] + (["MRN"] if "MRN" in df.columns else [])
    df_model = pd.concat(
        [
            df[left_cols].reset_index(drop=True),
            X_all[feat_all_cols].reset_index(drop=True),
        ],
        axis=1,
    )

    # Candidate feature sets for three models
    cand_glob = [c for c in global_cols if c in df_model.columns]
    cand_local = [c for c in local_cols if c in df_model.columns]
    cand_all = [c for c in feat_all_cols if c in df_model.columns]

    # Forward selection per model on an inner split; then refit on full data and infer on full data
    seeds_for_stability = list(range(10))
    _log_progress("Selecting features for Global model", True)
    sel_glob = _stability_select_features_lifelines(
        df_model, cand_glob, time_col="PE_Time", event_col="VT/VF/SCD", seeds=seeds_for_stability, threshold=0.4, verbose=False
    ) if len(cand_glob) > 0 else []
    _log_progress("Selecting features for Local model", True)
    sel_local = _stability_select_features_lifelines(
        df_model, cand_local, time_col="PE_Time", event_col="VT/VF/SCD", seeds=seeds_for_stability, threshold=0.4, verbose=False
    ) if len(cand_local) > 0 else []
    _log_progress("Selecting features for All-features model", True)
    sel_all = _stability_select_features_lifelines(
        df_model, cand_all, time_col="PE_Time", event_col="VT/VF/SCD", seeds=seeds_for_stability, threshold=0.4, verbose=False
    ) if len(cand_all) > 0 else []

    # Ensure non-empty by falling back to candidate lists
    use_glob = sel_glob if sel_glob else cand_glob
    use_local = sel_local if sel_local else cand_local
    use_all = sel_all if sel_all else cand_all

    # Fit on all data
    _log_progress("Fitting Global/Local/All models on full data", True)
    model_gl = _fit_cox_lifelines(df_model, use_glob, time_col="PE_Time", event_col="VT/VF/SCD") if len(use_glob) > 0 else None
    model_lo = _fit_cox_lifelines(df_model, use_local, time_col="PE_Time", event_col="VT/VF/SCD") if len(use_local) > 0 else None
    model_all = _fit_cox_lifelines(df_model, use_all, time_col="PE_Time", event_col="VT/VF/SCD") if len(use_all) > 0 else None

    # Inference on all data
    risk_gl_full = _predict_risk_lifelines(model_gl, df_model) if model_gl is not None else np.zeros(len(df_model))
    risk_lo_full = _predict_risk_lifelines(model_lo, df_model) if model_lo is not None else np.zeros(len(df_model))
    risk_all_full = _predict_risk_lifelines(model_all, df_model) if model_all is not None else np.zeros(len(df_model))

    # Assignment on all data based on events
    evt = df_model["VT/VF/SCD"].values.astype(int)
    risks_stack = np.vstack([risk_gl_full, risk_lo_full, risk_all_full])
    best_idx = np.zeros(len(df_model), dtype=int)
    best_idx[evt == 1] = np.argmax(risks_stack[:, evt == 1], axis=0)
    best_idx[evt == 0] = np.argmin(risks_stack[:, evt == 0], axis=0)

    # Train/test split for classifier and evaluation（用于评估，不影响全体标签的产生与保存）
    tr_idx, te_idx = train_test_split(
        np.arange(len(df_model)),
        test_size=test_size,
        random_state=random_state,
        stratify=evt,
    )
    _log_progress("Splitting data for classifier evaluation", True)
    tr_df = df_model.iloc[tr_idx].copy()
    te_df = df_model.iloc[te_idx].copy()

    # Generate assignment labels without leakage:
    # - Train OOF risks on the TRAIN split using the previously selected features
    # - Assign TRAIN labels from OOF risks; TEST labels from models trained on TRAIN
    from sklearn.model_selection import KFold as _KFold

    n_tr = len(tr_df)
    n_splits_tr = max(2, min(5, n_tr))
    risk_gl_tr = np.full(n_tr, np.nan, dtype=float)
    risk_lo_tr = np.full(n_tr, np.nan, dtype=float)
    risk_all_tr = np.full(n_tr, np.nan, dtype=float)
    kf_tr = _KFold(n_splits=n_splits_tr, shuffle=True, random_state=random_state)

    use_glob_tr = [c for c in use_glob if c in tr_df.columns]
    use_local_tr = [c for c in use_local if c in tr_df.columns]
    use_all_tr = [c for c in use_all if c in tr_df.columns]

    for _fold_idx, (tr_i, va_i) in enumerate(kf_tr.split(tr_df), start=1):
        tr_part = tr_df.iloc[tr_i]
        va_part = tr_df.iloc[va_i]
        m_gl = _fit_cox_lifelines(tr_part, use_glob_tr, time_col="PE_Time", event_col="VT/VF/SCD") if len(use_glob_tr) > 0 else None
        m_lo = _fit_cox_lifelines(tr_part, use_local_tr, time_col="PE_Time", event_col="VT/VF/SCD") if len(use_local_tr) > 0 else None
        m_all = _fit_cox_lifelines(tr_part, use_all_tr, time_col="PE_Time", event_col="VT/VF/SCD") if len(use_all_tr) > 0 else None
        risk_gl_tr[va_i] = _predict_risk_lifelines(m_gl, va_part) if m_gl is not None else 0.0
        risk_lo_tr[va_i] = _predict_risk_lifelines(m_lo, va_part) if m_lo is not None else 0.0
        risk_all_tr[va_i] = _predict_risk_lifelines(m_all, va_part) if m_all is not None else 0.0

    # Replace NaNs with zeros to keep downstream logic simple
    for arr in (risk_gl_tr, risk_lo_tr, risk_all_tr):
        mask = ~np.isfinite(arr)
        if mask.any():
            arr[mask] = 0.0

    evt_tr = tr_df["VT/VF/SCD"].values.astype(int)
    risks_stack_tr = np.vstack([risk_gl_tr, risk_lo_tr, risk_all_tr])
    y_tr_assign = np.zeros(n_tr, dtype=int)
    y_tr_assign[evt_tr == 1] = np.argmax(risks_stack_tr[:, evt_tr == 1], axis=0)
    y_tr_assign[evt_tr == 0] = np.argmin(risks_stack_tr[:, evt_tr == 0], axis=0)

    # Fit three lifelines Cox models on training set for test-time metrics
    _log_progress("Fitting three lifelines models on training split", True)
    feat_gl_tr = [c for c in use_glob if c in tr_df.columns]
    feat_lo_tr = [c for c in use_local if c in tr_df.columns]
    feat_all_cols_tr = [c for c in use_all if c in tr_df.columns]
    model_gl = _fit_cox_lifelines(
        tr_df,
        feat_gl_tr,
        time_col="PE_Time",
        event_col="VT/VF/SCD",
    ) if len(feat_gl_tr) > 0 else None
    model_lo = _fit_cox_lifelines(
        tr_df,
        feat_lo_tr,
        time_col="PE_Time",
        event_col="VT/VF/SCD",
    ) if len(feat_lo_tr) > 0 else None
    model_all = _fit_cox_lifelines(
        tr_df,
        feat_all_cols_tr,
        time_col="PE_Time",
        event_col="VT/VF/SCD",
    ) if len(feat_all_cols_tr) > 0 else None

    _log_progress("Predicting risks on test split", True)
    risk_gl_te = _predict_risk_lifelines(model_gl, te_df)
    risk_lo_te = _predict_risk_lifelines(model_lo, te_df)
    risk_all_te = _predict_risk_lifelines(model_all, te_df)

    # Derive TEST assignment labels from trained TRAIN models (no leakage)
    evt_te = te_df["VT/VF/SCD"].values.astype(int)
    risks_stack_te = np.vstack([risk_gl_te, risk_lo_te, risk_all_te])
    y_te_assign = np.zeros(len(te_df), dtype=int)
    y_te_assign[evt_te == 1] = np.argmax(risks_stack_te[:, evt_te == 1], axis=0)
    y_te_assign[evt_te == 0] = np.argmin(risks_stack_te[:, evt_te == 0], axis=0)

    # Test-set C-index for three models
    cidx_gl = concordance_index(
        te_df["PE_Time"].values, -risk_gl_te, te_df["VT/VF/SCD"].values
    )
    cidx_lo = concordance_index(
        te_df["PE_Time"].values, -risk_lo_te, te_df["VT/VF/SCD"].values
    )
    cidx_all = concordance_index(
        te_df["PE_Time"].values, -risk_all_te, te_df["VT/VF/SCD"].values
    )

    # Train classifier to predict assignment label (0=Global,1=Local,2=All)
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.preprocessing import StandardScaler as SkStandardScaler

    clf = SkPipeline(
        [
            ("scaler", SkStandardScaler()),
            (
                "clf",
                LogisticRegression(
                    multi_class="multinomial", solver="lbfgs", max_iter=2000
                ),
            ),
        ]
    )
    X_tr_cls = tr_df.drop(columns=["PE_Time", "VT/VF/SCD", "MRN"], errors="ignore")
    X_te_cls = te_df.drop(columns=["PE_Time", "VT/VF/SCD", "MRN"], errors="ignore")
    _log_progress("Training assignment classifier", True)
    unique_labels = np.unique(y_tr_assign)
    if unique_labels.size < 2:
        # Fallback: constant predictor when only one class present in training labels
        const_label = int(unique_labels[0]) if unique_labels.size == 1 else 0
        y_pred = np.full_like(y_te_assign, const_label)
        acc = float(accuracy_score(y_te_assign, y_pred))
        f1_macro = float(
            f1_score(y_te_assign, y_pred, average="macro", zero_division=0)
        )
    else:
        clf.fit(X_tr_cls, y_tr_assign)
        y_pred = clf.predict(X_te_cls)
        acc = float(accuracy_score(y_te_assign, y_pred))
        f1_macro = float(
            f1_score(y_te_assign, y_pred, average="macro", zero_division=0)
        )

    print("\nThree-model assignment via lifelines CoxPH (train/test split):")
    print(
        f"- Selected | Global: {len(use_glob)} | Local: {len(use_local)} | All: {len(use_all)}"
    )
    print(
        f"- Test C-index | Global: {cidx_gl:.4f} | Local: {cidx_lo:.4f} | All: {cidx_all:.4f}"
    )
    print(f"- Assignment classifier | Acc: {acc:.4f} | F1-macro: {f1_macro:.4f}")
    try:
        print(
            classification_report(
                y_te_assign, y_pred, target_names=["Global", "Local", "All"]
            )
        )
    except Exception:
        pass

    # Visualize classifier top coefficients per class
    try:
        lr = clf.named_steps.get("clf")
        if lr is not None and hasattr(lr, "coef_"):
            coef = lr.coef_
            feat = X_tr_cls.columns.to_list()
            coef_df = pd.DataFrame(
                coef, columns=feat, index=["Global", "Local", "All"]
            ).T
            if output_dir:
                _ensure_dir(output_dir)
                coef_df.to_csv(
                    os.path.join(output_dir, "assignment_lifelines_coef.csv")
                )
            topk = 15
            for cls in ["Global", "Local", "All"]:
                ser = coef_df[cls].abs().sort_values(ascending=False).head(topk)
                _plot_series_barh(
                    ser,
                    topn=len(ser),
                    title=f"Assignment classifier top features for {cls}",
                    xlabel="|coefficient|",
                    output_dir=output_dir,
                    filename=f"assignment_lifelines_top_{cls.lower()}.png",
                    color=(
                        "#2ca02c"
                        if cls == "Global"
                        else ("#ff7f0e" if cls == "Local" else "#1f77b4")
                    ),
                )
    except Exception:
        pass

    # Counts plot of assigned groups (based on all OOF labels)
    try:
        counts = pd.Series(
            {
                "Global": int((best_idx == 0).sum()),
                "Local": int((best_idx == 1).sum()),
                "All": int((best_idx == 2).sum()),
            }
        )
        _plot_series_barh(
            counts,
            topn=len(counts),
            title="Assigned best model counts (OOF)",
            xlabel="Count",
            output_dir=output_dir,
            filename="assignment_lifelines_counts.png",
            color="#9467bd",
        )
    except Exception:
        pass

    # Save and return per-sample assignment labels for the full dataset
    try:
        mapping = {0: "Global", 1: "Local", 2: "All"}
        id_series = (
            df_model["MRN"]
            if "MRN" in df_model.columns
            else pd.Series(np.arange(len(df_model)), name="index")
        )
        assign_df = pd.DataFrame(
            {
                id_series.name: id_series.values,
                "assignment_label": best_idx.astype(int),
                "assignment_group": pd.Series(best_idx).map(mapping).fillna("").values,
            }
        )
        if output_dir:
            _ensure_dir(output_dir)
            assign_df.to_csv(
                os.path.join(output_dir, "assignment_labels_all.csv"), index=False
            )
            with open(os.path.join(output_dir, "selected_features.txt"), "w") as f:
                f.write("Global:\n" + ",".join(use_glob) + "\n")
                f.write("Local:\n" + ",".join(use_local) + "\n")
                f.write("All:\n" + ",".join(use_all) + "\n")
    except Exception:
        pass

    # Reverse feature analysis on ALL samples: train classifier on full X to predict assignment
    try:
        from sklearn.pipeline import Pipeline as SkPipeline
        from sklearn.preprocessing import StandardScaler as SkStandardScaler

        X_cls_all = df_model.drop(
            columns=["PE_Time", "VT/VF/SCD", "MRN"], errors="ignore"
        )
        y_cls_all = best_idx.astype(int)
        clf_all = SkPipeline(
            [
                ("scaler", SkStandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        multi_class="multinomial", solver="lbfgs", max_iter=2000
                    ),
                ),
            ]
        )
        clf_all.fit(X_cls_all, y_cls_all)
        lr_all = clf_all.named_steps.get("clf")
        if lr_all is not None and hasattr(lr_all, "coef_"):
            coef_all = lr_all.coef_
            feat_all = X_cls_all.columns.to_list()
            coef_df_all = pd.DataFrame(
                coef_all, columns=feat_all, index=["Global", "Local", "All"]
            ).T
            if output_dir:
                _ensure_dir(output_dir)
                coef_df_all.to_csv(
                    os.path.join(output_dir, "assignment_coefficients_all.csv")
                )
            # Plot top features per class
            topk = 20
            for cls in ["Global", "Local", "All"]:
                ser = coef_df_all[cls].abs().sort_values(ascending=False).head(topk)
                _plot_series_barh(
                    ser,
                    topn=len(ser),
                    title=f"Assignment predictor (ALL data): top features for {cls}",
                    xlabel="|coefficient|",
                    output_dir=output_dir,
                    filename=f"assignment_full_top_{cls.lower()}.png",
                    color=(
                        "#2ca02c"
                        if cls == "Global"
                        else ("#ff7f0e" if cls == "Local" else "#1f77b4")
                    ),
                )
    except Exception:
        pass

    return {
        "c_index_global": float(cidx_gl),
        "c_index_local": float(cidx_lo),
        "c_index_all": float(cidx_all),
        "accuracy": acc,
        "macro_f1": f1_macro,
        "n_train": int(len(df_model)),
        "n_test": int(len(df_model)),
        "assignment_all_labels": [int(v) for v in best_idx.tolist()],
        "assignment_all_ids": (
            df_model["MRN"].tolist()
            if "MRN" in df_model.columns
            else list(range(len(df_model)))
        ),
        "selected_features": {
            "global": use_glob,
            "local": use_local,
            "all": use_all,
        },
    }


def run_assignment_experiments(
    clean_df: pd.DataFrame,
    runs: int = 50,
    test_size: float = 0.30,
    random_state: int = 42,
    output_dir: Optional[str] = None,
) -> Dict[str, object]:
    """Run multiple seeds of three-model assignment and aggregate metrics.

    Aggregates mean and 95% CI (normal approx) for C-index and classifier metrics.
    """
    seeds = [int(random_state + i) for i in range(max(1, int(runs)))]
    per_run: List[Dict[str, float]] = []

    for i, s in enumerate(seeds, start=1):
        _log_progress(f"Run {i}/{len(seeds)} (seed={s})", True)
        res = evaluate_three_model_assignment_and_classifier(
            clean_df=clean_df,
            test_size=test_size,
            random_state=s,
            output_dir=output_dir,
        )
        per_run.append(
            {
                "seed": float(s),
                "c_index_global": float(res.get("c_index_global", np.nan)),
                "c_index_local": float(res.get("c_index_local", np.nan)),
                "c_index_all": float(res.get("c_index_all", np.nan)),
                "accuracy": float(res.get("accuracy", np.nan)),
                "macro_f1": float(res.get("macro_f1", np.nan)),
            }
        )

    def _mean_ci(x: List[float]) -> Tuple[float, float, float]:
        arr = np.asarray(x, dtype=float)
        mask = np.isfinite(arr)
        if not mask.any():
            return float("nan"), float("nan"), float("nan")
        vals = arr[mask]
        n = len(vals)
        mu = float(np.nanmean(vals))
        if n <= 1:
            return mu, float("nan"), float("nan")
        se = float(np.nanstd(vals, ddof=1) / np.sqrt(n))
        ci = 1.96 * se
        return mu, mu - ci, mu + ci

    # Build summary
    keys = [
        "c_index_global",
        "c_index_local",
        "c_index_all",
        "accuracy",
        "macro_f1",
    ]
    summary: Dict[str, Tuple[float, float, float]] = {}
    for k in keys:
        summary[k] = _mean_ci([r.get(k, np.nan) for r in per_run])

    # Print summary
    print("\n=== Multi-seed summary (mean [95% CI]) ===")
    for k in keys:
        mu, lo, hi = summary[k]
        if np.isfinite(mu) and np.isfinite(lo) and np.isfinite(hi):
            print(f"- {k}: {mu:.3f} ({lo:.3f}, {hi:.3f})")
        else:
            print(f"- {k}: {mu}")

    # Optional export
    try:
        if output_dir:
            _ensure_dir(output_dir)
            runs_df = pd.DataFrame(per_run)
            runs_df.to_csv(os.path.join(output_dir, "assignment_multi_seed_runs.csv"), index=False)
            # Expand summary to a table for convenient saving
            sm_rows = []
            for k, (mu, lo, hi) in summary.items():
                sm_rows.append({"metric": k, "mean": mu, "ci_lower": lo, "ci_upper": hi})
            pd.DataFrame(sm_rows).to_csv(
                os.path.join(output_dir, "assignment_multi_seed_summary.csv"), index=False
            )
    except Exception:
        pass

    return {
        "per_run": per_run,
        "summary": summary,
        "runs": int(len(seeds)),
    }


def _clean_X_for_cox(X: pd.DataFrame) -> pd.DataFrame:
    """Make design matrix numeric, finite, and reasonably conditioned for CoxPH.

    - Coerces non-numeric columns to numeric (invalid parsed as NaN)
    - Replaces inf/-inf with NaN, fills NaNs with column median (or 0.0 if median invalid)
    - Drops columns with all-NaN
    - Drops near-constant columns (std <= 1e-12)
    """
    Xc = X.copy()
    for col in Xc.columns:
        if not pd.api.types.is_numeric_dtype(Xc[col]):
            Xc[col] = pd.to_numeric(Xc[col], errors="coerce")
    Xc = Xc.replace([np.inf, -np.inf], np.nan)
    if Xc.shape[1] > 0:
        all_nan_cols = Xc.columns[Xc.isnull().all()].tolist()
        if all_nan_cols:
            Xc = Xc.drop(columns=all_nan_cols)
    for col in Xc.columns:
        if Xc[col].isnull().any():
            med = Xc[col].median()
            if not np.isfinite(med):
                med = 0.0
            Xc[col] = Xc[col].fillna(med)
    if Xc.shape[1] > 0:
        try:
            std = Xc.std(ddof=0)
            keep_cols = std[std > 1e-12].index.tolist()
            Xc = Xc[keep_cols] if len(keep_cols) > 0 else Xc.iloc[:, :0]
        except Exception:
            pass
    # Final cast to float64 to ensure compatibility with scikit-survival/sklearn
    try:
        Xc = Xc.astype(float)
    except Exception:
        for c in Xc.columns:
            try:
                Xc[c] = Xc[c].astype(float)
            except Exception:
                Xc[c] = pd.to_numeric(Xc[c], errors="coerce").astype(float).fillna(0.0)
    # Additional sanitization: drop constant/duplicate/highly-correlated columns
    Xc = _sanitize_cox_features_matrix(Xc, corr_threshold=0.995, verbose=False)
    return Xc


def _fit_coxph_clean(X: pd.DataFrame, y: np.ndarray) -> Optional[object]:
    """Fit CoxPH with defensive cleaning and scaling. Returns a pipeline or None."""
    Xc = _clean_X_for_cox(X)
    if Xc.shape[0] < 2 or Xc.shape[1] == 0:
        return None
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.preprocessing import StandardScaler as SkStandardScaler

    # Use scaling to prevent exp overflow in risk computations
    pipe = SkPipeline(
        [
            ("scaler", SkStandardScaler()),
            ("coxph", CoxPHSurvivalAnalysis()),
        ]
    )
    try:
        pipe.fit(Xc, y)
        return pipe
    except Exception:
        # Retry with stricter sanitization threshold
        Xc2 = _sanitize_cox_features_matrix(Xc, corr_threshold=0.95, verbose=False)
        if Xc2.shape[1] == 0:
            return None
        try:
            pipe.fit(Xc2, y)
            return pipe
        except Exception:
            return None


def _binary_outcome_at_time(y_arr, t: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (y_bin, known_mask) at time t.
    y_bin = 1 if event occurred by t, else 0 if survival past t is observed.
    If censored before t, label is unknown (known_mask = False).
    """
    e_field, tm_field = _surv_field_names(y_arr)
    evt = y_arr[e_field].astype(bool)
    tm = y_arr[tm_field].astype(float)
    known = (evt & (tm <= t)) | ((~evt) & (tm > t))
    y_bin = np.where(evt & (tm <= t), 1.0, 0.0)
    return y_bin, known


def _optimize_gate_quantiles(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    global_cols: List[str],
    local_cols: List[str],
    gating: Optional[str],
    random_state: int,
    time_horizon_days: float,
    q_low_grid: Optional[List[float]] = None,
    q_high_grid: Optional[List[float]] = None,
    min_gap: float = 0.10,
    inner_val_size: float = 0.33,
) -> Tuple[Optional[float], Optional[float], Dict[str, float]]:
    """Deprecated: gating functionality removed. Keep signature for backward compatibility."""
    return None, None, {}


def analyze_benefit_subgroup(
    clean_df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    percent_for_time: float = 0.75,
    margin: float = 0.0,
    topk_local_importance: int = 12,
    output_dir: Optional[str] = None,
) -> Dict[str, object]:
    """
    Identify patients who benefit from local features and assess local-feature importance in that subgroup.

    Approach:
    - Generate out-of-fold risks at a fold-specific time horizon t (percentile of train times).
    - Define per-sample benefit label by squared-error improvement at t.
    - Evaluate OOF C-index in benefit vs non-benefit groups (local vs global risks).
    - Fit local-only model on benefit subgroup and report permutation importances.
    """
    X_all, y_all, feature_names = _prepare_survival_xy(clean_df)
    global_cols, local_cols, _ = _find_feature_groups(feature_names)
    have_global = len(global_cols) > 0
    have_local = len(local_cols) > 0
    n = len(X_all)
    if n == 0 or not (have_global and have_local):
        print(
            "Benefit analysis skipped: insufficient data or feature groups not found."
        )
        return {
            "n": int(n),
            "have_global": have_global,
            "have_local": have_local,
        }

    risk_glob_oof = np.full(n, np.nan, dtype=float)
    risk_loc_oof = np.full(n, np.nan, dtype=float)
    risk_all_oof = np.full(n, np.nan, dtype=float)
    y_bin_oof = np.full(n, np.nan, dtype=float)
    known_oof = np.zeros(n, dtype=bool)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def _fit(X, y) -> Optional[object]:
        return _fit_coxph_clean(X, y)

    for fold_idx, (tr_idx, va_idx) in _maybe_tqdm(list(enumerate(kf.split(X_all), start=1)), total=n_splits, desc="Benefit CV", leave=False):
        _log_progress(f"Benefit CV fold {fold_idx}/{n_splits} start", True)
        X_tr, X_va = X_all.iloc[tr_idx], X_all.iloc[va_idx]
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]

        # Fold-specific time horizon
        e_field, t_field = _surv_field_names(y_tr)
        t_hor = (
            float(np.percentile(y_tr[t_field], percent_for_time * 100.0))
            if len(y_tr)
            else 365.0
        )
        if not np.isfinite(t_hor) or t_hor <= 0:
            t_hor = 365.0

        # Train models
        model_gl = _fit(X_tr[global_cols], y_tr) if have_global else None
        model_lo = _fit(X_tr[local_cols], y_tr) if have_local else None
        model_all = _fit(X_tr, y_tr)

        # OOF risk predictions at t
        risk_gl = (
            _risk_at_time(model_gl, X_va[global_cols], t_hor)
            if model_gl is not None
            else np.zeros(len(X_va))
        )
        risk_lo = (
            _risk_at_time(model_lo, X_va[local_cols], t_hor)
            if model_lo is not None
            else np.zeros(len(X_va))
        )
        risk_all = (
            _risk_at_time(model_all, X_va, t_hor)
            if model_all is not None
            else np.zeros(len(X_va))
        )

        # Binary outcome at t with known mask
        y_bin, known = _binary_outcome_at_time(y_va, t_hor)

        # Store
        risk_glob_oof[va_idx] = risk_gl
        risk_loc_oof[va_idx] = risk_lo
        risk_all_oof[va_idx] = risk_all
        y_bin_oof[va_idx] = y_bin
        known_oof[va_idx] = known
        _log_progress(f"Benefit CV fold {fold_idx}/{n_splits} done", True)

    # Define benefit by squared-error improvement with optional margin
    err_gl = (risk_glob_oof - y_bin_oof) ** 2
    err_lo = (risk_loc_oof - y_bin_oof) ** 2
    err_all = (risk_all_oof - y_bin_oof) ** 2

    valid = known_oof & np.isfinite(err_gl) & np.isfinite(err_lo) & np.isfinite(err_all)
    # Benefit definitions (incremental):
    benefit_local = valid & (err_gl - err_all > margin)  # Adding local to global helps
    benefit_global = valid & (err_lo - err_all > margin)  # Adding global to local helps
    # Best-of-three winner per sample
    best_idx = np.full(len(X_all), -1, dtype=int)
    if valid.any():
        triple = np.vstack(
            [err_gl[valid], err_lo[valid], err_all[valid]]
        )  # rows: G, L, A
        best = np.argmin(triple, axis=0)
        best_idx[np.where(valid)[0]] = best
    best_g = best_idx == 0
    best_l = best_idx == 1
    best_a = best_idx == 2

    # C-index within groups using OOF risks
    def _c_index(y, risk):
        e_field, t_field = _surv_field_names(y)
        evt = y[e_field].astype(bool)
        tm = y[t_field].astype(float)
        return float(concordance_index_censored(evt, tm, risk)[0])

    metrics: Dict[str, object] = {
        "n": int(n),
        "n_labeled": int(known_oof.sum()),
        "n_valid": int(valid.sum()),
        "n_benefit_local": int(benefit_local.sum()),
        "n_benefit_global": int(benefit_global.sum()),
        "n_best_global": int((valid & best_g).sum()),
        "n_best_local": int((valid & best_l).sum()),
        "n_best_all": int((valid & best_a).sum()),
    }

    if benefit_local.sum() > 1:
        metrics["c_index_global_in_benefitLocal"] = _c_index(
            y_all[benefit_local], risk_glob_oof[benefit_local]
        )
        metrics["c_index_all_in_benefitLocal"] = _c_index(
            y_all[benefit_local], risk_all_oof[benefit_local]
        )
    if benefit_global.sum() > 1:
        metrics["c_index_local_in_benefitGlobal"] = _c_index(
            y_all[benefit_global], risk_loc_oof[benefit_global]
        )
        metrics["c_index_all_in_benefitGlobal"] = _c_index(
            y_all[benefit_global], risk_all_oof[benefit_global]
        )

    # Train local-only model on benefit subgroup and compute permutation importance
    try:
        if benefit_local.sum() >= 10:
            # Importance under ALL-features model, restricted to local features (conditional on globals)
            X_ben_all = X_all.loc[benefit_local, :]
            y_ben = y_all[benefit_local]
            model_ben_all = _fit_coxph_clean(X_ben_all, y_ben)
            if model_ben_all is not None:
                try:
                    X_pi_all = _align_X_to_model(model_ben_all, X_ben_all)
                    names_all = _model_feature_names(model_ben_all) or list(
                        X_pi_all.columns
                    )
                    perm_all = permutation_importance(
                        model_ben_all,
                        X_pi_all,
                        y_ben,
                        n_repeats=20,
                        random_state=random_state,
                        n_jobs=-1,
                    )
                    fi_all = pd.Series(
                        perm_all.importances_mean, index=names_all
                    ).sort_values(ascending=False)
                except Exception:
                    fi_all = pd.Series(dtype=float)
            else:
                fi_all = pd.Series(dtype=float)
            fi_local_cond = fi_all[fi_all.index.isin(local_cols)].sort_values(
                ascending=False
            )
            metrics["local_feature_importance_in_benefit_conditional"] = (
                fi_local_cond.head(topk_local_importance)
            )
            try:
                _plot_series_barh(
                    fi_local_cond,
                    topn=topk_local_importance,
                    title="Benefit (A>G): Local features (conditional)",
                    xlabel="Importance (mean)",
                    output_dir=output_dir,
                    filename="benefit_local_conditional_importance.png",
                    color="#1f77b4",
                )
            except Exception:
                pass

            # Also report local-only model importance within benefit subgroup (pure local effect)
            X_ben_loc = X_all.loc[benefit_local, local_cols]
            model_ben_loc = _fit_coxph_clean(X_ben_loc, y_ben)
            if model_ben_loc is not None:
                try:
                    X_pi_loc = _align_X_to_model(model_ben_loc, X_ben_loc)
                    names_loc = _model_feature_names(model_ben_loc) or list(
                        X_pi_loc.columns
                    )
                    perm_loc = permutation_importance(
                        model_ben_loc,
                        X_pi_loc,
                        y_ben,
                        n_repeats=20,
                        random_state=random_state,
                        n_jobs=-1,
                    )
                    fi_loc = pd.Series(
                        perm_loc.importances_mean, index=names_loc
                    ).sort_values(ascending=False)
                except Exception:
                    fi_loc = pd.Series(dtype=float)
            else:
                fi_loc = pd.Series(dtype=float)
            metrics["local_feature_importance_in_benefit_localOnly"] = fi_loc.head(
                topk_local_importance
            )
            try:
                _plot_series_barh(
                    fi_loc,
                    topn=topk_local_importance,
                    title="Benefit (A>G): Local-only model importance",
                    xlabel="Importance (mean)",
                    output_dir=output_dir,
                    filename="benefit_local_only_importance.png",
                    color="#ff7f0e",
                )
            except Exception:
                pass

            print(
                "\nBenefit subgroup (A better than G): local feature importance (conditional on globals, top):"
            )
            print(metrics["local_feature_importance_in_benefit_conditional"])
            print("\nBenefit subgroup: local-only model feature importance (top):")
            print(metrics["local_feature_importance_in_benefit_localOnly"])
    except Exception:
        pass

    print("\nBenefit subgroup analysis:")
    print(
        f"- Labeled at t: {metrics['n_labeled']} / {metrics['n']} (valid={metrics['n_valid']})"
    )
    print(f"- Benefit (A better than G): {metrics['n_benefit_local']}")
    print(f"- Benefit (A better than L): {metrics['n_benefit_global']}")
    print(
        f"- Best model counts [G/L/A]: {metrics['n_best_global']} / {metrics['n_best_local']} / {metrics['n_best_all']}"
    )
    if "c_index_global_in_benefitLocal" in metrics:
        print(
            f"- C-index in BENEFIT(A>G): all={metrics['c_index_all_in_benefitLocal']:.4f}, global={metrics['c_index_global_in_benefitLocal']:.4f}"
        )
    if "c_index_local_in_benefitGlobal" in metrics:
        print(
            f"- C-index in BENEFIT(A>L): all={metrics['c_index_all_in_benefitGlobal']:.4f}, local={metrics['c_index_local_in_benefitGlobal']:.4f}"
        )

    # Counts visualization
    try:
        counts = pd.Series(
            {
                "Benefit(A>G)": metrics.get("n_benefit_local", 0),
                "Benefit(A>L)": metrics.get("n_benefit_global", 0),
                "Best=Global": metrics.get("n_best_global", 0),
                "Best=Local": metrics.get("n_best_local", 0),
                "Best=All": metrics.get("n_best_all", 0),
            }
        )
        _plot_series_barh(
            counts,
            topn=len(counts),
            title="Benefit subgroup counts",
            xlabel="Count",
            output_dir=output_dir,
            filename="benefit_counts.png",
            color="#2ca02c",
        )
    except Exception:
        pass

    return metrics


def evaluate_two_stage_strategy(
    clean_df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
    q_low: float = 0.40,
    q_high: float = 0.75,
    optimize_thresholds: bool = False,
    q_low_grid: Optional[List[float]] = None,
    q_high_grid: Optional[List[float]] = None,
    min_gap: float = 0.10,
    inner_val_size: float = 0.33,
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Deprecated: two-stage model and gating removed. Returns a marker dict."""
    # Removed in simplified codebase; kept as stub for backward compatibility
    print("[INFO] Two-stage strategy evaluation is not available in the simplified script.")
    return {"removed": True}


def _compute_oof_three_model_risks(
    X: pd.DataFrame,
    y: np.ndarray,
    global_cols: List[str],
    local_cols: List[str],
    n_splits: int,
    random_state: int,
    percent_for_time: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute out-of-fold risk scores at a fold-specific time horizon t for three models:
    - Global-only
    - Local-only
    - All-features

    Returns: (risk_glob_oof, risk_loc_oof, risk_all_oof, y_bin_oof, known_oof, t_mean)
    where y_bin_oof and known_oof are defined at the fold-specific t for each fold,
    collected into global arrays; t_mean is the average horizon across folds.
    """
    n = len(X)
    risk_glob_oof = np.full(n, np.nan, dtype=float)
    risk_loc_oof = np.full(n, np.nan, dtype=float)
    risk_all_oof = np.full(n, np.nan, dtype=float)
    y_bin_oof = np.full(n, np.nan, dtype=float)
    known_oof = np.zeros(n, dtype=bool)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def _fit(Xtr, ytr) -> Optional[CoxPHSurvivalAnalysis]:
        return _fit_coxph_clean(Xtr, ytr)

    t_values: List[float] = []
    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # Fold-specific horizon
        e_field, t_field = _surv_field_names(y_tr)
        t_hor = (
            float(np.percentile(y_tr[t_field], percent_for_time * 100.0))
            if len(y_tr)
            else 365.0
        )
        if not np.isfinite(t_hor) or t_hor <= 0:
            t_hor = 365.0
        t_values.append(t_hor)

        # Train models on fold-train
        model_gl = _fit(X_tr[global_cols], y_tr) if len(global_cols) > 0 else None
        model_lo = _fit(X_tr[local_cols], y_tr) if len(local_cols) > 0 else None
        model_all = _fit(X_tr, y_tr)

        # Predict risks on fold-val
        risk_gl = (
            _risk_at_time(model_gl, X_va[global_cols], t_hor)
            if model_gl is not None
            else np.zeros(len(X_va))
        )
        risk_lo = (
            _risk_at_time(model_lo, X_va[local_cols], t_hor)
            if model_lo is not None
            else np.zeros(len(X_va))
        )
        risk_all = (
            _risk_at_time(model_all, X_va, t_hor)
            if model_all is not None
            else np.zeros(len(X_va))
        )

        # Binary outcomes and known mask at t
        y_bin, known = _binary_outcome_at_time(y_va, t_hor)

        # Store OOF
        risk_glob_oof[va_idx] = risk_gl
        risk_loc_oof[va_idx] = risk_lo
        risk_all_oof[va_idx] = risk_all
        y_bin_oof[va_idx] = y_bin
        known_oof[va_idx] = known

    t_mean = float(np.nanmean(np.asarray(t_values))) if len(t_values) > 0 else 365.0
    return risk_glob_oof, risk_loc_oof, risk_all_oof, y_bin_oof, known_oof, t_mean


def evaluate_three_model_grouping_and_rule(
    clean_df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
    n_splits_oof: int = 5,
    percent_for_time: float = 0.75,
    q_low_grid: Optional[List[float]] = None,
    q_high_grid: Optional[List[float]] = None,
    min_gap: float = 0.10,
    output_dir: Optional[str] = None,
) -> Dict[str, object]:
    """Deprecated: three-zone gating rule removed. Returns a marker dict."""
    # Removed in simplified codebase; kept as stub for backward compatibility
    print("[INFO] Three-model grouping and gating rule is not available in the simplified script.")
    return {"removed": True}


def train_assignment_classifier_and_tableone(
    clean_df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
    n_splits_oof: int = 5,
    percent_for_time: float = 0.75,
    topk_coef: int = 12,
    output_dir: Optional[str] = None,
) -> Dict[str, object]:
    """
    Stage-1 (enhanced):
    - Determine per-patient best model among {Global, Local, All} using OOF (train) and held-out risks (test)
      at a fixed time horizon defined from training data.
    - Train a multiclass classifier to predict assignment from baseline features.
    - Evaluate assignment prediction on test and generate a TableOne-style summary across the three groups.
    """
    X_all, y_all, feature_names = _prepare_survival_xy(clean_df)
    global_cols, local_cols, _ = _find_feature_groups(feature_names)
    have_global = len(global_cols) > 0
    have_local = len(local_cols) > 0
    if not (have_global and have_local):
        print("Assignment training skipped: missing global or local feature groups.")
        return {"have_global": have_global, "have_local": have_local}

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=random_state
    )

    # OOF risks on train
    risk_gl_tr, risk_lo_tr, risk_all_tr, ybin_tr, known_tr, t_mean = (
        _compute_oof_three_model_risks(
            X_train,
            y_train,
            global_cols,
            local_cols,
            n_splits_oof,
            random_state,
            percent_for_time,
        )
    )
    err_gl = (risk_gl_tr - ybin_tr) ** 2
    err_lo = (risk_lo_tr - ybin_tr) ** 2
    err_all = (risk_all_tr - ybin_tr) ** 2
    valid_tr = (
        known_tr & np.isfinite(err_gl) & np.isfinite(err_lo) & np.isfinite(err_all)
    )
    best_idx_tr = np.full(len(X_train), -1, dtype=int)
    if valid_tr.any():
        triple = np.vstack([err_gl[valid_tr], err_lo[valid_tr], err_all[valid_tr]])
        best = np.argmin(triple, axis=0)
        best_idx_tr[np.where(valid_tr)[0]] = best

    # Fixed horizon on test = 75th percentile of train times
    e_field, t_field = _surv_field_names(y_train)
    t_hor = (
        float(np.percentile(y_train[t_field], percent_for_time * 100.0))
        if len(y_train)
        else 365.0
    )
    if not np.isfinite(t_hor) or t_hor <= 0:
        t_hor = 365.0

    # Train CoxPH models on full training
    def _fit(X, y):
        return _fit_coxph_clean(X, y)

    model_all = _fit(X_train, y_train)
    model_gl = _fit(X_train[global_cols], y_train) if have_global else None
    model_lo = _fit(X_train[local_cols], y_train) if have_local else None

    # Risks and assignment on test
    risk_all_te = (
        _risk_at_time(model_all, X_test, t_hor)
        if model_all is not None
        else np.zeros(len(X_test))
    )
    risk_gl_te = (
        _risk_at_time(model_gl, X_test[global_cols], t_hor)
        if model_gl is not None
        else np.zeros(len(X_test))
    )
    risk_lo_te = (
        _risk_at_time(model_lo, X_test[local_cols], t_hor)
        if model_lo is not None
        else np.zeros(len(X_test))
    )
    ybin_te, known_te = _binary_outcome_at_time(y_test, t_hor)
    err_gl_te = (risk_gl_te - ybin_te) ** 2
    err_lo_te = (risk_lo_te - ybin_te) ** 2
    err_all_te = (risk_all_te - ybin_te) ** 2
    valid_te = (
        known_te
        & np.isfinite(err_gl_te)
        & np.isfinite(err_lo_te)
        & np.isfinite(err_all_te)
    )
    best_idx_te = np.full(len(X_test), -1, dtype=int)
    if valid_te.any():
        triple = np.vstack(
            [err_gl_te[valid_te], err_lo_te[valid_te], err_all_te[valid_te]]
        )
        best = np.argmin(triple, axis=0)
        best_idx_te[np.where(valid_te)[0]] = best

    # Prepare labels for classifier: 0=Global,1=Local,2=All
    X_tr_cls = X_train.loc[valid_tr, :]
    y_tr_cls = best_idx_tr[valid_tr]
    X_te_cls = X_test.loc[valid_te, :]
    y_te_cls = best_idx_te[valid_te]

    # Multinomial logistic regression (with scaling)
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    multi_class="multinomial",
                    solver="lbfgs",
                    max_iter=2000,
                    n_jobs=None,
                ),
            ),
        ]
    )
    clf.fit(X_tr_cls, y_tr_cls)
    y_pred = clf.predict(X_te_cls) if len(X_te_cls) else np.array([], dtype=int)
    acc = float(accuracy_score(y_te_cls, y_pred)) if len(y_te_cls) else np.nan
    f1_macro = (
        float(f1_score(y_te_cls, y_pred, average="macro")) if len(y_te_cls) else np.nan
    )

    print("\nAssignment prediction (multinomial logistic):")
    print(
        f"- Train labeled: {len(X_tr_cls)} / {len(X_train)} | Test labeled: {len(X_te_cls)} / {len(X_test)}"
    )
    if np.isfinite(acc):
        print(f"- Accuracy (test): {acc:.4f}, Macro-F1: {f1_macro:.4f}")
        try:
            print(
                classification_report(
                    y_te_cls, y_pred, target_names=["Global", "Local", "All"]
                )
            )
        except Exception:
            pass

    # Coefficients as feature importance (per class)
    try:
        lr = clf.named_steps.get("clf")
        if lr is not None and hasattr(lr, "coef_"):
            coef = lr.coef_  # shape (3, n_features)
            feat = X_tr_cls.columns.to_list()
            coef_df = pd.DataFrame(
                coef, columns=feat, index=["Global", "Local", "All"]
            ).T
            # Top features per class by absolute coefficient
            top_dict: Dict[str, pd.Series] = {}
            for cls in ["Global", "Local", "All"]:
                top_dict[cls] = (
                    coef_df[cls].abs().sort_values(ascending=False).head(topk_coef)
                )
            # Save CSVs
            if output_dir:
                _ensure_dir(output_dir)
                coef_df.to_csv(os.path.join(output_dir, "assignment_logreg_coef.csv"))
                with open(
                    os.path.join(output_dir, "assignment_logreg_metrics.txt"), "w"
                ) as f:
                    f.write(f"accuracy={acc}\nmacro_f1={f1_macro}\n")
            # Plot per class top features
            try:
                for cls, ser in top_dict.items():
                    _plot_series_barh(
                        ser,
                        topn=len(ser),
                        title=f"Assignment predictor: top features for {cls}",
                        xlabel="|coefficient|",
                        output_dir=output_dir,
                        filename=f"assignment_top_{cls.lower()}.png",
                        color=(
                            "#2ca02c"
                            if cls == "Global"
                            else ("#ff7f0e" if cls == "Local" else "#1f77b4")
                        ),
                    )
            except Exception:
                pass
    except Exception:
        pass

    # Build TableOne-style summary across three groups (using available labeled samples)
    mapping = {0: "Global", 1: "Local", 2: "All"}
    df_train_groups = X_tr_cls.copy()
    df_train_groups["assignment"] = pd.Series(
        y_tr_cls, index=df_train_groups.index
    ).map(mapping)
    df_test_groups = X_te_cls.copy()
    df_test_groups["assignment"] = pd.Series(y_te_cls, index=df_test_groups.index).map(
        mapping
    )
    df_groups = pd.concat([df_train_groups, df_test_groups], axis=0)

    # Try to use tableone if available
    tableone_df: Optional[pd.DataFrame] = None
    try:
        from tableone import TableOne  # type: ignore

        # Heuristics: categorical if object dtype or low unique count (<=5)
        cats = [
            c
            for c in df_groups.columns
            if c != "assignment"
            and (df_groups[c].dtype == "object" or df_groups[c].nunique() <= 5)
        ]
        conts = [c for c in df_groups.columns if c != "assignment" and c not in cats]
        t1 = TableOne(
            df_groups,
            columns=cats + conts,
            categorical=cats,
            groupby="assignment",
            pval=True,
        )
        tableone_df = t1.tableone.reset_index()
    except Exception:
        # Fallback: simple describe by group (mean±std for numeric, % for binary-like)
        try:
            parts = []
            for grp, sub in df_groups.groupby("assignment"):
                desc = sub.describe().T
                desc["group"] = grp
                parts.append(desc)
            tableone_df = pd.concat(parts)
        except Exception:
            tableone_df = None

    if output_dir and tableone_df is not None:
        try:
            _ensure_dir(output_dir)
            tableone_df.to_csv(
                os.path.join(output_dir, "tableone_assignment.csv"), index=False
            )
        except Exception:
            pass

    result: Dict[str, object] = {
        "n_train_labeled": int(len(X_tr_cls)),
        "n_test_labeled": int(len(X_te_cls)),
        "assignment_train_index": X_tr_cls.index.tolist(),
        "assignment_train_labels": [int(v) for v in y_tr_cls],
        "assignment_test_index": X_te_cls.index.tolist(),
        "assignment_test_labels": [int(v) for v in y_te_cls],
        "accuracy_test": acc,
        "macro_f1_test": f1_macro,
        "time_horizon_days": float(t_hor),
    }
    print(
        "TableOne saved to tableone_assignment.csv"
        if output_dir and tableone_df is not None
        else "TableOne not generated."
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Three-model assignment and reverse feature analysis"
    )
    parser.add_argument(
        "--figs-dir",
        type=str,
        default="/home/sunx/data/aiiih/projects/sunx/projects/ICD/fig",
        help="Output directory for figures and CSVs; takes precedence over env FIGURES_DIR",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Maximum iterations for lifelines CoxPH (Newton–Raphson steps).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars and progress logs.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of random seeds to run for aggregation (1 = single run).",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=42,
        help="Base random seed; seeds will be seed_base + i per run.",
    )
    args = parser.parse_args()

    clean_df = load_dataframes()
    # Resolve figures directory: CLI > global setting > environment variable > default
    figs_dir_cli = args.figs_dir
    figs_dir_env = os.environ.get("FIGURES_DIR")
    figs_dir_glb = FIGURES_DIR
    figs_dir = (
        figs_dir_cli
        or figs_dir_glb
        or figs_dir_env
        or os.path.join("figures", "three_model_assignment")
    )
    # Apply CLI controls
    set_max_iterations(args.max_iter)
    if args.no_progress:
        try:
            global PROGRESS
            PROGRESS = False
        except Exception:
            pass
    # Run experiments: multi-seed aggregation when --runs > 1, else single run
    if int(args.runs) > 1:
        _ = run_assignment_experiments(
            clean_df,
            runs=int(args.runs),
            test_size=0.30,
            random_state=int(args.seed_base),
            output_dir=figs_dir,
        )
    else:
        _ = evaluate_three_model_assignment_and_classifier(
            clean_df,
            output_dir=figs_dir,
        )


if __name__ == "__main__":
    main()
