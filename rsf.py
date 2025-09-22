from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from sksurv.linear_model import CoxPHSurvivalAnalysis
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
            fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
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


def _plot_cindex_bars(metrics: Dict[str, float], output_dir: Optional[str]) -> None:
    labels = ["All", "Global", "Local", "Two-stage"]
    keys = [
        "c_index_all",
        "c_index_global_only",
        "c_index_local_only",
        "c_index_two_stage",
    ]
    vals = [float(metrics.get(k, np.nan)) for k in keys]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    # Avoid seaborn deprecation of palette without hue
    sns.barplot(x=labels, y=vals, hue=labels, ax=ax, palette="Set2", legend=False)
    ax.set_ylabel("C-index (test)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("C-index comparison (CoxPH)")
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, output_dir, "cindex_comparison.png")


def _plot_gating_hist(
    values: np.ndarray,
    thr_low: float,
    thr_high: float,
    label: str,
    output_dir: Optional[str],
) -> None:
    if values is None or len(values) == 0 or not np.isfinite(thr_low) or not np.isfinite(thr_high):
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    sns.histplot(values, kde=False, bins=30, color="#4c78a8", ax=ax)
    ax.axvline(thr_low, ls=":", lw=2, color="#f58518", label=f"low thr = {thr_low:.3g}")
    ax.axvline(thr_high, ls=":", lw=2, color="#e45756", label=f"high thr = {thr_high:.3g}")
    ax.set_title(f"Gating distribution ({label})")
    ax.set_xlabel(label)
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, output_dir, "gating_distribution.png")


def _plot_zone_counts(metrics: Dict[str, float], output_dir: Optional[str]) -> None:
    n_low = int(metrics.get("n_zone_low", 0))
    n_mid = int(metrics.get("n_zone_mid", 0))
    n_high = int(metrics.get("n_zone_high", 0))
    if (n_low + n_mid + n_high) == 0:
        return
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    labels = ["Low", "Mid", "High"]
    vals = [n_low, n_mid, n_high]
    sns.barplot(
        x=labels,
        y=vals,
        hue=labels,
        ax=ax,
        palette=["#4daf4a", "#377eb8", "#e41a1c"],
        legend=False,
    )
    ax.set_ylabel("Count (test)")
    ax.set_title("Zone counts by gating thresholds")
    for i, v in enumerate(vals):
        ax.text(i, v + max(1, 0.02 * max(vals)), str(v), ha="center", va="bottom")
    fig.tight_layout()
    _save_fig(fig, output_dir, "zone_counts.png")


def _plot_cindex_bars_generic(
    labels: List[str], vals: List[float], title: str, output_dir: Optional[str], filename: str
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    sns.barplot(x=labels, y=vals, hue=labels, ax=ax, palette="Set2", legend=False)
    ax.set_ylabel("C-index (test)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, output_dir, filename)


def _plot_gating_hist_by_best(
    gate_values: np.ndarray,
    best_idx: np.ndarray,
    thr_low: float,
    thr_high: float,
    label: str,
    output_dir: Optional[str],
    filename: str = "gating_by_best_hist.png",
) -> None:
    if gate_values is None or len(gate_values) == 0:
        return
    if best_idx is None or len(best_idx) != len(gate_values):
        return
    mask = np.isfinite(gate_values) & (best_idx >= 0)
    if not mask.any():
        return
    df = pd.DataFrame({
        "gate": gate_values[mask],
        "best": best_idx[mask].astype(int),
    })
    mapping = {0: "Global", 1: "Local", 2: "All"}
    try:
        df["best_label"] = df["best"].map(mapping)
    except Exception:
        df["best_label"] = df["best"].astype(str)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    sns.histplot(
        data=df,
        x="gate",
        hue="best_label",
        bins=30,
        element="step",
        stat="count",
        common_norm=False,
        palette="Set2",
        ax=ax,
    )
    if np.isfinite(thr_low):
        ax.axvline(thr_low, ls=":", lw=2, color="#f58518", label=f"low thr = {thr_low:.3g}")
    if np.isfinite(thr_high):
        ax.axvline(thr_high, ls=":", lw=2, color="#e45756", label=f"high thr = {thr_high:.3g}")
    ax.set_title(f"Gating distribution by best model ({label})")
    ax.set_xlabel(label)
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, output_dir, filename)


def _plot_zone_best_model_stack(
    counts_per_zone: Dict[str, Dict[str, int]], output_dir: Optional[str], filename: str = "zone_best_model_stack.png"
) -> None:
    # counts_per_zone: {"low": {"Global": n, "Local": n, "All": n}, ...}
    zones = ["Low", "Mid", "High"]
    keys_norm = {"Low": "low", "Mid": "mid", "High": "high"}
    models = ["Global", "Local", "All"]
    data = []
    for z in zones:
        zkey = keys_norm[z]
        sub = counts_per_zone.get(zkey, {})
        total = float(sum(int(sub.get(m, 0)) for m in models))
        if total <= 0:
            data.append([0.0, 0.0, 0.0])
        else:
            data.append([int(sub.get(m, 0)) / total for m in models])

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    bottoms = np.zeros(len(zones))
    colors = {"Global": "#4daf4a", "Local": "#377eb8", "All": "#e41a1c"}
    for i, m in enumerate(models):
        vals = [row[i] for row in data]
        ax.bar(zones, vals, bottom=bottoms, color=colors.get(m, None), label=m)
        bottoms += np.array(vals)
        # annotate
        for xi, v, b in zip(range(len(zones)), vals, bottoms):
            if v > 0.02:
                ax.text(xi, b - v / 2.0, f"{int(round(v*100))}%", ha="center", va="center", color="white")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Proportion within zone")
    ax.set_title("Best model composition by zone (train, OOF)")
    ax.legend(loc="upper right")
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
    test_c_index = float(concordance_index_censored(y_test[e_field].astype(bool), y_test[t_field].astype(float), risk_scores)[0])
    # Use permutation importance on the hold-out set as a model-agnostic alternative.
    if model is not None:
        try:
            names_in = _model_feature_names(model)
            if names_in is None:
                names_in = list(X_train.columns)
            X_pi = _align_X_to_model(model, X_test)
            perm_result = permutation_importance(
                model,
                X_pi,
                y_test,
                n_repeats=20,
                random_state=random_state,
                n_jobs=-1,
            )
            feat_imp = pd.Series(perm_result.importances_mean, index=names_in).sort_values(
                ascending=False
            )
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
            last = steps.get("coxph") or steps.get("final") or steps.get(list(steps.keys())[-1])
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
    pipe = SkPipeline([
        ("scaler", SkStandardScaler()),
        ("coxph", CoxPHSurvivalAnalysis()),
    ])
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
    """
    Choose (q_low, q_high) by maximizing validation C-index on an inner split of the training set.

    Returns (best_q_low, best_q_high, info_dict). If not applicable, returns (None, None, {}).
    """
    # Basic availability checks
    have_global = len(global_cols) > 0
    have_local = len(local_cols) > 0
    if gating is None and not have_local:
        return None, None, {}

    # Default grids
    if q_low_grid is None:
        q_low_grid = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    if q_high_grid is None:
        q_high_grid = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    # Inner split for threshold tuning
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=inner_val_size, random_state=random_state + 1
    )

    # Helper: robust field names for y
    def _get_surv_field_names(y_arr) -> Tuple[str, str]:
        names = getattr(y_arr.dtype, "names", None)
        if not names or len(names) < 2:
            return "event", "time"
        event_field = "event" if "event" in names else names[0]
        time_candidates = [n for n in names if n != event_field]
        time_field = "time" if "time" in names else time_candidates[0]
        return event_field, time_field

    def _c_index(y, risk):
        e_field, t_field = _get_surv_field_names(y)
        evt = y[e_field].astype(bool)
        tm = y[t_field].astype(float)
        c = concordance_index_censored(evt, tm, risk)[0]
        return float(c)

    # Prepare gating values
    if gating is not None and gating in X_tr.columns and gating in X_val.columns:
        gate_tr_vals = X_tr[gating].astype(float).values
        gate_val_vals = X_val[gating].astype(float).values
    else:
        if have_local:
            gate_tr_vals = X_tr[local_cols].astype(float).sum(axis=1).values
            gate_val_vals = X_val[local_cols].astype(float).sum(axis=1).values
        else:
            return None, None, {}

    # Fit inner models on X_tr
    def _fit(X, y) -> Optional[object]:
        return _fit_coxph_clean(X, y)

    model_glob_in = _fit(X_tr[global_cols], y_tr) if have_global else None
    model_loc_in = _fit(X_tr[local_cols], y_tr) if have_local else None

    # Precompute validation risks
    risk_glob_val = (
        _risk_at_time(model_glob_in, X_val[global_cols], time_horizon_days)
        if model_glob_in is not None
        else np.zeros(len(X_val), dtype=float)
    )
    risk_loc_val = (
        _risk_at_time(model_loc_in, X_val[local_cols], time_horizon_days)
        if model_loc_in is not None
        else np.zeros(len(X_val), dtype=float)
    )

    # Grid search
    best_score = -np.inf
    best_pair: Tuple[Optional[float], Optional[float]] = (None, None)
    tried = 0
    for ql in q_low_grid:
        for qh in q_high_grid:
            if qh - ql < min_gap:
                continue
            thr_l = float(np.nanquantile(gate_tr_vals, ql))
            thr_h = float(np.nanquantile(gate_tr_vals, qh))
            if not np.isfinite(thr_l) or not np.isfinite(thr_h) or thr_l >= thr_h:
                continue
            zone_high = gate_val_vals >= thr_h
            zone_low = gate_val_vals < thr_l
            zone_mid = ~(zone_high | zone_low)
            risk_two = np.zeros(len(X_val), dtype=float)
            risk_two[zone_high] = risk_glob_val[zone_high]
            risk_two[zone_low] = risk_glob_val[zone_low]
            risk_two[zone_mid] = risk_loc_val[zone_mid]
            score = _c_index(y_val, risk_two)
            tried += 1
            if score > best_score:
                best_score = score
                best_pair = (ql, qh)

    info: Dict[str, float] = {"tried": float(tried), "best_c_index_val": float(best_score)}
    return best_pair[0], best_pair[1], info


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
        print("Benefit analysis skipped: insufficient data or feature groups not found.")
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

    for tr_idx, va_idx in kf.split(X_all):
        X_tr, X_va = X_all.iloc[tr_idx], X_all.iloc[va_idx]
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]

        # Fold-specific time horizon
        e_field, t_field = _surv_field_names(y_tr)
        t_hor = float(np.percentile(y_tr[t_field], percent_for_time * 100.0)) if len(y_tr) else 365.0
        if not np.isfinite(t_hor) or t_hor <= 0:
            t_hor = 365.0

        # Train models
        model_gl = _fit(X_tr[global_cols], y_tr) if have_global else None
        model_lo = _fit(X_tr[local_cols], y_tr) if have_local else None
        model_all = _fit(X_tr, y_tr)

        # OOF risk predictions at t
        risk_gl = (
            _risk_at_time(model_gl, X_va[global_cols], t_hor) if model_gl is not None else np.zeros(len(X_va))
        )
        risk_lo = (
            _risk_at_time(model_lo, X_va[local_cols], t_hor) if model_lo is not None else np.zeros(len(X_va))
        )
        risk_all = _risk_at_time(model_all, X_va, t_hor) if model_all is not None else np.zeros(len(X_va))

        # Binary outcome at t with known mask
        y_bin, known = _binary_outcome_at_time(y_va, t_hor)

        # Store
        risk_glob_oof[va_idx] = risk_gl
        risk_loc_oof[va_idx] = risk_lo
        risk_all_oof[va_idx] = risk_all
        y_bin_oof[va_idx] = y_bin
        known_oof[va_idx] = known

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
        triple = np.vstack([err_gl[valid], err_lo[valid], err_all[valid]])  # rows: G, L, A
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
        metrics["c_index_global_in_benefitLocal"] = _c_index(y_all[benefit_local], risk_glob_oof[benefit_local])
        metrics["c_index_all_in_benefitLocal"] = _c_index(y_all[benefit_local], risk_all_oof[benefit_local])
    if benefit_global.sum() > 1:
        metrics["c_index_local_in_benefitGlobal"] = _c_index(y_all[benefit_global], risk_loc_oof[benefit_global])
        metrics["c_index_all_in_benefitGlobal"] = _c_index(y_all[benefit_global], risk_all_oof[benefit_global])

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
                    names_all = _model_feature_names(model_ben_all) or list(X_pi_all.columns)
                    perm_all = permutation_importance(
                        model_ben_all,
                        X_pi_all,
                        y_ben,
                        n_repeats=20,
                        random_state=random_state,
                        n_jobs=-1,
                    )
                    fi_all = pd.Series(perm_all.importances_mean, index=names_all).sort_values(ascending=False)
                except Exception:
                    fi_all = pd.Series(dtype=float)
            else:
                fi_all = pd.Series(dtype=float)
            fi_local_cond = fi_all[fi_all.index.isin(local_cols)].sort_values(ascending=False)
            metrics["local_feature_importance_in_benefit_conditional"] = fi_local_cond.head(topk_local_importance)
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
                    names_loc = _model_feature_names(model_ben_loc) or list(X_pi_loc.columns)
                    perm_loc = permutation_importance(
                        model_ben_loc,
                        X_pi_loc,
                        y_ben,
                        n_repeats=20,
                        random_state=random_state,
                        n_jobs=-1,
                    )
                    fi_loc = pd.Series(perm_loc.importances_mean, index=names_loc).sort_values(ascending=False)
                except Exception:
                    fi_loc = pd.Series(dtype=float)
            else:
                fi_loc = pd.Series(dtype=float)
            metrics["local_feature_importance_in_benefit_localOnly"] = fi_loc.head(topk_local_importance)
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

            print("\nBenefit subgroup (A better than G): local feature importance (conditional on globals, top):")
            print(metrics["local_feature_importance_in_benefit_conditional"])
            print("\nBenefit subgroup: local-only model feature importance (top):")
            print(metrics["local_feature_importance_in_benefit_localOnly"])
    except Exception:
        pass

    print("\nBenefit subgroup analysis:")
    print(f"- Labeled at t: {metrics['n_labeled']} / {metrics['n']} (valid={metrics['n_valid']})")
    print(f"- Benefit (A better than G): {metrics['n_benefit_local']}")
    print(f"- Benefit (A better than L): {metrics['n_benefit_global']}")
    print(f"- Best model counts [G/L/A]: {metrics['n_best_global']} / {metrics['n_best_local']} / {metrics['n_best_all']}")
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
    # Be robust to different sksurv structured array field names
    def _get_surv_field_names(y_arr) -> Tuple[str, str]:
        names = getattr(y_arr.dtype, "names", None)
        if not names or len(names) < 2:
            return "event", "time"
        event_field = "event" if "event" in names else names[0]
        time_candidates = [n for n in names if n != event_field]
        time_field = "time" if "time" in names else time_candidates[0]
        return event_field, time_field

    evt_field, time_field = _get_surv_field_names(y_train)
    t_hor = float(np.percentile(y_train[time_field], 75)) if len(y_train) else 365.0
    if not np.isfinite(t_hor) or t_hor <= 0:
        t_hor = 365.0

    # Train CoxPH models
    def _fit(X, y) -> Optional[object]:
        return _fit_coxph_clean(X, y)

    model_all = _fit(X_train, y_train)
    model_global = _fit(X_train[global_cols], y_train) if have_global else None
    model_local = _fit(X_train[local_cols], y_train) if have_local else None

    # Baseline risks at horizon
    risk_all = _risk_at_time(model_all, X_test, t_hor) if model_all is not None else np.zeros(len(X_test))
    risk_glob = _risk_at_time(model_global, X_test[global_cols], t_hor) if model_global is not None else np.zeros(len(X_test))
    risk_loc = _risk_at_time(model_local, X_test[local_cols], t_hor) if model_local is not None else np.zeros(len(X_test))

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

    selected_q_low = np.nan
    selected_q_high = np.nan
    if gating is not None:
        if optimize_thresholds:
            ql_opt, qh_opt, _info = _optimize_gate_quantiles(
                X_train,
                y_train,
                global_cols,
                local_cols,
                gating,
                random_state,
                t_hor,
                q_low_grid=q_low_grid,
                q_high_grid=q_high_grid,
                min_gap=min_gap,
                inner_val_size=inner_val_size,
            )
            if ql_opt is not None and qh_opt is not None:
                selected_q_low = float(ql_opt)
                selected_q_high = float(qh_opt)
        # Fallback to provided q_low/q_high if no optimized pair
        if not np.isfinite(selected_q_low) or not np.isfinite(selected_q_high):
            selected_q_low = float(q_low)
            selected_q_high = float(q_high)

        thr_low = float(np.nanquantile(gate_train_vals, selected_q_low))
        thr_high = float(np.nanquantile(gate_train_vals, selected_q_high))
        if not np.isfinite(thr_low):
            thr_low = float(np.nanmedian(gate_train_vals))
        if not np.isfinite(thr_high):
            thr_high = float(np.nanmedian(gate_train_vals))
        if thr_low >= thr_high:
            # Enforce separation via fixed percentiles if degenerate
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
        e_field, t_field = _get_surv_field_names(y)
        evt = y[e_field].astype(bool)
        tm = y[t_field].astype(float)
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
        "q_low": float(selected_q_low) if np.isfinite(selected_q_low) else np.nan,
        "q_high": float(selected_q_high) if np.isfinite(selected_q_high) else np.nan,
        "n_zone_low": int(zone_low.sum()) if gating is not None else 0,
        "n_zone_mid": int(zone_mid.sum()) if gating is not None else 0,
        "n_zone_high": int(zone_high.sum()) if gating is not None else 0,
        "n_test": int(len(X_test)),
    }

    # Also return top permutation importances (optional, compact)
    try:
        if model_global is not None:
            try:
                X_pi_g = _align_X_to_model(model_global, X_test[global_cols])
                names_g = _model_feature_names(model_global) or list(X_pi_g.columns)
                perm_glob = permutation_importance(
                    model_global, X_pi_g, y_test, n_repeats=10, random_state=random_state, n_jobs=-1
                )
                fi_glob = pd.Series(perm_glob.importances_mean, index=names_g).sort_values(ascending=False)
            except Exception:
                fi_glob = pd.Series(dtype=float)
            topg = fi_glob.head(8)
            print("Global features (top 8 by permutation importance):")
            print(topg)
            try:
                _plot_series_barh(
                    fi_glob,
                    topn=15,
                    title="Global features (perm importance)",
                    xlabel="Importance (mean)",
                    output_dir=output_dir,
                    filename="global_perm_importance.png",
                    color="#9467bd",
                )
            except Exception:
                pass
        if model_local is not None:
            try:
                X_pi_l = _align_X_to_model(model_local, X_test[local_cols])
                names_l = _model_feature_names(model_local) or list(X_pi_l.columns)
                perm_loc = permutation_importance(
                    model_local, X_pi_l, y_test, n_repeats=10, random_state=random_state, n_jobs=-1
                )
                fi_loc = pd.Series(perm_loc.importances_mean, index=names_l).sort_values(ascending=False)
            except Exception:
                fi_loc = pd.Series(dtype=float)
            topl = fi_loc.head(8)
            print("Local features (top 8 by permutation importance):")
            print(topl)
            try:
                _plot_series_barh(
                    fi_loc,
                    topn=15,
                    title="Local features (perm importance)",
                    xlabel="Importance (mean)",
                    output_dir=output_dir,
                    filename="local_perm_importance.png",
                    color="#8c564b",
                )
            except Exception:
                pass
    except Exception:
        pass

    print("\nTwo-stage strategy evaluation (CoxPH):")
    print(f"- Gating feature: {metrics['gating_feature']}")
    if np.isfinite(metrics.get("q_low", np.nan)) and np.isfinite(metrics.get("q_high", np.nan)):
        print(
            f"- Thresholds: low={metrics['thr_low']:.4g} (q={metrics['q_low']:.2f}), high={metrics['thr_high']:.4g} (q={metrics['q_high']:.2f})"
        )
    else:
        print(f"- Thresholds: low={metrics['thr_low']:.4g}, high={metrics['thr_high']:.4g}")
    print(f"- Time horizon (days): {metrics['time_horizon_days']:.1f}")
    if gating is not None:
        print(
            f"- Zone counts (low/mid/high): {metrics['n_zone_low']}/{metrics['n_zone_mid']}/{metrics['n_zone_high']} of {metrics['n_test']}"
        )
    print("- C-index baselines and two-stage (test):")
    print(f"  * All-features CoxPH:   {metrics['c_index_all']:.4f}")
    print(f"  * Global-only CoxPH:    {metrics['c_index_global_only']:.4f}")
    print(f"  * Local-only CoxPH:     {metrics['c_index_local_only']:.4f}")
    print(f"  * Two-stage (g->l mid): {metrics['c_index_two_stage']:.4f}")

    # Visualizations for two-stage
    try:
        _plot_cindex_bars(metrics, output_dir)
    except Exception:
        pass
    try:
        if gating is not None:
            _plot_gating_hist(gate_train_vals, thr_low, thr_high, label=str(gating), output_dir=output_dir)
            _plot_zone_counts(metrics, output_dir)
    except Exception:
        pass

    return metrics


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
        t_hor = float(np.percentile(y_tr[t_field], percent_for_time * 100.0)) if len(y_tr) else 365.0
        if not np.isfinite(t_hor) or t_hor <= 0:
            t_hor = 365.0
        t_values.append(t_hor)

        # Train models on fold-train
        model_gl = _fit(X_tr[global_cols], y_tr) if len(global_cols) > 0 else None
        model_lo = _fit(X_tr[local_cols], y_tr) if len(local_cols) > 0 else None
        model_all = _fit(X_tr, y_tr)

        # Predict risks on fold-val
        risk_gl = _risk_at_time(model_gl, X_va[global_cols], t_hor) if model_gl is not None else np.zeros(len(X_va))
        risk_lo = _risk_at_time(model_lo, X_va[local_cols], t_hor) if model_lo is not None else np.zeros(len(X_va))
        risk_all = _risk_at_time(model_all, X_va, t_hor) if model_all is not None else np.zeros(len(X_va))

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
    """
    Stage-1: Compare three CoxPH models (Global-only, Local-only, All-features) per patient.
    - Use OOF risks on the training set to assign each patient to the best-performing model
      (by minimal squared error at a fixed time horizon).
    - Learn an interpretable three-zone rule based on a single gating feature to predict the
      best model group (Low/Mid/High zones -> mapped to one of {Global, Local, All}).
    - Evaluate this rule on a held-out test set by selecting the model per patient and
      computing C-index.

    Returns a dict with thresholds, zone mapping, counts, and C-index metrics.
    """
    # Prep data and feature groups
    X_all, y_all, feature_names = _prepare_survival_xy(clean_df)
    global_cols, local_cols, gating = _find_feature_groups(feature_names)
    have_global = len(global_cols) > 0
    have_local = len(local_cols) > 0
    if not (have_global and have_local):
        print("Three-model evaluation skipped: missing global or local feature groups.")
        return {
            "have_global": have_global,
            "have_local": have_local,
        }

    # Split train/test once
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=random_state
    )

    # OOF on training to determine best-of-three group labels
    risk_gl_tr, risk_lo_tr, risk_all_tr, ybin_tr, known_tr, t_mean = _compute_oof_three_model_risks(
        X_train, y_train, global_cols, local_cols, n_splits_oof, random_state, percent_for_time
    )

    # Define best model per training sample (only where known)
    err_gl = (risk_gl_tr - ybin_tr) ** 2
    err_lo = (risk_lo_tr - ybin_tr) ** 2
    err_all = (risk_all_tr - ybin_tr) ** 2
    valid_tr = known_tr & np.isfinite(err_gl) & np.isfinite(err_lo) & np.isfinite(err_all)
    best_idx_tr = np.full(len(X_train), -1, dtype=int)
    if valid_tr.any():
        triple = np.vstack([err_gl[valid_tr], err_lo[valid_tr], err_all[valid_tr]])
        best = np.argmin(triple, axis=0)  # 0=Global, 1=Local, 2=All
        best_idx_tr[np.where(valid_tr)[0]] = best

    # Pick gating values (prefer explicit gating feature)
    if gating is not None and gating in X_train.columns:
        gate_train = X_train[gating].astype(float).values
        gate_test = X_test[gating].astype(float).values
        gate_label = str(gating)
    else:
        gate_label = "(sum of local segments)"
        gate_train = X_train[local_cols].astype(float).sum(axis=1).values
        gate_test = X_test[local_cols].astype(float).sum(axis=1).values

    # Threshold grids
    if q_low_grid is None:
        q_low_grid = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    if q_high_grid is None:
        q_high_grid = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    # Train CoxPH models on full training for test-time inference
    def _fit(X, y) -> Optional[CoxPHSurvivalAnalysis]:
        return _fit_coxph_clean(X, y)

    # Fixed horizon for test evaluation = 75th percentile of train times
    e_field, t_field = _surv_field_names(y_train)
    t_hor = float(np.percentile(y_train[t_field], percent_for_time * 100.0)) if len(y_train) else 365.0
    if not np.isfinite(t_hor) or t_hor <= 0:
        t_hor = 365.0

    model_all = _fit(X_train, y_train)
    model_gl = _fit(X_train[global_cols], y_train)
    model_lo = _fit(X_train[local_cols], y_train)

    # Precompute test risks
    risk_all_te = _risk_at_time(model_all, X_test, t_hor) if model_all is not None else np.zeros(len(X_test))
    risk_gl_te = _risk_at_time(model_gl, X_test[global_cols], t_hor) if model_gl is not None else np.zeros(len(X_test))
    risk_lo_te = _risk_at_time(model_lo, X_test[local_cols], t_hor) if model_lo is not None else np.zeros(len(X_test))

    # Helper for C-index
    def _c_index(y_arr, risk_scores) -> float:
        e_name, t_name = _surv_field_names(y_arr)
        evt = y_arr[e_name].astype(bool)
        tm = y_arr[t_name].astype(float)
        return float(concordance_index_censored(evt, tm, risk_scores)[0])

    # Optimize thresholds by maximizing OOF C-index of rule-applied risks on training
    best_val_c = -np.inf
    best_thr_low = np.nan
    best_thr_high = np.nan
    best_zone_to_model: Dict[str, int] = {}

    # Build OOF risks dict for convenience (training time horizon varies per fold, so
    # we will evaluate rule quality using classification agreement to best_idx_tr first,
    # then break ties by OOF C-index constructed at fold horizons approximately by
    # using risk_all_tr as a proxy when needed.)
    # Primary score: classification accuracy of zone -> best model on valid_tr samples.
    for ql in q_low_grid:
        for qh in q_high_grid:
            if qh - ql < min_gap:
                continue
            thr_l = float(np.nanquantile(gate_train, ql))
            thr_h = float(np.nanquantile(gate_train, qh))
            if not np.isfinite(thr_l) or not np.isfinite(thr_h) or thr_l >= thr_h:
                continue

            zone_low = gate_train < thr_l
            zone_high = gate_train >= thr_h
            zone_mid = ~(zone_low | zone_high)

            # Majority label in each zone among valid training samples
            z2m: Dict[str, int] = {}
            acc_parts = []
            for name, mask in ("low", zone_low), ("mid", zone_mid), ("high", zone_high):
                m = mask & valid_tr
                if m.sum() == 0:
                    # Fallback to Global if zone empty
                    z2m[name] = 0
                    continue
                labels = best_idx_tr[m]
                # majority vote among {0,1,2}
                vals, counts = np.unique(labels, return_counts=True)
                maj = int(vals[np.argmax(counts)])
                z2m[name] = maj
                acc_parts.append((labels == maj).mean())

            # Primary score: mean per-zone accuracy where zone has data
            if len(acc_parts) == 0:
                continue
            score_primary = float(np.mean(acc_parts))

            # Secondary (tie-breaker): OOF pseudo C-index by composing per-zone chosen risks
            # We approximate by mixing the three OOF risks using the same mapping and evaluate
            # concordance against original survival times at varying fold horizons; since OOF
            # was computed with fold-specific horizons for y_bin, use a crude proxy here by
            # ranking capability measured via Kendall-like concordance on available finite values.
            # To keep robust and simple, we will not overfit this tie-breaker.
            score_secondary = score_primary  # identical fallback

            if (score_primary > best_val_c + 1e-12) or (
                abs(score_primary - best_val_c) <= 1e-12 and score_secondary > best_val_c
            ):
                best_val_c = score_primary
                best_thr_low = thr_l
                best_thr_high = thr_h
                best_zone_to_model = dict(z2m)

    if not np.isfinite(best_thr_low) or not np.isfinite(best_thr_high):
        # Fallback to fixed percentiles
        best_thr_low = float(np.nanpercentile(gate_train, 40))
        best_thr_high = float(np.nanpercentile(gate_train, 75))
        best_zone_to_model = {"low": 0, "mid": 2, "high": 0}  # heuristic: global, all, global

    # Apply rule on test set to pick model per patient
    zone_low_te = gate_test < best_thr_low
    zone_high_te = gate_test >= best_thr_high
    zone_mid_te = ~(zone_low_te | zone_high_te)

    # Map chosen model index per zone to risk arrays
    # 0 -> global, 1 -> local, 2 -> all
    rule_risk_te = np.zeros(len(X_test), dtype=float)
    for name, mask in ("low", zone_low_te), ("mid", zone_mid_te), ("high", zone_high_te):
        chosen = int(best_zone_to_model.get(name, 0))
        if chosen == 0:
            rule_risk_te[mask] = risk_gl_te[mask]
        elif chosen == 1:
            rule_risk_te[mask] = risk_lo_te[mask]
        else:
            rule_risk_te[mask] = risk_all_te[mask]

    # Metrics on test
    cidx_all = _c_index(y_test, risk_all_te)
    cidx_gl = _c_index(y_test, risk_gl_te)
    cidx_lo = _c_index(y_test, risk_lo_te)
    cidx_rule = _c_index(y_test, rule_risk_te)

    # Package results
    mapping_human = {
        "low": ["Global", "Local", "All"][int(best_zone_to_model.get("low", 0))],
        "mid": ["Global", "Local", "All"][int(best_zone_to_model.get("mid", 0))],
        "high": ["Global", "Local", "All"][int(best_zone_to_model.get("high", 0))],
    }
    out: Dict[str, object] = {
        "gating_feature": gate_label,
        "thr_low": float(best_thr_low),
        "thr_high": float(best_thr_high),
        "zone_mapping": mapping_human,
        "n_zone_low_test": int(zone_low_te.sum()),
        "n_zone_mid_test": int(zone_mid_te.sum()),
        "n_zone_high_test": int(zone_high_te.sum()),
        "time_horizon_days": float(t_hor),
        "c_index_all": float(cidx_all),
        "c_index_global_only": float(cidx_gl),
        "c_index_local_only": float(cidx_lo),
        "c_index_rule": float(cidx_rule),
        "n_test": int(len(X_test)),
        "n_train_valid_for_grouping": int(valid_tr.sum()),
        "t_mean_oof": float(t_mean),
    }

    print("\nThree-model grouping and rule-based selection (test):")
    print(f"- Gating feature: {out['gating_feature']}")
    print(
        f"- Thresholds: low={out['thr_low']:.4g}, high={out['thr_high']:.4g}; zone mapping = "
        f"low->{mapping_human['low']}, mid->{mapping_human['mid']}, high->{mapping_human['high']}"
    )
    print(f"- Time horizon (days): {out['time_horizon_days']:.1f}")
    print(
        f"- Zone counts (low/mid/high): {out['n_zone_low_test']}/{out['n_zone_mid_test']}/{out['n_zone_high_test']} of {out['n_test']}"
    )
    print("- C-index baselines and rule:")
    print(f"  * All-features CoxPH:   {out['c_index_all']:.4f}")
    print(f"  * Global-only CoxPH:    {out['c_index_global_only']:.4f}")
    print(f"  * Local-only CoxPH:     {out['c_index_local_only']:.4f}")
    print(f"  * Rule (3-zone select): {out['c_index_rule']:.4f}")

    # Optional visuals
    try:
        # 1) Gating histogram colored by best model (train, OOF best)
        _plot_gating_hist_by_best(
            gate_train,
            best_idx_tr,
            float(out["thr_low"]),
            float(out["thr_high"]),
            label=str(gate_label),
            output_dir=output_dir,
        )
        # 2) Zone-wise composition (train)
        thr_l_v = float(out["thr_low"])
        thr_h_v = float(out["thr_high"])
        z_low_tr = gate_train < thr_l_v
        z_high_tr = gate_train >= thr_h_v
        z_mid_tr = ~(z_low_tr | z_high_tr)
        mapping = {0: "Global", 1: "Local", 2: "All"}
        comp_counts: Dict[str, Dict[str, int]] = {"low": {}, "mid": {}, "high": {}}
        for name, mask in ("low", z_low_tr), ("mid", z_mid_tr), ("high", z_high_tr):
            m = mask & valid_tr
            sub = best_idx_tr[m]
            cc: Dict[str, int] = {}
            if len(sub) > 0:
                vals, counts = np.unique(sub, return_counts=True)
                for v, c in zip(vals, counts):
                    cc[mapping.get(int(v), str(int(v)))] = int(c)
            comp_counts[name] = cc
        _plot_zone_best_model_stack(comp_counts, output_dir)
        # 3) C-index comparison including Rule
        _plot_cindex_bars_generic(
            labels=["All", "Global", "Local", "Rule"],
            vals=[out["c_index_all"], out["c_index_global_only"], out["c_index_local_only"], out["c_index_rule"]],
            title="C-index comparison: All vs Global vs Local vs Rule",
            output_dir=output_dir,
            filename="cindex_all_global_local_rule.png",
        )
    except Exception:
        pass

    return out


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
    risk_gl_tr, risk_lo_tr, risk_all_tr, ybin_tr, known_tr, t_mean = _compute_oof_three_model_risks(
        X_train, y_train, global_cols, local_cols, n_splits_oof, random_state, percent_for_time
    )
    err_gl = (risk_gl_tr - ybin_tr) ** 2
    err_lo = (risk_lo_tr - ybin_tr) ** 2
    err_all = (risk_all_tr - ybin_tr) ** 2
    valid_tr = known_tr & np.isfinite(err_gl) & np.isfinite(err_lo) & np.isfinite(err_all)
    best_idx_tr = np.full(len(X_train), -1, dtype=int)
    if valid_tr.any():
        triple = np.vstack([err_gl[valid_tr], err_lo[valid_tr], err_all[valid_tr]])
        best = np.argmin(triple, axis=0)
        best_idx_tr[np.where(valid_tr)[0]] = best

    # Fixed horizon on test = 75th percentile of train times
    e_field, t_field = _surv_field_names(y_train)
    t_hor = float(np.percentile(y_train[t_field], percent_for_time * 100.0)) if len(y_train) else 365.0
    if not np.isfinite(t_hor) or t_hor <= 0:
        t_hor = 365.0

    # Train CoxPH models on full training
    def _fit(X, y):
        return _fit_coxph_clean(X, y)

    model_all = _fit(X_train, y_train)
    model_gl = _fit(X_train[global_cols], y_train) if have_global else None
    model_lo = _fit(X_train[local_cols], y_train) if have_local else None

    # Risks and assignment on test
    risk_all_te = _risk_at_time(model_all, X_test, t_hor) if model_all is not None else np.zeros(len(X_test))
    risk_gl_te = _risk_at_time(model_gl, X_test[global_cols], t_hor) if model_gl is not None else np.zeros(len(X_test))
    risk_lo_te = _risk_at_time(model_lo, X_test[local_cols], t_hor) if model_lo is not None else np.zeros(len(X_test))
    ybin_te, known_te = _binary_outcome_at_time(y_test, t_hor)
    err_gl_te = (risk_gl_te - ybin_te) ** 2
    err_lo_te = (risk_lo_te - ybin_te) ** 2
    err_all_te = (risk_all_te - ybin_te) ** 2
    valid_te = known_te & np.isfinite(err_gl_te) & np.isfinite(err_lo_te) & np.isfinite(err_all_te)
    best_idx_te = np.full(len(X_test), -1, dtype=int)
    if valid_te.any():
        triple = np.vstack([err_gl_te[valid_te], err_lo_te[valid_te], err_all_te[valid_te]])
        best = np.argmin(triple, axis=0)
        best_idx_te[np.where(valid_te)[0]] = best

    # Prepare labels for classifier: 0=Global,1=Local,2=All
    X_tr_cls = X_train.loc[valid_tr, :]
    y_tr_cls = best_idx_tr[valid_tr]
    X_te_cls = X_test.loc[valid_te, :]
    y_te_cls = best_idx_te[valid_te]

    # Multinomial logistic regression (with scaling)
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=2000, n_jobs=None))
    ])
    clf.fit(X_tr_cls, y_tr_cls)
    y_pred = clf.predict(X_te_cls) if len(X_te_cls) else np.array([], dtype=int)
    acc = float(accuracy_score(y_te_cls, y_pred)) if len(y_te_cls) else np.nan
    f1_macro = float(f1_score(y_te_cls, y_pred, average="macro")) if len(y_te_cls) else np.nan

    print("\nAssignment prediction (multinomial logistic):")
    print(f"- Train labeled: {len(X_tr_cls)} / {len(X_train)} | Test labeled: {len(X_te_cls)} / {len(X_test)}")
    if np.isfinite(acc):
        print(f"- Accuracy (test): {acc:.4f}, Macro-F1: {f1_macro:.4f}")
        try:
            print(classification_report(y_te_cls, y_pred, target_names=["Global","Local","All"]))
        except Exception:
            pass

    # Coefficients as feature importance (per class)
    try:
        lr = clf.named_steps.get("clf")
        if lr is not None and hasattr(lr, "coef_"):
            coef = lr.coef_  # shape (3, n_features)
            feat = X_tr_cls.columns.to_list()
            coef_df = pd.DataFrame(coef, columns=feat, index=["Global","Local","All"]).T
            # Top features per class by absolute coefficient
            top_dict: Dict[str, pd.Series] = {}
            for cls in ["Global","Local","All"]:
                top_dict[cls] = coef_df[cls].abs().sort_values(ascending=False).head(topk_coef)
            # Save CSVs
            if output_dir:
                _ensure_dir(output_dir)
                coef_df.to_csv(os.path.join(output_dir, "assignment_logreg_coef.csv"))
                with open(os.path.join(output_dir, "assignment_logreg_metrics.txt"), "w") as f:
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
                        color="#2ca02c" if cls=="Global" else ("#ff7f0e" if cls=="Local" else "#1f77b4"),
                    )
            except Exception:
                pass
    except Exception:
        pass

    # Build TableOne-style summary across three groups (using available labeled samples)
    mapping = {0: "Global", 1: "Local", 2: "All"}
    df_train_groups = X_tr_cls.copy()
    df_train_groups["assignment"] = pd.Series(y_tr_cls, index=df_train_groups.index).map(mapping)
    df_test_groups = X_te_cls.copy()
    df_test_groups["assignment"] = pd.Series(y_te_cls, index=df_test_groups.index).map(mapping)
    df_groups = pd.concat([df_train_groups, df_test_groups], axis=0)

    # Try to use tableone if available
    tableone_df: Optional[pd.DataFrame] = None
    try:
        from tableone import TableOne  # type: ignore

        # Heuristics: categorical if object dtype or low unique count (<=5)
        cats = [c for c in df_groups.columns if c != "assignment" and (df_groups[c].dtype == "object" or df_groups[c].nunique() <= 5)]
        conts = [c for c in df_groups.columns if c != "assignment" and c not in cats]
        t1 = TableOne(df_groups, columns=cats + conts, categorical=cats, groupby="assignment", pval=True)
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
            tableone_df.to_csv(os.path.join(output_dir, "tableone_assignment.csv"), index=False)
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
    print("TableOne saved to tableone_assignment.csv" if output_dir and tableone_df is not None else "TableOne not generated.")
    return result


def main():
    clean_df = load_dataframes()
    figs_dir = os.path.join("figures", "coxph")
    model, metrics, feat_imp = train_coxph_model(clean_df, output_dir=figs_dir)
    print(f"Test C-index: {metrics['test_c_index']:.4f}")
    print("Top feature importances:")
    topn = 15 if len(feat_imp) >= 15 else len(feat_imp)
    print(feat_imp.head(topn))

    # Validate the global-first, local-then hypothesis
    _ = evaluate_two_stage_strategy(
        clean_df,
        optimize_thresholds=True,
        output_dir=figs_dir,
    )

    # Analyze benefit subgroup and local-feature importance within it
    _ = analyze_benefit_subgroup(clean_df, output_dir=figs_dir)

    # Stage-1: best-of-three grouping and rule learning/evaluation
    _ = evaluate_three_model_grouping_and_rule(
        clean_df,
        output_dir=figs_dir,
    )

    # Stage-1 enhanced: train assignment classifier and produce TableOne
    _ = train_assignment_classifier_and_tableone(
        clean_df,
        output_dir=figs_dir,
    )


if __name__ == "__main__":
    main()

