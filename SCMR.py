import os
import warnings
import re
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend("Agg")
from typing import Dict, List, Tuple, Any

# ==========================================
# Global store for selected features (Proposed & Advanced only)
# ==========================================
SELECTED_FEATURES_STORE: Dict[str, Any] = {
    # Proposed
    "proposed_sex_agnostic": None,  # type: List[str] | None
    "proposed_sex_specific": {  # type: Dict[str, List[str]] | None
        "male": None,
        "female": None,
    },
    # Advanced
    "advanced_sex_agnostic": None,  # type: List[str] | None
    "advanced_sex_specific": {  # type: Dict[str, List[str]] | None
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


def _progress(message: str) -> None:
    """Lightweight progress printer that always flushes."""
    try:
        print(f"[Progress] {message}", flush=True)
    except Exception:
        pass


def _normalize_column_name(name: Any) -> Any:
    """Trim and collapse internal whitespace for column labels."""
    if not isinstance(name, str):
        return name
    name = name.strip()
    # Collapse multiple whitespace to single spaces
    name = " ".join(name.split())
    return name


def _simplify_name(name: str) -> str:
    """Lowercase and strip non-alphanumeric to enable fuzzy column matching."""
    if not isinstance(name, str):
        return ""
    lowered = name.lower()
    simplified = re.sub(r"[^a-z0-9]+", " ", lowered).strip()
    return simplified


def _build_alias_map(columns: List[str]) -> Dict[str, str]:
    """Map simplified column names to their original labels.

    If duplicates occur after simplification, the first occurrence is kept.
    """
    alias: Dict[str, str] = {}
    for col in columns:
        key = _simplify_name(col)
        if key and key not in alias:
            alias[key] = col
    return alias


# ==========================================
# Advanced LGE nominal/binary feature names (normalized)
# ==========================================
ADV_LGE_NOMINAL: List[str] = [
    _normalize_column_name(s)
    for s in [
        "LGE_Basal anterior (0; No, 1; yes)",
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
        "LGE_Circumural",
        "LGE_Ring-Like",
    ]
]



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
    save_path: str = None,
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
    plt.xticks(range(len(feats)), feats, rotation=45, fontsize=11, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        except Exception:
            pass
        try:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        except Exception:
            pass
    plt.close()


def plot_km_two_subplots_by_gender(merged_df: pd.DataFrame, save_path: str = None) -> None:
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
        # Ensure legend with larger font on each subplot
        ax.legend(loc="best", fontsize=12)

    plt.suptitle("Primary Endpoint - Survival by Gender and Risk Group", y=1.02)
    plt.tight_layout()
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        except Exception:
            pass
        try:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        except Exception:
            pass
    plt.close()


# ==========================================
# Persistence and evaluation helpers
# ==========================================


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _save_pickle(obj: Any, path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    try:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    except Exception:
        warnings.warn(f"[Save] Failed to pickle object to {path}")


def _save_json(obj: Any, path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
    except Exception:
        warnings.warn(f"[Save] Failed to write JSON to {path}")


def _safe_cindex(t: np.ndarray, e: np.ndarray, r: np.ndarray) -> float:
    try:
        if t is None or e is None or r is None:
            return float("nan")
        t = np.asarray(t)
        e = np.asarray(e)
        r = np.asarray(r)
        if len(t) < 2:
            return float("nan")
        if np.all(~np.isfinite(r)) or np.allclose(r, r[0], equal_nan=True):
            return float("nan")
        return float(concordance_index(t, -r, e))
    except Exception:
        return float("nan")


def _compute_logrank_pvalues_by_gender(df_with_preds: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {"male": float("nan"), "female": float("nan")}
    try:
        ep_time_col, ep_event_col = "PE_Time", "PE"
        for sex_val, key in [(0, "male"), (1, "female")]:
            subset = df_with_preds[df_with_preds["Female"] == sex_val]
            if subset.empty:
                out[key] = float("nan")
                continue
            low = subset[subset["pred_label"] == 0]
            high = subset[subset["pred_label"] == 1]
            if low.empty or high.empty:
                out[key] = float("nan")
                continue
            lr = logrank_test(
                low[ep_time_col], high[ep_time_col], low[ep_event_col], high[ep_event_col]
            )
            out[key] = float(lr.p_value)
    except Exception:
        pass
    return out


def _compute_hr_by_gender(df_with_preds: pd.DataFrame) -> Dict[str, Tuple[float, float, float]]:
    """Compute univariate HR (High vs Low) by gender using pred_label as exposure.

    Returns dict: {"male": (HR, CI_low, CI_high), "female": (...)} with NaNs when unavailable.
    """
    out: Dict[str, Tuple[float, float, float]] = {
        "male": (float("nan"), float("nan"), float("nan")),
        "female": (float("nan"), float("nan"), float("nan")),
    }
    try:
        ep_time_col, ep_event_col = "PE_Time", "PE"
        for sex_val, key in [(0, "male"), (1, "female")]:
            sub = df_with_preds[df_with_preds["Female"] == sex_val]
            if sub.empty or sub["pred_label"].nunique() < 2:
                continue
            # Fit Cox with binary pred_label (1=High risk, 0=Low risk)
            cph = CoxPHFitter()
            cph.fit(
                sub[[ep_time_col, ep_event_col, "pred_label"]],
                duration_col=ep_time_col,
                event_col=ep_event_col,
                robust=True,
            )
            hr = float(np.exp(cph.params_["pred_label"]))
            ci = cph.confidence_intervals_.loc["pred_label"].values
            ci_low, ci_high = float(np.exp(ci[0])), float(np.exp(ci[1]))
            out[key] = (hr, ci_low, ci_high)
    except Exception:
        pass
    return out


def stratified_train_test_split_by_columns(
    df: pd.DataFrame,
    cols: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratify by combined categories in cols. Robustly falls back if needed."""
    df_local = df.copy()
    for c in cols:
        if c not in df_local.columns:
            raise ValueError(f"Missing stratification column: {c}")
    # Build combined label; convert to string to avoid NaNs
    combo = (
        df_local[cols]
        .astype(str)
        .agg("|".join, axis=1)
        .fillna("missing")
    )
    try:
        tr, te = train_test_split(
            df_local, test_size=test_size, random_state=random_state, stratify=combo
        )
        return tr.reset_index(drop=True), te.reset_index(drop=True)
    except Exception:
        # iterative fallback: use fewer columns
        for k in range(len(cols) - 1, -1, -1):
            try_cols = cols[:k]
            if not try_cols:
                break
            try:
                combo2 = (
                    df_local[try_cols]
                    .astype(str)
                    .agg("|".join, axis=1)
                    .fillna("missing")
                )
                tr, te = train_test_split(
                    df_local,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=combo2,
                )
                return tr.reset_index(drop=True), te.reset_index(drop=True)
            except Exception:
                continue
        # last resort no stratify
        tr, te = train_test_split(
            df_local, test_size=test_size, random_state=random_state
        )
        return tr.reset_index(drop=True), te.reset_index(drop=True)


def run_heldout_training_and_evaluation(
    df: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    time_col: str = "PE_Time",
    event_col: str = "VT/VF/SCD",
    results_dir: str = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train models on a single held-out split and evaluate on test.

    - Stratifies by Female/PE/ICD when possible
    - Persists models and thresholds from training
    - Evaluates on held-out test set (c-index, log-rank) and saves KM plots
    - Returns a summary dataframe of metrics
    """
    if results_dir is None:
        results_dir = os.path.join(os.getcwd(), "results")
    models_dir = os.path.join(results_dir, "heldout", "models")
    figs_dir = os.path.join(results_dir, "heldout", "figs")
    preds_dir = os.path.join(results_dir, "heldout", "preds")
    meta_dir = os.path.join(results_dir, "heldout", "meta")
    for d in [models_dir, figs_dir, preds_dir, meta_dir]:
        _ensure_dir(d)

    _progress("[Heldout] Splitting train/test and creating output directories...")
    # Split
    tr, te = stratified_train_test_split_by_columns(
        df.dropna(subset=[time_col, event_col]).copy(),
        cols=["Female", event_col, "ICD"],
        test_size=test_size,
        random_state=random_state,
    )

    # Save distributions
    try:
        def distro(x: pd.DataFrame) -> Dict[str, int]:
            key = (
                x[["Female", event_col, "ICD"]]
                .astype(int)
                .astype(str)
                .agg("|".join, axis=1)
            )
            return key.value_counts().to_dict()

        split_meta = {
            "train_size": int(len(tr)),
            "test_size": int(len(te)),
            "train_distribution": distro(tr),
            "test_distribution": distro(te),
        }
        _save_json(split_meta, os.path.join(meta_dir, "split_distribution.json"))
    except Exception:
        pass

    # Collect metrics
    rows: List[Dict[str, Any]] = []

    # Stability selection seeds limited for speed if small data
    seeds_for_stability = list(range(20))

    for featset_name, feature_cols in feature_sets.items():
        _progress(f"[Heldout] Processing feature set: {featset_name}")
        # Stabilize features on TRAIN ONLY for Proposed/Advanced
        used_agnostic_feats = list(feature_cols)
        used_shared_specific_feats: List[str] = [f for f in feature_cols if f != "Female"]
        is_proposed = set(feature_cols) == set(FEATURE_SETS.get("Proposed", []))
        is_advanced = set(feature_cols) == set(FEATURE_SETS.get("Advanced", []))
        try:
            if is_proposed or is_advanced:
                # sex-agnostic selection (includes Female)
                _progress(f"[Heldout] {featset_name}: Running sex-agnostic stability selection...")
                sel_agn = stability_select_features(
                    df=tr,
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
                    used_agnostic_feats = list(sel_agn)
                # sex-specific shared selection (exclude Female)
                base_feats = [f for f in feature_cols if f != "Female"]
                _progress(f"[Heldout] {featset_name}: Running sex-specific shared stability selection...")
                sel_shared = stability_select_features(
                    df=tr,
                    candidate_features=list(base_feats),
                    time_col=time_col,
                    event_col=event_col,
                    seeds=seeds_for_stability,
                    max_features=None,
                    threshold=0.4,
                    min_features=None,
                    verbose=True,
                )
                if sel_shared:
                    used_shared_specific_feats = list(sel_shared)
        except Exception:
            pass

        # ===== Train and persist: sex-agnostic =====
        try:
            _progress(f"[Heldout] {featset_name}: Training sex-agnostic model and saving artifacts...")
            tr_agn_base = create_undersampled_dataset(tr, event_col, random_state)
            tr_agn = tr_agn_base.dropna(subset=[time_col, event_col]).copy()
            if not tr_agn.empty:
                cph = fit_cox_model(tr_agn, used_agnostic_feats, time_col, event_col)
                tr_risk = predict_risk(cph, tr_agn, used_agnostic_feats)
                thr = float(np.nanmedian(tr_risk))

                # Persist
                base_path = os.path.join(models_dir, featset_name, "sex_agnostic")
                _ensure_dir(base_path)
                _save_pickle(cph, os.path.join(base_path, "model.pkl"))
                _save_json(
                    {"threshold": thr, "features": used_agnostic_feats},
                    os.path.join(base_path, "meta.json"),
                )
                # Coef plot
                plot_cox_coefficients(
                    cph,
                    f"{featset_name} Sex-Agnostic Cox Coefficients",
                    reference_df=tr_agn,
                    effect_scale="per_sd",
                    save_path=os.path.join(figs_dir, f"{featset_name}_sex_agnostic_coefs.png"),
                )

                # Evaluate on heldout test
                _progress(f"[Heldout] {featset_name}: Evaluating sex-agnostic on test and exporting plots...")
                te_eval = te.copy()
                risk = predict_risk(cph, te_eval, used_agnostic_feats)
                te_eval["pred_prob"] = risk
                te_eval["pred_label"] = (risk >= thr).astype(int)
                merged = (
                    te_eval[["MRN", "Female", "pred_label", time_col, event_col]]
                    .dropna(subset=[time_col, event_col])
                    .drop_duplicates(subset=["MRN"])  # one row per MRN
                    .rename(columns={event_col: "PE"})
                )
                # Save preds
                merged.to_csv(
                    os.path.join(
                        preds_dir, f"{featset_name}_sex_agnostic_test_preds.csv"
                    ),
                    index=False,
                )
                # KM + log-rank
                plot_km_two_subplots_by_gender(
                    merged,
                    save_path=os.path.join(
                        figs_dir, f"{featset_name}_sex_agnostic_km_by_gender.png"
                    ),
                )
                # Metrics
                mask_m = merged["Female"].values == 0
                mask_f = merged["Female"].values == 1
                c_all = _safe_cindex(
                    merged["PE_Time"].values, merged["PE"].values, merged["pred_prob"].values
                )
                c_m = _safe_cindex(
                    merged.loc[mask_m, "PE_Time"].values,
                    merged.loc[mask_m, "PE"].values,
                    merged.loc[mask_m, "pred_prob"].values,
                )
                c_f = _safe_cindex(
                    merged.loc[mask_f, "PE_Time"].values,
                    merged.loc[mask_f, "PE"].values,
                    merged.loc[mask_f, "pred_prob"].values,
                )
            pvals = _compute_logrank_pvalues_by_gender(merged)
            hrs = _compute_hr_by_gender(merged)
            rows.append(
                    {
                        "feature_set": featset_name,
                        "mode": "sex_agnostic",
                        "c_index_all": c_all,
                        "c_index_male": c_m,
                        "c_index_female": c_f,
                        "logrank_p_male": pvals.get("male"),
                        "logrank_p_female": pvals.get("female"),
                    "hr_male": hrs.get("male", (np.nan, np.nan, np.nan))[0],
                    "hr_male_ci_low": hrs.get("male", (np.nan, np.nan, np.nan))[1],
                    "hr_male_ci_high": hrs.get("male", (np.nan, np.nan, np.nan))[2],
                    "hr_female": hrs.get("female", (np.nan, np.nan, np.nan))[0],
                    "hr_female_ci_low": hrs.get("female", (np.nan, np.nan, np.nan))[1],
                    "hr_female_ci_high": hrs.get("female", (np.nan, np.nan, np.nan))[2],
                    }
                )
        except Exception as e:
            warnings.warn(f"[Heldout][{featset_name}] sex-agnostic failed: {e}")

        # ===== Train and persist: sex-specific =====
        try:
            # Only Proposed/Advanced run sex-specific per requirement
            if featset_name not in {"Proposed", "Advanced"}:
                raise RuntimeError("Skip sex-specific for non P/A feature sets")
            _progress(f"[Heldout] {featset_name}: Training sex-specific models (male/female) and saving artifacts...")
            tr_m = tr[tr["Female"] == 0].dropna(subset=[time_col, event_col]).copy()
            tr_f = tr[tr["Female"] == 1].dropna(subset=[time_col, event_col]).copy()
            te_m = te[te["Female"] == 0].copy()
            te_f = te[te["Female"] == 1].copy()

            base_path = os.path.join(models_dir, featset_name, "sex_specific")
            _ensure_dir(base_path)

            models_specific: Dict[str, CoxPHFitter] = {}
            thresholds_specific: Dict[str, float] = {}

            if not tr_m.empty:
                cph_m = fit_cox_model(tr_m, used_shared_specific_feats, time_col, event_col)
                tr_risk_m = predict_risk(cph_m, tr_m, used_shared_specific_feats)
                thr_m = float(np.nanmedian(tr_risk_m))
                models_specific["male"] = cph_m
                thresholds_specific["male"] = thr_m
                _save_pickle(cph_m, os.path.join(base_path, "male_model.pkl"))
                # coefs
                plot_cox_coefficients(
                    cph_m,
                    f"{featset_name} Male Cox Coefficients",
                    reference_df=tr_m,
                    effect_scale="per_sd",
                    save_path=os.path.join(figs_dir, f"{featset_name}_male_coefs.png"),
                )
                # coef table
                try:
                    cph_m.summary.to_csv(
                        os.path.join(base_path, "male_summary.csv"), index=True
                    )
                except Exception:
                    pass
                try:
                    cph_m.params_.to_frame("coef").to_csv(
                        os.path.join(base_path, "male_coef.csv")
                    )
                except Exception:
                    pass

            if not tr_f.empty:
                cph_f = fit_cox_model(tr_f, used_shared_specific_feats, time_col, event_col)
                tr_risk_f = predict_risk(cph_f, tr_f, used_shared_specific_feats)
                thr_f = float(np.nanmedian(tr_risk_f))
                models_specific["female"] = cph_f
                thresholds_specific["female"] = thr_f
                _save_pickle(cph_f, os.path.join(base_path, "female_model.pkl"))
                plot_cox_coefficients(
                    cph_f,
                    f"{featset_name} Female Cox Coefficients",
                    reference_df=tr_f,
                    effect_scale="per_sd",
                    save_path=os.path.join(figs_dir, f"{featset_name}_female_coefs.png"),
                )
                try:
                    cph_f.summary.to_csv(
                        os.path.join(base_path, "female_summary.csv"), index=True
                    )
                except Exception:
                    pass
                try:
                    cph_f.params_.to_frame("coef").to_csv(
                        os.path.join(base_path, "female_coef.csv")
                    )
                except Exception:
                    pass

            # Persist thresholds and features used
            _save_json(
                {
                    "features": used_shared_specific_feats,
                    "thresholds": thresholds_specific,
                },
                os.path.join(base_path, "meta.json"),
            )

            # Evaluate on test
            _progress(f"[Heldout] {featset_name}: Evaluating sex-specific on test and exporting plots...")
            te_out = te.copy()
            if "male" in models_specific and not te_m.empty:
                r_m = predict_risk(
                    models_specific["male"], te_m, used_shared_specific_feats
                )
                te_out.loc[te_out["Female"] == 0, "pred_prob"] = r_m
                te_out.loc[te_out["Female"] == 0, "pred_label"] = (
                    r_m >= thresholds_specific.get("male", float("nan"))
                ).astype(int)
            if "female" in models_specific and not te_f.empty:
                r_f = predict_risk(
                    models_specific["female"], te_f, used_shared_specific_feats
                )
                te_out.loc[te_out["Female"] == 1, "pred_prob"] = r_f
                te_out.loc[te_out["Female"] == 1, "pred_label"] = (
                    r_f >= thresholds_specific.get("female", float("nan"))
                ).astype(int)

            merged = (
                te_out[["MRN", "Female", "pred_label", time_col, event_col]]
                .dropna(subset=[time_col, event_col])
                .drop_duplicates(subset=["MRN"])
                .rename(columns={event_col: "PE"})
            )
            # Save preds
            merged.to_csv(
                os.path.join(preds_dir, f"{featset_name}_sex_specific_test_preds.csv"),
                index=False,
            )
            # KM + log-rank
            plot_km_two_subplots_by_gender(
                merged,
                save_path=os.path.join(
                    figs_dir, f"{featset_name}_sex_specific_km_by_gender.png"
                ),
            )
            # Metrics
            mask_m = merged["Female"].values == 0
            mask_f = merged["Female"].values == 1
            c_all = _safe_cindex(
                merged["PE_Time"].values, merged["PE"].values, merged["pred_prob"].values
            )
            c_m = _safe_cindex(
                merged.loc[mask_m, "PE_Time"].values,
                merged.loc[mask_m, "PE"].values,
                merged.loc[mask_m, "pred_prob"].values,
            )
            c_f = _safe_cindex(
                merged.loc[mask_f, "PE_Time"].values,
                merged.loc[mask_f, "PE"].values,
                merged.loc[mask_f, "pred_prob"].values,
            )
            pvals = _compute_logrank_pvalues_by_gender(merged)
            hrs = _compute_hr_by_gender(merged)
            rows.append(
                {
                    "feature_set": featset_name,
                    "mode": "sex_specific",
                    "c_index_all": c_all,
                    "c_index_male": c_m,
                    "c_index_female": c_f,
                    "logrank_p_male": pvals.get("male"),
                    "logrank_p_female": pvals.get("female"),
                    "hr_male": hrs.get("male", (np.nan, np.nan, np.nan))[0],
                    "hr_male_ci_low": hrs.get("male", (np.nan, np.nan, np.nan))[1],
                    "hr_male_ci_high": hrs.get("male", (np.nan, np.nan, np.nan))[2],
                    "hr_female": hrs.get("female", (np.nan, np.nan, np.nan))[0],
                    "hr_female_ci_low": hrs.get("female", (np.nan, np.nan, np.nan))[1],
                    "hr_female_ci_high": hrs.get("female", (np.nan, np.nan, np.nan))[2],
                }
            )
        except Exception as e:
            warnings.warn(f"[Heldout][{featset_name}] sex-specific skipped/failed: {e}")

    _progress("[Heldout] Saving summary metrics to CSV/XLSX...")
    summary_df = pd.DataFrame(rows)
    try:
        summary_path_csv = os.path.join(results_dir, "heldout", "summary_metrics.csv")
        _ensure_dir(os.path.dirname(summary_path_csv))
        summary_df.to_csv(summary_path_csv, index=False)
    except Exception:
        pass

    try:
        summary_path_xlsx = os.path.join(results_dir, "heldout", "summary_metrics.xlsx")
        _ensure_dir(os.path.dirname(summary_path_xlsx))
        summary_df.to_excel(summary_path_xlsx, index=False)
    except Exception:
        pass

    return summary_df


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
    # Extend categorical set with Advanced LGE nominal/binary variables if present
    try:
        adv_lge_cats = [c for c in ADV_LGE_NOMINAL if c in variables]
    except Exception:
        adv_lge_cats = []
    categorical_cols = [c for c in known_cats if c in variables] + adv_lge_cats

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
        print(f"[Cox] Dropping constant/no-information columns: {constant_cols}")
    X = X.drop(columns=constant_cols, errors="ignore")

    if X.shape[1] == 0:
        return X, []

    # Remove exactly duplicated columns
    X_filled = X.fillna(0.0)
    duplicated_mask = X_filled.T.duplicated(keep="first")
    if duplicated_mask.any():
        dup_cols = X.columns[duplicated_mask.values].tolist()
        if verbose:
            print(f"[Cox] Dropping duplicated columns: {dup_cols}")
        X = X.loc[:, ~duplicated_mask.values]

    if X.shape[1] <= 1:
        kept = list(X.columns)
        if verbose:
            removed = [c for c in original_features if c not in kept]
            if removed:
                print(f"[Cox] After sanitization kept {kept}, removed: {removed}")
        return X, kept

    # Remove highly correlated columns (keep the first in order)
    corr = X.fillna(0.0).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if (upper[col] >= corr_threshold).any()]
    if to_drop and verbose:
        print(f"[Cox] Dropping highly correlated columns (|r|>={corr_threshold}): {to_drop}")
    X = X.drop(columns=to_drop, errors="ignore")

    kept = list(X.columns)
    if verbose:
        removed = [c for c in original_features if c not in kept]
        if removed:
            print(f"[Cox] Kept {len(kept)}/{len(original_features)} features: {kept}")
            print(f"[Cox] Total removed: {removed}")
    return X, kept


# ==========================================
# CoxPH training/inference blocks
# ==========================================


def fit_cox_model(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    time_col: str,
    event_col: str,
    robust: bool = True,
) -> CoxPHFitter:
    # Guard against missing columns in feature list
    present_features = [c for c in feature_cols if c in train_df.columns]
    X_sanitized, kept_features = _sanitize_cox_features_matrix(
        train_df, present_features, corr_threshold=0.995
    )
    if len(kept_features) == 0:
        candidates = [
            c for c in present_features if train_df[c].nunique(dropna=False) > 1
        ]
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
            cph.fit(
                df_fit, duration_col=time_col, event_col=event_col, robust=robust
            )
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
            cph.fit(
                df_fit2, duration_col=time_col, event_col=event_col, robust=robust
            )
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
                # Use non-robust fitting for speed during feature selection
                cph = fit_cox_model(
                    inner_tr, trial_feats, time_col, event_col, robust=False
                )
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

    # Allow environment overrides for selection verbosity and max features
    try:
        if max_features is None:
            _mf = os.environ.get("SCMR_FS_MAX_FEATURES")
            if _mf is not None and _mf.strip() != "":
                _mf_int = int(_mf)
                max_features = None if _mf_int <= 0 else _mf_int
    except Exception:
        pass

    env_verbose = str(os.environ.get("SCMR_FS_VERBOSE", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
    }
    _verbose = bool(verbose or env_verbose)

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
            if _verbose:
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
                verbose=_verbose,
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

    if _verbose:
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


def compute_shared_features(
    df: pd.DataFrame,
    base_features: List[str],
    time_col: str = "PE_Time",
    event_col: str = "VT/VF/SCD",
    seeds: List[int] = None,
    threshold: float = 0.4,
    verbose: bool = True,
) -> List[str]:
    """Compute a single, shared feature set to be used for both sexes.

    - Excludes `Female` from candidates.
    - Runs stability selection on the full cohort (male+female together) to
      produce a consistent subset. Falls back to the provided candidates when
      selection yields empty.
    """
    if seeds is None:
        seeds = list(range(20))
    candidates = [f for f in base_features if f != "Female"]
    try:
        sel = stability_select_features(
            df=df.dropna(subset=[time_col, event_col]).copy(),
            candidate_features=list(candidates),
            time_col=time_col,
            event_col=event_col,
            seeds=seeds,
            max_features=None,
            threshold=threshold,
            min_features=None,
            verbose=verbose,
        )
        if sel:
            return list(sel)
    except Exception:
        pass
    return list(candidates)


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
    # Build one shared feature set for both sexes (exclude Female)
    base_candidates = [f for f in features if f != "Female"]
    try:
        # Unify features for Proposed/Advanced using stability selection once
        if set(features) == set(FEATURE_SETS.get("Proposed", [])) or set(features) == set(
            FEATURE_SETS.get("Advanced", [])
        ):
            is_proposed = set(features) == set(FEATURE_SETS.get("Proposed", []))
            shared_features = compute_shared_features(
                df=train_df,
                base_features=base_candidates,
                time_col="PE_Time",
                event_col="VT/VF/SCD",
                seeds=list(range(20)),
                threshold=0.4,
                verbose=True,
            )
            # Cache into store for traceability/consistency
            store_key = "proposed_sex_specific" if is_proposed else "advanced_sex_specific"
            try:
                SELECTED_FEATURES_STORE[store_key]["male"] = list(shared_features)
                SELECTED_FEATURES_STORE[store_key]["female"] = list(shared_features)
            except Exception:
                pass
        else:
            shared_features = list(base_candidates)
    except Exception:
        shared_features = list(base_candidates)

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
        cph_m = fit_cox_model(train_m, shared_features, "PE_Time", "VT/VF/SCD")
        tr_risk_m = predict_risk(cph_m, train_m, shared_features)
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
        cph_f = fit_cox_model(train_f, shared_features, "PE_Time", "VT/VF/SCD")
        tr_risk_f = predict_risk(cph_f, train_f, shared_features)
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
        risk_m = predict_risk(models["male"], test_m, shared_features)
        df_out.loc[df_out["Female"] == 0, "pred_prob"] = risk_m
        df_out.loc[df_out["Female"] == 0, "pred_label"] = (
            risk_m >= thresholds["male"]
        ).astype(int)
    if "female" in models and not test_f.empty:
        risk_f = predict_risk(models["female"], test_f, shared_features)
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
    _progress("[Full] Sex-specific: Preparing shared features and running stability selection (if enabled)...")
    # Build one shared feature set for both sexes (exclude Female)
    base_candidates = [f for f in features if f != "Female"]
    try:
        if set(features) == set(FEATURE_SETS.get("Proposed", [])) or set(features) == set(
            FEATURE_SETS.get("Advanced", [])
        ):
            is_proposed = set(features) == set(FEATURE_SETS.get("Proposed", []))
            shared_features = compute_shared_features(
                df=df,
                base_features=base_candidates,
                time_col="PE_Time",
                event_col="VT/VF/SCD",
                seeds=list(range(20)),
                threshold=0.4,
                verbose=True,
            )
            store_key = "proposed_sex_specific" if is_proposed else "advanced_sex_specific"
            try:
                SELECTED_FEATURES_STORE[store_key]["male"] = list(shared_features)
                SELECTED_FEATURES_STORE[store_key]["female"] = list(shared_features)
            except Exception:
                pass
        else:
            shared_features = list(base_candidates)
    except Exception:
        shared_features = list(base_candidates)

    data_m = df[df["Female"] == 0].dropna(subset=["PE_Time", "VT/VF/SCD"]).copy()
    data_f = df[df["Female"] == 1].dropna(subset=["PE_Time", "VT/VF/SCD"]).copy()

    models = {}
    thresholds = {}

    if not data_m.empty:
        _progress("[Full] Sex-specific: Training male model and plotting...")
        used_features_m = list(shared_features)
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
        _progress("[Full] Sex-specific: Training female model and plotting...")
        used_features_f = list(shared_features)
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
    _progress("[Full] Sex-specific: Generated KM plots and finished")
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
    _progress("[Full] Sex-agnostic: Starting data prep and feature stability selection...")
    df_base = (
        create_undersampled_dataset(df, label_col, 42) if use_undersampling else df
    )
    used_features = features
    # If features correspond to Proposed/Advanced, perform stability selection ONCE and freeze
    try:
        if set(features) == set(FEATURE_SETS.get("Proposed", [])) or set(features) == set(FEATURE_SETS.get("Advanced", [])):
            is_proposed = set(features) == set(FEATURE_SETS.get("Proposed", []))
            store_key = "proposed_sex_agnostic" if is_proposed else "advanced_sex_agnostic"
            selected = SELECTED_FEATURES_STORE.get(store_key)
            if not selected:
                seeds_for_stability = list(range(20))
                selected = stability_select_features(
                    df=df_base.dropna(subset=["PE_Time", label_col]).copy(),
                    candidate_features=list(features),
                    time_col="PE_Time",
                    event_col=label_col,
                    seeds=seeds_for_stability,
                    max_features=None,
                    threshold=0.4,
                    min_features=None,
                    verbose=True,
                )
                if selected:
                    SELECTED_FEATURES_STORE[store_key] = list(selected)
                    print(
                        f"[FS][Store] {'Proposed' if is_proposed else 'Advanced'} sex-agnostic features stored: {len(selected)}"
                    )
            if selected:
                used_features = list(selected)
    except Exception:
        pass
    data = df_base.dropna(subset=["PE_Time", label_col]).copy()
    if data.empty:
        return pd.DataFrame()

    _progress("[Full] Sex-agnostic: Training model and plotting coefficients...")
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
    _progress("[Full] Sex-agnostic: Generated KM plots and finished")
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
        # Feature selection only for Proposed/Advanced set
        if not disable_within_split_feature_selection:
            try:
                if set(feature_cols) == set(FEATURE_SETS.get("Proposed", [])) or set(feature_cols) == set(FEATURE_SETS.get("Advanced", [])):
                    selected = select_features_max_cindex_forward(
                        tr, list(feature_cols), time_col, event_col, random_state=seed
                    )
                    if selected:
                        used_features = selected
                        model_name = (
                            "Proposed"
                            if set(feature_cols) == set(FEATURE_SETS.get("Proposed", []))
                            else "Advanced"
                        )
                        print(
                            f"[FS] Sex-agnostic: selected {len(selected)} features for {model_name}: {selected}"
                        )
            except Exception:
                pass
        cph = fit_cox_model(tr, used_features, time_col, event_col)
        # Threshold must be derived from training risks (avoid test leakage)
        tr_risk = predict_risk(cph, tr, used_features)
        thr = threshold_by_top_quantile(tr_risk, 0.5)
        risk_scores = predict_risk(cph, test_df, used_features)
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
        # Threshold from training male subset
        tr_risk_m = predict_risk(cph, tr_m, used_features)
        thr = threshold_by_top_quantile(tr_risk_m, 0.5)
        risk_m = predict_risk(cph, te_m, used_features)
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
        # Threshold from training female subset
        tr_risk_f = predict_risk(cph, tr_f, used_features)
        thr = threshold_by_top_quantile(tr_risk_f, 0.5)
        risk_f = predict_risk(cph, te_f, used_features)
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
                    # Feature selection for Proposed/Advanced per sex (male)
                    try:
                        if set(feature_cols) == set(FEATURE_SETS.get("Proposed", [])) or set(feature_cols) == set(FEATURE_SETS.get("Advanced", [])):
                            selected_m = select_features_max_cindex_forward(
                                tr_m,
                                list(used_features),
                                time_col,
                                event_col,
                                random_state=seed,
                            )
                            if selected_m:
                                used_features_m = selected_m
                                model_name = (
                                    "Proposed"
                                    if set(feature_cols) == set(FEATURE_SETS.get("Proposed", []))
                                    else "Advanced"
                                )
                                print(
                                    f"[FS] Sex-specific (male): selected {len(selected_m)} features for {model_name}: {selected_m}"
                                )
                    except Exception:
                        pass
            cph_m = fit_cox_model(tr_m, used_features_m, time_col, event_col)
            # Threshold from training male subset
            tr_risk_m = predict_risk(cph_m, tr_m, used_features_m)
            thr_m = threshold_by_top_quantile(tr_risk_m, 0.5)
            risk_m = predict_risk(cph_m, te_m, used_features_m)
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
                    # Feature selection for Proposed/Advanced per sex (female)
                    try:
                        if set(feature_cols) == set(FEATURE_SETS.get("Proposed", [])) or set(feature_cols) == set(FEATURE_SETS.get("Advanced", [])):
                            selected_f = select_features_max_cindex_forward(
                                tr_f,
                                list(used_features),
                                time_col,
                                event_col,
                                random_state=seed,
                            )
                            if selected_f:
                                used_features_f = selected_f
                                model_name = (
                                    "Proposed"
                                    if set(feature_cols) == set(FEATURE_SETS.get("Proposed", []))
                                    else "Advanced"
                                )
                                print(
                                    f"[FS] Sex-specific (female): selected {len(selected_f)} features for {model_name}: {selected_f}"
                                )
                    except Exception:
                        pass
            cph_f = fit_cox_model(tr_f, used_features_f, time_col, event_col)
            # Threshold from training female subset
            tr_risk_f = predict_risk(cph_f, tr_f, used_features_f)
            thr_f = threshold_by_top_quantile(tr_risk_f, 0.5)
            risk_f = predict_risk(cph_f, te_f, used_features_f)
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

    # Normalize column labels to reduce breakage due to stray spaces
    df.columns = [_normalize_column_name(c) for c in df.columns]

    # Resolve aliases for requested columns (features + labels)
    requested = list(dict.fromkeys(list(features) + list(labels)))
    alias_map = _build_alias_map(df.columns.tolist())

    resolved_columns: List[str] = []
    rename_map: Dict[str, str] = {}
    missing_requested: List[str] = []

    for col in requested:
        # Exact match first
        if col in df.columns:
            resolved_columns.append(col)
            continue
        # Try simplified alias mapping
        key = _simplify_name(col)
        matched = alias_map.get(key)
        if matched is not None:
            resolved_columns.append(matched)
            # Rename matched column to the canonical requested name
            if matched != col:
                rename_map[matched] = col
        else:
            missing_requested.append(col)

    if missing_requested:
        warnings.warn(
            "The following requested columns were not found and will be ignored: "
            + ", ".join(missing_requested)
        )

    # Keep only the resolved columns (present in df)
    if not resolved_columns:
        return pd.DataFrame()
    df = df[resolved_columns]
    if rename_map:
        df = df.rename(columns=rename_map)

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
        # Newly added binary variables
        "LGE_Circumural",
        "LGE_Ring-Like",
    ]
    exist_bin = [c for c in binary_cols if c in df.columns]
    for c in exist_bin:
        if df[c].dtype == "object":
            df[c] = df[c].replace(
                {"Yes": 1, "No": 0, "Y": 1, "N": 0, "True": 1, "False": 0}
            )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Broadly coerce all requested feature columns to numeric when possible
    # This covers Advanced/DERIVATIVE columns that are binary-like but not in known list
    yes_no_map = {"Yes": 1, "No": 0, "Y": 1, "N": 0, "True": 1, "False": 0}
    for c in features:
        if c in df.columns and df[c].dtype == "object":
            df[c] = df[c].replace(yes_no_map)
            # Extract numerics from strings like "1 superior" -> 1
            try:
                df[c] = df[c].astype(str).str.extract(r"(-?\d+(?:\.\d+)?)").astype(float)
            except Exception:
                pass
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Imputation on feature matrix
    X = df[features].copy()
    # Optional: downcast float64 to float32 before MissForest to speed up
    try:
        X = X.apply(lambda s: s.astype(np.float32) if s.dtype.kind == 'f' else s)
    except Exception:
        pass
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
    # Resolve NICM.xlsx location with env overrides and safe fallbacks
    base_default = "/home/sunx/data/aiiih/projects/sunx/projects/ICD"
    base = os.environ.get("ICD_DATA_BASE", base_default)
    nicm_path = os.environ.get("NICM_XLSX", os.path.join(base, "NICM.xlsx"))
    if not os.path.exists(nicm_path):
        alt = os.path.join(os.getcwd(), "NICM.xlsx")
        if os.path.exists(alt):
            nicm_path = alt
        else:
            raise FileNotFoundError(
                f"NICM.xlsx not found at '{nicm_path}'. Set env 'NICM_XLSX' or 'ICD_DATA_BASE'."
            )

    icd = pd.read_excel(nicm_path, sheet_name="ICD")
    noicd = pd.read_excel(nicm_path, sheet_name="No_ICD")

    # Normalize column labels to avoid subtle whitespace mismatches
    icd.columns = [_normalize_column_name(c) for c in icd.columns]
    noicd.columns = [_normalize_column_name(c) for c in noicd.columns]

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

    # Align to the UNION of columns to retain variables unique to either sheet
    all_cols = sorted(set(icd.columns).union(set(noicd.columns)))
    icd_aligned = icd.reindex(columns=all_cols).copy()
    noicd_aligned = noicd.reindex(columns=all_cols).copy()
    icd_aligned["ICD"] = 1
    noicd_aligned["ICD"] = 0
    nicm = pd.concat([icd_aligned, noicd_aligned], ignore_index=True)

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
    # Extend with newly introduced nominal/binary LGE columns when present (normalized names)
    categorical += [c for c in ADV_LGE_NOMINAL if c in nicm.columns]
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

    # Seeds for per-split stability selection (train-only)
    # Allow override via environment variable SCMR_STABILITY_SEEDS
    try:
        _ss = os.environ.get("SCMR_STABILITY_SEEDS")
        if _ss is not None and _ss.strip() != "":
            ss_count = max(1, int(_ss))
        else:
            ss_count = min(N, 20)
    except Exception:
        ss_count = min(N, 20)
    seeds_for_stability = list(range(ss_count))

    _progress(f"Starting multi-split experiments: N={N}, feature_sets={list(feature_sets.keys())}")
    for seed in range(N):
        if seed % max(1, N // 10) == 0:
            _progress(f"Progress {seed}/{N} splits...")
        tr, te = train_test_split(
            df, test_size=0.3, random_state=seed, stratify=df[event_col]
        )
        tr = tr.dropna(subset=[time_col, event_col])
        te = te.dropna(subset=[time_col, event_col])

        for featset_name, feature_cols in feature_sets.items():
            for cfg in model_configs:
                name = f"{featset_name} - {cfg['name']}"
                use_undersampling = cfg["mode"] == "sex_agnostic"
                # Only Proposed/Advanced run sex-specific; others skip
                if cfg["mode"] == "sex_specific" and featset_name not in {"Proposed", "Advanced"}:
                    continue
                # Per-split stability selection on train only for Proposed/Advanced
                if featset_name in {"Proposed", "Advanced"}:
                    _progress(
                        f"seed={seed}: Feature set [{featset_name}] stability selection (sex-agnostic/shared)"
                    )
                    # sex-agnostic candidates include Female
                    sel_agn = stability_select_features(
                        df=tr,
                        candidate_features=list(feature_cols),
                        time_col=time_col,
                        event_col=event_col,
                        seeds=seeds_for_stability,
                        max_features=None,
                        threshold=0.4,
                        min_features=None,
                        verbose=False,
                    ) or list(feature_cols)
                    # sex-specific shared candidates exclude Female
                    base_feats = [f for f in feature_cols if f != "Female"]
                    sel_shared = stability_select_features(
                        df=tr,
                        candidate_features=list(base_feats),
                        time_col=time_col,
                        event_col=event_col,
                        seeds=seeds_for_stability,
                        max_features=None,
                        threshold=0.4,
                        min_features=None,
                        verbose=False,
                    ) or list(base_feats)
                else:
                    sel_agn = list(feature_cols)
                    sel_shared = [f for f in feature_cols if f != "Female"]

                # Use stabilized features and disable per-split selection
                if cfg["mode"] == "sex_agnostic":
                    _progress(
                        f"seed={seed}: Train/Eval {name} (sex-agnostic)"
                    )
                    pred, risk, met = evaluate_split(
                        tr,
                        te,
                        sel_agn,
                        time_col,
                        event_col,
                        mode=cfg["mode"],
                        seed=seed,
                        use_undersampling=use_undersampling,
                        disable_within_split_feature_selection=True,
                    )
                else:
                    overrides = {"male": list(sel_shared), "female": list(sel_shared)}
                    _progress(
                        f"seed={seed}: Train/Eval {name} (sex-specific)"
                    )
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

    _progress("All splits finished, summarizing results...")
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
    "DERIVATIVE": ["Female", "LVEF", "LGE_Score"],
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
    "Advanced": [
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
        # Advanced LGE nominal/binary (use normalized list to avoid whitespace variants)
        *ADV_LGE_NOMINAL,
    ],
}


if __name__ == "__main__":
    # Load and prepare data
    try:
        df = load_dataframes()
    except Exception as e:
        warnings.warn(f"[Main] Data loading failed: {e}")
        raise

    # Output directory override
    results_dir = os.environ.get("SCMR_RESULTS_DIR") or os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Generate TableOne grouped by Sex x ICD (four groups)
    try:
        generate_tableone_by_sex_icd(
            df,
            output_excel_path=os.path.join(results_dir, "results_tableone_sex_icd.xlsx"),
        )
    except Exception as e:
        warnings.warn(f"[Main] TableOne generation skipped due to error: {e}")

    # Toggle legacy multi-split experiments
    enable_legacy = os.environ.get("SCMR_ENABLE_LEGACY_EXPERIMENTS", "1") == "1"
    if enable_legacy:
        try:
            n_splits_env = int(os.environ.get("SCMR_N_SPLITS", "10"))
        except Exception:
            n_splits_env = 10
        export_path = os.path.join(results_dir, "results_cox.xlsx")
        _, summary = run_cox_experiments(
            df=df,
            feature_sets=FEATURE_SETS,
            N=n_splits_env,
            time_col="PE_Time",
            event_col="VT/VF/SCD",
            export_excel_path=export_path,
        )
        print("Saved Excel:", export_path)

    # Held-out single split training + evaluation across all feature sets
    enable_heldout = os.environ.get("SCMR_ENABLE_HELDOUT", "1") == "1"
    if enable_heldout:
        try:
            heldout_test_size = float(os.environ.get("SCMR_HELDOUT_TEST_SIZE", "0.3"))
        except Exception:
            heldout_test_size = 0.3
        try:
            heldout_seed = int(os.environ.get("SCMR_HELDOUT_SEED", "42"))
        except Exception:
            heldout_seed = 42
        print("Running held-out training and evaluation...")
        heldout_summary = run_heldout_training_and_evaluation(
            df=df,
            feature_sets=FEATURE_SETS,
            time_col="PE_Time",
            event_col="VT/VF/SCD",
            results_dir=results_dir,
            test_size=heldout_test_size,
            random_state=heldout_seed,
        )
        print("Held-out summary saved to results/heldout/")

    # Toggle legacy full-data inference blocks (for visualizations not saved by default)
    enable_full_data_blocks = os.environ.get("SCMR_ENABLE_FULLDATA_BLOCKS", "0") == "1"
    if enable_full_data_blocks:
        # Full-data inference and analysis - Guideline
        features = FEATURE_SETS["Guideline"]
        print("Running sex-agnostic full-data inference (includes Female)...")
        _ = sex_agnostic_full_inference(df, features, use_undersampling=False)
        # Guideline: skip sex-specific per requirement

        # Full-data inference and analysis - Benchmark
        features = FEATURE_SETS["Benchmark"]
        print("Running sex-agnostic full-data inference (includes Female)...")
        _ = sex_agnostic_full_inference(df, features, use_undersampling=False)
        # Benchmark: skip sex-specific per requirement

        # Full-data inference and analysis - DERIVATIVE
        features = FEATURE_SETS["DERIVATIVE"]
        print("Running sex-agnostic full-data inference (includes Female)...")
        _ = sex_agnostic_full_inference(df, features, use_undersampling=False)
        # DERIVATIVE: skip sex-specific per requirement

        # Full-data inference and analysis - Proposed
        features = FEATURE_SETS["Proposed"]
        print("Running sex-agnostic full-data inference (includes Female)...")
        _ = sex_agnostic_full_inference(df, features, use_undersampling=False)
        print("Running sex-specific full-data inference (excludes Female in submodels)...")
        _ = sex_specific_full_inference(df, features)

        # Full-data inference and analysis - Advanced
        features = FEATURE_SETS["Advanced"]
        print("Running sex-agnostic full-data inference (includes Female)...")
        _ = sex_agnostic_full_inference(df, features, use_undersampling=False)
        print("Running sex-specific full-data inference (excludes Female in submodels)...")
        _ = sex_specific_full_inference(df, features)

    # =============================
    # Print features used by models
    # =============================
    try:
        print("\n==== Features used by each model ====")
        # Guideline/Benchmark/DERIVATIVE: no feature selection
        gl_feats = FEATURE_SETS.get("Guideline", [])
        bm_feats = FEATURE_SETS.get("Benchmark", [])
        dv_feats = FEATURE_SETS.get("DERIVATIVE", [])
        print(f"Guideline (sex-agnostic): {gl_feats}")
        print(f"Benchmark (sex-agnostic): {bm_feats}")
        print(f"DERIVATIVE (sex-agnostic): {dv_feats}")

        # Proposed: sex-agnostic + sex-specific if available
        prop_agn = SELECTED_FEATURES_STORE.get("proposed_sex_agnostic")
        prop_agn = prop_agn if prop_agn else FEATURE_SETS.get("Proposed", [])
        prop_m = SELECTED_FEATURES_STORE.get("proposed_sex_specific", {}).get("male")
        prop_f = SELECTED_FEATURES_STORE.get("proposed_sex_specific", {}).get("female")
        if not prop_m:
            prop_m = [f for f in FEATURE_SETS.get("Proposed", []) if f != "Female"]
        if not prop_f:
            prop_f = [f for f in FEATURE_SETS.get("Proposed", []) if f != "Female"]
        print(f"Proposed (sex-agnostic): {prop_agn}")
        print(f"Proposed (male-specific): {prop_m}")
        print(f"Proposed (female-specific): {prop_f}")

        # Advanced: sex-agnostic + sex-specific if available
        adv_agn = SELECTED_FEATURES_STORE.get("advanced_sex_agnostic")
        adv_agn = adv_agn if adv_agn else FEATURE_SETS.get("Advanced", [])
        adv_m = SELECTED_FEATURES_STORE.get("advanced_sex_specific", {}).get("male")
        adv_f = SELECTED_FEATURES_STORE.get("advanced_sex_specific", {}).get("female")
        if not adv_m:
            adv_m = [f for f in FEATURE_SETS.get("Advanced", []) if f != "Female"]
        if not adv_f:
            adv_f = [f for f in FEATURE_SETS.get("Advanced", []) if f != "Female"]
        print(f"Advanced (sex-agnostic): {adv_agn}")
        print(f"Advanced (male-specific): {adv_m}")
        print(f"Advanced (female-specific): {adv_f}")
    except Exception as e:
        warnings.warn(f"[Main] Failed to print feature sets: {e}")
