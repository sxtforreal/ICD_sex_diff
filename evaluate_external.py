import os
import re
import json
import argparse
import warnings
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

# Lifelines for metrics/plots
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

# Reuse the training pipeline’s preprocessing and helpers
import SCMR as scmr


def _progress(msg: str) -> None:
    try:
        print(f"[EvalExt] {msg}", flush=True)
    except Exception:
        pass


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _read_external_dataframe(path: str, sheet: Optional[str] = None) -> pd.DataFrame:
    """Read external dataset from CSV/XLSX, normalize column names to match SCMR."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"External dataset not found: {path}")
    try:
        if path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(path, sheet_name=sheet if sheet else 0)
        else:
            df = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read external dataset: {e}")

    # Normalize column labels similar to SCMR
    try:
        df.columns = [scmr._normalize_column_name(c) for c in df.columns]
    except Exception:
        df.columns = [str(c).strip() if isinstance(c, str) else c for c in df.columns]
    return df


def _derive_endpoints_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """Derive PE_Time if missing but date columns exist, following SCMR logic."""
    df = df.copy()
    if "PE_Time" in df.columns:
        return df

    # Try to locate likely date columns via simplified names
    def _simplify(name: Any) -> str:
        s = str(name) if not isinstance(name, str) else name
        s = s.strip().lower()
        s = re.sub(r"\s+", " ", s)
        return s

    cols = {_simplify(c): c for c in df.columns}
    mri_col = cols.get("mri date") or cols.get("date of mri") or cols.get("mri_date")
    event_date_col = cols.get("date vt/vf/scd") or cols.get("vt/vf/scd date")
    end_fu_col = cols.get("end follow-up date") or cols.get("end of follow-up") or cols.get("follow-up end date")

    if mri_col is None or (event_date_col is None and end_fu_col is None):
        # Cannot derive
        return df

    try:
        df[mri_col] = pd.to_datetime(df[mri_col])
    except Exception:
        return df

    if event_date_col and event_date_col in df.columns:
        df[event_date_col] = pd.to_datetime(df[event_date_col], errors="coerce")
    if end_fu_col and end_fu_col in df.columns:
        df[end_fu_col] = pd.to_datetime(df[end_fu_col], errors="coerce")

    # Need event indicator to decide which date to use; if not present, try to coerce from strings
    evt_col = "VT/VF/SCD" if "VT/VF/SCD" in df.columns else None
    if evt_col is None:
        # Attempt to create from yes/no-like column names
        for candidate in df.columns:
            key = _simplify(candidate)
            if key in {"vt/vf/scd", "event", "primary endpoint", "pe"}:
                evt_col = candidate
                break
    if evt_col is not None and df[evt_col].dtype == "object":
        # Use SCMR binary coercion
        try:
            df[evt_col] = scmr._coerce_yes_no_to_binary(df[evt_col])
        except Exception:
            df[evt_col] = pd.to_numeric(df[evt_col], errors="coerce")

    # Compute PE_Time following SCMR.load_dataframes
    try:
        df["PE_Time"] = df.apply(
            lambda row: (
                (row[event_date_col] - row[mri_col]).days
                if (evt_col and pd.to_numeric(row.get(evt_col, np.nan), errors="coerce") == 1)
                and pd.notna(row.get(event_date_col))
                and pd.notna(row.get(mri_col))
                else (
                    (row[end_fu_col] - row[mri_col]).days
                    if pd.notna(row.get(end_fu_col)) and pd.notna(row.get(mri_col))
                    else np.nan
                )
            ),
            axis=1,
        )
    except Exception:
        pass
    return df


def _discover_models(models_root: str) -> List[Dict[str, Any]]:
    """Discover saved models and metadata under models_root structure created by SCMR.

    Returns list of model specs:
      { 'feature_set': str,
        'mode': 'sex_agnostic' | 'sex_specific',
        'paths': { ... },
        'features': List[str],
        'threshold' or 'thresholds': ... }
    """
    found: List[Dict[str, Any]] = []
    if not os.path.isdir(models_root):
        return found

    for featset in sorted(os.listdir(models_root)):
        feat_dir = os.path.join(models_root, featset)
        if not os.path.isdir(feat_dir):
            continue
        # Sex-agnostic
        agn_dir = os.path.join(feat_dir, "sex_agnostic")
        meta_path = os.path.join(agn_dir, "meta.json")
        model_path = os.path.join(agn_dir, "model.pkl")
        if os.path.exists(model_path):
            spec = {"feature_set": featset, "mode": "sex_agnostic", "paths": {"model": model_path, "meta": meta_path}}
            try:
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                else:
                    meta = {}
                spec["features"] = list(meta.get("features", []))
                if "threshold" in meta:
                    spec["threshold"] = float(meta.get("threshold"))
            except Exception:
                spec["features"] = []
            found.append(spec)
        # Sex-specific
        sp_dir = os.path.join(feat_dir, "sex_specific")
        male_path = os.path.join(sp_dir, "male_model.pkl")
        female_path = os.path.join(sp_dir, "female_model.pkl")
        meta_path2 = os.path.join(sp_dir, "meta.json")
        if os.path.exists(male_path) or os.path.exists(female_path):
            spec = {
                "feature_set": featset,
                "mode": "sex_specific",
                "paths": {"male": male_path, "female": female_path, "meta": meta_path2},
            }
            try:
                if os.path.exists(meta_path2):
                    with open(meta_path2, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                else:
                    meta = {}
                spec["features"] = list(meta.get("features", []))  # shared features
                spec["thresholds"] = dict(meta.get("thresholds", {}))
            except Exception:
                spec["features"] = []
            found.append(spec)
    return found


def _safe_cindex(t: np.ndarray, e: np.ndarray, r: np.ndarray) -> float:
    try:
        if t.size == 0 or e.size == 0 or r.size == 0:
            return float("nan")
        if np.all(~np.isfinite(r)) or np.allclose(r, r[0], equal_nan=True):
            return float("nan")
        return float(concordance_index(t, -r, e))
    except Exception:
        return float("nan")


def _compute_logrank_p_by_gender(df_with_preds: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {"male": float("nan"), "female": float("nan")}
    try:
        for sex_val, key in [(0, "male"), (1, "female")]:
            subset = df_with_preds[df_with_preds["Female"] == sex_val]
            if subset.empty or subset["pred_label"].nunique() < 2:
                out[key] = float("nan")
                continue
            low = subset[subset["pred_label"] == 0]
            high = subset[subset["pred_label"] == 1]
            lr = logrank_test(low["PE_Time"], high["PE_Time"], low["PE"], high["PE"])
            out[key] = float(lr.p_value)
    except Exception:
        pass
    return out


def _compute_hr_by_gender(df_with_preds: pd.DataFrame) -> Dict[str, Tuple[float, float, float]]:
    # Univariate HR between High(vs Low) groups via Cox is available in SCMR; keep minimal here
    # We report HR approximations via logrank statistic if lifelines CoxPH not imported here
    out: Dict[str, Tuple[float, float, float]] = {
        "male": (float("nan"), float("nan"), float("nan")),
        "female": (float("nan"), float("nan"), float("nan")),
    }
    try:
        from lifelines import CoxPHFitter

        for sex_val, key in [(0, "male"), (1, "female")]:
            sub = df_with_preds[df_with_preds["Female"] == sex_val]
            if sub.empty or sub["pred_label"].nunique() < 2:
                continue
            cph = CoxPHFitter()
            cph.fit(sub[["PE_Time", "PE", "pred_label"]], duration_col="PE_Time", event_col="PE", robust=True)
            hr = float(np.exp(cph.params_["pred_label"]))
            ci = cph.confidence_intervals_.loc["pred_label"].values
            out[key] = (hr, float(ci[0]), float(ci[1]))
    except Exception:
        pass
    return out


def _plot_km_by_gender(df_with_preds: pd.DataFrame, save_path: Optional[str] = None) -> None:
    if df_with_preds.empty:
        return
    needed = {"PE_Time", "PE", "Female", "pred_label"}
    if not needed.issubset(df_with_preds.columns):
        return

    kmf = KaplanMeierFitter()
    fig, axes = None, None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        for ax_idx, (sex_val, sex_name) in enumerate([(0, "Male"), (1, "Female")]):
            ax = axes[ax_idx]
            subset = df_with_preds[df_with_preds["Female"] == sex_val]
            if subset.empty:
                ax.set_title(f"{sex_name} (n=0)")
                continue
            for pred_val, color in [(0, "steelblue"), (1, "tomato")]:
                grp = subset[subset["pred_label"] == pred_val]
                if grp.empty:
                    continue
                n_samples = int(len(grp))
                events = int(grp["PE"].sum())
                label = f"{'Low' if pred_val == 0 else 'High'} risk (n={n_samples}, events={events})"
                kmf.fit(durations=grp["PE_Time"], event_observed=grp["PE"], label=label)
                kmf.plot(ax=ax, color=color)
            ax.set_title(sex_name)
            ax.set_xlabel("Days")
            if ax_idx == 0:
                ax.set_ylabel("Survival Probability")
            ax.grid(alpha=0.3)
            ax.legend(loc="best", fontsize=10)

        plt.suptitle("Primary Endpoint - Survival by Gender and Risk Group", y=1.02)
        plt.tight_layout()
        if save_path:
            _ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    except Exception:
        # If plotting fails (missing MPL), silently skip
        if fig is not None:
            try:
                import matplotlib.pyplot as plt  # type: ignore
                plt.close()
            except Exception:
                pass


def _plot_calibration_curve_3yr(
    df_with_preds: pd.DataFrame,
    save_path: Optional[str] = None,
    bins: int = 10,
) -> Optional[pd.DataFrame]:
    """Plot 3-year calibration curve using KM-based observed risk per bin.

    Returns a dataframe with per-bin calibration if computed; None otherwise.
    """
    needed = {"pred_prob", "PE_Time", "PE"}
    if df_with_preds.empty or not needed.issubset(df_with_preds.columns):
        return None

    df = df_with_preds.dropna(subset=["pred_prob", "PE_Time", "PE"]).copy()
    if df.empty:
        return None

    # Create quantile bins on predicted risk
    try:
        df["calib_bin"], bin_edges = pd.qcut(
            df["pred_prob"], q=bins, labels=False, retbins=True, duplicates="drop"
        )
    except Exception:
        # Fallback to equal-width bins when qcut fails
        try:
            df["calib_bin"] = pd.cut(df["pred_prob"], bins=bins, labels=False, include_lowest=True)
            bin_edges = None
        except Exception:
            return None

    rows: List[Dict[str, Any]] = []
    horizon_days = getattr(scmr, "HORIZON_DAYS", 3 * 365)

    # Compute observed risk via KM at 3 years per bin
    for bin_id, grp in df.groupby("calib_bin"):
        if grp.empty:
            continue
        kmf = KaplanMeierFitter()
        try:
            kmf.fit(durations=grp["PE_Time"], event_observed=grp["PE"])
            s_at_h = float(kmf.predict(horizon_days))
            obs_risk = max(0.0, min(1.0, 1.0 - s_at_h))
        except Exception:
            obs_risk = float("nan")
        rows.append(
            {
                "bin": int(bin_id) if pd.notna(bin_id) else -1,
                "n": int(len(grp)),
                "pred_mean": float(np.nanmean(grp["pred_prob"].values)),
                "obs_risk": obs_risk,
            }
        )

    if not rows:
        return None

    calib_df = pd.DataFrame(rows).sort_values("pred_mean").reset_index(drop=True)

    # Plot
    fig = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
        ax.plot(calib_df["pred_mean"], calib_df["obs_risk"], marker="o", color="steelblue", label="Observed")
        ax.set_xlabel("Predicted 3-year risk")
        ax.set_ylabel("Observed 3-year risk (KM)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        plt.tight_layout()
        if save_path:
            _ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    except Exception:
        if fig is not None:
            try:
                import matplotlib.pyplot as plt  # type: ignore
                plt.close()
            except Exception:
                pass

    return calib_df

def _generate_tableone_by_cohort(
    ref_df: pd.DataFrame,
    ext_df: pd.DataFrame,
    variables: List[str],
    output_excel_path: Optional[str],
) -> None:
    """Generate a TableOne-like 2-cohort comparison for selected variables.

    Prefers tableone.TableOne; falls back to a simple summary if unavailable.
    """
    # Keep only columns that exist in at least one dataset to avoid errors
    vars_present = [c for c in variables if (c in ref_df.columns) or (c in ext_df.columns)]
    ref = ref_df.copy()
    ext = ext_df.copy()
    ref["Cohort"] = "SCMR"
    ext["Cohort"] = "External"
    both = pd.concat([
        ref[[c for c in vars_present + ["Cohort"] if c in ref.columns]],
        ext[[c for c in vars_present + ["Cohort"] if c in ext.columns]],
    ], axis=0, ignore_index=True)

    # Categorical variables from SCMR declarations
    try:
        known_cats = [
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
            "CrCl>45",
            "Significant LGE",
        ]
        adv_lge_cats = [c for c in scmr.ADV_LGE_NOMINAL if c in vars_present]
        categorical_cols = [c for c in known_cats if c in vars_present] + adv_lge_cats
    except Exception:
        categorical_cols = [c for c in vars_present if both[c].dtype == "object"]

    # Prefer TableOne
    has_tableone = False
    try:
        from tableone import TableOne  # type: ignore
        has_tableone = True
    except Exception:
        has_tableone = False

    if has_tableone:
        try:
            tab1 = TableOne(
                both,
                columns=vars_present,
                categorical=categorical_cols,
                groupby="Cohort",
                pval=True,
                overall=False,
                missing=True,
                label_suffix=True,
            )
            table_df = getattr(tab1, "tableone", None)
            if table_df is not None and output_excel_path:
                _ensure_dir(os.path.dirname(output_excel_path))
                table_df.to_excel(output_excel_path)
            return
        except Exception as e:
            warnings.warn(f"[TableOne] Failed to generate cohort TableOne: {e}")

    # Fallback: simple summary
    try:
        rows: List[Dict[str, Any]] = []
        for var in vars_present:
            s_ref = ref_df[var] if var in ref_df.columns else pd.Series(dtype=float)
            s_ext = ext_df[var] if var in ext_df.columns else pd.Series(dtype=float)
            is_cat = var in categorical_cols
            if is_cat:
                levels = list(
                    pd.unique(pd.concat([
                        s_ref.astype("object"), s_ext.astype("object")
                    ], axis=0, ignore_index=True))
                )
                for level in levels:
                    row = {"variable": f"{var}, n (%)", "level": str(level)}
                    for label, s in [("SCMR", s_ref), ("External", s_ext)]:
                        non_missing = s.notna().sum()
                        if non_missing == 0:
                            row[label] = ""
                        else:
                            count = int((s == level).sum())
                            pct = count / max(1, non_missing) * 100.0
                            row[label] = f"{count} ({pct:.1f}%)"
                    rows.append(row)
            else:
                row = {"variable": var, "level": ""}
                row["SCMR_mean"] = float(pd.to_numeric(s_ref, errors="coerce").mean()) if len(s_ref) > 0 else np.nan
                row["SCMR_sd"] = float(pd.to_numeric(s_ref, errors="coerce").std()) if len(s_ref) > 0 else np.nan
                row["External_mean"] = float(pd.to_numeric(s_ext, errors="coerce").mean()) if len(s_ext) > 0 else np.nan
                row["External_sd"] = float(pd.to_numeric(s_ext, errors="coerce").std()) if len(s_ext) > 0 else np.nan
                rows.append(row)
        out_df = pd.DataFrame(rows)
        if output_excel_path:
            _ensure_dir(os.path.dirname(output_excel_path))
            out_df.to_excel(output_excel_path, index=False)
    except Exception as e:
        warnings.warn(f"[TableOne Fallback] Failed to build summary: {e}")


def _prepare_external_for_features(ext_raw: pd.DataFrame, features: List[str], labels: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze and preprocess external data for the requested features/labels.

    Returns a tuple of (pre_impute_df, clean_imputed_df) both aligned to requested.
    pre_impute_df is after alias resolution and type coercion but before MissForest.
    """
    # First, ensure PE_Time if possible
    ext_raw2 = _derive_endpoints_if_needed(ext_raw)

    # Use SCMR diagnostics to analyze mapping and missingness
    try:
        analysis = scmr.analyze_column_resolution(ext_raw2, features, labels)
    except Exception:
        analysis = {
            "resolved_columns": [c for c in ext_raw2.columns if c in features + labels],
            "rename_map": {},
            "missing_requested": [c for c in features + labels if c not in ext_raw2.columns],
            "present_features": [c for c in features if c in ext_raw2.columns],
            "present_labels": [c for c in labels if c in ext_raw2.columns],
        }

    # Build a dataframe with just resolved columns, renamed to canonical names
    cols = analysis.get("resolved_columns", [])
    rename_map = analysis.get("rename_map", {})
    pre_df = ext_raw2[cols].copy() if cols else pd.DataFrame()
    if not pre_df.empty and rename_map:
        pre_df = pre_df.rename(columns=rename_map)

    # Preprocess + impute using SCMR conversion_and_imputation
    try:
        clean_df = scmr.conversion_and_imputation(ext_raw2, features, labels)
    except Exception as e:
        warnings.warn(f"[Impute] SCMR.conversion_and_imputation failed, will try minimal pipeline: {e}")
        clean_df = pre_df.copy()

    return pre_df, clean_df


def _evaluate_sex_agnostic(
    model_obj: Any,
    features: List[str],
    threshold: Optional[float],
    ext_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # Risk at SCMR default horizon
    risk = scmr.predict_absolute_risk_at_horizon(model_obj, ext_df, features, scmr.HORIZON_DAYS)
    out = ext_df.copy()
    out["pred_prob"] = risk
    thr = float(np.nanmedian(risk)) if threshold is None or not np.isfinite(threshold) else float(threshold)
    out["pred_label"] = (out["pred_prob"] >= thr).astype(int)

    merged = (
        out[[c for c in ["MRN", "Female", "pred_label", "pred_prob", "PE_Time", "VT/VF/SCD"] if c in out.columns]]
        .dropna(subset=["PE_Time", "VT/VF/SCD"], how="any")
        .drop_duplicates(subset=["MRN"]) if "MRN" in out.columns else out.dropna(subset=["PE_Time", "VT/VF/SCD"])  # type: ignore
        .rename(columns={"VT/VF/SCD": "PE"})
    )

    mask_m = merged["Female"].values == 0 if "Female" in merged.columns else np.array([False] * len(merged))
    mask_f = merged["Female"].values == 1 if "Female" in merged.columns else np.array([False] * len(merged))

    c_all = _safe_cindex(merged["PE_Time"].values, merged["PE"].values, merged["pred_prob"].values)
    c_m = _safe_cindex(merged.loc[mask_m, "PE_Time"].values, merged.loc[mask_m, "PE"].values, merged.loc[mask_m, "pred_prob"].values) if mask_m.any() else float("nan")
    c_f = _safe_cindex(merged.loc[mask_f, "PE_Time"].values, merged.loc[mask_f, "PE"].values, merged.loc[mask_f, "pred_prob"].values) if mask_f.any() else float("nan")
    pvals = _compute_logrank_p_by_gender(merged)
    hrs = _compute_hr_by_gender(merged)

    metrics = {
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
        "threshold_used": thr,
    }
    return merged, metrics


def _evaluate_sex_specific(
    male_model: Optional[Any],
    female_model: Optional[Any],
    shared_features: List[str],
    thresholds: Dict[str, Any],
    ext_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = ext_df.copy()
    out["pred_prob"] = np.nan
    out["pred_label"] = np.nan

    if male_model is not None and not out[out.get("Female", pd.Series(dtype=int)) == 0].empty:
        te_m = out[out["Female"] == 0]
        r_m = scmr.predict_absolute_risk_at_horizon(male_model, te_m, shared_features, scmr.HORIZON_DAYS)
        out.loc[out["Female"] == 0, "pred_prob"] = r_m
        thr_m = float(thresholds.get("male", np.nan))
        if not np.isfinite(thr_m):
            thr_m = float(np.nanmedian(r_m))
        out.loc[out["Female"] == 0, "pred_label"] = (r_m >= thr_m).astype(int)

    if female_model is not None and not out[out.get("Female", pd.Series(dtype=int)) == 1].empty:
        te_f = out[out["Female"] == 1]
        r_f = scmr.predict_absolute_risk_at_horizon(female_model, te_f, shared_features, scmr.HORIZON_DAYS)
        out.loc[out["Female"] == 1, "pred_prob"] = r_f
        thr_f = float(thresholds.get("female", np.nan))
        if not np.isfinite(thr_f):
            thr_f = float(np.nanmedian(r_f))
        out.loc[out["Female"] == 1, "pred_label"] = (r_f >= thr_f).astype(int)

    merged = (
        out[[c for c in ["MRN", "Female", "pred_label", "pred_prob", "PE_Time", "VT/VF/SCD"] if c in out.columns]]
        .dropna(subset=["PE_Time", "VT/VF/SCD"], how="any")
        .drop_duplicates(subset=["MRN"]) if "MRN" in out.columns else out.dropna(subset=["PE_Time", "VT/VF/SCD"])  # type: ignore
        .rename(columns={"VT/VF/SCD": "PE"})
    )

    mask_m = merged["Female"].values == 0 if "Female" in merged.columns else np.array([False] * len(merged))
    mask_f = merged["Female"].values == 1 if "Female" in merged.columns else np.array([False] * len(merged))

    c_all = _safe_cindex(merged["PE_Time"].values, merged["PE"].values, merged["pred_prob"].values)
    c_m = _safe_cindex(merged.loc[mask_m, "PE_Time"].values, merged.loc[mask_m, "PE"].values, merged.loc[mask_m, "pred_prob"].values) if mask_m.any() else float("nan")
    c_f = _safe_cindex(merged.loc[mask_f, "PE_Time"].values, merged.loc[mask_f, "PE"].values, merged.loc[mask_f, "pred_prob"].values) if mask_f.any() else float("nan")
    pvals = _compute_logrank_p_by_gender(merged)
    hrs = _compute_hr_by_gender(merged)

    metrics = {
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
    return merged, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained SCMR models on an external dataset (no retraining)")
    parser.add_argument("--external_path", required=True, help="Path to external dataset (CSV/XLSX)")
    parser.add_argument("--external_sheet", default=None, help="Sheet name for Excel (optional)")
    parser.add_argument("--results_dir", default=os.path.join(os.getcwd(), "results"), help="Results directory (default: ./results)")
    parser.add_argument("--models_dir", default=None, help="Models root directory (default: <results_dir>/heldout/models)")
    parser.add_argument("--limit_feature_sets", nargs="*", default=None, help="Optional: limit to specific feature sets (e.g., Proposed Advanced)")
    args = parser.parse_args()

    results_dir = args.results_dir
    models_root = args.models_dir or os.path.join(results_dir, "heldout", "models")
    out_root = os.path.join(results_dir, "external_eval")
    out_preds = os.path.join(out_root, "preds")
    out_figs = os.path.join(out_root, "figs")
    out_metrics = os.path.join(out_root, "metrics")
    out_meta = os.path.join(out_root, "meta")
    out_tables = os.path.join(out_root, "tableone")
    for d in [out_root, out_preds, out_figs, out_metrics, out_meta, out_tables]:
        _ensure_dir(d)

    _progress("Loading SCMR reference dataset (preprocessed)...")
    ref_df = scmr.load_dataframes()

    _progress("Reading external dataset...")
    ext_raw = _read_external_dataframe(args.external_path, args.external_sheet)

    _progress("Discovering trained models and metadata...")
    model_specs = _discover_models(models_root)
    if args.limit_feature_sets:
        keep = set(args.limit_feature_sets)
        model_specs = [m for m in model_specs if m.get("feature_set") in keep]
    if not model_specs:
        raise RuntimeError(f"No trained model artifacts found under: {models_root}")

    # Build a union of all requested features across discovered models
    requested_features: List[str] = []
    for spec in model_specs:
        feats = list(spec.get("features") or [])
        if not feats:
            # fallback to default feature sets mapping if meta missing
            try:
                feats = list(scmr.FEATURE_SETS.get(spec["feature_set"], []))
            except Exception:
                feats = []
        requested_features.extend(feats)
    # Deduplicate while preserving order
    requested_features = list(dict.fromkeys(requested_features))

    # Always include Female for cohort stratification if present in either dataset
    if "Female" not in requested_features:
        requested_features.insert(0, "Female")

    # Labels needed for evaluation
    labels = ["VT/VF/SCD", "PE_Time", "ICD", "MRN"]

    _progress("Preprocessing external dataset to align features/labels (before and after imputation)...")
    pre_ext_df, ext_df = _prepare_external_for_features(ext_raw, requested_features, labels)

    # Report feature presence and missingness before imputation
    missing_report_rows: List[Dict[str, Any]] = []
    present_cols = set(pre_ext_df.columns.tolist()) if not pre_ext_df.empty else set()
    for f in requested_features:
        status = {
            "feature": f,
            "present": f in present_cols,
            "missing_rate": float(pd.to_numeric(pre_ext_df[f], errors="coerce").isna().mean()) if f in present_cols else 1.0,
        }
        missing_report_rows.append(status)
    missing_report = pd.DataFrame(missing_report_rows)
    missing_csv = os.path.join(out_meta, "external_missingness_before_impute.csv")
    try:
        missing_report.to_csv(missing_csv, index=False)
    except Exception:
        pass
    # Also dump missing requested features for quick inspection
    try:
        with open(os.path.join(out_meta, "missing_features.json"), "w", encoding="utf-8") as f:
            json.dump({"missing_features": [r["feature"] for r in missing_report_rows if not r["present"]]}, f, indent=2)
    except Exception:
        pass

    # Drift/SMD report between reference and external for requested features
    try:
        drift_df = scmr.compute_smd_drift_report(ref_df, ext_df, requested_features)
        drift_df.to_csv(os.path.join(out_meta, "smd_drift_report.csv"), index=False)
    except Exception as e:
        warnings.warn(f"[Drift] Failed to compute SMD drift report: {e}")

    # Evaluate each discovered model on the imputed external dataset
    summary_rows: List[Dict[str, Any]] = []
    for spec in model_specs:
        featset = spec.get("feature_set")
        mode = spec.get("mode")
        feats = list(spec.get("features") or [])
        if not feats:
            try:
                feats = list(scmr.FEATURE_SETS.get(featset, []))
            except Exception:
                feats = []
        # Ensure features exist in ext_df; warn if not
        missing_in_ext = [c for c in feats if c not in ext_df.columns]
        if missing_in_ext:
            warnings.warn(f"[{featset}][{mode}] Missing features in external dataset: {missing_in_ext}")

        # Load model objects
        model_obj = None
        male_model = None
        female_model = None
        try:
            import pickle
            if mode == "sex_agnostic" and os.path.exists(spec["paths"].get("model", "")):
                with open(spec["paths"]["model"], "rb") as f:
                    model_obj = pickle.load(f)
            if mode == "sex_specific":
                if os.path.exists(spec["paths"].get("male", "")):
                    with open(spec["paths"]["male"], "rb") as f:
                        male_model = pickle.load(f)
                if os.path.exists(spec["paths"].get("female", "")):
                    with open(spec["paths"]["female"], "rb") as f:
                        female_model = pickle.load(f)
        except Exception as e:
            warnings.warn(f"[{featset}][{mode}] Failed to load model pickle(s): {e}")

        # Produce per-model TableOne comparing SCMR vs External for actually used features
        # Do this BEFORE evaluation to diagnose potential drift impacting performance
        try:
            _generate_tableone_by_cohort(
                ref_df,
                ext_df,
                variables=[c for c in feats if c in ref_df.columns or c in ext_df.columns],
                output_excel_path=os.path.join(out_tables, f"tableone_{featset}_{mode}_cohort.xlsx"),
            )
        except Exception as e:
            warnings.warn(f"[TableOne] Failed for {featset} {mode}: {e}")

        # Evaluate
        pred_csv_name = f"{featset}_{mode}_external_preds.csv"
        fig_path = os.path.join(out_figs, f"{featset}_{mode}_km_by_gender.png")
        if mode == "sex_agnostic" and model_obj is not None:
            m_thr = spec.get("threshold")
            merged, metrics = _evaluate_sex_agnostic(model_obj, feats, m_thr, ext_df)
            _plot_km_by_gender(merged, fig_path)
            # 3-year calibration curve
            calib_path = os.path.join(out_figs, f"{featset}_{mode}_calibration_3yr.png")
            calib_df = _plot_calibration_curve_3yr(merged, calib_path, bins=10)
            if calib_df is not None:
                try:
                    calib_df.to_csv(
                        os.path.join(out_metrics, f"{featset}_{mode}_calibration_3yr.csv"),
                        index=False,
                    )
                except Exception:
                    pass
            try:
                merged.to_csv(os.path.join(out_preds, pred_csv_name), index=False)
            except Exception:
                pass
            summary_rows.append({
                "feature_set": featset,
                "mode": mode,
                **metrics,
            })
        elif mode == "sex_specific" and (male_model is not None or female_model is not None):
            thresholds = dict(spec.get("thresholds", {}))
            merged, metrics = _evaluate_sex_specific(male_model, female_model, feats, thresholds, ext_df)
            _plot_km_by_gender(merged, fig_path)
            # 3-year calibration curve
            calib_path = os.path.join(out_figs, f"{featset}_{mode}_calibration_3yr.png")
            calib_df = _plot_calibration_curve_3yr(merged, calib_path, bins=10)
            if calib_df is not None:
                try:
                    calib_df.to_csv(
                        os.path.join(out_metrics, f"{featset}_{mode}_calibration_3yr.csv"),
                        index=False,
                    )
                except Exception:
                    pass
            try:
                merged.to_csv(os.path.join(out_preds, pred_csv_name), index=False)
            except Exception:
                pass
            summary_rows.append({
                "feature_set": featset,
                "mode": mode,
                **metrics,
            })
        else:
            warnings.warn(f"[{featset}][{mode}] Skipped: model objects not available")
            continue


    # Save summary metrics
    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        try:
            summary.to_csv(os.path.join(out_metrics, "external_eval_summary.csv"), index=False)
        except Exception:
            pass
        try:
            summary.to_excel(os.path.join(out_metrics, "external_eval_summary.xlsx"), index=False)
        except Exception:
            pass

    _progress("Done. Outputs saved under results/external_eval/")


if __name__ == "__main__":
    main()
