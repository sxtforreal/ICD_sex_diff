import os
import json
import pickle
import warnings
from typing import Dict, List, Any, Tuple

import pandas as pd
import numpy as np

# Reuse the existing pipeline utilities without modifying repo code
import SCMR


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ensure_models_exist(
    df: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    results_dir: str,
    test_size: float = 0.3,
    random_seed: int = 42,
) -> pd.DataFrame:
    models_dir = os.path.join(results_dir, "heldout", "models")
    # Heuristic: if any feature set directory exists with expected files, assume trained
    trained = False
    if os.path.isdir(models_dir):
        for feat in os.listdir(models_dir):
            agn = os.path.join(models_dir, feat, "sex_agnostic", "model.pkl")
            spe_m = os.path.join(models_dir, feat, "sex_specific", "male_model.pkl")
            spe_f = os.path.join(models_dir, feat, "sex_specific", "female_model.pkl")
            if os.path.exists(agn) or (os.path.exists(spe_m) or os.path.exists(spe_f)):
                trained = True
                break
    if trained:
        # Try to read previously saved summary
        summary_path_xlsx = os.path.join(results_dir, "heldout", "summary_metrics.xlsx")
        if os.path.exists(summary_path_xlsx):
            try:
                return pd.read_excel(summary_path_xlsx)
            except Exception:
                pass
        # Fallback: return empty df; we'll recompute after external eval
        return pd.DataFrame()

    # Train and evaluate on held-out to persist models and baseline test summary
    summary_df = SCMR.run_heldout_training_and_evaluation(
        df=df,
        feature_sets=feature_sets,
        time_col="PE_Time",
        event_col="VT/VF/SCD",
        results_dir=results_dir,
        test_size=test_size,
        random_state=random_seed,
    )
    return summary_df


def _read_excel_robust(path: str) -> pd.DataFrame:
    # Read all sheets if present and attempt to combine similar to NICM loader
    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except Exception as e:
        raise FileNotFoundError(f"Failed to read Excel at {path}: {e}")

    # Normalize column labels to reduce whitespace/case issues
    def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = [SCMR._normalize_column_name(c) for c in out.columns]
        return out

    if isinstance(sheets, dict) and len(sheets) >= 2 and {"ICD", "No_ICD"}.issubset({k for k in sheets}):
        icd = _norm_cols(sheets["ICD"]).copy()
        noicd = _norm_cols(sheets["No_ICD"]).copy()
        all_cols = sorted(set(icd.columns).union(set(noicd.columns)))
        icd_aligned = icd.reindex(columns=all_cols).copy()
        noicd_aligned = noicd.reindex(columns=all_cols).copy()
        icd_aligned["ICD"] = 1
        noicd_aligned["ICD"] = 0
        combined = pd.concat([icd_aligned, noicd_aligned], ignore_index=True)
        return combined

    # Otherwise use the first sheet
    if isinstance(sheets, dict):
        first_name = next(iter(sheets.keys()))
        return _norm_cols(sheets[first_name]).copy()
    else:
        # Unlikely branch: pandas returned a single DataFrame
        return _norm_cols(sheets).copy()


def load_harvard_dataframe(harvard_xlsx: str, restrict_to_feature_set: str | None = None) -> pd.DataFrame:
    if not os.path.exists(harvard_xlsx):
        raise FileNotFoundError(
            f"Harvard dataset not found at '{harvard_xlsx}'. Set --harvard_xlsx or HARVARD_XLSX."
        )

    raw = _read_excel_robust(harvard_xlsx)

    # Align with SCMR preprocessing: fill Cockcroft-Gault CrCl if missing
    try:
        def _fill_cg(row: pd.Series) -> Any:
            try:
                val = row["Cockcroft-Gault Creatinine Clearance (mL/min)"]
            except Exception:
                val = np.nan
            if pd.notna(val):
                return val
            try:
                return SCMR.CG_equation(
                    row["Age at CMR"],
                    row["Weight (Kg)"],
                    row["Female"],
                    row["Serum creatinine (within 3 months of MRI)"],
                )
            except Exception:
                return np.nan

        raw["Cockcroft-Gault Creatinine Clearance (mL/min)"] = raw.apply(_fill_cg, axis=1)
    except Exception:
        # Best-effort; proceed if inputs not available
        pass

    # Attempt PE_Time derivation if missing
    if "PE_Time" not in raw.columns:
        try:
            raw["MRI Date"] = pd.to_datetime(raw.get("MRI Date", pd.NaT))
            raw["Date VT/VF/SCD"] = pd.to_datetime(raw.get("Date VT/VF/SCD", pd.NaT))
            raw["End follow-up date"] = pd.to_datetime(raw.get("End follow-up date", pd.NaT))
            raw["PE_Time"] = raw.apply(
                lambda row: (
                    (row["Date VT/VF/SCD"] - row["MRI Date"]).days
                    if pd.notna(row.get("VT/VF/SCD", np.nan))
                    and int(row.get("VT/VF/SCD", 0)) == 1
                    and pd.notna(row["Date VT/VF/SCD"]) \
                    and pd.notna(row["MRI Date"]) else (
                        (row["End follow-up date"] - row["MRI Date"]).days
                        if pd.notna(row["End follow-up date"]) and pd.notna(row["MRI Date"]) else np.nan
                    )
                ),
                axis=1,
            )
        except Exception:
            pass

    # Build candidate features: strictly the requested model set if provided; otherwise union
    labels = ["MRN", "VT/VF/SCD", "ICD", "PE_Time"]
    feature_superset: List[str] = []
    if restrict_to_feature_set and restrict_to_feature_set in SCMR.FEATURE_SETS:
        feature_superset = list(SCMR.FEATURE_SETS[restrict_to_feature_set])
    else:
        for v in SCMR.FEATURE_SETS.values():
            feature_superset.extend(v)
        # Keep unique order
        feature_superset = list(dict.fromkeys(feature_superset))

    # Exclude engineered-after-imputation features from the preprocessing request
    # They will be created below once the base matrix is clean/imputed
    derived_after_imputation = {"Age by decade", "CrCl>45", "NYHA>2", "Significant LGE"}
    feature_superset_for_impute = [f for f in feature_superset if f not in derived_after_imputation]

    clean = SCMR.conversion_and_imputation(raw, feature_superset_for_impute, labels)

    # Engineer additional features as in SCMR.load_dataframes
    if "Age at CMR" in clean.columns:
        clean["Age by decade"] = (clean["Age at CMR"] // 10).values
    if "Cockcroft-Gault Creatinine Clearance (mL/min)" in clean.columns:
        clean["CrCl>45"] = (
            (clean["Cockcroft-Gault Creatinine Clearance (mL/min)"] > 45)
            .fillna(False)
            .astype(int)
        )
    if "NYHA Class" in clean.columns:
        clean["NYHA>2"] = (clean["NYHA Class"] > 2).fillna(False).astype(int)
    if "LGE Burden 5SD" in clean.columns:
        clean["Significant LGE"] = (
            (clean["LGE Burden 5SD"] > 2).fillna(False).astype(int)
        )

    return clean


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def evaluate_with_saved_models(
    df: pd.DataFrame,
    results_dir: str,
    dataset_name: str,
    model_name: str | None = None,
) -> pd.DataFrame:
    models_dir = os.path.join(results_dir, "heldout", "models")
    preds_dir = os.path.join(results_dir, "heldout", "preds")
    _ensure_dir(preds_dir)

    rows: List[Dict[str, Any]] = []

    if not os.path.isdir(models_dir):
        warnings.warn(f"Models directory not found: {models_dir}")
        return pd.DataFrame()

    # Loop over feature sets (or restrict to one model_name)
    if model_name:
        # Match folder name case-insensitively for robustness
        names = os.listdir(models_dir)
        chosen = [n for n in names if n.lower() == str(model_name).lower()]
        featset_names = chosen if chosen else [model_name]
    else:
        featset_names = sorted(os.listdir(models_dir))
    for featset_name in featset_names:
        if featset_name is None:
            continue
        feat_dir = os.path.join(models_dir, featset_name)
        if not os.path.isdir(feat_dir):
            continue

        # Sex-agnostic
        agn_dir = os.path.join(feat_dir, "sex_agnostic")
        model_pkl = os.path.join(agn_dir, "model.pkl")
        meta_json = os.path.join(agn_dir, "meta.json")
        if os.path.exists(model_pkl) and (os.path.exists(meta_json) or True):
            try:
                with open(model_pkl, "rb") as f:
                    cph = pickle.load(f)
                used_feats: List[str] = []
                thr: float = float("nan")
                if os.path.exists(meta_json):
                    try:
                        meta = _load_json(meta_json)
                        used_feats = list(meta.get("features", []))
                        thr = float(meta.get("threshold", np.nan))
                    except Exception:
                        pass
                # Fallback: infer features from model if meta missing or empty
                if not used_feats:
                    try:
                        used_feats = [str(x) for x in getattr(cph, "params_", pd.Series()).index.tolist()]
                    except Exception:
                        used_feats = []

                te_eval = df.copy()
                # Ensure an identifier column exists even if MRN is missing
                id_col = "MRN" if "MRN" in te_eval.columns else "_row_id"
                if id_col == "_row_id" and "_row_id" not in te_eval.columns:
                    te_eval["_row_id"] = np.arange(len(te_eval))
                risk = SCMR.predict_absolute_risk_at_horizon(
                    cph, te_eval, used_feats, SCMR.HORIZON_DAYS
                )
                te_eval["pred_prob"] = risk
                te_eval["pred_label"] = (risk >= thr).astype(int)
                cols = [id_col, "pred_label", "pred_prob", "PE_Time", "VT/VF/SCD"]
                if "Female" in te_eval.columns:
                    cols.insert(1, "Female")
                merged = (
                    te_eval[[c for c in cols if c in te_eval.columns]]
                    .dropna(subset=["PE_Time", "VT/VF/SCD"]).drop_duplicates(subset=[id_col])
                    .rename(columns={"VT/VF/SCD": "PE"})
                )
                merged.to_csv(
                    os.path.join(preds_dir, f"{featset_name}_sex_agnostic_{dataset_name}_preds.csv"),
                    index=False,
                )
                c_all = SCMR._safe_cindex(
                    merged["PE_Time"].values, merged["PE"].values, merged["pred_prob"].values
                )
                c_m = float("nan")
                c_f = float("nan")
                if "Female" in merged.columns:
                    mask_m = merged["Female"].values == 0
                    mask_f = merged["Female"].values == 1
                    if mask_m.any():
                        c_m = SCMR._safe_cindex(
                            merged.loc[mask_m, "PE_Time"].values,
                            merged.loc[mask_m, "PE"].values,
                            merged.loc[mask_m, "pred_prob"].values,
                        )
                    if mask_f.any():
                        c_f = SCMR._safe_cindex(
                            merged.loc[mask_f, "PE_Time"].values,
                            merged.loc[mask_f, "PE"].values,
                            merged.loc[mask_f, "pred_prob"].values,
                        )
                pvals = SCMR._compute_logrank_pvalues_by_gender(merged)
                hrs = SCMR._compute_hr_by_gender(merged)
                rows.append(
                    {
                        "dataset": dataset_name,
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
                warnings.warn(f"[{featset_name}] sex-agnostic external eval failed: {e}")

        # Sex-specific
        spe_dir = os.path.join(feat_dir, "sex_specific")
        meta_json = os.path.join(spe_dir, "meta.json")
        model_m = os.path.join(spe_dir, "male_model.pkl")
        model_f = os.path.join(spe_dir, "female_model.pkl")
        if (os.path.exists(model_m) or os.path.exists(model_f)) and (os.path.exists(meta_json) or True):
            try:
                used_feats: List[str] = []
                thresholds: Dict[str, float] = {}
                if os.path.exists(meta_json):
                    try:
                        meta = _load_json(meta_json)
                        used_feats = list(meta.get("features", []))
                        thresholds = {k: float(v) for k, v in meta.get("thresholds", {}).items()}
                    except Exception:
                        pass

                # Require Female column for sex-specific evaluation; otherwise skip
                if "Female" not in df.columns:
                    raise RuntimeError("Female column not found; skipping sex-specific evaluation for this feature set.")

                te_out = df.copy()
                if "pred_prob" not in te_out.columns:
                    te_out["pred_prob"] = np.nan
                if "pred_label" not in te_out.columns:
                    te_out["pred_label"] = np.nan

                if os.path.exists(model_m):
                    with open(model_m, "rb") as f:
                        cph_m = pickle.load(f)
                    if not used_feats:
                        try:
                            used_feats = [str(x) for x in getattr(cph_m, "params_", pd.Series()).index.tolist()]
                        except Exception:
                            used_feats = []
                    te_m = te_out[te_out["Female"] == 0]
                    if not te_m.empty:
                        r_m = SCMR.predict_absolute_risk_at_horizon(cph_m, te_m, used_feats, SCMR.HORIZON_DAYS)
                        te_out.loc[te_out["Female"] == 0, "pred_prob"] = r_m
                        te_out.loc[te_out["Female"] == 0, "pred_label"] = (r_m >= thresholds.get("male", np.nan)).astype(int)

                if os.path.exists(model_f):
                    with open(model_f, "rb") as f:
                        cph_f = pickle.load(f)
                    if not used_feats:
                        try:
                            used_feats = [str(x) for x in getattr(cph_f, "params_", pd.Series()).index.tolist()]
                        except Exception:
                            used_feats = []
                    te_f = te_out[te_out["Female"] == 1]
                    if not te_f.empty:
                        r_f = SCMR.predict_absolute_risk_at_horizon(cph_f, te_f, used_feats, SCMR.HORIZON_DAYS)
                        te_out.loc[te_out["Female"] == 1, "pred_prob"] = r_f
                        te_out.loc[te_out["Female"] == 1, "pred_label"] = (r_f >= thresholds.get("female", np.nan)).astype(int)

                # Build merged output with robust ID handling
                id_col = "MRN" if "MRN" in te_out.columns else "_row_id"
                if id_col == "_row_id" and "_row_id" not in te_out.columns:
                    te_out["_row_id"] = np.arange(len(te_out))
                merged = (
                    te_out[[c for c in [id_col, "Female", "pred_label", "pred_prob", "PE_Time", "VT/VF/SCD"] if c in te_out.columns]]
                    .dropna(subset=["PE_Time", "VT/VF/SCD"]).drop_duplicates(subset=[id_col])
                    .rename(columns={"VT/VF/SCD": "PE"})
                )
                merged.to_csv(
                    os.path.join(preds_dir, f"{featset_name}_sex_specific_{dataset_name}_preds.csv"),
                    index=False,
                )
                mask_m = merged["Female"].values == 0
                mask_f = merged["Female"].values == 1
                c_all = SCMR._safe_cindex(
                    merged["PE_Time"].values, merged["PE"].values, merged["pred_prob"].values
                )
                c_m = SCMR._safe_cindex(
                    merged.loc[mask_m, "PE_Time"].values,
                    merged.loc[mask_m, "PE"].values,
                    merged.loc[mask_m, "pred_prob"].values,
                )
                c_f = SCMR._safe_cindex(
                    merged.loc[mask_f, "PE_Time"].values,
                    merged.loc[mask_f, "PE"].values,
                    merged.loc[mask_f, "pred_prob"].values,
                )
                pvals = SCMR._compute_logrank_pvalues_by_gender(merged)
                hrs = SCMR._compute_hr_by_gender(merged)
                rows.append(
                    {
                        "dataset": dataset_name,
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
                warnings.warn(f"[{featset_name}] sex-specific external eval failed: {e}")

    summary_df = pd.DataFrame(rows)
    # Save Harvard-only summary
    out_xlsx = os.path.join(results_dir, "heldout", f"summary_metrics_{dataset_name}.xlsx")
    _ensure_dir(os.path.dirname(out_xlsx))
    try:
        summary_df.to_excel(out_xlsx, index=False)
    except Exception:
        pass
    return summary_df


def combine_and_export(
    internal_summary: pd.DataFrame,
    external_summary: pd.DataFrame,
    results_dir: str,
) -> str:
    # Ensure dataset column exists in internal summary
    if not internal_summary.empty:
        if "dataset" not in internal_summary.columns:
            internal_summary = internal_summary.copy()
            internal_summary["dataset"] = "internal_test"
    combined = pd.concat([internal_summary, external_summary], ignore_index=True)
    out = os.path.join(results_dir, "heldout", "summary_metrics_combined.xlsx")
    _ensure_dir(os.path.dirname(out))
    combined.to_excel(out, index=False)
    return out


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate external Harvard dataset using saved held-out models and compare with internal test results.")
    parser.add_argument("--harvard_xlsx", default=os.environ.get("HARVARD_XLSX", os.path.join(os.getcwd(), "harvard.xlsx")), help="Path to Harvard Excel file (default: ./harvard.xlsx or HARVARD_XLSX env)")
    parser.add_argument("--results_dir", default=os.environ.get("SCMR_RESULTS_DIR", os.path.join(os.getcwd(), "results")), help="Results directory (default: ./results)")
    parser.add_argument("--retrain", action="store_true", help="Force retraining held-out models before external evaluation")
    parser.add_argument("--test_size", type=float, default=0.3, help="Held-out test size for internal split if retraining")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for held-out split if retraining")
    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", None), help="Only evaluate this model folder name under results/heldout/models")

    args = parser.parse_args()

    results_dir = args.results_dir
    _ensure_dir(results_dir)

    # 1) Load internal data and ensure models exist
    try:
        df_internal = SCMR.load_dataframes()
    except Exception as e:
        raise RuntimeError(f"Failed to load internal NICM.xlsx data: {e}")

    internal_summary = pd.DataFrame()
    if args.retrain:
        internal_summary = ensure_models_exist(df_internal, SCMR.FEATURE_SETS, results_dir, args.test_size, args.seed)
    else:
        # If summary exists, read; else ensure models exist (train once)
        summary_path_xlsx = os.path.join(results_dir, "heldout", "summary_metrics.xlsx")
        if os.path.exists(summary_path_xlsx):
            try:
                internal_summary = pd.read_excel(summary_path_xlsx)
            except Exception:
                internal_summary = pd.DataFrame()
        if internal_summary.empty:
            internal_summary = ensure_models_exist(df_internal, SCMR.FEATURE_SETS, results_dir, args.test_size, args.seed)

    # 2) Load Harvard and evaluate with saved models
    if not os.path.exists(args.harvard_xlsx):
        raise FileNotFoundError(
            f"Harvard dataset not found at '{args.harvard_xlsx}'. Place it there or pass --harvard_xlsx."
        )

    # Restrict preprocessing to the selected model's declared columns when provided
    df_harvard = load_harvard_dataframe(args.harvard_xlsx, restrict_to_feature_set=args.model_name)

    harv_summary = evaluate_with_saved_models(df_harvard, results_dir, dataset_name="harvard", model_name=args.model_name)

    # 3) Combine and export
    combined_path = combine_and_export(internal_summary, harv_summary, results_dir)

    print("== Done ==")
    print(f"- Internal test summary: {os.path.join(results_dir, 'heldout', 'summary_metrics.xlsx')}")
    print(f"- Harvard summary:       {os.path.join(results_dir, 'heldout', 'summary_metrics_harvard.xlsx')}")
    print(f"- Combined summary:      {combined_path}")


if __name__ == "__main__":
    main()
