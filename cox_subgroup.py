import warnings
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

# Reuse utilities from cox.py to keep behavior consistent
from cox import (
    FEATURE_SETS,
    fit_cox_model,
    predict_risk,
    plot_km_two_subplots_by_gender,
    plot_cox_coefficients,
    load_dataframes,
)


SubgroupFeatureMap = Dict[Union[str, int, float], Union[str, List[str]]]


def _resolve_feature_list(
    subgroup_value: Union[str, int, float],
    subgroup_to_features: SubgroupFeatureMap,
    default_features: Union[str, List[str], None] = None,
) -> List[str]:
    """
    Resolve the concrete feature list for a subgroup.

    Accepts either:
      - a list of feature names, or
      - a string key that refers to FEATURE_SETS (e.g., "Guideline", "Benchmark", "Proposed").
    """
    raw = subgroup_to_features.get(subgroup_value, default_features)
    if raw is None:
        raise ValueError(f"No feature mapping found for subgroup={subgroup_value} and default_features is None.")

    if isinstance(raw, str):
        if raw not in FEATURE_SETS:
            raise ValueError(
                f"Feature set name '{raw}' for subgroup={subgroup_value} not found in FEATURE_SETS."
            )
        feats = FEATURE_SETS[raw]
    else:
        feats = list(raw)

    # For single-sex models, exclude constant sex indicator to avoid singularities
    feats = [f for f in feats if f != "Female"]
    return feats


def sex_specific_inference_by_subgroup(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    subgroup_col: str,
    subgroup_to_features: SubgroupFeatureMap,
    default_features: Union[str, List[str], None] = None,
    time_col: str = "PE_Time",
    event_col: str = "VT/VF/SCD",
    plot_coefficients: bool = True,
    gray_features: List[str] = None,
    red_features: List[str] = None,
) -> pd.DataFrame:
    """
    Train sex-specific Cox models, further split by subgroup, each with its own feature subset.

    For each combination of sex in {0,1} and subgroup value in train_df[subgroup_col],
    trains a Cox model using features resolved via subgroup_to_features and predicts on
    the corresponding test subset.

    Returns a dataframe containing MRN, Female, pred_label, time_col, event_col for KM plotting.
    """
    if subgroup_col not in train_df.columns or subgroup_col not in test_df.columns:
        raise ValueError(f"subgroup_col '{subgroup_col}' must exist in both train and test dataframes.")

    # Prepare containers
    models: Dict[Tuple[int, Union[str, int, float]], object] = {}
    thresholds: Dict[Tuple[int, Union[str, int, float]], float] = {}

    # Fit models per sex x subgroup
    for sex_val in (0, 1):
        tr_sex = train_df[(train_df["Female"] == sex_val)].copy()
        if tr_sex.empty:
            continue
        # Iterate subgroups present in training for this sex
        for subgroup_value in sorted(tr_sex[subgroup_col].dropna().unique().tolist()):
            tr_subset = tr_sex[tr_sex[subgroup_col] == subgroup_value]
            tr_subset = tr_subset.dropna(subset=[time_col, event_col])
            if tr_subset.empty:
                continue
            try:
                used_features = _resolve_feature_list(subgroup_value, subgroup_to_features, default_features)
            except Exception as e:
                warnings.warn(str(e))
                continue

            try:
                cph = fit_cox_model(tr_subset, used_features, time_col, event_col)
            except Exception as e:
                warnings.warn(f"[Cox] Fit skipped for sex={sex_val}, subgroup={subgroup_value}: {e}")
                continue

            # Compute median risk on training subset as threshold
            tr_risk = predict_risk(cph, tr_subset, used_features)
            thr = float(np.nanmedian(tr_risk))
            models[(sex_val, subgroup_value)] = cph
            thresholds[(sex_val, subgroup_value)] = thr

            if plot_coefficients:
                title = f"Cox Coefficients (log HR) - sex={'F' if sex_val==1 else 'M'}, {subgroup_col}={subgroup_value}"
                plot_cox_coefficients(cph, title, gray_features, red_features)

    # Predict on test per sex x subgroup
    out = test_df.copy()
    for sex_val in (0, 1):
        te_sex = out[out["Female"] == sex_val]
        if te_sex.empty:
            continue
        for subgroup_value in sorted(te_sex[subgroup_col].dropna().unique().tolist()):
            key = (sex_val, subgroup_value)
            if key not in models:
                # Try to fall back to default features if provided and model missing
                try:
                    used_features = _resolve_feature_list(subgroup_value, subgroup_to_features, default_features)
                except Exception:
                    continue
                # Need training data for this sex to at least build a model
                tr_fallback = train_df[(train_df["Female"] == sex_val)].dropna(subset=[time_col, event_col])
                if tr_fallback.empty:
                    continue
                try:
                    cph = fit_cox_model(tr_fallback, used_features, time_col, event_col)
                    risk_tr = predict_risk(cph, tr_fallback, used_features)
                    thr = float(np.nanmedian(risk_tr))
                    models[key] = cph
                    thresholds[key] = thr
                except Exception:
                    continue

            # Predict
            try:
                used_features = list(models[key].params_.index)
                te_mask = (out["Female"] == sex_val) & (out[subgroup_col] == subgroup_value)
                te_subset = out.loc[te_mask]
                if te_subset.empty:
                    continue
                risk = predict_risk(models[key], te_subset, used_features)
                out.loc[te_mask, "pred_prob"] = risk
                out.loc[te_mask, "pred_label"] = (risk >= thresholds[key]).astype(int)
            except Exception as e:
                warnings.warn(f"[Cox] Predict skipped for sex={sex_val}, subgroup={subgroup_value}: {e}")
                continue

    merged_df = (
        out[["MRN", "Female", "pred_label", time_col, event_col]]
        .dropna(subset=[time_col, event_col])
        .drop_duplicates(subset=["MRN"])
        .rename(columns={event_col: "PE", time_col: "PE_Time"})
    )

    plot_km_two_subplots_by_gender(merged_df)
    return merged_df


def sex_specific_full_inference_by_subgroup(
    df: pd.DataFrame,
    subgroup_col: str,
    subgroup_to_features: SubgroupFeatureMap,
    default_features: Union[str, List[str], None] = None,
    time_col: str = "PE_Time",
    event_col: str = "VT/VF/SCD",
    plot_coefficients: bool = True,
    gray_features: List[str] = None,
    red_features: List[str] = None,
) -> pd.DataFrame:
    """
    Variant that trains and evaluates on the full dataset per sex x subgroup.
    """
    return sex_specific_inference_by_subgroup(
        train_df=df,
        test_df=df,
        subgroup_col=subgroup_col,
        subgroup_to_features=subgroup_to_features,
        default_features=default_features,
        time_col=time_col,
        event_col=event_col,
        plot_coefficients=plot_coefficients,
        gray_features=gray_features,
        red_features=red_features,
    )


if __name__ == "__main__":
    # Example usage with ICD as subgroup
    try:
        df = load_dataframes()
    except Exception as e:
        raise SystemExit(f"Failed to load dataframes from cox.py: {e}")

    # Map subgroup values to feature sets or explicit feature lists
    # Here we demonstrate using FEATURE_SETS names for convenience
    subgroup_col = "ICD"
    subgroup_to_features: SubgroupFeatureMap = {
        1: "Guideline",   # Patients with ICD: use guideline features
        0: "Proposed",    # Patients without ICD: use proposed features
    }

    print("Running sex-specific by subgroup (full-data) with ICD mapping...")
    _ = sex_specific_full_inference_by_subgroup(
        df=df,
        subgroup_col=subgroup_col,
        subgroup_to_features=subgroup_to_features,
        default_features="Benchmark",  # fallback
        time_col="PE_Time",
        event_col="VT/VF/SCD",
        plot_coefficients=True,
        gray_features=FEATURE_SETS.get("Guideline", []) + ["NYHA>2"],
        red_features=FEATURE_SETS.get("Proposed", []),
    )