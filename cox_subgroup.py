import os
import warnings
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from lifelines.utils import concordance_index

# Reuse core utilities from cox.py to avoid duplication
from cox import (
    fit_cox_model,
    predict_risk,
    plot_cox_coefficients,
    plot_km_two_subplots_by_gender,
    load_dataframes,
    FEATURE_SETS,
)


def _resolve_feature_list(feature_ref: Union[str, List[str]]) -> List[str]:
    """Resolve a feature reference to a concrete feature list.

    - If feature_ref is a string, look up in FEATURE_SETS
    - If already a list, return a copy
    - Otherwise return empty list
    """
    if isinstance(feature_ref, str):
        return list(FEATURE_SETS.get(feature_ref, []))
    if isinstance(feature_ref, list):
        return list(feature_ref)
    return []


def _filter_features_for_df(df: pd.DataFrame, features: List[str]) -> List[str]:
    """Keep only features that exist in df, and drop 'Female' for single-sex models."""
    present = [f for f in features if f in df.columns]
    return [f for f in present if f != "Female"]


def evaluate_split_sex_specific_subsets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    subgroup_features: Dict[str, Union[str, List[str]]],
    time_col: str,
    event_col: str,
    make_plots: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Train sex-specific Cox models using subgroup-specific feature subsets.

    subgroup_features keys: 'male' and 'female'. Values can be feature list or
    a string key into FEATURE_SETS. 'Female' column is excluded from submodels.
    """
    # Resolve features per subgroup
    male_features = _resolve_feature_list(subgroup_features.get("male", []))
    female_features = _resolve_feature_list(subgroup_features.get("female", []))

    # Prepare containers
    pred = np.zeros(len(test_df), dtype=int)
    risk_scores = np.zeros(len(test_df))

    # Male branch (Female == 0)
    tr_m = train_df[train_df["Female"] == 0]
    te_m = test_df[test_df["Female"] == 0]
    if not tr_m.empty and not te_m.empty:
        used_m = _filter_features_for_df(tr_m, male_features)
        if len(used_m) > 0:
            cph_m = fit_cox_model(tr_m, used_m, time_col, event_col)
            risk_m = predict_risk(cph_m, te_m, used_m)
            thr_m = float(np.nanmedian(risk_m))
            pred_m = (risk_m >= thr_m).astype(int)
            mask_m = test_df["Female"].values == 0
            pred[mask_m] = pred_m
            risk_scores[mask_m] = risk_m
            if make_plots:
                try:
                    plot_cox_coefficients(cph_m, "Male Cox Coefficients (log HR)")
                except Exception:
                    pass

    # Female branch (Female == 1)
    tr_f = train_df[train_df["Female"] == 1]
    te_f = test_df[test_df["Female"] == 1]
    if not tr_f.empty and not te_f.empty:
        used_f = _filter_features_for_df(tr_f, female_features)
        if len(used_f) > 0:
            cph_f = fit_cox_model(tr_f, used_f, time_col, event_col)
            risk_f = predict_risk(cph_f, te_f, used_f)
            thr_f = float(np.nanmedian(risk_f))
            pred_f = (risk_f >= thr_f).astype(int)
            mask_f = test_df["Female"].values == 1
            pred[mask_f] = pred_f
            risk_scores[mask_f] = risk_f
            if make_plots:
                try:
                    plot_cox_coefficients(cph_f, "Female Cox Coefficients (log HR)")
                except Exception:
                    pass

    # Metrics
    try:
        cidx_all = concordance_index(test_df[time_col], -risk_scores, test_df[event_col])
    except Exception:
        cidx_all = np.nan

    try:
        mask_m = test_df["Female"].values == 0
        mask_f = test_df["Female"].values == 1

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
            test_df.loc[mask_m, time_col].values,
            test_df.loc[mask_m, event_col].values,
            np.asarray(risk_scores)[mask_m],
        )
        cidx_f = _safe_cidx(
            test_df.loc[mask_f, time_col].values,
            test_df.loc[mask_f, event_col].values,
            np.asarray(risk_scores)[mask_f],
        )
    except Exception:
        cidx_m, cidx_f = np.nan, np.nan

    return pred, risk_scores, {"c_index": cidx_all, "c_index_male": cidx_m, "c_index_female": cidx_f}


def run_cox_experiments_sex_specific_subsets(
    df: pd.DataFrame,
    subgroup_features: Dict[str, Union[str, List[str]]],
    N: int = 50,
    time_col: str = "PE_Time",
    event_col: str = "VT/VF/SCD",
    export_excel_path: str = None,
) -> Tuple[Dict[str, Dict[str, List[float]]], pd.DataFrame]:
    """Run N random 70/30 splits with sex-specific feature subsets and summarize C-indexes."""
    from sklearn.model_selection import train_test_split

    results: Dict[str, Dict[str, List[float]]] = {"Sex-specific subsets": {"c_index_all": [], "c_index_male": [], "c_index_female": []}}

    data = df.dropna(subset=[time_col, event_col])
    for seed in range(N):
        tr, te = train_test_split(
            data,
            test_size=0.3,
            random_state=seed,
            stratify=data[event_col] if event_col in data.columns else None,
        )
        pred, risk, met = evaluate_split_sex_specific_subsets(
            tr,
            te,
            subgroup_features=subgroup_features,
            time_col=time_col,
            event_col=event_col,
            make_plots=False,
        )
        results["Sex-specific subsets"]["c_index_all"].append(met.get("c_index", np.nan))
        results["Sex-specific subsets"]["c_index_male"].append(met.get("c_index_male", np.nan))
        results["Sex-specific subsets"]["c_index_female"].append(met.get("c_index_female", np.nan))

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


def sex_specific_full_inference_with_subsets(
    df: pd.DataFrame,
    subgroup_features: Dict[str, Union[str, List[str]]],
    time_col: str = "PE_Time",
    event_col: str = "VT/VF/SCD",
) -> pd.DataFrame:
    """Train sex-specific models on all available data for each sex with subgroup-specific features,
    score all samples, dichotomize by sex-specific medians, and plot KM curves by sex.
    """
    male_features = _filter_features_for_df(df[df["Female"] == 0], _resolve_feature_list(subgroup_features.get("male", [])))
    female_features = _filter_features_for_df(df[df["Female"] == 1], _resolve_feature_list(subgroup_features.get("female", [])))

    out = df.copy()

    # Male
    data_m = out[(out["Female"] == 0)].dropna(subset=[time_col, event_col])
    if not data_m.empty and len(male_features) > 0:
        cph_m = fit_cox_model(data_m, male_features, time_col, event_col)
        r_m = predict_risk(cph_m, data_m, male_features)
        thr_m = float(np.nanmedian(r_m))
        out.loc[out["Female"] == 0, "pred_prob"] = r_m
        out.loc[out["Female"] == 0, "pred_label"] = (r_m >= thr_m).astype(int)
        try:
            plot_cox_coefficients(cph_m, "Male Cox Coefficients (log HR)")
        except Exception:
            pass

    # Female
    data_f = out[(out["Female"] == 1)].dropna(subset=[time_col, event_col])
    if not data_f.empty and len(female_features) > 0:
        cph_f = fit_cox_model(data_f, female_features, time_col, event_col)
        r_f = predict_risk(cph_f, data_f, female_features)
        thr_f = float(np.nanmedian(r_f))
        out.loc[out["Female"] == 1, "pred_prob"] = r_f
        out.loc[out["Female"] == 1, "pred_label"] = (r_f >= thr_f).astype(int)
        try:
            plot_cox_coefficients(cph_f, "Female Cox Coefficients (log HR)")
        except Exception:
            pass

    merged_df = (
        out[["MRN", "Female", "pred_label", time_col, event_col]]
        .dropna(subset=[time_col, event_col])
        .drop_duplicates(subset=["MRN"])
        .rename(columns={event_col: "PE"})
    )
    plot_km_two_subplots_by_gender(merged_df)
    return merged_df


if __name__ == "__main__":
    # Load data (reuse cox.py loader)
    df = load_dataframes()

    # Define subgroup-specific feature subsets. Values can be lists or names in FEATURE_SETS.
    # Example: Male uses 'Benchmark', Female uses 'Proposed'. Adjust as needed.
    SUBGROUP_FEATURES = {
        "male": "Benchmark",
        "female": "Proposed",
    }

    # Run experiments
    export_path = "/home/sunx/data/aiiih/projects/sunx/projects/ICD_sex_diff/results_cox_subgroup.xlsx"
    _, summary = run_cox_experiments_sex_specific_subsets(
        df=df,
        subgroup_features=SUBGROUP_FEATURES,
        N=100,
        time_col="PE_Time",
        event_col="VT/VF/SCD",
        export_excel_path=export_path,
    )
    print("Saved Excel:", export_path)

    # Full-data inference and analysis
    _ = sex_specific_full_inference_with_subsets(
        df,
        subgroup_features=SUBGROUP_FEATURES,
        time_col="PE_Time",
        event_col="VT/VF/SCD",
    )

