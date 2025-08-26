import argparse
import os
import warnings
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def CG_equation(age: float, weight: float, female: int, serum_creatinine: float) -> float:
    constant = 0.85 if bool(female) else 1.0
    return ((140 - age) * weight * constant) / (72 * serum_creatinine)


def load_data(
    nicm_path: str,
    icd_sheet: str = "ICD",
    no_icd_sheet: str = "No_ICD",
    icd_survival_path: str = None,
    no_icd_survival_path: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load features and survival dataframes following a.py conventions."""
    # Survival: ICD
    icd_survival = pd.read_excel(icd_survival_path)
    icd_survival["PE_Time"] = icd_survival.apply(
        lambda row: (
            row["Time from ICD Implant to Primary Endpoint (in days)"]
            if row["Was Primary Endpoint Reached? (Appropriate ICD Therapy)"] == 1
            else row["Time from ICD Implant to Last Cardiology Encounter (in days)"]
        ),
        axis=1,
    )
    icd_survival["SE_Time"] = icd_survival.apply(
        lambda row: (
            row["Time from ICD Implant to Secondary Endpoint (in days)"]
            if row["Was Secondary Endpoint Reached?"] == 1
            else row["Time from ICD Implant to Last Cardiology Encounter (in days)"]
        ),
        axis=1,
    )
    icd_survival = icd_survival[
        [
            "MRN",
            "Was Primary Endpoint Reached? (Appropriate ICD Therapy)",
            "PE_Time",
            "Was Secondary Endpoint Reached?",
            "SE_Time",
        ]
    ].rename(
        columns={
            "Was Primary Endpoint Reached? (Appropriate ICD Therapy)": "PE",
            "Was Secondary Endpoint Reached?": "SE",
        }
    )

    # Survival: No ICD
    no_icd_survival = pd.read_csv(no_icd_survival_path)
    no_icd_survival["PE_Time"] = no_icd_survival.apply(
        lambda row: (
            row["days_MRI_to_VTVFSCD"] if row["VT/VF/SCD"] == 1 else row["days_MRI_to_followup"]
        ),
        axis=1,
    )
    no_icd_survival["SE_Time"] = no_icd_survival.apply(
        lambda row: (
            row["days_MRI_to_death"] if row["Death"] == 1 else row["days_MRI_to_followup"]
        ),
        axis=1,
    )
    no_icd_survival = no_icd_survival[
        [
            "MRN",
            "VT/VF/SCD",
            "PE_Time",
            "Death",
            "SE_Time",
        ]
    ].rename(columns={"VT/VF/SCD": "PE", "Death": "SE"})

    survival_df = pd.concat([icd_survival, no_icd_survival], ignore_index=True)

    # Features: with and without ICD
    with_icd = pd.read_excel(nicm_path, sheet_name=icd_sheet)
    with_icd["ICD"] = 1
    without_icd = pd.read_excel(nicm_path, sheet_name=no_icd_sheet)
    without_icd["ICD"] = 0
    without_icd["Cockcroft-Gault Creatinine Clearance (mL/min)"] = without_icd.apply(
        lambda row: CG_equation(
            row["Age at CMR"],
            row["Weight (Kg)"],
            row["Female"],
            row["Serum creatinine (within 3 months of MRI)"],
        ),
        axis=1,
    )
    common_cols = with_icd.columns.intersection(without_icd.columns)
    df = pd.concat([with_icd[common_cols], without_icd[common_cols]], ignore_index=True)
    df = df.drop(columns=[c for c in ["Date VT/VF/SCD", "End follow-up date", "CRT Date", "QRS"] if c in df.columns], errors="ignore")

    return df, survival_df


FEATURE_SETS: Dict[str, List[str]] = {
    'guideline': ["NYHA Class", "LVEF"],
    'benchmark': ["Female", "Age by decade", "BMI", "AF", "Beta Blocker", "CrCl>45",
                  "LVEF", "QTc", "NYHA>2", "CRT", "AAD", "Significant LGE"],
    'proposed': ["Female", "Age by decade", "BMI", "AF", "Beta Blocker", "CrCl>45",
                 "LVEF", "QTc", "NYHA>2", "CRT", "AAD", "Significant LGE", "DM", "HTN",
                 "HLP", "LVEDVi", "LV Mass Index", "RVEDVi", "RVEF", "LA EF", "LAVi",
                 "MRF (%)", "Sphericity Index", "Relative Wall Thickness",
                 "MV Annular Diameter", "ACEi/ARB/ARNi", "Aldosterone Antagonist"],
    'real_proposed': ["Female", "Age by decade", "BMI", "AF", "Beta Blocker", "CrCl>45",
                      "LVEF", "QTc", "CRT", "AAD", "LGE Burden 5SD", "DM", "HTN",
                      "HLP", "LVEDVi", "LV Mass Index", "RVEDVi", "RVEF", "LA EF", "LAVi",
                      "MRF (%)", "Sphericity Index", "Relative Wall Thickness",
                      "MV Annular Diameter", "ACEi/ARB/ARNi", "Aldosterone Antagonist", "NYHA Class"]
}


def prepare_cox_dataset(features_df: pd.DataFrame, survival_df: pd.DataFrame, features: List[str], endpoint: str) -> pd.DataFrame:
    merged = features_df.merge(survival_df[["MRN", endpoint, f"{endpoint}_Time"]], on="MRN", how="inner")
    cols = [c for c in features if c in merged.columns]
    model_df = merged[cols + ["Female", "MRN", endpoint, f"{endpoint}_Time"]].dropna()
    # Ensure numeric features
    for col in cols:
        if not np.issubdtype(model_df[col].dtype, np.number):
            model_df[col] = pd.Categorical(model_df[col]).codes
    model_df = model_df.rename(columns={endpoint: "event", f"{endpoint}_Time": "duration"})
    return model_df.reset_index(drop=True)


def balance_sex_undersample(train_df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    grp_male = train_df[train_df["Female"] == 0]
    grp_female = train_df[train_df["Female"] == 1]
    n_target = min(len(grp_male), len(grp_female))
    if n_target == 0:
        return train_df.copy()
    grp_male_s = grp_male.sample(n=n_target, random_state=random_state, replace=False)
    grp_female_s = grp_female.sample(n=n_target, random_state=random_state, replace=False)
    return pd.concat([grp_male_s, grp_female_s], axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def fit_cox_and_predict(train_df: pd.DataFrame, val_df: pd.DataFrame, features: List[str], penalizer: float = 0.1):
    cph = CoxPHFitter(penalizer=penalizer)
    use_cols = [c for c in features if c in train_df.columns]
    cph.fit(train_df[use_cols + ["duration", "event"]], duration_col="duration", event_col="event")
    train_risk = cph.predict_partial_hazard(train_df[use_cols]).values.reshape(-1)
    val_risk = cph.predict_partial_hazard(val_df[use_cols]).values.reshape(-1)
    thr = float(np.median(train_risk))
    return cph, val_risk, thr


def cox_multiple_random_splits(df_all: pd.DataFrame, survival_df: pd.DataFrame, N: int, endpoint: str) -> Tuple[Dict, pd.DataFrame]:
    model_configs = [
        {'name': 'Guideline Cox (sex-agnostic)', 'features': 'guideline', 'type': 'cox_all'},
        {'name': 'Benchmark Cox (sex-agnostic)', 'features': 'benchmark', 'type': 'cox_all'},
        {'name': 'Benchmark Cox (sex-agnostic undersampled)', 'features': 'benchmark', 'type': 'cox_all_us'},
        {'name': 'Benchmark Cox (male)', 'features': 'benchmark', 'type': 'cox_male'},
        {'name': 'Benchmark Cox (female)', 'features': 'benchmark', 'type': 'cox_female'},
        {'name': 'Benchmark Cox (sex-specific)', 'features': 'benchmark', 'type': 'cox_sex_specific'},
        {'name': 'Proposed Cox (sex-agnostic)', 'features': 'proposed', 'type': 'cox_all'},
        {'name': 'Proposed Cox (sex-agnostic undersampled)', 'features': 'proposed', 'type': 'cox_all_us'},
        {'name': 'Proposed Cox (male)', 'features': 'proposed', 'type': 'cox_male'},
        {'name': 'Proposed Cox (female)', 'features': 'proposed', 'type': 'cox_female'},
        {'name': 'Proposed Cox (sex-specific)', 'features': 'proposed', 'type': 'cox_sex_specific'},
        {'name': 'Real Proposed Cox (sex-agnostic)', 'features': 'real_proposed', 'type': 'cox_all'},
        {'name': 'Real Proposed Cox (sex-agnostic undersampled)', 'features': 'real_proposed', 'type': 'cox_all_us'},
        {'name': 'Real Proposed Cox (male)', 'features': 'real_proposed', 'type': 'cox_male'},
        {'name': 'Real Proposed Cox (female)', 'features': 'real_proposed', 'type': 'cox_female'},
        {'name': 'Real Proposed Cox (sex-specific)', 'features': 'real_proposed', 'type': 'cox_sex_specific'},
    ]

    metrics = ['cindex_overall', 'cindex_male', 'cindex_female', 'logrank_p', 'hr_high_vs_low']
    results = {config['name']: {m: [] for m in metrics} for config in model_configs}

    for seed in range(N):
        train_df, val_df = train_test_split(df_all, test_size=0.3, random_state=seed, stratify=df_all['Female'])
        mask_m = val_df["Female"].values == 0
        mask_f = val_df["Female"].values == 1

        for config in model_configs:
            model_name = config['name']
            feat_names = FEATURE_SETS[config['features']]
            train_model = prepare_cox_dataset(train_df, survival_df, feat_names, endpoint)
            val_model = prepare_cox_dataset(val_df, survival_df, feat_names, endpoint)
            if len(train_model) == 0 or len(val_model) == 0:
                for m in metrics:
                    results[model_name][m].append(np.nan)
                continue

            try:
                # Train/eval per type
                if config['type'] == 'cox_all':
                    cph, val_risk, thr = fit_cox_and_predict(train_model, val_model, feat_names)
                    pred_label = (val_risk >= thr).astype(int)
                    risk_vec = val_risk
                elif config['type'] == 'cox_all_us':
                    balanced = balance_sex_undersample(train_model, seed)
                    cph, val_risk, thr = fit_cox_and_predict(balanced, val_model, feat_names)
                    pred_label = (val_risk >= thr).astype(int)
                    risk_vec = val_risk
                elif config['type'] == 'cox_male':
                    tr_m = train_model[train_model['Female'] == 0]
                    va_m = val_model[val_model['Female'] == 0]
                    if len(tr_m) == 0 or len(va_m) == 0:
                        raise ValueError('no male data')
                    cph_m, risk_m, thr_m = fit_cox_and_predict(tr_m, va_m, feat_names)
                    pred_label = np.zeros(len(val_model), dtype=int)
                    pred_label[val_model['Female'].values == 0] = (risk_m >= thr_m).astype(int)
                    risk_vec = pred_label.astype(float)
                elif config['type'] == 'cox_female':
                    tr_f = train_model[train_model['Female'] == 1]
                    va_f = val_model[val_model['Female'] == 1]
                    if len(tr_f) == 0 or len(va_f) == 0:
                        raise ValueError('no female data')
                    cph_f, risk_f, thr_f = fit_cox_and_predict(tr_f, va_f, feat_names)
                    pred_label = np.zeros(len(val_model), dtype=int)
                    pred_label[val_model['Female'].values == 1] = (risk_f >= thr_f).astype(int)
                    risk_vec = pred_label.astype(float)
                elif config['type'] == 'cox_sex_specific':
                    tr_m = train_model[train_model['Female'] == 0]
                    va_m = val_model[val_model['Female'] == 0]
                    tr_f = train_model[train_model['Female'] == 1]
                    va_f = val_model[val_model['Female'] == 1]
                    pred_label = np.zeros(len(val_model), dtype=int)
                    if len(tr_m) > 0 and len(va_m) > 0:
                        cph_m, risk_m, thr_m = fit_cox_and_predict(tr_m, va_m, feat_names)
                        pred_label[val_model['Female'].values == 0] = (risk_m >= thr_m).astype(int)
                    if len(tr_f) > 0 and len(va_f) > 0:
                        cph_f, risk_f, thr_f = fit_cox_and_predict(tr_f, va_f, feat_names)
                        pred_label[val_model['Female'].values == 1] = (risk_f >= thr_f).astype(int)
                    risk_vec = pred_label.astype(float)
                else:
                    raise ValueError('unknown type')

                # Metrics
                val_times = val_model['duration'].values
                val_events = val_model['event'].values
                cidx_all = concordance_index(val_times, risk_vec, val_events)
                cidx_m = concordance_index(val_times[mask_m], risk_vec[mask_m], val_events[mask_m]) if mask_m.sum() > 1 else np.nan
                cidx_f = concordance_index(val_times[mask_f], risk_vec[mask_f], val_events[mask_f]) if mask_f.sum() > 1 else np.nan

                high = pred_label == 1
                if high.any() and (~high).any():
                    lr = logrank_test(val_times[high], val_times[~high], event_observed_A=val_events[high], event_observed_B=val_events[~high])
                    p_lr = float(lr.p_value)
                    tmp = pd.DataFrame({'duration': val_times, 'event': val_events, 'risk_high': pred_label.astype(int)})
                    cph_tmp = CoxPHFitter()
                    cph_tmp.fit(tmp, duration_col='duration', event_col='event')
                    hr = float(np.exp(cph_tmp.params_['risk_high']))
                else:
                    p_lr = np.nan
                    hr = np.nan

                results[model_name]['cindex_overall'].append(cidx_all)
                results[model_name]['cindex_male'].append(cidx_m)
                results[model_name]['cindex_female'].append(cidx_f)
                results[model_name]['logrank_p'].append(p_lr)
                results[model_name]['hr_high_vs_low'].append(hr)
            except Exception:
                for m in metrics:
                    results[model_name][m].append(np.nan)

    summary = {}
    for model, md in results.items():
        summary[model] = {}
        for metric, values in md.items():
            arr = np.array(values, dtype=float)
            mu = np.nanmean(arr)
            n_eff = np.sum(~np.isnan(arr))
            se = np.nanstd(arr, ddof=1) / np.sqrt(n_eff) if n_eff > 1 else np.nan
            ci = 1.96 * se if se == se else np.nan
            summary[model][metric] = (mu, mu - ci if ci == ci else np.nan, mu + ci if ci == ci else np.nan)

    summary_df = pd.concat({
        model: pd.DataFrame.from_dict(metrics_dict, orient='index', columns=['mean', 'ci_lower', 'ci_upper'])
        for model, metrics_dict in summary.items()
    }, axis=0)

    formatted = summary_df.apply(lambda row: f"{row['mean']:.3f} ({row['ci_lower']:.3f}, {row['ci_upper']:.3f})" if row.notna().all() else "nan", axis=1)
    summary_table = formatted.unstack(level=1)
    return results, summary_table


def main():
    parser = argparse.ArgumentParser(description="Run CoxPH evaluation (sex-agnostic balanced and sex-specific) with multiple 70/30 splits and export to Excel.")
    parser.add_argument("--splits", type=int, default=50, help="Number of random 70/30 splits (default: 50)")
    parser.add_argument("--endpoint", type=str, default="both", choices=["PE", "SE", "both"], help="Endpoint to evaluate (PE, SE, or both)")
    parser.add_argument("--nicm", type=str, default="/home/sunx/data/aiiih/projects/sunx/projects/ICD_sex_diff/NICM.xlsx", help="NICM.xlsx path")
    parser.add_argument("--icd_survival", type=str, default="/home/sunx/data/aiiih/projects/sunx/projects/ICD_sex_diff/icd_survival.xlsx", help="ICD survival Excel path")
    parser.add_argument("--no_icd_survival", type=str, default="/home/sunx/data/aiiih/projects/sunx/projects/ICD_sex_diff/no_icd_survival.csv", help="No-ICD survival CSV path")
    parser.add_argument("--out", type=str, default="/home/sunx/data/aiiih/projects/sunx/projects/ICD_sex_diff/results_cox.xlsx", help="Output Excel path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df, survival_df = load_data(
        nicm_path=args.nicm,
        icd_survival_path=args.icd_survival,
        no_icd_survival_path=args.no_icd_survival,
    )

    if args.endpoint == "both":
        _, summary_pe = cox_multiple_random_splits(df, survival_df, args.splits, endpoint="PE")
        _, summary_se = cox_multiple_random_splits(df, survival_df, args.splits, endpoint="SE")
        with pd.ExcelWriter(args.out) as writer:
            summary_pe.to_excel(writer, sheet_name='PE', index=True, index_label='RowName')
            summary_se.to_excel(writer, sheet_name='SE', index=True, index_label='RowName')
        print(f"Wrote PE and SE summaries to {args.out}")
    else:
        _, summary = cox_multiple_random_splits(df, survival_df, args.splits, endpoint=args.endpoint)
        with pd.ExcelWriter(args.out) as writer:
            summary.to_excel(writer, sheet_name=args.endpoint, index=True, index_label='RowName')
        print(f"Wrote {args.endpoint} summary to {args.out}")


if __name__ == "__main__":
    main()