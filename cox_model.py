import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import sklearn.neighbors._base
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
import sys
import warnings
from sklearn.exceptions import UndefinedMetricWarning

sys.modules["sklearn.neighbors.base"] = sklearn.neighbors._base
from missingpy import MissForest

pd.set_option("future.no_silent_downcasting", True)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def CG_equation(age, weight, female, serum_creatinine):
    """Cockcroft-Gault Equation."""
    constant = 0.85 if female else 1.0
    return ((140 - age) * weight * constant) / (72 * serum_creatinine)


# Load and prepare data
icd_survival = pd.read_excel(
    "/home/sunx/data/aiiih/projects/sunx/projects/ICD_sex_diff/icd_survival.xlsx",
)
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

no_icd_survival = pd.read_csv('/home/sunx/data/aiiih/projects/sunx/projects/ICD_sex_diff/no_icd_survival.csv')
no_icd_survival["PE_Time"] = no_icd_survival.apply(
    lambda row: (
        row["days_MRI_to_VTVFSCD"]
        if row["VT/VF/SCD"] == 1
        else row["days_MRI_to_followup"]
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
].rename(
    columns={
        "VT/VF/SCD": "PE",
        "Death": "SE"
    }
)

# Merged survival
survival_df = pd.concat([icd_survival, no_icd_survival], ignore_index=True)

# Load features data
with_icd = pd.read_excel(
    "/home/sunx/data/aiiih/projects/sunx/projects/ICD_sex_diff/NICM.xlsx",
    sheet_name="ICD",
)
with_icd["ICD"] = 1
without_icd = pd.read_excel(
    "/home/sunx/data/aiiih/projects/sunx/projects/ICD_sex_diff/NICM.xlsx",
    sheet_name="No_ICD",
)
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
df.drop(
    [
        "Date VT/VF/SCD",
        "End follow-up date",
        "CRT Date",
        "QRS",
    ],
    axis=1,
    inplace=True,
)

# Variables
var = df.columns.tolist()
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
numerical = list(set(var) - set(categorical))
df[categorical] = df[categorical].astype("object")

# Labels
labels = ["MRN", "Female", "VT/VF/SCD", "ICD"]
features = [c for c in var if c not in labels]


def impute_misforest(X, random_seed):
    """Non-parametric iterative imputation method based on random forest."""
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
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

    # Convert multiclass categorical variables to numerical format
    ordinal = "NYHA Class"
    le = LabelEncoder()
    df[ordinal] = le.fit_transform(df[ordinal])

    # Convert binary columns to int
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
    ]
    exist_bin = [col for col in binary_cols if col in df.columns]
    for c in exist_bin:
        if df[c].dtype == "object":
            df[c] = df[c].replace(
                {"Yes": 1, "No": 0, "Y": 1, "N": 0, "True": 1, "False": 0}
            )
        df[c] = df[c].astype("float")

    # Imputation
    X = df[features]
    imputed_X = impute_misforest(X, 0)
    imputed_X.index = df.index
    imputed_X["MRN"] = df["MRN"].values
    imputed_X["Female"] = df["Female"].values
    imputed_X["VT/VF/SCD"] = df["VT/VF/SCD"].values
    imputed_X["ICD"] = df["ICD"].values

    # Map to 0 and 1 using 0.5 as threshold
    for c in exist_bin:
        imputed_X[c] = (imputed_X[c] >= 0.5).astype("float")

    return imputed_X


clean_df = conversion_and_imputation(df, features, labels)

# Additional features
clean_df["Age by decade"] = df["Age at CMR"] // 10
clean_df["CrCl>45"] = (
    clean_df["Cockcroft-Gault Creatinine Clearance (mL/min)"] > 45
).astype(int)
clean_df["NYHA>2"] = (clean_df["NYHA Class"] > 2).astype(int)
clean_df["Significant LGE"] = (clean_df["LGE Burden 5SD"] > 2).astype(int)

# Merge with survival data for Cox model
clean_df_with_survival = clean_df.merge(survival_df, on="MRN", how="inner")

print("\nDistribution of Sex")
print(clean_df["Female"].value_counts())
print("\nDistribution of Arrhythmia")
print(clean_df["VT/VF/SCD"].value_counts())

# Train-test split
stratify_column = clean_df['Female'].astype(str) + '_' + clean_df['VT/VF/SCD'].astype(str) + '_' + clean_df['ICD'].astype(str)
train_df, test_df = train_test_split(
    clean_df,
    test_size=0.2,
    stratify=stratify_column,
    random_state=100
)

print("\nOverall Female proportion:", clean_df['Female'].mean())
print("Train Female proportion:", train_df['Female'].mean())
print("Test Female proportion:", test_df['Female'].mean())


def create_undersampled_dataset(train_df, label_col, random_state, pos_to_neg_ratio=1.0):
    """Create per-sex undersampled dataset with desired positive:negative ratio."""
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
        r = float(pos_to_neg_ratio) if pos_to_neg_ratio > 0 else 1.0
        if r >= 1.0:
            target_pos = min(P, int(np.floor(r * N)))
            if target_pos <= 0:
                target_pos = min(P, 1)
            target_neg = min(N, int(np.floor(target_pos / r)))
            if target_neg <= 0:
                target_neg = min(N, 1)
        else:
            target_neg = min(N, int(np.floor(P / r)))
            if target_neg <= 0:
                target_neg = min(N, 1)
            target_pos = min(P, int(np.floor(r * target_neg)))
            if target_pos <= 0:
                target_pos = min(P, 1)
        samp_pos = pos.sample(n=target_pos, replace=False, random_state=random_state)
        samp_neg = neg.sample(n=target_neg, replace=False, random_state=random_state)
        sampled_parts.append(pd.concat([samp_pos, samp_neg]))
    return pd.concat(sampled_parts).sample(frac=1, random_state=random_state).reset_index(drop=True)


def prepare_cox_data(df, features, time_col, event_col):
    """Prepare data for Cox regression."""
    cox_df = df[features + [time_col, event_col]].copy()
    # Ensure positive times
    cox_df[time_col] = cox_df[time_col].clip(lower=0.1)
    return cox_df


def train_cox_model(train_df, features, time_col, event_col, penalizer=0.1):
    """Train Cox Proportional Hazards model."""
    cox_df = prepare_cox_data(train_df, features, time_col, event_col)
    
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(cox_df, duration_col=time_col, event_col=event_col)
    
    return cph


def cox_predict_risk(model, test_df, features, time_point=None):
    """
    Predict risk scores using Cox model.
    Returns hazard ratios and predicted probabilities.
    """
    X_test = test_df[features]
    
    # Get partial hazard (exp(linear predictor))
    hazard_ratios = model.predict_partial_hazard(X_test)
    
    # If time_point is specified, get survival probability at that time
    if time_point is not None:
        survival_probs = model.predict_survival_function(X_test, times=[time_point]).T
        # Convert survival to risk probability
        risk_probs = 1 - survival_probs.values.flatten()
    else:
        # Use median survival time if no time point specified
        median_time = model.summary['coef'].index[0]  # Placeholder
        risk_probs = hazard_ratios.values
    
    return hazard_ratios.values, risk_probs


def compute_cox_concordance(model, test_df, features, time_col, event_col):
    """Compute concordance index for Cox model."""
    cox_test_df = prepare_cox_data(test_df, features, time_col, event_col)
    predictions = model.predict_partial_hazard(cox_test_df[features])
    
    c_index = concordance_index(
        cox_test_df[time_col],
        -predictions,  # Negative because higher hazard = worse outcome
        cox_test_df[event_col]
    )
    return c_index


def find_optimal_risk_threshold(hazard_ratios, events, percentile=75):
    """Find optimal threshold for risk stratification based on hazard ratios."""
    # Use percentile-based threshold
    threshold = np.percentile(hazard_ratios, percentile)
    return threshold


def evaluate_cox_performance(y_true, y_pred, y_prob, mask_m, mask_f, overall_mask=None):
    """Evaluate model performance for overall, male, and female subsets."""
    # Overall performance
    if overall_mask is None:
        y_true_overall = y_true
        y_pred_overall = y_pred
        y_prob_overall = y_prob
    else:
        y_true_overall = y_true[overall_mask]
        y_pred_overall = y_pred[overall_mask]
        y_prob_overall = y_prob[overall_mask]
    
    acc = accuracy_score(y_true_overall, y_pred_overall)
    auc = roc_auc_score(y_true_overall, y_prob_overall) if len(np.unique(y_true_overall)) > 1 else np.nan
    f1 = f1_score(y_true_overall, y_pred_overall)
    
    tn, fp, fn, tp = confusion_matrix(y_true_overall, y_pred_overall).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    
    # Male subset
    y_true_m = y_true[mask_m]
    y_pred_m = y_pred[mask_m]
    y_prob_m = y_prob[mask_m]
    
    male_acc = accuracy_score(y_true_m, y_pred_m) if len(y_true_m) > 0 else np.nan
    male_auc = roc_auc_score(y_true_m, y_prob_m) if len(y_true_m) > 1 and len(np.unique(y_true_m)) > 1 else np.nan
    male_f1 = f1_score(y_true_m, y_pred_m) if len(y_true_m) > 0 else np.nan
    
    if len(y_true_m) > 0:
        tn_m, fp_m, fn_m, tp_m = confusion_matrix(y_true_m, y_pred_m).ravel()
        male_sens = tp_m / (tp_m + fn_m) if (tp_m + fn_m) > 0 else np.nan
        male_spec = tn_m / (tn_m + fp_m) if (tn_m + fp_m) > 0 else np.nan
    else:
        male_sens, male_spec = np.nan, np.nan
    
    # Female subset
    y_true_f = y_true[mask_f]
    y_pred_f = y_pred[mask_f]
    y_prob_f = y_prob[mask_f]
    
    female_acc = accuracy_score(y_true_f, y_pred_f) if len(y_true_f) > 0 else np.nan
    female_auc = roc_auc_score(y_true_f, y_prob_f) if len(y_true_f) > 1 and len(np.unique(y_true_f)) > 1 else np.nan
    female_f1 = f1_score(y_true_f, y_pred_f) if len(y_true_f) > 0 else np.nan
    
    if len(y_true_f) > 0:
        tn_f, fp_f, fn_f, tp_f = confusion_matrix(y_true_f, y_pred_f).ravel()
        female_sens = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else np.nan
        female_spec = tn_f / (tn_f + fp_f) if (tn_f + fp_f) > 0 else np.nan
    else:
        female_sens, female_spec = np.nan, np.nan
    
    return {
        'accuracy': acc, 'auc': auc, 'f1': f1, 'sensitivity': sens, 'specificity': spec,
        'male_accuracy': male_acc, 'male_auc': male_auc, 'male_f1': male_f1, 
        'male_sensitivity': male_sens, 'male_specificity': male_spec,
        'female_accuracy': female_acc, 'female_auc': female_auc, 'female_f1': female_f1,
        'female_sensitivity': female_sens, 'female_specificity': female_spec
    }


def incidence_rate(df, pred_col, label_col):
    """Compute incidence rate for males and females."""
    def rate(sub):
        n_pred = (sub[pred_col] == 1).sum()
        n_true = (sub[label_col] == 1).sum()
        return n_true / n_pred if n_pred > 0 else np.nan

    male_rate = rate(df[df["Female"] == 0])
    female_rate = rate(df[df["Female"] == 1])
    return male_rate, female_rate


def plot_cox_feature_importance(model, features, title, gray_features=None, red_features=None):
    """Plot feature importance from Cox model using hazard ratios."""
    # Get coefficients and hazard ratios
    coefs = model.params_
    hazard_ratios = np.exp(coefs)
    
    # Sort by absolute coefficient value
    abs_coefs = np.abs(coefs)
    idx = np.argsort(abs_coefs)[::-1]
    
    sorted_features = [features[i] for i in idx]
    sorted_hrs = hazard_ratios.iloc[idx].values
    sorted_coefs = coefs.iloc[idx].values
    
    # Determine colors
    if gray_features is not None or red_features is not None:
        gray_features_set = set(gray_features) if gray_features is not None else set()
        red_features_set = set(red_features) if red_features is not None else set()
        colors = []
        for f in sorted_features:
            if f in red_features_set:
                colors.append("red")
            elif f in gray_features_set:
                colors.append("gray")
            else:
                colors.append("blue")
    else:
        colors = [
            "red" if f in {"LVEF", "NYHA Class", "NYHA>2"}
            else "gold" if f in {"Significant LGE", "LGE Burden 5SD"} 
            else "lightgray"
            for f in sorted_features
        ]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot hazard ratios
    bars1 = ax1.barh(range(len(sorted_features)), sorted_hrs, color=colors)
    ax1.set_yticks(range(len(sorted_features)))
    ax1.set_yticklabels(sorted_features)
    ax1.axvline(x=1, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel("Hazard Ratio")
    ax1.set_title(f"{title} - Hazard Ratios")
    ax1.set_xlim(0, max(sorted_hrs) * 1.1)
    
    # Plot coefficients
    bars2 = ax2.barh(range(len(sorted_features)), sorted_coefs, color=colors)
    ax2.set_yticks(range(len(sorted_features)))
    ax2.set_yticklabels(sorted_features)
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel("Coefficient")
    ax2.set_title(f"{title} - Coefficients")
    
    # Add legend
    from matplotlib.patches import Patch
    present_colors = set(colors)
    legend_elements = []
    if 'red' in present_colors:
        legend_elements.append(Patch(facecolor='red', label='guideline features'))
    if 'gray' in present_colors:
        legend_elements.append(Patch(facecolor='gray', label='standard cmr features'))
    if 'blue' in present_colors:
        legend_elements.append(Patch(facecolor='blue', label='advanced cmr features'))
    if len(legend_elements) > 0:
        ax1.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.show()


def multiple_random_splits_cox(df, N, label="VT/VF/SCD", time_col="PE_Time", event_col="PE"):
    """
    Perform multiple random train-test splits and evaluate Cox models.
    Includes sex-agnostic and sex-specific models with undersampling.
    """
    # Merge with survival data for Cox models
    df_with_survival = df.merge(survival_df, on="MRN", how="inner")
    
    # Feature definitions
    feature_sets = {
        'guideline': ["NYHA Class", "LVEF"],
        'benchmark': ["Age by decade", "BMI", "AF", "Beta Blocker", "CrCl>45", 
                     "LVEF", "QTc", "NYHA>2", "CRT", "AAD", "Significant LGE"],
        'proposed': ["Age by decade", "BMI", "AF", "Beta Blocker", "CrCl>45", 
                    "LVEF", "QTc", "NYHA>2", "CRT", "AAD", "Significant LGE", "DM", "HTN", 
                    "HLP", "LVEDVi", "LV Mass Index", "RVEDVi", "RVEF", "LA EF", "LAVi", 
                    "MRF (%)", "Sphericity Index", "Relative Wall Thickness", 
                    "MV Annular Diameter", "ACEi/ARB/ARNi", "Aldosterone Antagonist"],
        'real_proposed': ["Age by decade", "BMI", "AF", "Beta Blocker", "CrCl>45", 
                         "LVEF", "QTc", "CRT", "AAD", "LGE Burden 5SD", "DM", "HTN", 
                         "HLP", "LVEDVi", "LV Mass Index", "RVEDVi", "RVEF", "LA EF", "LAVi", 
                         "MRF (%)", "Sphericity Index", "Relative Wall Thickness", 
                         "MV Annular Diameter", "ACEi/ARB/ARNi", "Aldosterone Antagonist", "NYHA Class"]
    }
    
    # Model configurations
    model_configs = [
        {'name': 'Guideline', 'features': 'guideline', 'type': 'rule_based'},
        {'name': 'Cox Benchmark Sex-agnostic', 'features': 'benchmark', 'type': 'cox'},
        {'name': 'Cox Benchmark Sex-agnostic (undersampled)', 'features': 'benchmark', 'type': 'cox_undersampled'},
        {'name': 'Cox Benchmark Male', 'features': 'benchmark', 'type': 'cox_male_only'},
        {'name': 'Cox Benchmark Female', 'features': 'benchmark', 'type': 'cox_female_only'},
        {'name': 'Cox Benchmark Sex-specific', 'features': 'benchmark', 'type': 'cox_sex_specific'},
        {'name': 'Cox Proposed Sex-agnostic', 'features': 'proposed', 'type': 'cox'},
        {'name': 'Cox Proposed Sex-agnostic (undersampled)', 'features': 'proposed', 'type': 'cox_undersampled'},
        {'name': 'Cox Proposed Male', 'features': 'proposed', 'type': 'cox_male_only'},
        {'name': 'Cox Proposed Female', 'features': 'proposed', 'type': 'cox_female_only'},
        {'name': 'Cox Proposed Sex-specific', 'features': 'proposed', 'type': 'cox_sex_specific'},
        {'name': 'Cox Real Proposed Sex-agnostic', 'features': 'real_proposed', 'type': 'cox'},
        {'name': 'Cox Real Proposed Sex-agnostic (undersampled)', 'features': 'real_proposed', 'type': 'cox_undersampled'},
        {'name': 'Cox Real Proposed Male', 'features': 'real_proposed', 'type': 'cox_male_only'},
        {'name': 'Cox Real Proposed Female', 'features': 'real_proposed', 'type': 'cox_female_only'},
        {'name': 'Cox Real Proposed Sex-specific', 'features': 'real_proposed', 'type': 'cox_sex_specific'}
    ]
    
    # Metrics to track
    metrics = ['accuracy', 'auc', 'f1', 'sensitivity', 'specificity', 
               'male_accuracy', 'male_auc', 'male_f1', 'male_sensitivity', 'male_specificity',
               'female_accuracy', 'female_auc', 'female_f1', 'female_sensitivity', 'female_specificity',
               'male_rate', 'female_rate', 'c_index']
    
    # Initialize results storage
    results = {config['name']: {met: [] for met in metrics} for config in model_configs}
    
    for seed in range(N):
        print(f"\nRunning split #{seed+1}/{N}")
        
        # Split data
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=seed, stratify=df[label])
        
        # Merge with survival data
        train_surv = train_df.merge(survival_df, on="MRN", how="inner")
        test_surv = test_df.merge(survival_df, on="MRN", how="inner")
        
        tr_m = train_surv[train_surv["Female"] == 0]
        tr_f = train_surv[train_surv["Female"] == 1]
        te_m = test_surv[test_surv["Female"] == 0]
        te_f = test_surv[test_surv["Female"] == 1]
        
        # Create undersampled dataset
        us_train_surv = create_undersampled_dataset(train_surv, label, seed)
        
        # Create masks for gender subsets
        mask_m = test_surv["Female"].values == 0
        mask_f = test_surv["Female"].values == 1
        y_true = test_surv[label].values
        
        # Evaluate each model configuration
        for config in model_configs:
            model_name = config['name']
            feature_set = feature_sets[config['features']]
            model_type = config['type']
            
            mask_m_eval = mask_m
            mask_f_eval = mask_f
            overall_mask_override = None
            
            if model_type == 'rule_based':
                # Guideline model
                pred = ((test_surv["NYHA Class"] >= 2) & (test_surv["LVEF"] <= 35)).astype(int).values
                prob = np.full_like(pred, 0.5, dtype=float)
                c_index = np.nan
                
            elif model_type == 'cox':
                # Standard Cox model
                cox_model = train_cox_model(train_surv, feature_set, time_col, event_col)
                hazard_ratios, risk_probs = cox_predict_risk(cox_model, test_surv, feature_set)
                threshold = find_optimal_risk_threshold(hazard_ratios, train_surv[event_col])
                pred = (hazard_ratios >= threshold).astype(int)
                prob = risk_probs
                c_index = compute_cox_concordance(cox_model, test_surv, feature_set, time_col, event_col)
                
            elif model_type == 'cox_undersampled':
                # Undersampled Cox model
                cox_model = train_cox_model(us_train_surv, feature_set, time_col, event_col)
                hazard_ratios, risk_probs = cox_predict_risk(cox_model, test_surv, feature_set)
                threshold = find_optimal_risk_threshold(hazard_ratios, us_train_surv[event_col])
                pred = (hazard_ratios >= threshold).astype(int)
                prob = risk_probs
                c_index = compute_cox_concordance(cox_model, test_surv, feature_set, time_col, event_col)
                
            elif model_type == 'cox_male_only':
                # Male-only Cox model
                if not tr_m.empty and not te_m.empty:
                    cox_model = train_cox_model(tr_m, feature_set, time_col, event_col)
                    hazard_ratios_m, risk_probs_m = cox_predict_risk(cox_model, te_m, feature_set)
                    threshold = find_optimal_risk_threshold(hazard_ratios_m, tr_m[event_col])
                    pred_m = (hazard_ratios_m >= threshold).astype(int)
                    
                    # Create full test set predictions
                    pred = np.zeros(len(test_surv), dtype=int)
                    prob = np.zeros(len(test_surv), dtype=float)
                    pred[mask_m] = pred_m
                    prob[mask_m] = risk_probs_m
                    mask_f_eval = np.zeros_like(mask_f, dtype=bool)
                    overall_mask_override = mask_m
                    c_index = compute_cox_concordance(cox_model, te_m, feature_set, time_col, event_col)
                else:
                    pred = np.zeros(len(test_surv), dtype=int)
                    prob = np.zeros(len(test_surv), dtype=float)
                    c_index = np.nan
                    
            elif model_type == 'cox_female_only':
                # Female-only Cox model
                if not tr_f.empty and not te_f.empty:
                    cox_model = train_cox_model(tr_f, feature_set, time_col, event_col)
                    hazard_ratios_f, risk_probs_f = cox_predict_risk(cox_model, te_f, feature_set)
                    threshold = find_optimal_risk_threshold(hazard_ratios_f, tr_f[event_col])
                    pred_f = (hazard_ratios_f >= threshold).astype(int)
                    
                    # Create full test set predictions
                    pred = np.zeros(len(test_surv), dtype=int)
                    prob = np.zeros(len(test_surv), dtype=float)
                    pred[mask_f] = pred_f
                    prob[mask_f] = risk_probs_f
                    mask_m_eval = np.zeros_like(mask_m, dtype=bool)
                    overall_mask_override = mask_f
                    c_index = compute_cox_concordance(cox_model, te_f, feature_set, time_col, event_col)
                else:
                    pred = np.zeros(len(test_surv), dtype=int)
                    prob = np.zeros(len(test_surv), dtype=float)
                    c_index = np.nan
                    
            elif model_type == 'cox_sex_specific':
                # Sex-specific Cox models
                pred = np.empty(len(test_surv), dtype=int)
                prob = np.empty(len(test_surv), dtype=float)
                c_indices = []
                
                if not tr_m.empty and not te_m.empty:
                    cox_model_m = train_cox_model(tr_m, feature_set, time_col, event_col)
                    hazard_ratios_m, risk_probs_m = cox_predict_risk(cox_model_m, te_m, feature_set)
                    threshold_m = find_optimal_risk_threshold(hazard_ratios_m, tr_m[event_col])
                    pred[mask_m] = (hazard_ratios_m >= threshold_m).astype(int)
                    prob[mask_m] = risk_probs_m
                    c_indices.append(compute_cox_concordance(cox_model_m, te_m, feature_set, time_col, event_col))
                
                if not tr_f.empty and not te_f.empty:
                    cox_model_f = train_cox_model(tr_f, feature_set, time_col, event_col)
                    hazard_ratios_f, risk_probs_f = cox_predict_risk(cox_model_f, te_f, feature_set)
                    threshold_f = find_optimal_risk_threshold(hazard_ratios_f, tr_f[event_col])
                    pred[mask_f] = (hazard_ratios_f >= threshold_f).astype(int)
                    prob[mask_f] = risk_probs_f
                    c_indices.append(compute_cox_concordance(cox_model_f, te_f, feature_set, time_col, event_col))
                
                c_index = np.mean(c_indices) if c_indices else np.nan
            
            # Evaluate performance
            eval_df = test_surv[[label, "Female"]].reset_index(drop=True).copy()
            eval_df["pred"] = pred
            m_rate, f_rate = incidence_rate(eval_df, "pred", label)
            
            perf_metrics = evaluate_cox_performance(y_true, pred, prob, mask_m_eval, mask_f_eval, overall_mask=overall_mask_override)
            
            # Store results
            for metric, value in perf_metrics.items():
                results[model_name][metric].append(value)
            
            results[model_name]['male_rate'].append(m_rate)
            results[model_name]['female_rate'].append(f_rate)
            results[model_name]['c_index'].append(c_index)
    
    # Compute summary statistics
    summary = {}
    for model, metrics_dict in results.items():
        summary[model] = {}
        for metric, values in metrics_dict.items():
            arr = np.array(values, dtype=float)
            mu = np.nanmean(arr)
            se = np.nanstd(arr, ddof=1) / np.sqrt(np.sum(~np.isnan(arr)))
            ci = 1.96 * se
            summary[model][metric] = (mu, mu - ci, mu + ci)
    
    # Create summary DataFrame
    summary_df = pd.concat(
        {
            model: pd.DataFrame.from_dict(
                metrics_dict, orient="index", columns=["mean", "ci_lower", "ci_upper"]
            )
            for model, metrics_dict in summary.items()
        },
        axis=0,
    )
    
    # Format summary table
    formatted = summary_df.apply(
        lambda row: f"{row['mean']:.3f} ({row['ci_lower']:.3f}, {row['ci_upper']:.3f})",
        axis=1,
    )
    summary_table = formatted.unstack(level=1)
    
    return results, summary_table


def sex_specific_cox_inference(train_df, test_df, features, label_col, survival_df, seed, 
                               time_col="PE_Time", event_col="PE", gray_features=None, red_features=None):
    """
    Sex-specific Cox model inference with survival analysis and visualization.
    """
    # Merge with survival data
    train_surv = train_df.merge(survival_df, on="MRN", how="inner")
    test_surv = test_df.merge(survival_df, on="MRN", how="inner")
    
    # Separate by gender
    train_m = train_surv[train_surv["Female"] == 0]
    train_f = train_surv[train_surv["Female"] == 1]
    test_m = test_surv[test_surv["Female"] == 0]
    test_f = test_surv[test_surv["Female"] == 1]
    
    # Train gender-specific Cox models
    print("Training Male Cox Model...")
    cox_male = train_cox_model(train_m, features, time_col, event_col)
    
    print("Training Female Cox Model...")
    cox_female = train_cox_model(train_f, features, time_col, event_col)
    
    # Make predictions
    df = test_surv.copy()
    
    if not test_m.empty:
        hazard_m, prob_m = cox_predict_risk(cox_male, test_m, features)
        threshold_m = find_optimal_risk_threshold(hazard_m, train_m[event_col])
        pred_m = (hazard_m >= threshold_m).astype(int)
        df.loc[df["Female"] == 0, "pred_label"] = pred_m
        df.loc[df["Female"] == 0, "pred_prob"] = prob_m
    
    if not test_f.empty:
        hazard_f, prob_f = cox_predict_risk(cox_female, test_f, features)
        threshold_f = find_optimal_risk_threshold(hazard_f, train_f[event_col])
        pred_f = (hazard_f >= threshold_f).astype(int)
        df.loc[df["Female"] == 1, "pred_label"] = pred_f
        df.loc[df["Female"] == 1, "pred_prob"] = prob_f
    
    # Feature importance visualization
    plot_cox_feature_importance(cox_male, features, "Male Cox Model", gray_features, red_features)
    plot_cox_feature_importance(cox_female, features, "Female Cox Model", gray_features, red_features)
    
    # Survival analysis
    print(f"\n=== Cox Model Summary ===")
    print(f"Total test samples: {len(df)}")
    
    # Calculate incidence rates
    for gender_val, gender_name in [(0, "Male"), (1, "Female")]:
        gender_data = df[df["Female"] == gender_val]
        if not gender_data.empty:
            for pred_val in [0, 1]:
                group_data = gender_data[gender_data["pred_label"] == pred_val]
                if not group_data.empty:
                    pe_rate = group_data[event_col].sum() / len(group_data)
                    print(f"{gender_name}-Pred{pred_val}: {event_col} rate = {pe_rate:.4f}")
    
    # Plot Kaplan-Meier curves
    plot_km_curves_cox(df, time_col, event_col)
    
    return df


def plot_km_curves_cox(df, time_col, event_col):
    """Plot Kaplan-Meier curves for Cox model predictions."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    kmf = KaplanMeierFitter()
    
    # Plot low risk vs high risk
    for pred_val, risk_label, color in [(0, "Low Risk", "blue"), (1, "High Risk", "red")]:
        risk_data = df[df["pred_label"] == pred_val]
        
        if risk_data.empty:
            continue
            
        n_samples = len(risk_data)
        events = risk_data[event_col].sum()
        label = f"{risk_label} (n={n_samples}, events={events})"
        
        kmf.fit(
            durations=risk_data[time_col],
            event_observed=risk_data[event_col],
            label=label
        )
        kmf.plot(ax=ax, color=color)
    
    # Log-rank test
    low_risk = df[df["pred_label"] == 0]
    high_risk = df[df["pred_label"] == 1]
    
    if not low_risk.empty and not high_risk.empty:
        lr_test = logrank_test(
            low_risk[time_col], high_risk[time_col],
            low_risk[event_col], high_risk[event_col]
        )
        ax.text(0.95, 0.05, f"Log-rank p = {lr_test.p_value:.4f}", 
               transform=ax.transAxes, ha="right", va="bottom",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_title("Cox Model Survival Analysis: Low Risk vs High Risk")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival Probability")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.close()


# Run Cox model evaluation
print("\n=== Running Cox Proportional Hazards Model Evaluation ===")
print("This will take several minutes for 50 iterations...")

# Run with 50 iterations
results_cox, summary_cox = multiple_random_splits_cox(clean_df, N=50)

# Save results to Excel
summary_cox.to_excel('/workspace/cox_results.xlsx', index=True, index_label='Model/Metric')
print("\nResults saved to cox_results.xlsx")

# Example inference for visualization
print("\n=== Example Cox Model Inference ===")
# Define features for inference
inference_features = ["Age by decade", "BMI", "AF", "Beta Blocker", "CrCl>45", 
                     "LVEF", "QTc", "NYHA>2", "CRT", "AAD", "Significant LGE"]

gray_features = ["Age by decade", "BMI", "AF", "Beta Blocker", "CrCl>45", "QTc", "CRT", "AAD"]
red_features = ["LVEF", "NYHA>2"]

# Run inference
cox_inference_df = sex_specific_cox_inference(
    train_df=train_df,
    test_df=test_df,
    features=inference_features,
    label_col="VT/VF/SCD",
    survival_df=survival_df,
    seed=42,
    gray_features=gray_features,
    red_features=red_features
)