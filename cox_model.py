import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    cross_val_predict
)
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    make_scorer,
    f1_score,
    average_precision_score,
    precision_recall_curve,
    accuracy_score,
    confusion_matrix,
    recall_score,
    c_index
)
import sklearn.neighbors._base
from scipy.stats import randint
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
import sys

sys.modules["sklearn.neighbors.base"] = sklearn.neighbors._base
from missingpy import MissForest

pd.set_option("future.no_silent_downcasting", True)
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=FutureWarning)


def CG_equation(age, weight, female, serum_creatinine):
    """Cockcroft-Gault Equation."""
    constant = 0.85 if female else 1.0
    return ((140 - age) * weight * constant) / (72 * serum_creatinine)

# ICD SURVIVAL:
# PE: ICD IMPLANT -> APPROPRIATE ICD SHOCK
# SE: ICD IMPLANT -> DEATH
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

# NO ICD SURVIVAL:
# PE: MRI -> VT/VF/SCD
# SE: MRI -> DEATH
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

# MERGED SURVIVAL
survival_df = pd.concat([icd_survival, no_icd_survival], ignore_index=True)

# ICD & NO ICD FEATURES
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

# Missing percentage
missing_pct = df.isnull().sum() / len(df) * 100
print(missing_pct)


def impute_misforest(X, random_seed):
    """
    Non-parametric iterative imputation method based on random forest.
    """
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

# Additional
clean_df["Age by decade"] = df["Age at CMR"] // 10
clean_df["CrCl>45"] = (
    clean_df["Cockcroft-Gault Creatinine Clearance (mL/min)"] > 45
).astype(int)
clean_df["NYHA>2"] = (clean_df["NYHA Class"] > 2).astype(int)
clean_df["Significant LGE"] = (clean_df["LGE Burden 5SD"] > 2).astype(int)

# Distribution of sex
print("\nDistribution of Sex")
print(clean_df["Female"].value_counts())

# Distribution of true label
print("\nDistribution of Arrhythmia")
print(clean_df["VT/VF/SCD"].value_counts())

# Proportion in ICD population that follows the rule-based guideline
icd_df = clean_df[clean_df["ICD"] == 1]
cond = (icd_df["NYHA Class"] >= 2) & (icd_df["LVEF"] <= 35)
pct = cond.sum() / len(icd_df) * 100
print(
    f"\nProportion in ICD population that follows the rule-based guideline: {pct:.2f}%"
)

df = clean_df.copy()
stratify_column = df['Female'].astype(str) + '_' + df['VT/VF/SCD'].astype(str)+ '_' + df['ICD'].astype(str)

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=stratify_column,
    random_state=100
)

print("Overall Female proportion:", df['Female'].mean())
print("Train Female proportion:", train_df['Female'].mean())
print("Test Female proportion:", test_df['Female'].mean())

print("Overall VT proportion:", df['VT/VF/SCD'].mean())
print("Train VT proportion:", train_df['VT/VF/SCD'].mean())
print("Test VT proportion:", test_df['VT/VF/SCD'].mean())

print("Overall ICD proportion:", df['ICD'].mean())
print("Train ICD proportion:", train_df['ICD'].mean())
print("Test ICD proportion:", test_df['ICD'].mean())


def prepare_survival_data(df, features, time_col, event_col):
    """Prepare data for Cox model training."""
    # Create survival dataframe
    survival_data = df[features + [time_col, event_col]].copy()
    
    # Ensure positive survival times
    survival_data[time_col] = np.maximum(survival_data[time_col], 0.01)
    
    return survival_data


def cox_evaluate(X_train, y_train_df, X_test, y_test_df, feat_names, survival_df, 
                 random_state=None, visualize_importance=False, gray_features=None, red_features=None):
    """Train Cox Proportional Hazard model and return risk predictions.
    
    Args:
        X_train: Training features
        y_train_df: Training labels with MRN and Female columns
        X_test: Test features  
        y_test_df: Test labels with MRN and Female columns
        feat_names: Feature names
        survival_df: Survival data with MRN, PE_Time, PE, SE_Time, SE columns
        random_state: Random seed
        visualize_importance: Whether to plot feature importance
        gray_features: Features to color gray in importance plot
        red_features: Features to color red in importance plot
    """
    # Merge training data with survival information
    train_with_mrn = X_train.copy()
    train_with_mrn['MRN'] = y_train_df['MRN'].values
    train_survival = train_with_mrn.merge(survival_df, on='MRN', how='inner')
    
    if train_survival.empty:
        print("Warning: No survival data found for training samples")
        # Return dummy predictions
        y_pred = np.zeros(len(X_test))
        y_prob = np.full(len(X_test), 0.5)
        return y_pred, y_prob
    
    # Prepare survival data for Cox model (using primary endpoint)
    cox_data = train_survival[feat_names + ['PE_Time', 'PE']].copy()
    cox_data['PE_Time'] = np.maximum(cox_data['PE_Time'], 0.01)  # Ensure positive times
    
    # Train Cox model
    cph = CoxPHFitter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph.fit(cox_data, duration_col='PE_Time', event_col='PE')
    except Exception as e:
        print(f"Cox model fitting failed: {e}")
        # Return dummy predictions
        y_pred = np.zeros(len(X_test))
        y_prob = np.full(len(X_test), 0.5)
        return y_pred, y_prob
    
    # Get partial hazards (risk scores) for test set
    test_risk_scores = cph.predict_partial_hazard(X_test[feat_names])
    
    # Convert risk scores to probabilities using sigmoid transformation
    y_prob = 1 / (1 + np.exp(-np.log(test_risk_scores)))
    y_prob = np.clip(y_prob, 0.001, 0.999)  # Avoid extreme values
    
    # Determine optimal threshold using training data cross-validation
    # Get risk scores for training data
    train_risk_scores = cph.predict_partial_hazard(cox_data[feat_names])
    train_probs = 1 / (1 + np.exp(-np.log(train_risk_scores)))
    train_probs = np.clip(train_probs, 0.001, 0.999)
    
    # Find best threshold using F1 score
    threshold = find_best_threshold(cox_data['PE'].values, train_probs)
    y_pred = (y_prob >= threshold).astype(int)
    
    # Feature importance visualization
    if visualize_importance:
        plot_cox_feature_importance(cph, feat_names, "Cox Model Feature Importances", 
                                   gray_features, red_features)
    
    return y_pred, y_prob


def plot_cox_feature_importance(cph, features, title, gray_features=None, red_features=None):
    """Plot Cox model coefficients as feature importance."""
    if cph.params_ is None or cph.params_.empty:
        print("No coefficients available for plotting")
        return
        
    # Get coefficients (log hazard ratios)
    coeffs = cph.params_.values
    abs_coeffs = np.abs(coeffs)
    
    # Sort by absolute coefficient values
    idx = np.argsort(abs_coeffs)[::-1]
    sorted_features = [features[i] for i in idx]
    
    # Color scheme
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
        # Default coloring scheme
        colors = [
            "red" if f in {"LVEF", "NYHA Class", "NYHA>2"}
            else "gold" if f in {"Significant LGE", "LGE Burden 5SD"} 
            else "lightgray"
            for f in sorted_features
        ]
    
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(sorted_features)), abs_coeffs[idx], color=colors)
    plt.xticks(range(len(sorted_features)), sorted_features, rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("|Coefficient|")
    plt.title(title)
    
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
        plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()


def find_best_threshold(y_true, y_scores, beta=2.0):
    """Find the probability threshold that maximizes the F-beta score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    fbeta_scores = (1 + beta**2) * precisions * recalls / (beta**2 * precisions + recalls + 1e-8)
    best_idx = np.nanargmax(fbeta_scores[:-1])
    return thresholds[best_idx]


def compute_sensitivity_specificity(y_true, y_pred):
    """Compute sensitivity and specificity from binary predictions."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return sensitivity, specificity


def incidence_rate(df, pred_col, label_col):
    """Compute incidence rate for males and females."""
    def rate(sub):
        n_pred = (sub[pred_col] == 1).sum()
        n_true = (sub[label_col] == 1).sum()
        return n_true / n_pred if n_pred > 0 else np.nan

    male_rate = rate(df[df["Female"] == 0])
    female_rate = rate(df[df["Female"] == 1])
    return male_rate, female_rate


def evaluate_model_performance(y_true, y_pred, y_prob, mask_m, mask_f, overall_mask=None):
    """Evaluate model performance for overall, male, and female subsets."""
    # Overall performance (optionally on a subset)
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
    sens, spec = compute_sensitivity_specificity(y_true_overall, y_pred_overall)
    
    # Male subset
    y_true_m = y_true[mask_m]
    y_pred_m = y_pred[mask_m]
    y_prob_m = y_prob[mask_m]
    
    male_acc = accuracy_score(y_true_m, y_pred_m) if len(y_true_m) > 0 else np.nan
    male_auc = roc_auc_score(y_true_m, y_prob_m) if len(y_true_m) > 1 and len(np.unique(y_true_m)) > 1 else np.nan
    male_f1 = f1_score(y_true_m, y_pred_m) if len(y_true_m) > 0 else np.nan
    male_sens, male_spec = compute_sensitivity_specificity(y_true_m, y_pred_m) if len(y_true_m) > 0 else (np.nan, np.nan)
    
    # Female subset
    y_true_f = y_true[mask_f]
    y_pred_f = y_pred[mask_f]
    y_prob_f = y_prob[mask_f]
    
    female_acc = accuracy_score(y_true_f, y_pred_f) if len(y_true_f) > 0 else np.nan
    female_auc = roc_auc_score(y_true_f, y_prob_f) if len(y_true_f) > 1 and len(np.unique(y_true_f)) > 1 else np.nan
    female_f1 = f1_score(y_true_f, y_pred_f) if len(y_true_f) > 0 else np.nan
    female_sens, female_spec = compute_sensitivity_specificity(y_true_f, y_pred_f) if len(y_true_f) > 0 else (np.nan, np.nan)
    
    return {
        'accuracy': acc, 'auc': auc, 'f1': f1, 'sensitivity': sens, 'specificity': spec,
        'male_accuracy': male_acc, 'male_auc': male_auc, 'male_f1': male_f1, 
        'male_sensitivity': male_sens, 'male_specificity': male_spec,
        'female_accuracy': female_acc, 'female_auc': female_auc, 'female_f1': female_f1,
        'female_sensitivity': female_sens, 'female_specificity': female_spec
    }


def create_undersampled_dataset(train_df, label_col, random_state, pos_to_neg_ratio=1.0):
    """Create per-sex undersampled dataset with desired positive:negative ratio (default 1:1)."""
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


def run_guideline_model(X_test, y_test, label_col):
    """Run guideline-based model using NYHA Class and LVEF."""
    pred = ((X_test["NYHA Class"] >= 2) & (X_test["LVEF"] <= 35)).astype(int).values
    y_true = y_test[label_col].values
    
    # Create evaluation dataframe for incidence rate calculation
    eval_df = y_test.reset_index(drop=True).copy()
    eval_df["pred"] = pred
    m_rate, f_rate = incidence_rate(eval_df, "pred", label_col)
    
    return pred, np.full_like(pred, 0.5, dtype=float), m_rate, f_rate


def run_sex_specific_cox_models(train_m, train_f, test_m, test_f, features, label_col, survival_df, random_state):
    """Train and evaluate sex-specific Cox models."""
    results = {}
    
    # Train male model
    if not train_m.empty and not test_m.empty:
        pred_m, prob_m = cox_evaluate(
            train_m[features], train_m[["MRN", label_col, "Female"]], 
            test_m[features], test_m[["MRN", label_col, "Female"]], 
            features, survival_df, random_state
        )
        results['male'] = {'pred': pred_m, 'prob': prob_m}
    
    # Train female model
    if not train_f.empty and not test_f.empty:
        pred_f, prob_f = cox_evaluate(
            train_f[features], train_f[["MRN", label_col, "Female"]], 
            test_f[features], test_f[["MRN", label_col, "Female"]], 
            features, survival_df, random_state
        )
        results['female'] = {'pred': pred_f, 'prob': prob_f}
    
    return results


def multiple_random_splits_cox(df, survival_df, N=50, label="VT/VF/SCD"):
    """
    Cox Proportional Hazard model version of multiple random splits evaluation.
    
    Replaces Random Forest models with Cox PH models for all feature selection methods.
    Implements sex-agnostic and sex-specific training with undersampling.
    Uses 70% train, 30% validation splits as requested.
    
    Args:
        df: Training dataframe
        survival_df: Survival data with MRN, PE_Time, PE, SE_Time, SE columns
        N: Number of random splits to perform (default: 50)
        label: Target variable name (default: "VT/VF/SCD")
    
    Returns:
        results: Dictionary with detailed results for each model
        summary: Formatted summary table with confidence intervals
    """
    
    # Feature definitions
    feature_sets = {
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
    
    # Model configurations
    model_configs = [
        {'name': 'Guideline', 'features': 'guideline', 'type': 'rule_based'},
        {'name': 'Cox Guideline', 'features': 'guideline', 'type': 'cox'},
        {'name': 'Benchmark Sex-agnostic', 'features': 'benchmark', 'type': 'cox'},
        {'name': 'Benchmark Sex-agnostic (undersampled)', 'features': 'benchmark', 'type': 'cox_undersampled'},
        {'name': 'Benchmark Male', 'features': 'benchmark', 'type': 'male_only'},
        {'name': 'Benchmark Female', 'features': 'benchmark', 'type': 'female_only'},
        {'name': 'Benchmark Sex-specific', 'features': 'benchmark', 'type': 'sex_specific'},
        {'name': 'Proposed Sex-agnostic', 'features': 'proposed', 'type': 'cox'},
        {'name': 'Proposed Sex-agnostic (undersampled)', 'features': 'proposed', 'type': 'cox_undersampled'},
        {'name': 'Proposed Male', 'features': 'proposed', 'type': 'male_only'},
        {'name': 'Proposed Female', 'features': 'proposed', 'type': 'female_only'},
        {'name': 'Proposed Sex-specific', 'features': 'proposed', 'type': 'sex_specific'},
        {'name': 'Real Proposed Sex-agnostic', 'features': 'real_proposed', 'type': 'cox'},
        {'name': 'Real Proposed Sex-agnostic (undersampled)', 'features': 'real_proposed', 'type': 'cox_undersampled'},
        {'name': 'Real Proposed Male', 'features': 'real_proposed', 'type': 'male_only'},
        {'name': 'Real Proposed Female', 'features': 'real_proposed', 'type': 'female_only'},
        {'name': 'Real Proposed Sex-specific', 'features': 'real_proposed', 'type': 'sex_specific'}
    ]
    
    # Metrics to track
    metrics = ['accuracy', 'auc', 'f1', 'sensitivity', 'specificity', 
               'male_accuracy', 'male_auc', 'male_f1', 'male_sensitivity', 'male_specificity',
               'female_accuracy', 'female_auc', 'female_f1', 'female_sensitivity', 'female_specificity',
               'male_rate', 'female_rate']
    
    # Initialize results storage
    results = {config['name']: {met: [] for met in metrics} for config in model_configs}
    
    print(f"Starting Cox PH model evaluation with {N} random splits (70% train, 30% validation)")
    
    for seed in range(N):
        print(f"Running split #{seed+1}/{N}")
        
        # Split data (70% train, 30% validation as requested)
        train_df_split, test_df_split = train_test_split(df, test_size=0.3, random_state=seed, stratify=df[label])
        tr_m = train_df_split[train_df_split["Female"] == 0]
        tr_f = train_df_split[train_df_split["Female"] == 1]
        te_m = test_df_split[test_df_split["Female"] == 0]
        te_f = test_df_split[test_df_split["Female"] == 1]
        
        # Create undersampled dataset once per split
        us_train_df = create_undersampled_dataset(train_df_split, label, seed)
        
        # Create masks for gender subsets
        mask_m = test_df_split["Female"].values == 0
        mask_f = test_df_split["Female"].values == 1
        y_true = test_df_split[label].values
        
        # Evaluate each model configuration
        for config in model_configs:
            model_name = config['name']
            feature_set = feature_sets[config['features']]
            model_type = config['type']
            
            # Default: evaluate overall on full test and use true male/female masks
            mask_m_eval = mask_m
            mask_f_eval = mask_f
            overall_mask_override = None
            
            if model_type == 'rule_based':
                # Guideline model
                pred, prob, m_rate, f_rate = run_guideline_model(
                    test_df_split[feature_set], test_df_split[[label, "Female"]], label
                )
                
            elif model_type == 'cox':
                # Standard Cox model
                pred, prob = cox_evaluate(
                    train_df_split[feature_set], train_df_split[["MRN", label, "Female"]],
                    test_df_split[feature_set], test_df_split[["MRN", label, "Female"]],
                    feature_set, survival_df, seed
                )
                eval_df = test_df_split[[label, "Female"]].reset_index(drop=True).copy()
                eval_df["pred"] = pred
                m_rate, f_rate = incidence_rate(eval_df, "pred", label)
                
            elif model_type == 'cox_undersampled':
                # Undersampled Cox model
                pred, prob = cox_evaluate(
                    us_train_df[feature_set], us_train_df[["MRN", label, "Female"]],
                    test_df_split[feature_set], test_df_split[["MRN", label, "Female"]],
                    feature_set, survival_df, seed
                )
                eval_df = test_df_split[[label, "Female"]].reset_index(drop=True).copy()
                eval_df["pred"] = pred
                m_rate, f_rate = incidence_rate(eval_df, "pred", label)
                
            elif model_type == 'male_only':
                # Male-only Cox model
                if not tr_m.empty and not te_m.empty:
                    pred, prob = cox_evaluate(
                        tr_m[feature_set], tr_m[["MRN", label, "Female"]],
                        te_m[feature_set], te_m[["MRN", label, "Female"]],
                        feature_set, survival_df, seed
                    )
                    # Create full test set predictions (only male predictions)
                    full_pred = np.zeros(len(test_df_split), dtype=int)
                    full_prob = np.zeros(len(test_df_split), dtype=float)
                    full_pred[mask_m] = pred
                    full_prob[mask_m] = prob
                    pred, prob = full_pred, full_prob
                    eval_df = test_df_split[[label, "Female"]].reset_index(drop=True).copy()
                    eval_df["pred"] = pred
                    m_rate, f_rate = incidence_rate(eval_df, "pred", label)
                    # Evaluate overall metrics only on the male subset
                    mask_f_eval = np.zeros_like(mask_f, dtype=bool)
                    overall_mask_override = mask_m
                else:
                    pred = np.zeros(len(test_df_split), dtype=int)
                    prob = np.zeros(len(test_df_split), dtype=float)
                    m_rate, f_rate = 0.0, 0.0
                    mask_f_eval = np.zeros_like(mask_f, dtype=bool)
                    overall_mask_override = mask_m
                
            elif model_type == 'female_only':
                # Female-only Cox model
                if not tr_f.empty and not te_f.empty:
                    pred, prob = cox_evaluate(
                        tr_f[feature_set], tr_f[["MRN", label, "Female"]],
                        te_f[feature_set], te_f[["MRN", label, "Female"]],
                        feature_set, survival_df, seed
                    )
                    # Create full test set predictions (only female predictions)
                    full_pred = np.zeros(len(test_df_split), dtype=int)
                    full_prob = np.zeros(len(test_df_split), dtype=float)
                    full_pred[mask_f] = pred
                    full_prob[mask_f] = prob
                    pred, prob = full_pred, full_prob
                    eval_df = test_df_split[[label, "Female"]].reset_index(drop=True).copy()
                    eval_df["pred"] = pred
                    m_rate, f_rate = incidence_rate(eval_df, "pred", label)
                    # Evaluate overall metrics only on the female subset
                    mask_m_eval = np.zeros_like(mask_m, dtype=bool)
                    overall_mask_override = mask_f
                else:
                    pred = np.zeros(len(test_df_split), dtype=int)
                    prob = np.zeros(len(test_df_split), dtype=float)
                    m_rate, f_rate = 0.0, 0.0
                    mask_m_eval = np.zeros_like(mask_m, dtype=bool)
                    overall_mask_override = mask_f
                
            elif model_type == 'sex_specific':
                # Sex-specific Cox models
                sex_results = run_sex_specific_cox_models(
                    tr_m, tr_f, te_m, te_f, feature_set, label, survival_df, seed
                )
                
                # Combine predictions
                combined_pred = np.empty(len(test_df_split), dtype=int)
                combined_prob = np.empty(len(test_df_split), dtype=float)
                
                if 'male' in sex_results:
                    combined_pred[mask_m] = sex_results['male']['pred']
                    combined_prob[mask_m] = sex_results['male']['prob']
                if 'female' in sex_results:
                    combined_pred[mask_f] = sex_results['female']['pred']
                    combined_prob[mask_f] = sex_results['female']['prob']
                
                pred, prob = combined_pred, combined_prob
                eval_df = test_df_split[[label, "Female"]].reset_index(drop=True).copy()
                eval_df["pred"] = pred
                m_rate, f_rate = incidence_rate(eval_df, "pred", label)
            
            # Evaluate performance
            perf_metrics = evaluate_model_performance(y_true, pred, prob, mask_m_eval, mask_f_eval, overall_mask=overall_mask_override)
            
            # Store results
            for metric, value in perf_metrics.items():
                results[model_name][metric].append(value)
            
            # Store incidence rates
            results[model_name]['male_rate'].append(m_rate)
            results[model_name]['female_rate'].append(f_rate)
    
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


# Run the Cox model evaluation
print("Starting Cox Proportional Hazard model evaluation...")
cox_results, cox_summary = multiple_random_splits_cox(train_df, survival_df, N=50)

# Save results to Excel
output_path = '/home/sunx/data/aiiih/projects/sunx/projects/ICD_sex_diff/cox_results.xlsx'
cox_summary.to_excel(output_path, index=True, index_label='Model')
print(f"Cox model results saved to: {output_path}")


def train_cox_sex_specific_model(X_train, y_train, features, survival_df, seed):
    """Train a sex-specific Cox model."""
    # Merge training data with survival information
    train_with_mrn = X_train.copy()
    train_with_mrn['MRN'] = y_train.index if hasattr(y_train, 'index') else range(len(y_train))
    
    # If y_train is a Series, convert to DataFrame
    if hasattr(y_train, 'name'):
        train_with_mrn[y_train.name] = y_train.values
    else:
        train_with_mrn['target'] = y_train
    
    train_survival = train_with_mrn.merge(survival_df, on='MRN', how='inner')
    
    if train_survival.empty:
        print("Warning: No survival data found for training samples")
        return None, 0.5
    
    # Prepare survival data for Cox model
    cox_data = train_survival[features + ['PE_Time', 'PE']].copy()
    cox_data['PE_Time'] = np.maximum(cox_data['PE_Time'], 0.01)
    
    # Train Cox model
    cph = CoxPHFitter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph.fit(cox_data, duration_col='PE_Time', event_col='PE')
        
        # Get risk scores for training data to determine threshold
        train_risk_scores = cph.predict_partial_hazard(cox_data[features])
        train_probs = 1 / (1 + np.exp(-np.log(train_risk_scores)))
        train_probs = np.clip(train_probs, 0.001, 0.999)
        
        # Find best threshold
        best_threshold = find_best_threshold(cox_data['PE'].values, train_probs)
        
        return cph, best_threshold
        
    except Exception as e:
        print(f"Cox model fitting failed: {e}")
        return None, 0.5


def train_cox_sex_agnostic_model(train_df, features, label_col, survival_df, seed, use_undersampling=True):
    """Train a sex-agnostic Cox model using undersampling for fair comparison."""
    if use_undersampling:
        train_data = create_undersampled_dataset(train_df, label_col, seed)
        print(f"Using undersampled training data: {len(train_data)} samples")
    else:
        train_data = train_df.copy()
        print(f"Using full training data: {len(train_data)} samples")
    
    # Merge with survival data
    train_survival = train_data.merge(survival_df, on='MRN', how='inner')
    
    if train_survival.empty:
        print("Warning: No survival data found for training samples")
        return None, 0.5
    
    # Prepare Cox data
    cox_data = train_survival[features + ['PE_Time', 'PE']].copy()
    cox_data['PE_Time'] = np.maximum(cox_data['PE_Time'], 0.01)
    
    # Train Cox model
    cph = CoxPHFitter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph.fit(cox_data, duration_col='PE_Time', event_col='PE')
        
        # Get risk scores for training data to determine threshold
        train_risk_scores = cph.predict_partial_hazard(cox_data[features])
        train_probs = 1 / (1 + np.exp(-np.log(train_risk_scores)))
        train_probs = np.clip(train_probs, 0.001, 0.999)
        
        # Find best threshold
        best_threshold = find_best_threshold(cox_data['PE'].values, train_probs)
        
        return cph, best_threshold
        
    except Exception as e:
        print(f"Cox model fitting failed: {e}")
        return None, 0.5


def plot_km_by_gender_and_risk(merged_df, gender_col, risk_col, time_col, event_col, title_prefix):
    """Plot Kaplan-Meier curves separately for each gender and risk group."""
    genders = merged_df[gender_col].unique()
    
    for gender in genders:
        gender_data = merged_df[merged_df[gender_col] == gender]
        gender_label = "Female" if gender == 1 else "Male"
        
        if gender_data.empty:
            continue
            
        # Get risk groups for this gender
        risk_groups = gender_data[risk_col].unique()
        
        if len(risk_groups) < 2:
            continue
            
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
        
        for ax, (ep_name, ep_time_col, ep_event_col) in zip(axes, [
            ("Primary Endpoint", "PE_Time", "PE"),
            ("Secondary Endpoint", "SE_Time", "SE")
        ]):
            kmf = KaplanMeierFitter()
            
            for risk_group in sorted(risk_groups):
                mask = (gender_data[gender_col] == gender) & (gender_data[risk_col] == risk_group)
                risk_data = gender_data[mask]
                
                if risk_data.empty:
                    continue
                    
                n_risk = len(risk_data)
                events_risk = risk_data[ep_event_col].sum()
                risk_label = f"{'High' if risk_group == 1 else 'Low'} Risk (n={n_risk}, events={events_risk})"
                
                kmf.fit(
                    durations=risk_data[ep_time_col],
                    event_observed=risk_data[ep_event_col],
                    label=risk_label
                )
                kmf.plot(ax=ax)
            
            # Perform log-rank test if we have both risk groups
            if len(risk_groups) == 2:
                low_mask = (gender_data[gender_col] == gender) & (gender_data[risk_col] == 0)
                high_mask = (gender_data[gender_col] == gender) & (gender_data[risk_col] == 1)
                
                if low_mask.sum() > 0 and high_mask.sum() > 0:
                    lr = logrank_test(
                        gender_data.loc[low_mask, ep_time_col],
                        gender_data.loc[high_mask, ep_time_col],
                        gender_data.loc[low_mask, ep_event_col],
                        gender_data.loc[high_mask, ep_event_col]
                    )
                    ax.text(0.95, 0.05, f"Log-rank p = {lr.p_value:.5f}", 
                            transform=ax.transAxes, ha="right", va="bottom")
            
            ax.set_title(f"{ep_name} by Risk Group - {gender_label}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Survival Probability")
            ax.legend()
        
        plt.suptitle(f"{title_prefix} - {gender_label}")
        plt.tight_layout()
        plt.show()
        plt.close()


def analyze_survival_by_four_groups(merged_df):
    """Analyze survival outcomes by 4 groups: Male-Pred0, Male-Pred1, Female-Pred0, Female-Pred1."""
    print("\n=== Survival Analysis by Gender and Predicted Label ===")
    
    # Define the 4 groups
    groups = [
        (0, 0, "Male-Pred0"),
        (0, 1, "Male-Pred1"), 
        (1, 0, "Female-Pred0"),
        (1, 1, "Female-Pred1")
    ]
    
    group_data = {}
    
    for gender_val, pred_val, group_name in groups:
        mask = (merged_df["Female"] == gender_val) & (merged_df["pred_label"] == pred_val)
        group_df = merged_df[mask]
        group_data[group_name] = group_df
        
        if not group_df.empty:
            n_samples = len(group_df)
            
            # Primary endpoint analysis
            pe_events = group_df["PE"].sum()
            pe_rate = pe_events / n_samples if n_samples > 0 else 0
            
            # Secondary endpoint analysis  
            se_events = group_df["SE"].sum()
            se_rate = se_events / n_samples if n_samples > 0 else 0
            
            print(f"\n{group_name} (n={n_samples}):")
            print(f"  Primary Endpoint: {pe_events} events, Incidence Rate: {pe_rate:.4f}")
            print(f"  Secondary Endpoint: {se_events} events, Incidence Rate: {se_rate:.4f}")
        else:
            print(f"\n{group_name}: No samples")
    
    return group_data


def plot_km_curves_merged(merged_df):
    """Plot Kaplan-Meier curves with merged survival analysis: Low Risk vs High Risk (all genders combined)."""
    
    # Create 1x2 subplot layout for PE and SE
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define endpoints
    endpoints = [
        ("Primary Endpoint", "PE_Time", "PE"),
        ("Secondary Endpoint", "SE_Time", "SE")
    ]
    
    # Iterate through each endpoint
    for ax_idx, (ep_name, ep_time_col, ep_event_col) in enumerate(endpoints):
        ax = axes[ax_idx]
        kmf = KaplanMeierFitter()
        
        # Plot low risk (pred_label = 0) and high risk (pred_label = 1) for all patients
        for pred_val, risk_label, color in [(0, "Low Risk", "blue"), (1, "High Risk", "red")]:
            risk_data = merged_df[merged_df["pred_label"] == pred_val]
            
            if risk_data.empty:
                continue
                
            n_samples = len(risk_data)
            events = risk_data[ep_event_col].sum()
            label = f"{risk_label} (n={n_samples}, events={events})"
            
            kmf.fit(
                durations=risk_data[ep_time_col],
                event_observed=risk_data[ep_event_col],
                label=label
            )
            kmf.plot(ax=ax, color=color)
        
        # Perform log-rank test between low and high risk groups
        low_risk_data = merged_df[merged_df["pred_label"] == 0]
        high_risk_data = merged_df[merged_df["pred_label"] == 1]
        
        if not low_risk_data.empty and not high_risk_data.empty:
            lr_test = logrank_test(
                low_risk_data[ep_time_col], high_risk_data[ep_time_col],
                low_risk_data[ep_event_col], high_risk_data[ep_event_col]
            )
            # Add log-rank test result to bottom right corner
            ax.text(0.95, 0.05, f"Log-rank p = {lr_test.p_value:.4f}", 
                   transform=ax.transAxes, ha="right", va="bottom",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_title(f"{ep_name} - Combined Analysis")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Survival Probability")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Cox Model Survival Analysis: Low Risk vs High Risk")
    plt.tight_layout()
    plt.show()
    plt.close()


def cox_sex_specific_model_inference(train_df, test_df, features, labels, survival_df, seed, gray_features=None, red_features=None):
    """
    Cox sex-specific model inference function for survival analysis.
    """
    train = train_df.copy()
    test = test_df.copy()
    df = test.copy()

    # Separate training data by gender
    train_m = train[train["Female"] == 0].copy()
    train_f = train[train["Female"] == 1].copy()
    test_m = test[test["Female"] == 0].copy()
    test_f = test[test["Female"] == 1].copy()

    # Train gender-specific Cox models
    print("Training Male Cox Model...")
    best_male, best_thr_m = train_cox_sex_specific_model(
        train_m[features], train_m[labels], features, survival_df, seed
    )
    
    print("Training Female Cox Model...")
    best_female, best_thr_f = train_cox_sex_specific_model(
        train_f[features], train_f[labels], features, survival_df, seed
    )

    # Make predictions on test set
    if best_male is not None and not test_m.empty:
        risk_scores_m = best_male.predict_partial_hazard(test_m[features])
        prob_m = 1 / (1 + np.exp(-np.log(risk_scores_m)))
        prob_m = np.clip(prob_m, 0.001, 0.999)
        pred_m = (prob_m >= best_thr_m).astype(int)
        df.loc[df["Female"] == 0, "pred_label"] = pred_m
        df.loc[df["Female"] == 0, "pred_prob"] = prob_m
    
    if best_female is not None and not test_f.empty:
        risk_scores_f = best_female.predict_partial_hazard(test_f[features])
        prob_f = 1 / (1 + np.exp(-np.log(risk_scores_f)))
        prob_f = np.clip(prob_f, 0.001, 0.999)
        pred_f = (prob_f >= best_thr_f).astype(int)
        df.loc[df["Female"] == 1, "pred_label"] = pred_f
        df.loc[df["Female"] == 1, "pred_prob"] = prob_f

    # Feature importance visualization
    if best_male is not None:
        plot_cox_feature_importance(best_male, features, "Male Cox Model Feature Importances", gray_features, red_features)
    if best_female is not None:
        plot_cox_feature_importance(best_female, features, "Female Cox Model Feature Importances", gray_features, red_features)

    # Merge with survival data
    pred_labels = df[["MRN", "pred_label", "Female"]].drop_duplicates()
    merged_df = survival_df.merge(pred_labels, on="MRN", how="inner").drop_duplicates(subset=["MRN"])

    print(f"\n=== Cox Sex-Specific Model Summary ===")
    print(f"Total test samples: {len(df)}")
    print(f"Samples with survival data: {len(merged_df)}")
    
    # Calculate incidence rates for the 4 groups
    for gender_val, gender_name in [(0, "Male"), (1, "Female")]:
        gender_data = merged_df[merged_df["Female"] == gender_val]
        if not gender_data.empty:
            for pred_val in [0, 1]:
                group_data = gender_data[gender_data["pred_label"] == pred_val]
                if not group_data.empty:
                    pe_rate = group_data["PE"].sum() / len(group_data)
                    se_rate = group_data["SE"].sum() / len(group_data)
                    print(f"{gender_name}-Pred{pred_val}: PE rate = {pe_rate:.4f}, SE rate = {se_rate:.4f}")
    
    # Analyze and plot survival
    analyze_survival_by_four_groups(merged_df)
    plot_km_curves_merged(merged_df)

    return merged_df


def cox_sex_agnostic_model_inference(train_df, test_df, features, label_col, survival_df, seed, gray_features=None, red_features=None, use_undersampling=True):
    """
    Cox sex-agnostic model inference function.
    """
    test = test_df.copy()
    
    # Remove sex indicator for sex-agnostic training/inference
    used_features = [f for f in features if f != "Female"]
    
    # Train sex-agnostic Cox model
    print("Training Sex-Agnostic Cox Model...")
    best_model, best_threshold = train_cox_sex_agnostic_model(
        train_df, used_features, label_col, survival_df, seed, use_undersampling
    )
    print(f"Optimal probability threshold determined from training data: {best_threshold:.4f}")
    
    if best_model is None:
        print("Failed to train Cox model")
        return test
    
    # Make predictions on test set
    test_risk_scores = best_model.predict_partial_hazard(test[used_features])
    test_probs = 1 / (1 + np.exp(-np.log(test_risk_scores)))
    test_probs = np.clip(test_probs, 0.001, 0.999)
    
    print(f"Test probabilities – min: {test_probs.min():.4f}, max: {test_probs.max():.4f}, mean: {test_probs.mean():.4f}")
    test_preds = (test_probs >= best_threshold).astype(int)
    
    # Add predictions to test dataframe
    test["pred_label"] = test_preds
    test["pred_prob"] = test_probs
    
    # Feature importance visualization
    plot_cox_feature_importance(
        best_model, used_features, 
        "Sex-Agnostic Cox Model Feature Importances", 
        gray_features, red_features
    )
    
    # Merge with survival data
    pred_labels = test[["MRN", "pred_label", "Female"]].drop_duplicates()
    merged_df = survival_df.merge(pred_labels, on="MRN", how="inner").drop_duplicates(subset=["MRN"])
    
    print(f"\n=== Sex-Agnostic Cox Model Summary ===")
    print(f"Total test samples: {len(test)}")
    print(f"Samples with survival data: {len(merged_df)}")
    
    # Calculate overall prediction statistics
    n_high_risk = (merged_df["pred_label"] == 1).sum()
    n_low_risk = (merged_df["pred_label"] == 0).sum()
    print(f"High risk predictions: {n_high_risk}")
    print(f"Low risk predictions: {n_low_risk}")
    
    # Calculate incidence rates for the 4 groups (Male/Female × Low/High Risk)
    print(f"\n=== Incidence Rates by Gender and Risk ===")
    for gender_val, gender_name in [(0, "Male"), (1, "Female")]:
        gender_data = merged_df[merged_df["Female"] == gender_val]
        if not gender_data.empty:
            for pred_val, risk_name in [(0, "Low Risk"), (1, "High Risk")]:
                group_data = gender_data[gender_data["pred_label"] == pred_val]
                if not group_data.empty:
                    pe_rate = group_data["PE"].sum() / len(group_data)
                    se_rate = group_data["SE"].sum() / len(group_data)
                    print(f"{gender_name}-{risk_name}: PE rate = {pe_rate:.4f}, SE rate = {se_rate:.4f} (n={len(group_data)})")
    
    # Perform survival analysis
    analyze_survival_by_four_groups(merged_df)
    plot_km_curves_merged(merged_df)
    
    return merged_df


print("\nCox Proportional Hazard model evaluation completed!")
print("Key differences from Random Forest version:")
print("- Replaced all Random Forest models with Cox Proportional Hazard models")
print("- Uses survival time and event data for training")
print("- Converts hazard ratios to risk probabilities for binary classification")
print("- Feature importance based on Cox model coefficients")
print("- Maintains same evaluation framework with sex-agnostic/sex-specific models")
print("- Uses undersampling for fair comparison between model types")
print("- 50 random splits with 70% train, 30% validation")