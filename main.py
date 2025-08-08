import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
import seaborn as sns
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    make_scorer,
    f1_score,
    average_precision_score,
    precision_recall_curve,
    accuracy_score,
    confusion_matrix,
)
import sklearn.neighbors._base
from scipy.stats import randint
from tableone import TableOne
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import sys

sys.modules["sklearn.neighbors.base"] = sklearn.neighbors._base
from missingpy import MissForest
from itertools import combinations

pd.set_option("future.no_silent_downcasting", True)
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=FutureWarning)


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from tableone import TableOne
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import ceil
from scipy.stats import randint

def CG_equation(age, weight, female, serum_creatinine):
    """Cockcroft-Gault Equation."""
    constant = 0.85 if female else 1.0
    return ((140 - age) * weight * constant) / (72 * serum_creatinine)

# Load data
survival_df = pd.read_excel(
    "/home/sunx/data/aiiih/projects/sunx/projects/ICD_sex_diff/df.xlsx",
)
survival_df["PE_Time"] = survival_df.apply(
    lambda row: (
        row["Time from ICD Implant to Primary Endpoint (in days)"]
        if row["Was Primary Endpoint Reached? (Appropriate ICD Therapy)"] == 1
        else row["Time from ICD Implant to Last Cardiology Encounter (in days)"]
    ),
    axis=1,
)
survival_df["SE_Time"] = survival_df.apply(
    lambda row: (
        row["Time from ICD Implant to Secondary Endpoint (in days)"]
        if row["Was Secondary Endpoint Reached?"] == 1
        else row["Time from ICD Implant to Last Cardiology Encounter (in days)"]
    ),
    axis=1,
)
survival_df = survival_df[
    [
        "MRN",
        "Was Primary Endpoint Reached? (Appropriate ICD Therapy)",
        "PE_Time",
        "Was Secondary Endpoint Reached?",
        "SE_Time",
    ]
]
with_icd = pd.read_excel(
    "/home/sunx/data/aiiih/projects/sunx/projects/ICD_sex_diff/NICM Arrhythmia Cohort for Xiaotan Final.xlsx",
    sheet_name="ICD",
)
with_icd["ICD"] = 1
without_icd = pd.read_excel(
    "/home/sunx/data/aiiih/projects/sunx/projects/ICD_sex_diff/NICM Arrhythmia Cohort for Xiaotan Final.xlsx",
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

from sklearn.model_selection import train_test_split

df = clean_df.copy()
stratify_column = df['Female'].astype(str) + '_' + df['VT/VF/SCD'].astype(str)

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

def find_best_threshold(y_true, y_scores):
    """
    Find the probability threshold that maximizes the F1 score
    based on the precision-recall curve.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.nanargmax(f1_scores[:-1])
    return thresholds[best_idx]


def compute_sensitivity_specificity(y_true, y_pred):
    """
    Compute sensitivity (true positive rate) and specificity (true negative rate)
    from binary predictions.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return sensitivity, specificity


def incidence_rate(df, pred_col, label_col):
    """
    Compute the incidence rate defined as:
      #actually developed arrhythmia/#model predicted to develop arrhythmia,
    separately for males (Female==0) and females (Female==1).
    """

    def rate(sub):
        n_pred = (sub[pred_col] == 1).sum()
        n_true = (sub[label_col] == 1).sum()
        return n_true / n_pred if n_pred > 0 else np.nan

    male_rate = rate(df[df["Female"] == 0])
    female_rate = rate(df[df["Female"] == 1])
    return male_rate, female_rate


def rf_evaluate(
    X_train,
    y_train_df,
    X_test,
    y_test_df,
    feat_names,
    random_state=None,
    visualize_importance=False,
):
    """
    Train a RandomForest with randomized search optimizing average precision,
    then predict on X_test and return discrete predictions and probabilities.
    Threshold is now determined on the training set to avoid data leakage.
    """
    y_train = y_train_df["VT/VF/SCD"]
    y_test = y_test_df["VT/VF/SCD"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    param_dist = {
        "n_estimators": randint(100, 500),
        "max_depth": [None] + list(range(5, 26, 5)),
        "min_samples_split": randint(2, 11),
        "min_samples_leaf": randint(1, 5),
        "max_features": ["sqrt", "log2", None],
    }
    base_clf = RandomForestClassifier(
        random_state=random_state, n_jobs=-1, class_weight="balanced"
    )
    ap_scorer = make_scorer(average_precision_score, needs_proba=True)
    search = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=param_dist,
        n_iter=50,
        scoring=ap_scorer,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
        error_score="raise",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X_train, y_train)
    best_model = search.best_estimator_
    if visualize_importance:
        importances = best_model.feature_importances_
        idx = np.argsort(importances)[::-1]
        # highlight LVEF and NYHA Class in red
        highlight = {"LVEF", "NYHA"}
        colors = ["red" if feat_names[i] in highlight else "lightgray" for i in idx]
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(feat_names)), importances[idx], color=colors)
        plt.xticks(range(len(feat_names)), [feat_names[i] for i in idx], rotation=90)
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.title("Feature Importances")
        plt.tight_layout()
        plt.show()
    
    y_train_prob = best_model.predict_proba(X_train)[:, 1]
    threshold = find_best_threshold(y_train, y_train_prob)
    
    y_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return y_pred, y_prob


def multiple_random_splits(df, N, label="VT/VF/SCD"):
    """
    Perform N random train/test splits, fit several models, and collect metrics.

    For each random seed:
      1) Rule-based model using NYHA Class and LVEF.
      2) Random forest on the two guideline features.
      3) RF on all features (sex-agnostic).
      4) RF on male-only data.
      5) RF on female-only data.
      6) RF on sex-specific: combine male/female RF predictions on full test set.

    Returns:
      - results: nested dict {model_name: {metric: [values over seeds]}}
      - summary_df: DataFrame of mean and 95% CI for each metric and model.
    """
    # Features
    guideline_features = ["NYHA Class", "LVEF"]
    benchmark_features = [
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
    ]
    proposed_features = benchmark_features + [
        "DM",
        "HTN",
        "HLP",
        "LVEDVi",
        "LV Mass Index",
        "RVEDVi",
        "RVEF",
        "LA EF",
        "LAVi",
        "MRF (%)",
        "Sphericity Index",
        "Relative Wall Thickness",
        "MV Annular Diameter",
        "ACEi/ARB/ARNi",
        "Aldosterone Antagonist",
    ]
    real_proposed_features = proposed_features[:]
    real_proposed_features.remove("NYHA>2")
    real_proposed_features.remove("Significant LGE")
    real_proposed_features.extend(["LGE Burden 5SD", "NYHA Class"])

    # Models
    model_names = [
        "Guideline",
        "RF Guideline",
        "Benchmark Sex-agnostic",
        "Benchmark Sex-agnostic (undersampled)",
        "Benchmark Male",
        "Benchmark Female",
        "Benchmark Sex-specific",
        "Proposed Sex-agnostic",
        "Proposed Sex-agnostic (undersampled)",
        "Proposed Male",
        "Proposed Female",
        "Proposed Sex-specific",
        "Real Proposed Sex-agnostic",
        "Real Proposed Sex-agnostic (undersampled)",
        "Real Proposed Male",
        "Real Proposed Female",
        "Real Proposed Sex-specific",
    ]
    # Metrics (expanded to include male and female specific)
    metrics = [
        "accuracy",
        "auc",
        "f1",
        "sensitivity",
        "specificity",
        "male_accuracy",
        "male_auc",
        "male_f1",
        "male_sensitivity",
        "male_specificity",
        "female_accuracy",
        "female_auc",
        "female_f1",
        "female_sensitivity",
        "female_specificity",
        "male_rate",
        "female_rate",
    ]
    # Initialize result storage
    results = {m: {met: [] for met in metrics} for m in model_names}

    for seed in range(N):
        print(f"Running split #{seed+1}")
        # 1) Split into train/test + male/female
        train_df, test_df = train_test_split(
            df, test_size=0.3, random_state=seed, stratify=df[label]
        )
        tr_m = train_df[train_df["Female"] == 0]
        tr_f = train_df[train_df["Female"] == 1]
        te_m = test_df[test_df["Female"] == 0]
        te_f = test_df[test_df["Female"] == 1]

        # Undersampled
        n_male = (train_df["Female"] == 0).sum()
        n_female = (train_df["Female"] == 1).sum()
        n_target = ceil((n_male + n_female) / 2)

        sampled_parts = []
        for sex_val in (0, 1):
            grp = train_df[train_df["Female"] == sex_val]
            pos = grp[grp[label] == 1]
            neg = grp[grp[label] == 0]

            pos_n_target = int(round(len(pos) / len(grp) * n_target))
            neg_n_target = n_target - pos_n_target

            replace_pos = pos_n_target > len(pos)
            replace_neg = neg_n_target > len(neg)

            samp_pos = pos.sample(
                n=pos_n_target, replace=replace_pos, random_state=seed
            )
            samp_neg = neg.sample(
                n=neg_n_target, replace=replace_neg, random_state=seed
            )

            sampled_parts.append(pd.concat([samp_pos, samp_neg]))

        us_train_df = (
            pd.concat(sampled_parts)
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )

        # Guideline
        X_tr_g = train_df[guideline_features]
        y_tr_g = train_df[[label, "Female"]]
        X_te_g = test_df[guideline_features]
        y_te_g = test_df[[label, "Female"]]

        # Benchmark
        X_tr_b = train_df[benchmark_features]
        y_tr_b = train_df[[label, "Female"]]
        X_te_b = test_df[benchmark_features]
        y_te_b = test_df[[label, "Female"]]

        X_tr_b_m, y_tr_b_m = tr_m[benchmark_features], tr_m[[label, "Female"]]
        X_tr_b_f, y_tr_b_f = tr_f[benchmark_features], tr_f[[label, "Female"]]
        X_te_b_m, y_te_b_m = te_m[benchmark_features], te_m[[label, "Female"]]
        X_te_b_f, y_te_b_f = te_f[benchmark_features], te_f[[label, "Female"]]

        X_tr_b_us = us_train_df[benchmark_features]
        y_tr_b_us = us_train_df[[label, "Female"]]

        # Proposed
        X_tr_p = train_df[proposed_features]
        y_tr_p = train_df[[label, "Female"]]
        X_te_p = test_df[proposed_features]
        y_te_p = test_df[[label, "Female"]]

        X_tr_p_m, y_tr_p_m = tr_m[proposed_features], tr_m[[label, "Female"]]
        X_tr_p_f, y_tr_p_f = tr_f[proposed_features], tr_f[[label, "Female"]]
        X_te_p_m, y_te_p_m = te_m[proposed_features], te_m[[label, "Female"]]
        X_te_p_f, y_te_p_f = te_f[proposed_features], te_f[[label, "Female"]]

        X_tr_p_us = us_train_df[proposed_features]
        y_tr_p_us = us_train_df[[label, "Female"]]

        # Real proposed
        X_tr_r = train_df[real_proposed_features]
        y_tr_r = train_df[[label, "Female"]]
        X_te_r = test_df[real_proposed_features]
        y_te_r = test_df[[label, "Female"]]

        X_tr_r_m, y_tr_r_m = tr_m[real_proposed_features], tr_m[[label, "Female"]]
        X_tr_r_f, y_tr_r_f = tr_f[real_proposed_features], tr_f[[label, "Female"]]
        X_te_r_m, y_te_r_m = te_m[real_proposed_features], te_m[[label, "Female"]]
        X_te_r_f, y_te_r_f = te_f[real_proposed_features], te_f[[label, "Female"]]

        X_tr_r_us = us_train_df[real_proposed_features]
        y_tr_r_us = us_train_df[[label, "Female"]]

        # --- Guideline --- #
        pred_g = (
            ((X_te_g["NYHA Class"] >= 2) & (X_te_g["LVEF"] <= 35)).astype(int).values
        )
        y_true = y_te_g[label].values
        eval_df = y_te_g.reset_index(drop=True).copy()
        eval_df["pred"] = pred_g
        m_rate, f_rate = incidence_rate(eval_df, "pred", label)

        # Overall
        acc = accuracy_score(y_true, pred_g)
        auc = np.nan
        f1 = f1_score(y_true, pred_g)
        sens, spec = compute_sensitivity_specificity(y_true, pred_g)

        # Male subset
        mask_m = eval_df["Female"] == 0
        y_true_m = y_true[mask_m]
        pred_g_m = pred_g[mask_m]
        male_acc = accuracy_score(y_true_m, pred_g_m) if len(y_true_m) > 0 else np.nan
        male_auc = np.nan
        male_f1 = f1_score(y_true_m, pred_g_m) if len(y_true_m) > 0 else np.nan
        male_sens, male_spec = (
            compute_sensitivity_specificity(y_true_m, pred_g_m)
            if len(y_true_m) > 0
            else (np.nan, np.nan)
        )

        # Female subset
        mask_f = eval_df["Female"] == 1
        y_true_f = y_true[mask_f]
        pred_g_f = pred_g[mask_f]
        female_acc = accuracy_score(y_true_f, pred_g_f) if len(y_true_f) > 0 else np.nan
        female_auc = np.nan
        female_f1 = f1_score(y_true_f, pred_g_f) if len(y_true_f) > 0 else np.nan
        female_sens, female_spec = (
            compute_sensitivity_specificity(y_true_f, pred_g_f)
            if len(y_true_f) > 0
            else (np.nan, np.nan)
        )

        results["Guideline"]["accuracy"].append(acc)
        results["Guideline"]["auc"].append(auc)
        results["Guideline"]["f1"].append(f1)
        results["Guideline"]["sensitivity"].append(sens)
        results["Guideline"]["specificity"].append(spec)
        results["Guideline"]["male_accuracy"].append(male_acc)
        results["Guideline"]["male_auc"].append(male_auc)
        results["Guideline"]["male_f1"].append(male_f1)
        results["Guideline"]["male_sensitivity"].append(male_sens)
        results["Guideline"]["male_specificity"].append(male_spec)
        results["Guideline"]["female_accuracy"].append(female_acc)
        results["Guideline"]["female_auc"].append(female_auc)
        results["Guideline"]["female_f1"].append(female_f1)
        results["Guideline"]["female_sensitivity"].append(female_sens)
        results["Guideline"]["female_specificity"].append(female_spec)
        results["Guideline"]["male_rate"].append(m_rate)
        results["Guideline"]["female_rate"].append(f_rate)

        # --- RF Guideline --- #
        pred_g, prob_g = rf_evaluate(
            X_tr_g,
            y_tr_g,
            X_te_g,
            y_te_g,
            feat_names=guideline_features,
            random_state=seed,
        )
        eval_df = y_te_g.reset_index(drop=True).copy()
        eval_df["pred"] = pred_g
        m_rate, f_rate = incidence_rate(eval_df, "pred", label)

        # Overall
        acc = accuracy_score(y_true, pred_g)
        auc = roc_auc_score(y_true, prob_g)
        f1 = f1_score(y_true, pred_g)
        sens, spec = compute_sensitivity_specificity(y_true, pred_g)

        # Male subset
        y_true_m = y_true[mask_m]
        pred_g_m = pred_g[mask_m]
        prob_g_m = prob_g[mask_m]
        male_acc = accuracy_score(y_true_m, pred_g_m) if len(y_true_m) > 0 else np.nan
        male_auc = (
            roc_auc_score(y_true_m, prob_g_m)
            if len(y_true_m) > 1 and len(np.unique(y_true_m)) > 1
            else np.nan
        )
        male_f1 = f1_score(y_true_m, pred_g_m) if len(y_true_m) > 0 else np.nan
        male_sens, male_spec = (
            compute_sensitivity_specificity(y_true_m, pred_g_m)
            if len(y_true_m) > 0
            else (np.nan, np.nan)
        )

        # Female subset
        y_true_f = y_true[mask_f]
        pred_g_f = pred_g[mask_f]
        prob_g_f = prob_g[mask_f]
        female_acc = accuracy_score(y_true_f, pred_g_f) if len(y_true_f) > 0 else np.nan
        female_auc = (
            roc_auc_score(y_true_f, prob_g_f)
            if len(y_true_f) > 1 and len(np.unique(y_true_f)) > 1
            else np.nan
        )
        female_f1 = f1_score(y_true_f, pred_g_f) if len(y_true_f) > 0 else np.nan
        female_sens, female_spec = (
            compute_sensitivity_specificity(y_true_f, pred_g_f)
            if len(y_true_f) > 0
            else (np.nan, np.nan)
        )

        results["RF Guideline"]["accuracy"].append(acc)
        results["RF Guideline"]["auc"].append(auc)
        results["RF Guideline"]["f1"].append(f1)
        results["RF Guideline"]["sensitivity"].append(sens)
        results["RF Guideline"]["specificity"].append(spec)
        results["RF Guideline"]["male_accuracy"].append(male_acc)
        results["RF Guideline"]["male_auc"].append(male_auc)
        results["RF Guideline"]["male_f1"].append(male_f1)
        results["RF Guideline"]["male_sensitivity"].append(male_sens)
        results["RF Guideline"]["male_specificity"].append(male_spec)
        results["RF Guideline"]["female_accuracy"].append(female_acc)
        results["RF Guideline"]["female_auc"].append(female_auc)
        results["RF Guideline"]["female_f1"].append(female_f1)
        results["RF Guideline"]["female_sensitivity"].append(female_sens)
        results["RF Guideline"]["female_specificity"].append(female_spec)
        results["RF Guideline"]["male_rate"].append(m_rate)
        results["RF Guideline"]["female_rate"].append(f_rate)

        # --- Benchmark Sex-agnostic --- #
        pred_sa, prob_sa = rf_evaluate(
            X_tr_b,
            y_tr_b,
            X_te_b,
            y_te_b,
            feat_names=benchmark_features,
            random_state=seed,
        )
        eval_df = y_te_b.reset_index(drop=True).copy()
        eval_df["pred"] = pred_sa
        m_rate, f_rate = incidence_rate(eval_df, "pred", label)

        # Overall
        acc = accuracy_score(y_true, pred_sa)
        auc = roc_auc_score(y_true, prob_sa)
        f1 = f1_score(y_true, pred_sa)
        sens, spec = compute_sensitivity_specificity(y_true, pred_sa)

        # Male subset
        y_true_m = y_true[mask_m]
        pred_sa_m = pred_sa[mask_m]
        prob_sa_m = prob_sa[mask_m]
        male_acc = accuracy_score(y_true_m, pred_sa_m) if len(y_true_m) > 0 else np.nan
        male_auc = (
            roc_auc_score(y_true_m, prob_sa_m)
            if len(y_true_m) > 1 and len(np.unique(y_true_m)) > 1
            else np.nan
        )
        male_f1 = f1_score(y_true_m, pred_sa_m) if len(y_true_m) > 0 else np.nan
        male_sens, male_spec = (
            compute_sensitivity_specificity(y_true_m, pred_sa_m)
            if len(y_true_m) > 0
            else (np.nan, np.nan)
        )

        # Female subset
        y_true_f = y_true[mask_f]
        pred_sa_f = pred_sa[mask_f]
        prob_sa_f = prob_sa[mask_f]
        female_acc = (
            accuracy_score(y_true_f, pred_sa_f) if len(y_true_f) > 0 else np.nan
        )
        female_auc = (
            roc_auc_score(y_true_f, prob_sa_f)
            if len(y_true_f) > 1 and len(np.unique(y_true_f)) > 1
            else np.nan
        )
        female_f1 = f1_score(y_true_f, pred_sa_f) if len(y_true_f) > 0 else np.nan
        female_sens, female_spec = (
            compute_sensitivity_specificity(y_true_f, pred_sa_f)
            if len(y_true_f) > 0
            else (np.nan, np.nan)
        )

        results["Benchmark Sex-agnostic"]["accuracy"].append(acc)
        results["Benchmark Sex-agnostic"]["auc"].append(auc)
        results["Benchmark Sex-agnostic"]["f1"].append(f1)
        results["Benchmark Sex-agnostic"]["sensitivity"].append(sens)
        results["Benchmark Sex-agnostic"]["specificity"].append(spec)
        results["Benchmark Sex-agnostic"]["male_accuracy"].append(male_acc)
        results["Benchmark Sex-agnostic"]["male_auc"].append(male_auc)
        results["Benchmark Sex-agnostic"]["male_f1"].append(male_f1)
        results["Benchmark Sex-agnostic"]["male_sensitivity"].append(male_sens)
        results["Benchmark Sex-agnostic"]["male_specificity"].append(male_spec)
        results["Benchmark Sex-agnostic"]["female_accuracy"].append(female_acc)
        results["Benchmark Sex-agnostic"]["female_auc"].append(female_auc)
        results["Benchmark Sex-agnostic"]["female_f1"].append(female_f1)
        results["Benchmark Sex-agnostic"]["female_sensitivity"].append(female_sens)
        results["Benchmark Sex-agnostic"]["female_specificity"].append(female_spec)
        results["Benchmark Sex-agnostic"]["male_rate"].append(m_rate)
        results["Benchmark Sex-agnostic"]["female_rate"].append(f_rate)

        # --- Benchmark Sex-agnostic (undersampled) --- #
        pred_sa_us, prob_sa_us = rf_evaluate(
            X_tr_b_us,
            y_tr_b_us,
            X_te_b,
            y_te_b,
            feat_names=benchmark_features,
            random_state=seed,
        )
        eval_df = y_te_b.reset_index(drop=True).copy()
        eval_df["pred"] = pred_sa_us
        m_rate, f_rate = incidence_rate(eval_df, "pred", label)

        # Overall
        acc_us = accuracy_score(y_true, pred_sa_us)
        auc_us = roc_auc_score(y_true, prob_sa_us)
        f1_us = f1_score(y_true, pred_sa_us)
        sens_us, spec_us = compute_sensitivity_specificity(y_true, pred_sa_us)

        # Male subset
        pred_sa_us_m = pred_sa_us[mask_m]
        prob_sa_us_m = prob_sa_us[mask_m]
        male_acc_us = (
            accuracy_score(y_true_m, pred_sa_us_m) if len(y_true_m) > 0 else np.nan
        )
        male_auc_us = (
            roc_auc_score(y_true_m, prob_sa_us_m)
            if len(y_true_m) > 1 and len(np.unique(y_true_m)) > 1
            else np.nan
        )
        male_f1_us = f1_score(y_true_m, pred_sa_us_m) if len(y_true_m) > 0 else np.nan
        male_sens_us, male_spec_us = (
            compute_sensitivity_specificity(y_true_m, pred_sa_us_m)
            if len(y_true_m) > 0
            else (np.nan, np.nan)
        )

        # Female subset
        pred_sa_us_f = pred_sa_us[mask_f]
        prob_sa_us_f = prob_sa_us[mask_f]
        female_acc_us = (
            accuracy_score(y_true_f, pred_sa_us_f) if len(y_true_f) > 0 else np.nan
        )
        female_auc_us = (
            roc_auc_score(y_true_f, prob_sa_us_f)
            if len(y_true_f) > 1 and len(np.unique(y_true_f)) > 1
            else np.nan
        )
        female_f1_us = f1_score(y_true_f, pred_sa_us_f) if len(y_true_f) > 0 else np.nan
        female_sens_us, female_spec_us = (
            compute_sensitivity_specificity(y_true_f, pred_sa_us_f)
            if len(y_true_f) > 0
            else (np.nan, np.nan)
        )

        results["Benchmark Sex-agnostic (undersampled)"]["accuracy"].append(acc_us)
        results["Benchmark Sex-agnostic (undersampled)"]["auc"].append(auc_us)
        results["Benchmark Sex-agnostic (undersampled)"]["f1"].append(f1_us)
        results["Benchmark Sex-agnostic (undersampled)"]["sensitivity"].append(sens_us)
        results["Benchmark Sex-agnostic (undersampled)"]["specificity"].append(spec_us)
        results["Benchmark Sex-agnostic (undersampled)"]["male_accuracy"].append(
            male_acc_us
        )
        results["Benchmark Sex-agnostic (undersampled)"]["male_auc"].append(male_auc_us)
        results["Benchmark Sex-agnostic (undersampled)"]["male_f1"].append(male_f1_us)
        results["Benchmark Sex-agnostic (undersampled)"]["male_sensitivity"].append(
            male_sens_us
        )
        results["Benchmark Sex-agnostic (undersampled)"]["male_specificity"].append(
            male_spec_us
        )
        results["Benchmark Sex-agnostic (undersampled)"]["female_accuracy"].append(
            female_acc_us
        )
        results["Benchmark Sex-agnostic (undersampled)"]["female_auc"].append(
            female_auc_us
        )
        results["Benchmark Sex-agnostic (undersampled)"]["female_f1"].append(
            female_f1_us
        )
        results["Benchmark Sex-agnostic (undersampled)"]["female_sensitivity"].append(
            female_sens_us
        )
        results["Benchmark Sex-agnostic (undersampled)"]["female_specificity"].append(
            female_spec_us
        )
        results["Benchmark Sex-agnostic (undersampled)"]["male_rate"].append(m_rate)
        results["Benchmark Sex-agnostic (undersampled)"]["female_rate"].append(f_rate)

        # --- Benchmark Male-only --- #
        pred_m, prob_m = rf_evaluate(
            X_tr_b_m,
            y_tr_b_m,
            X_te_b_m,
            y_te_b_m,
            feat_names=benchmark_features,
            random_state=seed,
        )
        y_true_m = y_te_b_m[label].values
        eval_df = y_te_b_m.reset_index(drop=True).copy()
        eval_df["pred"] = pred_m
        m_rate_m, f_rate_m = incidence_rate(eval_df, "pred", label)

        acc = accuracy_score(y_true_m, pred_m)
        auc = roc_auc_score(y_true_m, prob_m)
        f1 = f1_score(y_true_m, pred_m)
        sens, spec = compute_sensitivity_specificity(y_true_m, pred_m)

        # For Male-only, overall = male, female = nan
        male_acc = acc
        male_auc = auc
        male_f1 = f1
        male_sens = sens
        male_spec = spec
        female_acc = np.nan
        female_auc = np.nan
        female_f1 = np.nan
        female_sens = np.nan
        female_spec = np.nan

        results["Benchmark Male"]["accuracy"].append(acc)
        results["Benchmark Male"]["auc"].append(auc)
        results["Benchmark Male"]["f1"].append(f1)
        results["Benchmark Male"]["sensitivity"].append(sens)
        results["Benchmark Male"]["specificity"].append(spec)
        results["Benchmark Male"]["male_accuracy"].append(male_acc)
        results["Benchmark Male"]["male_auc"].append(male_auc)
        results["Benchmark Male"]["male_f1"].append(male_f1)
        results["Benchmark Male"]["male_sensitivity"].append(male_sens)
        results["Benchmark Male"]["male_specificity"].append(male_spec)
        results["Benchmark Male"]["female_accuracy"].append(female_acc)
        results["Benchmark Male"]["female_auc"].append(female_auc)
        results["Benchmark Male"]["female_f1"].append(female_f1)
        results["Benchmark Male"]["female_sensitivity"].append(female_sens)
        results["Benchmark Male"]["female_specificity"].append(female_spec)
        results["Benchmark Male"]["male_rate"].append(m_rate_m)
        results["Benchmark Male"]["female_rate"].append(f_rate_m)

        # --- Benchmark Female-only --- #
        pred_f, prob_f = rf_evaluate(
            X_tr_b_f,
            y_tr_b_f,
            X_te_b_f,
            y_te_b_f,
            feat_names=benchmark_features,
            random_state=seed,
        )
        y_true_f = y_te_b_f[label].values
        eval_df = y_te_b_f.reset_index(drop=True).copy()
        eval_df["pred"] = pred_f
        m_rate_f, f_rate_f = incidence_rate(eval_df, "pred", label)

        acc = accuracy_score(y_true_f, pred_f)
        auc = roc_auc_score(y_true_f, prob_f)
        f1 = f1_score(y_true_f, pred_f)
        sens, spec = compute_sensitivity_specificity(y_true_f, pred_f)

        # For Female-only, overall = female, male = nan
        female_acc = acc
        female_auc = auc
        female_f1 = f1
        female_sens = sens
        female_spec = spec
        male_acc = np.nan
        male_auc = np.nan
        male_f1 = np.nan
        male_sens = np.nan
        male_spec = np.nan

        results["Benchmark Female"]["accuracy"].append(acc)
        results["Benchmark Female"]["auc"].append(auc)
        results["Benchmark Female"]["f1"].append(f1)
        results["Benchmark Female"]["sensitivity"].append(sens)
        results["Benchmark Female"]["specificity"].append(spec)
        results["Benchmark Female"]["male_accuracy"].append(male_acc)
        results["Benchmark Female"]["male_auc"].append(male_auc)
        results["Benchmark Female"]["male_f1"].append(male_f1)
        results["Benchmark Female"]["male_sensitivity"].append(male_sens)
        results["Benchmark Female"]["male_specificity"].append(male_spec)
        results["Benchmark Female"]["female_accuracy"].append(female_acc)
        results["Benchmark Female"]["female_auc"].append(female_auc)
        results["Benchmark Female"]["female_f1"].append(female_f1)
        results["Benchmark Female"]["female_sensitivity"].append(female_sens)
        results["Benchmark Female"]["female_specificity"].append(female_spec)
        results["Benchmark Female"]["male_rate"].append(m_rate_f)
        results["Benchmark Female"]["female_rate"].append(f_rate_f)

        # --- Benchmark Sex-specific --- #
        combined_pred = np.empty(len(test_df), dtype=int)
        combined_prob = np.empty(len(test_df), dtype=float)
        mask_m = test_df["Female"].values == 0
        mask_f = test_df["Female"].values == 1
        combined_pred[mask_m] = pred_m
        combined_pred[mask_f] = pred_f
        combined_prob[mask_m] = prob_m
        combined_prob[mask_f] = prob_f

        eval_df = y_te_b.reset_index(drop=True).copy()
        eval_df["pred"] = combined_pred
        m_rate_c, f_rate_c = incidence_rate(eval_df, "pred", label)

        # Overall
        acc = accuracy_score(y_true, combined_pred)
        auc = roc_auc_score(y_true, combined_prob)
        f1 = f1_score(y_true, combined_pred)
        sens, spec = compute_sensitivity_specificity(y_true, combined_pred)

        # Male subset
        combined_pred_m = combined_pred[mask_m]
        combined_prob_m = combined_prob[mask_m]
        male_acc = (
            accuracy_score(y_true_m, combined_pred_m) if len(y_true_m) > 0 else np.nan
        )
        male_auc = (
            roc_auc_score(y_true_m, combined_prob_m)
            if len(y_true_m) > 1 and len(np.unique(y_true_m)) > 1
            else np.nan
        )
        male_f1 = f1_score(y_true_m, combined_pred_m) if len(y_true_m) > 0 else np.nan
        male_sens, male_spec = (
            compute_sensitivity_specificity(y_true_m, combined_pred_m)
            if len(y_true_m) > 0
            else (np.nan, np.nan)
        )

        # Female subset
        combined_pred_f = combined_pred[mask_f]
        combined_prob_f = combined_prob[mask_f]
        female_acc = (
            accuracy_score(y_true_f, combined_pred_f) if len(y_true_f) > 0 else np.nan
        )
        female_auc = (
            roc_auc_score(y_true_f, combined_prob_f)
            if len(y_true_f) > 1 and len(np.unique(y_true_f)) > 1
            else np.nan
        )
        female_f1 = f1_score(y_true_f, combined_pred_f) if len(y_true_f) > 0 else np.nan
        female_sens, female_spec = (
            compute_sensitivity_specificity(y_true_f, combined_pred_f)
            if len(y_true_f) > 0
            else (np.nan, np.nan)
        )

        results["Benchmark Sex-specific"]["accuracy"].append(acc)
        results["Benchmark Sex-specific"]["auc"].append(auc)
        results["Benchmark Sex-specific"]["f1"].append(f1)
        results["Benchmark Sex-specific"]["sensitivity"].append(sens)
        results["Benchmark Sex-specific"]["specificity"].append(spec)
        results["Benchmark Sex-specific"]["male_accuracy"].append(male_acc)
        results["Benchmark Sex-specific"]["male_auc"].append(male_auc)
        results["Benchmark Sex-specific"]["male_f1"].append(male_f1)
        results["Benchmark Sex-specific"]["male_sensitivity"].append(male_sens)
        results["Benchmark Sex-specific"]["male_specificity"].append(male_spec)
        results["Benchmark Sex-specific"]["female_accuracy"].append(female_acc)
        results["Benchmark Sex-specific"]["female_auc"].append(female_auc)
        results["Benchmark Sex-specific"]["female_f1"].append(female_f1)
        results["Benchmark Sex-specific"]["female_sensitivity"].append(female_sens)
        results["Benchmark Sex-specific"]["female_specificity"].append(female_spec)
        results["Benchmark Sex-specific"]["male_rate"].append(m_rate_c)
        results["Benchmark Sex-specific"]["female_rate"].append(f_rate_c)

        # --- Proposed Sex-agnostic --- #
        pred_sa, prob_sa = rf_evaluate(
            X_tr_p,
            y_tr_p,
            X_te_p,
            y_te_p,
            feat_names=proposed_features,
            random_state=seed,
        )
        eval_df = y_te_p.reset_index(drop=True).copy()
        eval_df["pred"] = pred_sa
        m_rate, f_rate = incidence_rate(eval_df, "pred", label)

        # Overall
        acc = accuracy_score(y_true, pred_sa)
        auc = roc_auc_score(y_true, prob_sa)
        f1 = f1_score(y_true, pred_sa)
        sens, spec = compute_sensitivity_specificity(y_true, pred_sa)

        # Male subset
        pred_sa_m = pred_sa[mask_m]
        prob_sa_m = prob_sa[mask_m]
        male_acc = accuracy_score(y_true_m, pred_sa_m) if len(y_true_m) > 0 else np.nan
        male_auc = (
            roc_auc_score(y_true_m, prob_sa_m)
            if len(y_true_m) > 1 and len(np.unique(y_true_m)) > 1
            else np.nan
        )
        male_f1 = f1_score(y_true_m, pred_sa_m) if len(y_true_m) > 0 else np.nan
        male_sens, male_spec = (
            compute_sensitivity_specificity(y_true_m, pred_sa_m)
            if len(y_true_m) > 0
            else (np.nan, np.nan)
        )

        # Female subset
        pred_sa_f = pred_sa[mask_f]
        prob_sa_f = prob_sa[mask_f]
        female_acc = (
            accuracy_score(y_true_f, pred_sa_f) if len(y_true_f) > 0 else np.nan
        )
        female_auc = (
            roc_auc_score(y_true_f, prob_sa_f)
            if len(y_true_f) > 1 and len(np.unique(y_true_f)) > 1
            else np.nan
        )
        female_f1 = f1_score(y_true_f, pred_sa_f) if len(y_true_f) > 0 else np.nan
        female_sens, female_spec = (
            compute_sensitivity_specificity(y_true_f, pred_sa_f)
            if len(y_true_f) > 0
            else (np.nan, np.nan)
        )

        results["Proposed Sex-agnostic"]["accuracy"].append(acc)
        results["Proposed Sex-agnostic"]["auc"].append(auc)
        results["Proposed Sex-agnostic"]["f1"].append(f1)
        results["Proposed Sex-agnostic"]["sensitivity"].append(sens)
        results["Proposed Sex-agnostic"]["specificity"].append(spec)
        results["Proposed Sex-agnostic"]["male_accuracy"].append(male_acc)
        results["Proposed Sex-agnostic"]["male_auc"].append(male_auc)
        results["Proposed Sex-agnostic"]["male_f1"].append(male_f1)
        results["Proposed Sex-agnostic"]["male_sensitivity"].append(male_sens)
        results["Proposed Sex-agnostic"]["male_specificity"].append(male_spec)
        results["Proposed Sex-agnostic"]["female_accuracy"].append(female_acc)
        results["Proposed Sex-agnostic"]["female_auc"].append(female_auc)
        results["Proposed Sex-agnostic"]["female_f1"].append(female_f1)
        results["Proposed Sex-agnostic"]["female_sensitivity"].append(female_sens)
        results["Proposed Sex-agnostic"]["female_specificity"].append(female_spec)
        results["Proposed Sex-agnostic"]["male_rate"].append(m_rate)
        results["Proposed Sex-agnostic"]["female_rate"].append(f_rate)

        # --- Proposed Sex-agnostic (undersampled) --- #
        pred_sa_us, prob_sa_us = rf_evaluate(
            X_tr_p_us,
            y_tr_p_us,
            X_te_p,
            y_te_p,
            feat_names=proposed_features,
            random_state=seed,
        )
        eval_df = y_te_p.reset_index(drop=True).copy()
        eval_df["pred"] = pred_sa_us
        m_rate, f_rate = incidence_rate(eval_df, "pred", label)

        # Overall
        acc_us = accuracy_score(y_true, pred_sa_us)
        auc_us = roc_auc_score(y_true, prob_sa_us)
        f1_us = f1_score(y_true, pred_sa_us)
        sens_us, spec_us = compute_sensitivity_specificity(y_true, pred_sa_us)

        # Male subset
        pred_sa_us_m = pred_sa_us[mask_m]
        prob_sa_us_m = prob_sa_us[mask_m]
        male_acc_us = (
            accuracy_score(y_true_m, pred_sa_us_m) if len(y_true_m) > 0 else np.nan
        )
        male_auc_us = (
            roc_auc_score(y_true_m, prob_sa_us_m)
            if len(y_true_m) > 1 and len(np.unique(y_true_m)) > 1
            else np.nan
        )
        male_f1_us = f1_score(y_true_m, pred_sa_us_m) if len(y_true_m) > 0 else np.nan
        male_sens_us, male_spec_us = (
            compute_sensitivity_specificity(y_true_m, pred_sa_us_m)
            if len(y_true_m) > 0
            else (np.nan, np.nan)
        )

        # Female subset
        pred_sa_us_f = pred_sa_us[mask_f]
        prob_sa_us_f = prob_sa_us[mask_f]
        female_acc_us = (
            accuracy_score(y_true_f, pred_sa_us_f) if len(y_true_f) > 0 else np.nan
        )
        female_auc_us = (
            roc_auc_score(y_true_f, prob_sa_us_f)
            if len(y_true_f) > 1 and len(np.unique(y_true_f)) > 1
            else np.nan
        )
        female_f1_us = f1_score(y_true_f, pred_sa_us_f) if len(y_true_f) > 0 else np.nan
        female_sens_us, female_spec_us = (
            compute_sensitivity_specificity(y_true_f, pred_sa_us_f)
            if len(y_true_f) > 0
            else (np.nan, np.nan)
        )

        results["Proposed Sex-agnostic (undersampled)"]["accuracy"].append(acc_us)
        results["Proposed Sex-agnostic (undersampled)"]["auc"].append(auc_us)
        results["Proposed Sex-agnostic (undersampled)"]["f1"].append(f1_us)
        results["Proposed Sex-agnostic (undersampled)"]["sensitivity"].append(sens_us)
        results["Proposed Sex-agnostic (undersampled)"]["specificity"].append(spec_us)
        results["Proposed Sex-agnostic (undersampled)"]["male_accuracy"].append(
            male_acc_us
        )
        results["Proposed Sex-agnostic (undersampled)"]["male_auc"].append(male_auc_us)
        results["Proposed Sex-agnostic (undersampled)"]["male_f1"].append(male_f1_us)
        results["Proposed Sex-agnostic (undersampled)"]["male_sensitivity"].append(
            male_sens_us
        )
        results["Proposed Sex-agnostic (undersampled)"]["male_specificity"].append(
            male_spec_us
        )
        results["Proposed Sex-agnostic (undersampled)"]["female_accuracy"].append(
            female_acc_us
        )
        results["Proposed Sex-agnostic (undersampled)"]["female_auc"].append(
            female_auc_us
        )
        results["Proposed Sex-agnostic (undersampled)"]["female_f1"].append(
            female_f1_us
        )
        results["Proposed Sex-agnostic (undersampled)"]["female_sensitivity"].append(
            female_sens_us
        )
        results["Proposed Sex-agnostic (undersampled)"]["female_specificity"].append(
            female_spec_us
        )
        results["Proposed Sex-agnostic (undersampled)"]["male_rate"].append(m_rate)
        results["Proposed Sex-agnostic (undersampled)"]["female_rate"].append(f_rate)

        # --- Proposed Male-only --- #
        pred_m, prob_m = rf_evaluate(
            X_tr_p_m,
            y_tr_p_m,
            X_te_p_m,
            y_te_p_m,
            feat_names=proposed_features,
            random_state=seed,
        )
        y_true_m = y_te_p_m[label].values
        eval_df = y_te_p_m.reset_index(drop=True).copy()
        eval_df["pred"] = pred_m
        m_rate_m, f_rate_m = incidence_rate(eval_df, "pred", label)

        acc = accuracy_score(y_true_m, pred_m)
        auc = roc_auc_score(y_true_m, prob_m)
        f1 = f1_score(y_true_m, pred_m)
        sens, spec = compute_sensitivity_specificity(y_true_m, pred_m)

        # For Male-only, overall = male, female = nan
        male_acc = acc
        male_auc = auc
        male_f1 = f1
        male_sens = sens
        male_spec = spec
        female_acc = np.nan
        female_auc = np.nan
        female_f1 = np.nan
        female_sens = np.nan
        female_spec = np.nan

        results["Proposed Male"]["accuracy"].append(acc)
        results["Proposed Male"]["auc"].append(auc)
        results["Proposed Male"]["f1"].append(f1)
        results["Proposed Male"]["sensitivity"].append(sens)
        results["Proposed Male"]["specificity"].append(spec)
        results["Proposed Male"]["male_accuracy"].append(male_acc)
        results["Proposed Male"]["male_auc"].append(male_auc)
        results["Proposed Male"]["male_f1"].append(male_f1)
        results["Proposed Male"]["male_sensitivity"].append(male_sens)
        results["Proposed Male"]["male_specificity"].append(male_spec)
        results["Proposed Male"]["female_accuracy"].append(female_acc)
        results["Proposed Male"]["female_auc"].append(female_auc)
        results["Proposed Male"]["female_f1"].append(female_f1)
        results["Proposed Male"]["female_sensitivity"].append(female_sens)
        results["Proposed Male"]["female_specificity"].append(female_spec)
        results["Proposed Male"]["male_rate"].append(m_rate_m)
        results["Proposed Male"]["female_rate"].append(f_rate_m)

        # --- Proposed Female-only --- #
        pred_f, prob_f = rf_evaluate(
            X_tr_p_f,
            y_tr_p_f,
            X_te_p_f,
            y_te_p_f,
            feat_names=proposed_features,
            random_state=seed,
        )
        y_true_f = y_te_p_f[label].values
        eval_df = y_te_p_f.reset_index(drop=True).copy()
        eval_df["pred"] = pred_f
        m_rate_f, f_rate_f = incidence_rate(eval_df, "pred", label)

        acc = accuracy_score(y_true_f, pred_f)
        auc = roc_auc_score(y_true_f, prob_f)
        f1 = f1_score(y_true_f, pred_f)
        sens, spec = compute_sensitivity_specificity(y_true_f, pred_f)

        # For Female-only, overall = female, male = nan
        female_acc = acc
        female_auc = auc
        female_f1 = f1
        female_sens = sens
        female_spec = spec
        male_acc = np.nan
        male_auc = np.nan
        male_f1 = np.nan
        male_sens = np.nan
        male_spec = np.nan

        results["Proposed Female"]["accuracy"].append(acc)
        results["Proposed Female"]["auc"].append(auc)
        results["Proposed Female"]["f1"].append(f1)
        results["Proposed Female"]["sensitivity"].append(sens)
        results["Proposed Female"]["specificity"].append(spec)
        results["Proposed Female"]["male_accuracy"].append(male_acc)
        results["Proposed Female"]["male_auc"].append(male_auc)
        results["Proposed Female"]["male_f1"].append(male_f1)
        results["Proposed Female"]["male_sensitivity"].append(male_sens)
        results["Proposed Female"]["male_specificity"].append(male_spec)
        results["Proposed Female"]["female_accuracy"].append(female_acc)
        results["Proposed Female"]["female_auc"].append(female_auc)
        results["Proposed Female"]["female_f1"].append(female_f1)
        results["Proposed Female"]["female_sensitivity"].append(female_sens)
        results["Proposed Female"]["female_specificity"].append(female_spec)
        results["Proposed Female"]["male_rate"].append(m_rate_f)
        results["Proposed Female"]["female_rate"].append(f_rate_f)

        # --- Proposed Sex-specific --- #
        combined_pred = np.empty(len(test_df), dtype=int)
        combined_prob = np.empty(len(test_df), dtype=float)
        mask_m = test_df["Female"].values == 0
        mask_f = test_df["Female"].values == 1
        combined_pred[mask_m] = pred_m
        combined_pred[mask_f] = pred_f
        combined_prob[mask_m] = prob_m
        combined_prob[mask_f] = prob_f

        eval_df = y_te_p.reset_index(drop=True).copy()
        eval_df["pred"] = combined_pred
        m_rate_c, f_rate_c = incidence_rate(eval_df, "pred", label)

        # Overall
        acc = accuracy_score(y_true, combined_pred)
        auc = roc_auc_score(y_true, combined_prob)
        f1 = f1_score(y_true, combined_pred)
        sens, spec = compute_sensitivity_specificity(y_true, combined_pred)

        # Male subset
        combined_pred_m = combined_pred[mask_m]
        combined_prob_m = combined_prob[mask_m]
        male_acc = (
            accuracy_score(y_true_m, combined_pred_m) if len(y_true_m) > 0 else np.nan
        )
        male_auc = (
            roc_auc_score(y_true_m, combined_prob_m)
            if len(y_true_m) > 1 and len(np.unique(y_true_m)) > 1
            else np.nan
        )
        male_f1 = f1_score(y_true_m, combined_pred_m) if len(y_true_m) > 0 else np.nan
        male_sens, male_spec = (
            compute_sensitivity_specificity(y_true_m, combined_pred_m)
            if len(y_true_m) > 0
            else (np.nan, np.nan)
        )

        # Female subset
        combined_pred_f = combined_pred[mask_f]
        combined_prob_f = combined_prob[mask_f]
        female_acc = (
            accuracy_score(y_true_f, combined_pred_f) if len(y_true_f) > 0 else np.nan
        )
        female_auc = (
            roc_auc_score(y_true_f, combined_prob_f)
            if len(y_true_f) > 1 and len(np.unique(y_true_f)) > 1
            else np.nan
        )
        female_f1 = f1_score(y_true_f, combined_pred_f) if len(y_true_f) > 0 else np.nan
        female_sens, female_spec = (
            compute_sensitivity_specificity(y_true_f, combined_pred_f)
            if len(y_true_f) > 0
            else (np.nan, np.nan)
        )

        results["Proposed Sex-specific"]["accuracy"].append(acc)
        results["Proposed Sex-specific"]["auc"].append(auc)
        results["Proposed Sex-specific"]["f1"].append(f1)
        results["Proposed Sex-specific"]["sensitivity"].append(sens)
        results["Proposed Sex-specific"]["specificity"].append(spec)
        results["Proposed Sex-specific"]["male_accuracy"].append(male_acc)
        results["Proposed Sex-specific"]["male_auc"].append(male_auc)
        results["Proposed Sex-specific"]["male_f1"].append(male_f1)
        results["Proposed Sex-specific"]["male_sensitivity"].append(male_sens)
        results["Proposed Sex-specific"]["male_specificity"].append(male_spec)
        results["Proposed Sex-specific"]["female_accuracy"].append(female_acc)
        results["Proposed Sex-specific"]["female_auc"].append(female_auc)
        results["Proposed Sex-specific"]["female_f1"].append(female_f1)
        results["Proposed Sex-specific"]["female_sensitivity"].append(female_sens)
        results["Proposed Sex-specific"]["female_specificity"].append(female_spec)
        results["Proposed Sex-specific"]["male_rate"].append(m_rate_c)
        results["Proposed Sex-specific"]["female_rate"].append(f_rate_c)

        # --- Real proposed Sex-agnostic --- #
        pred_sa, prob_sa = rf_evaluate(
            X_tr_r,
            y_tr_r,
            X_te_r,
            y_te_r,
            feat_names=real_proposed_features,
            random_state=seed,
        )
        eval_df = y_te_r.reset_index(drop=True).copy()
        eval_df["pred"] = pred_sa
        m_rate, f_rate = incidence_rate(eval_df, "pred", label)

        # Overall
        acc = accuracy_score(y_true, pred_sa)
        auc = roc_auc_score(y_true, prob_sa)
        f1 = f1_score(y_true, pred_sa)
        sens, spec = compute_sensitivity_specificity(y_true, pred_sa)

        # Male subset
        pred_sa_m = pred_sa[mask_m]
        prob_sa_m = prob_sa[mask_m]
        male_acc = accuracy_score(y_true_m, pred_sa_m) if len(y_true_m) > 0 else np.nan
        male_auc = (
            roc_auc_score(y_true_m, prob_sa_m)
            if len(y_true_m) > 1 and len(np.unique(y_true_m)) > 1
            else np.nan
        )
        male_f1 = f1_score(y_true_m, pred_sa_m) if len(y_true_m) > 0 else np.nan
        male_sens, male_spec = (
            compute_sensitivity_specificity(y_true_m, pred_sa_m)
            if len(y_true_m) > 0
            else (np.nan, np.nan)
        )

        # Female subset
        pred_sa_f = pred_sa[mask_f]
        prob_sa_f = prob_sa[mask_f]
        female_acc = (
            accuracy_score(y_true_f, pred_sa_f) if len(y_true_f) > 0 else np.nan
        )
        female_auc = (
            roc_auc_score(y_true_f, prob_sa_f)
            if len(y_true_f) > 1 and len(np.unique(y_true_f)) > 1
            else np.nan
        )
        female_f1 = f1_score(y_true_f, pred_sa_f) if len(y_true_f) > 0 else np.nan
        female_sens, female_spec = (
            compute_sensitivity_specificity(y_true_f, pred_sa_f)
            if len(y_true_f) > 0
            else (np.nan, np.nan)
        )

        results["Real Proposed Sex-agnostic"]["accuracy"].append(acc)
        results["Real Proposed Sex-agnostic"]["auc"].append(auc)
        results["Real Proposed Sex-agnostic"]["f1"].append(f1)
        results["Real Proposed Sex-agnostic"]["sensitivity"].append(sens)
        results["Real Proposed Sex-agnostic"]["specificity"].append(spec)
        results["Real Proposed Sex-agnostic"]["male_accuracy"].append(male_acc)
        results["Real Proposed Sex-agnostic"]["male_auc"].append(male_auc)
        results["Real Proposed Sex-agnostic"]["male_f1"].append(male_f1)
        results["Real Proposed Sex-agnostic"]["male_sensitivity"].append(male_sens)
        results["Real Proposed Sex-agnostic"]["male_specificity"].append(male_spec)
        results["Real Proposed Sex-agnostic"]["female_accuracy"].append(female_acc)
        results["Real Proposed Sex-agnostic"]["female_auc"].append(female_auc)
        results["Real Proposed Sex-agnostic"]["female_f1"].append(female_f1)
        results["Real Proposed Sex-agnostic"]["female_sensitivity"].append(female_sens)
        results["Real Proposed Sex-agnostic"]["female_specificity"].append(female_spec)
        results["Real Proposed Sex-agnostic"]["male_rate"].append(m_rate)
        results["Real Proposed Sex-agnostic"]["female_rate"].append(f_rate)

        # --- Real Proposed Sex-agnostic (undersampled) --- #
        pred_sa_us, prob_sa_us = rf_evaluate(
            X_tr_r_us,
            y_tr_r_us,
            X_te_r,
            y_te_r,
            feat_names=real_proposed_features,
            random_state=seed,
        )
        eval_df = y_te_r.reset_index(drop=True).copy()
        eval_df["pred"] = pred_sa_us
        m_rate, f_rate = incidence_rate(eval_df, "pred", label)

        # Overall
        acc_us = accuracy_score(y_true, pred_sa_us)
        auc_us = roc_auc_score(y_true, prob_sa_us)
        f1_us = f1_score(y_true, pred_sa_us)
        sens_us, spec_us = compute_sensitivity_specificity(y_true, pred_sa_us)

        # Male subset
        pred_sa_us_m = pred_sa_us[mask_m]
        prob_sa_us_m = prob_sa_us[mask_m]
        male_acc_us = (
            accuracy_score(y_true_m, pred_sa_us_m) if len(y_true_m) > 0 else np.nan
        )
        male_auc_us = (
            roc_auc_score(y_true_m, prob_sa_us_m)
            if len(y_true_m) > 1 and len(np.unique(y_true_m)) > 1
            else np.nan
        )
        male_f1_us = f1_score(y_true_m, pred_sa_us_m) if len(y_true_m) > 0 else np.nan
        male_sens_us, male_spec_us = (
            compute_sensitivity_specificity(y_true_m, pred_sa_us_m)
            if len(y_true_m) > 0
            else (np.nan, np.nan)
        )

        # Female subset
        pred_sa_us_f = pred_sa_us[mask_f]
        prob_sa_us_f = prob_sa_us[mask_f]
        female_acc_us = (
            accuracy_score(y_true_f, pred_sa_us_f) if len(y_true_f) > 0 else np.nan
        )
        female_auc_us = (
            roc_auc_score(y_true_f, prob_sa_us_f)
            if len(y_true_f) > 1 and len(np.unique(y_true_f)) > 1
            else np.nan
        )
        female_f1_us = f1_score(y_true_f, pred_sa_us_f) if len(y_true_f) > 0 else np.nan
        female_sens_us, female_spec_us = (
            compute_sensitivity_specificity(y_true_f, pred_sa_us_f)
            if len(y_true_f) > 0
            else (np.nan, np.nan)
        )

        results["Real Proposed Sex-agnostic (undersampled)"]["accuracy"].append(acc_us)
        results["Real Proposed Sex-agnostic (undersampled)"]["auc"].append(auc_us)
        results["Real Proposed Sex-agnostic (undersampled)"]["f1"].append(f1_us)
        results["Real Proposed Sex-agnostic (undersampled)"]["sensitivity"].append(
            sens_us
        )
        results["Real Proposed Sex-agnostic (undersampled)"]["specificity"].append(
            spec_us
        )
        results["Real Proposed Sex-agnostic (undersampled)"]["male_accuracy"].append(
            male_acc_us
        )
        results["Real Proposed Sex-agnostic (undersampled)"]["male_auc"].append(
            male_auc_us
        )
        results["Real Proposed Sex-agnostic (undersampled)"]["male_f1"].append(
            male_f1_us
        )
        results["Real Proposed Sex-agnostic (undersampled)"]["male_sensitivity"].append(
            male_sens_us
        )
        results["Real Proposed Sex-agnostic (undersampled)"]["male_specificity"].append(
            male_spec_us
        )
        results["Real Proposed Sex-agnostic (undersampled)"]["female_accuracy"].append(
            female_acc_us
        )
        results["Real Proposed Sex-agnostic (undersampled)"]["female_auc"].append(
            female_auc_us
        )
        results["Real Proposed Sex-agnostic (undersampled)"]["female_f1"].append(
            female_f1_us
        )
        results["Real Proposed Sex-agnostic (undersampled)"][
            "female_sensitivity"
        ].append(female_sens_us)
        results["Real Proposed Sex-agnostic (undersampled)"][
            "female_specificity"
        ].append(female_spec_us)
        results["Real Proposed Sex-agnostic (undersampled)"]["male_rate"].append(m_rate)
        results["Real Proposed Sex-agnostic (undersampled)"]["female_rate"].append(
            f_rate
        )

        # --- Real Proposed Male-only --- #
        pred_m, prob_m = rf_evaluate(
            X_tr_r_m,
            y_tr_r_m,
            X_te_r_m,
            y_te_r_m,
            feat_names=real_proposed_features,
            random_state=seed,
        )
        y_true_m = y_te_r_m[label].values
        eval_df = y_te_r_m.reset_index(drop=True).copy()
        eval_df["pred"] = pred_m
        m_rate_m, f_rate_m = incidence_rate(eval_df, "pred", label)

        acc = accuracy_score(y_true_m, pred_m)
        auc = roc_auc_score(y_true_m, prob_m)
        f1 = f1_score(y_true_m, pred_m)
        sens, spec = compute_sensitivity_specificity(y_true_m, pred_m)

        # For Male-only, overall = male, female = nan
        male_acc = acc
        male_auc = auc
        male_f1 = f1
        male_sens = sens
        male_spec = spec
        female_acc = np.nan
        female_auc = np.nan
        female_f1 = np.nan
        female_sens = np.nan
        female_spec = np.nan

        results["Real Proposed Male"]["accuracy"].append(acc)
        results["Real Proposed Male"]["auc"].append(auc)
        results["Real Proposed Male"]["f1"].append(f1)
        results["Real Proposed Male"]["sensitivity"].append(sens)
        results["Real Proposed Male"]["specificity"].append(spec)
        results["Real Proposed Male"]["male_accuracy"].append(male_acc)
        results["Real Proposed Male"]["male_auc"].append(male_auc)
        results["Real Proposed Male"]["male_f1"].append(male_f1)
        results["Real Proposed Male"]["male_sensitivity"].append(male_sens)
        results["Real Proposed Male"]["male_specificity"].append(male_spec)
        results["Real Proposed Male"]["female_accuracy"].append(female_acc)
        results["Real Proposed Male"]["female_auc"].append(female_auc)
        results["Real Proposed Male"]["female_f1"].append(female_f1)
        results["Real Proposed Male"]["female_sensitivity"].append(female_sens)
        results["Real Proposed Male"]["female_specificity"].append(female_spec)
        results["Real Proposed Male"]["male_rate"].append(m_rate_m)
        results["Real Proposed Male"]["female_rate"].append(f_rate_m)

        # --- Real Proposed Female-only --- #
        pred_f, prob_f = rf_evaluate(
            X_tr_r_f,
            y_tr_r_f,
            X_te_r_f,
            y_te_r_f,
            feat_names=real_proposed_features,
            random_state=seed,
        )
        y_true_f = y_te_r_f[label].values
        eval_df = y_te_r_f.reset_index(drop=True).copy()
        eval_df["pred"] = pred_f
        m_rate_f, f_rate_f = incidence_rate(eval_df, "pred", label)

        acc = accuracy_score(y_true_f, pred_f)
        auc = roc_auc_score(y_true_f, prob_f)
        f1 = f1_score(y_true_f, pred_f)
        sens, spec = compute_sensitivity_specificity(y_true_f, pred_f)

        # For Female-only, overall = female, male = nan
        female_acc = acc
        female_auc = auc
        female_f1 = f1
        female_sens = sens
        female_spec = spec
        male_acc = np.nan
        male_auc = np.nan
        male_f1 = np.nan
        male_sens = np.nan
        male_spec = np.nan

        results["Real Proposed Female"]["accuracy"].append(acc)
        results["Real Proposed Female"]["auc"].append(auc)
        results["Real Proposed Female"]["f1"].append(f1)
        results["Real Proposed Female"]["sensitivity"].append(sens)
        results["Real Proposed Female"]["specificity"].append(spec)
        results["Real Proposed Female"]["male_accuracy"].append(male_acc)
        results["Real Proposed Female"]["male_auc"].append(male_auc)
        results["Real Proposed Female"]["male_f1"].append(male_f1)
        results["Real Proposed Female"]["male_sensitivity"].append(male_sens)
        results["Real Proposed Female"]["male_specificity"].append(male_spec)
        results["Real Proposed Female"]["female_accuracy"].append(female_acc)
        results["Real Proposed Female"]["female_auc"].append(female_auc)
        results["Real Proposed Female"]["female_f1"].append(female_f1)
        results["Real Proposed Female"]["female_sensitivity"].append(female_sens)
        results["Real Proposed Female"]["female_specificity"].append(female_spec)
        results["Real Proposed Female"]["male_rate"].append(m_rate_f)
        results["Real Proposed Female"]["female_rate"].append(f_rate_f)

        # --- Real Proposed Sex-specific --- #
        combined_pred = np.empty(len(test_df), dtype=int)
        combined_prob = np.empty(len(test_df), dtype=float)
        mask_m = test_df["Female"].values == 0
        mask_f = test_df["Female"].values == 1
        combined_pred[mask_m] = pred_m
        combined_pred[mask_f] = pred_f
        combined_prob[mask_m] = prob_m
        combined_prob[mask_f] = prob_f

        eval_df = y_te_r.reset_index(drop=True).copy()
        eval_df["pred"] = combined_pred
        m_rate_c, f_rate_c = incidence_rate(eval_df, "pred", label)

        # Overall
        acc = accuracy_score(y_true, combined_pred)
        auc = roc_auc_score(y_true, combined_prob)
        f1 = f1_score(y_true, combined_pred)
        sens, spec = compute_sensitivity_specificity(y_true, combined_pred)

        # Male subset
        combined_pred_m = combined_pred[mask_m]
        combined_prob_m = combined_prob[mask_m]
        male_acc = (
            accuracy_score(y_true_m, combined_pred_m) if len(y_true_m) > 0 else np.nan
        )
        male_auc = (
            roc_auc_score(y_true_m, combined_prob_m)
            if len(y_true_m) > 1 and len(np.unique(y_true_m)) > 1
            else np.nan
        )
        male_f1 = f1_score(y_true_m, combined_pred_m) if len(y_true_m) > 0 else np.nan
        male_sens, male_spec = (
            compute_sensitivity_specificity(y_true_m, combined_pred_m)
            if len(y_true_m) > 0
            else (np.nan, np.nan)
        )

        # Female subset
        combined_pred_f = combined_pred[mask_f]
        combined_prob_f = combined_prob[mask_f]
        female_acc = (
            accuracy_score(y_true_f, combined_pred_f) if len(y_true_f) > 0 else np.nan
        )
        female_auc = (
            roc_auc_score(y_true_f, combined_prob_f)
            if len(y_true_f) > 1 and len(np.unique(y_true_f)) > 1
            else np.nan
        )
        female_f1 = f1_score(y_true_f, combined_pred_f) if len(y_true_f) > 0 else np.nan
        female_sens, female_spec = (
            compute_sensitivity_specificity(y_true_f, combined_pred_f)
            if len(y_true_f) > 0
            else (np.nan, np.nan)
        )

        results["Real Proposed Sex-specific"]["accuracy"].append(acc)
        results["Real Proposed Sex-specific"]["auc"].append(auc)
        results["Real Proposed Sex-specific"]["f1"].append(f1)
        results["Real Proposed Sex-specific"]["sensitivity"].append(sens)
        results["Real Proposed Sex-specific"]["specificity"].append(spec)
        results["Real Proposed Sex-specific"]["male_accuracy"].append(male_acc)
        results["Real Proposed Sex-specific"]["male_auc"].append(male_auc)
        results["Real Proposed Sex-specific"]["male_f1"].append(male_f1)
        results["Real Proposed Sex-specific"]["male_sensitivity"].append(male_sens)
        results["Real Proposed Sex-specific"]["male_specificity"].append(male_spec)
        results["Real Proposed Sex-specific"]["female_accuracy"].append(female_acc)
        results["Real Proposed Sex-specific"]["female_auc"].append(female_auc)
        results["Real Proposed Sex-specific"]["female_f1"].append(female_f1)
        results["Real Proposed Sex-specific"]["female_sensitivity"].append(female_sens)
        results["Real Proposed Sex-specific"]["female_specificity"].append(female_spec)
        results["Real Proposed Sex-specific"]["male_rate"].append(m_rate_c)
        results["Real Proposed Sex-specific"]["female_rate"].append(f_rate_c)

    # After all seeds, compute mean and 95% CI
    summary = {}
    for model, mets in results.items():
        summary[model] = {}
        for met, vals in mets.items():
            arr = np.array(vals, dtype=float)
            mu = np.nanmean(arr)  # Use nanmean to handle nans
            se = np.nanstd(arr, ddof=1) / np.sqrt(np.sum(~np.isnan(arr)))
            ci = 1.96 * se
            summary[model][met] = (mu, mu - ci, mu + ci)

    summary_df = pd.concat(
        {
            model: pd.DataFrame.from_dict(
                metrics_dict, orient="index", columns=["mean", "ci_lower", "ci_upper"]
            )
            for model, metrics_dict in summary.items()
        },
        axis=0,
    )

    # Formatted summary
    formatted = summary_df.apply(
        lambda row: f"{row['mean']:.3f} ({row['ci_lower']:.3f}, {row['ci_upper']:.3f})",
        axis=1,
    )
    summary_table = formatted.unstack(level=1)
    rows_to_drop = [
        "Benchmark Male",
        "Benchmark Female",
        "Proposed Male",
        "Proposed Female",
        "Real Proposed Male",
        "Real Proposed Female",
    ]
    summary_table = summary_table.drop(index=rows_to_drop)

    # Save result
    output_dir = "/home/sunx/data/aiiih/projects/sunx/projects/ICD_sex_diff"
    output_file = "summary_results.xlsx"
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, output_file)
    summary_table.to_excel(full_path, index=True)
    print(f"Summary table saved to: {full_path}")

    return results, summary_table


res, summary = multiple_random_splits(train_df, 100)
print(summary)

from sklearn.tree import plot_tree
from sklearn.utils import resample
from sklearn.metrics import (
    make_scorer,
    average_precision_score,
    accuracy_score,
    roc_auc_score,
    f1_score,
    recall_score,
)
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import re


def full_model_inference(train_df, test_df, features, labels, survival_df, seed):

    train = train_df.copy()
    test = test_df.copy()
    df = test.copy()

    # male data
    train_m = train[train["Female"] == 0].copy()
    X_train_m = train_m[features]
    y_train_m = train_m[labels]

    test_m = test[test["Female"] == 0].copy()
    X_test_m = test_m[features]

    # female data
    train_f = train[train["Female"] == 1].copy()
    X_train_f = train_f[features]
    y_train_f = train_f[labels]

    test_f = test[test["Female"] == 1].copy()
    X_test_f = test_f[features]

    # RF params
    param_dist = {
        "n_estimators": randint(100, 500),
        "max_depth": [None] + list(range(5, 26, 5)),
        "min_samples_split": randint(2, 11),
        "min_samples_leaf": randint(1, 5),
        "max_features": ["sqrt", "log2", None],
    }
    base_clf = RandomForestClassifier(
        random_state=seed, n_jobs=-1, class_weight="balanced"
    )
    ap_scorer = make_scorer(average_precision_score, needs_proba=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=param_dist,
        n_iter=50,
        scoring=ap_scorer,
        cv=cv,
        random_state=seed,
        n_jobs=-1,
        verbose=0,
        error_score="raise",
    )

    # Male model
    search.fit(X_train_m, y_train_m)
    best_male = search.best_estimator_
    print("Best hyperparameters (Male):", search.best_params_)
    train_prob_m = best_male.predict_proba(X_train_m)[:, 1]
    best_thr_m = find_best_threshold(y_train_m, train_prob_m)
    prob_m = best_male.predict_proba(X_test_m)[:, 1]
    pred_m = (prob_m >= best_thr_m).astype(int)
    df.loc[df["Female"] == 0, "pred_male"] = pred_m
    df.loc[df["Female"] == 0, "prob_male"] = prob_m

    # Plot feature importances
    importances = best_male.feature_importances_
    idx = np.argsort(importances)[::-1]
    sorted_features = [features[i] for i in idx]
    colors = [
        (
            "red"
            if f in {"LVEF", "NYHA Class", "NYHA>2"}
            else "gold" if f in {"Significant LGE", "LGE Burden 5SD"} else "lightgray"
        )
        for f in sorted_features
    ]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(sorted_features)), importances[idx], color=colors)
    plt.xticks(range(len(sorted_features)), sorted_features, rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Male Model Feature Importances")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    plot_tree(
        best_male.estimators_[0],
        feature_names=features,
        class_names=True,
        filled=True,
        rounded=True,
    )
    plt.title("Visualization of One Decision Tree in Male Random Forest Model (Tree 0)")
    plt.show()

    # Female model
    search.fit(X_train_f, y_train_f)
    best_female = search.best_estimator_
    print("Best hyperparameters (Female):", search.best_params_)
    train_prob_f = best_female.predict_proba(X_train_f)[:, 1]
    best_thr_f = find_best_threshold(y_train_f, train_prob_f)
    prob_f = best_female.predict_proba(X_test_f)[:, 1]
    pred_f = (prob_f >= best_thr_f).astype(int)
    df.loc[df["Female"] == 1, "pred_female"] = pred_f
    df.loc[df["Female"] == 1, "prob_female"] = prob_f
    # Plot feature importances
    importances = best_female.feature_importances_
    idx = np.argsort(importances)[::-1]
    sorted_features = [features[i] for i in idx]
    colors = [
        (
            "red"
            if f in {"LVEF", "NYHA Class", "NYHA>2"}
            else "gold" if f in {"Significant LGE", "LGE Burden 5SD"} else "lightgray"
        )
        for f in sorted_features
    ]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(sorted_features)), importances[idx], color=colors)
    plt.xticks(range(len(sorted_features)), sorted_features, rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Female Model Feature Importances")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    plot_tree(
        best_female.estimators_[0],
        feature_names=features,
        class_names=True,
        filled=True,
        rounded=True,
    )
    plt.title(
        "Visualization of One Decision Tree in Female Random Forest Model (Tree 0)"
    )
    plt.show()

    # Survival analysis
    df["pred_sexspecific"] = np.nan
    df["prob_sexspecific"] = np.nan

    if "pred_male" in df.columns:
        df.loc[df["Female"] == 0, "pred_sexspecific"] = df.loc[
            df["Female"] == 0, "pred_male"
        ]
        df.loc[df["Female"] == 0, "prob_sexspecific"] = df.loc[
            df["Female"] == 0, "prob_male"
        ]

    if "pred_female" in df.columns:
        df.loc[df["Female"] == 1, "pred_sexspecific"] = df.loc[
            df["Female"] == 1, "pred_female"
        ]
        df.loc[df["Female"] == 1, "prob_sexspecific"] = df.loc[
            df["Female"] == 1, "prob_female"
        ]

    pred_labels = df[["MRN", "pred_sexspecific"]].drop_duplicates()  # Remove duplicates

    # Merge with deduplication
    merged_df = survival_df.merge(pred_labels, on="MRN", how="inner")
    merged_df = merged_df.drop_duplicates(subset=["MRN"])

    kmf = KaplanMeierFitter()
    endpoints = [
        (
            "Primary Endpoint",
            "PE_Time",
            "Was Primary Endpoint Reached? (Appropriate ICD Therapy)",
        ),
        ("Secondary Endpoint", "SE_Time", "Was Secondary Endpoint Reached?"),
    ]
    groupings = [
        ("Sex-Specific grouping", "pred_sexspecific"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

    for ax, (ep_name, time_col, event_col) in zip(axes, endpoints):
        title, pred_col = groupings[0]
        mask_low = merged_df[pred_col] == 0
        mask_high = merged_df[pred_col] == 1

        # Calculate group sizes and number of events
        n_low = mask_low.sum()
        n_high = mask_high.sum()
        total_n = n_low + n_high
        events_low = merged_df.loc[mask_low, event_col].sum()
        events_high = merged_df.loc[mask_high, event_col].sum()
        total_events = events_low + events_high

        print(f"{ep_name} - Low risk: n={n_low}, events={events_low}")
        print(f"{ep_name} - High risk: n={n_high}, events={events_high}")
        print(f"{ep_name} - Total: n={total_n}, events={total_events}")

        lr = logrank_test(
            merged_df.loc[mask_low, time_col],
            merged_df.loc[mask_high, time_col],
            merged_df.loc[mask_low, event_col],
            merged_df.loc[mask_high, event_col],
        )
        p_value = lr.p_value

        for mask, lbl, count, events in [
            (mask_low, f"Low Risk (n={n_low}, events={events_low})", n_low, events_low),
            (
                mask_high,
                f"High Risk (n={n_high}, events={events_high})",
                n_high,
                events_high,
            ),
        ]:
            kmf.fit(
                durations=merged_df.loc[mask, time_col],
                event_observed=merged_df.loc[mask, event_col],
                label=lbl,  # Include both n and events in label
            )
            kmf.plot(ax=ax)

        ax.set_title(f"{ep_name} by {title} (Total n={total_n}, events={total_events})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Survival Probability")
        ax.text(
            0.95,
            0.05,
            f"Log-rank p = {p_value:.5f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
        )
        ax.legend()

    plt.tight_layout()
    plt.show()
    plt.close()

    # Fit Cox PH model
    cph_feature = df[["MRN"] + features]
    cph_df = survival_df.merge(cph_feature, on="MRN", how="inner")
    cph_df = cph_df.drop_duplicates(subset=["MRN"])
    covariates = [
        col
        for col in cph_df.columns
        if col
        not in [
            "MRN",
            "PE_Time",
            "Was Primary Endpoint Reached? (Appropriate ICD Therapy)",
            "SE_Time",
            "Was Secondary Endpoint Reached?",
        ]
    ]
    formula_terms = [
        f"`{col}`" if re.search(r"[^a-zA-Z0-9_]", col) else col for col in covariates
    ]
    formula = " + ".join(formula_terms)

    cph_primary = CoxPHFitter()
    cph_primary.fit(
        cph_df,
        duration_col="PE_Time",
        event_col="Was Primary Endpoint Reached? (Appropriate ICD Therapy)",
        formula=formula,
    )
    print(f"\nCox PH Model for Primary endpoint:")
    print(cph_primary.summary)

    cph_secondary = CoxPHFitter()
    cph_secondary.fit(
        cph_df,
        duration_col="SE_Time",
        event_col="Was Secondary Endpoint Reached?",
        formula=formula,
    )
    print(f"\nCox PH Model for Secondary endpoint:")
    print(cph_secondary.summary)

    # Clustering for Male
    if not X_test_m.empty:
        # Elbow method for male
        inertias_m = []
        max_clusters = 10
        for k in range(1, max_clusters + 1):
            kmeans_m = KMeans(n_clusters=k, random_state=seed)
            kmeans_m.fit(X_test_m)
            inertias_m.append(kmeans_m.inertia_)

        diffs_m = np.diff(inertias_m)
        diff_ratios_m = diffs_m / inertias_m[:-1]
        best_k_m = np.argmin(diff_ratios_m) + 1
        if best_k_m < 2:
            best_k_m = 2

        print(
            f"Selected best number of clusters for Male using elbow method: {best_k_m}"
        )

        # Final KMeans for male
        kmeans_m = KMeans(n_clusters=best_k_m, random_state=seed)
        cluster_labels_m = kmeans_m.fit_predict(X_test_m)  # Use X_test_m for consistency
        test_m["cluster"] = cluster_labels_m  # Add cluster labels to test_m
        df.loc[df["Female"] == 0, "cluster"] = (
            cluster_labels_m  # Update df with cluster labels
        )

        # TSNE for male using original features
        reducer_m = TSNE(n_components=2, random_state=seed)
        embedding_m = reducer_m.fit_transform(X_test_m)  # Use X_test_m for TSNE

        # Average risk per cluster for male
        test_m["prob_sexspecific"] = prob_m  # Add prob_sexspecific to test_m
        cluster_risk_m = test_m.groupby("cluster")["prob_sexspecific"].mean().round(3)
        print("Average risk per cluster (Male):")
        print(cluster_risk_m)

        # Visualization for male
        plt.figure(figsize=(8, 6))
        palette_m = sns.color_palette("Set2", best_k_m)
        for c in range(best_k_m):
            mask_m = test_m["cluster"] == c
            plt.scatter(
                embedding_m[mask_m, 0],
                embedding_m[mask_m, 1],
                color=palette_m[c],
                s=30,
                alpha=0.7,
                label=f"Cluster {c} (avg risk = {cluster_risk_m[c]:.2f})",
            )
        plt.title("TSNE + KMeans Clustering with Average Risk (Male)")
        plt.xlabel("TSNE-1")
        plt.ylabel("TSNE-2")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Clustering for Female
    if not X_test_f.empty:
        # Elbow method for female
        inertias_f = []
        for k in range(1, max_clusters + 1):
            kmeans_f = KMeans(n_clusters=k, random_state=seed)
            kmeans_f.fit(X_test_f)
            inertias_f.append(kmeans_f.inertia_)

        diffs_f = np.diff(inertias_f)
        diff_ratios_f = diffs_f / inertias_f[:-1]
        best_k_f = np.argmin(diff_ratios_f) + 1
        if best_k_f < 2:
            best_k_f = 2

        print(
            f"Selected best number of clusters for Female using elbow method: {best_k_f}"
        )

        # Final KMeans for female
        kmeans_f = KMeans(n_clusters=best_k_f, random_state=seed)
        cluster_labels_f = kmeans_f.fit_predict(X_test_f)  # Use X_test_f for consistency
        test_f["cluster"] = cluster_labels_f  # Add cluster labels to test_f
        df.loc[df["Female"] == 1, "cluster"] = (
            cluster_labels_f  # Update df with cluster labels
        )

        # TSNE for female using original features
        reducer_f = TSNE(n_components=2, random_state=seed)
        embedding_f = reducer_f.fit_transform(X_test_f)  # Use X_test_f for TSNE

        # Average risk per cluster for female
        test_f["prob_sexspecific"] = prob_f  # Add prob_sexspecific to test_f
        cluster_risk_f = test_f.groupby("cluster")["prob_sexspecific"].mean().round(3)
        print("Average risk per cluster (Female):")
        print(cluster_risk_f)

        # Visualization for female
        plt.figure(figsize=(8, 6))
        palette_f = sns.color_palette("Set2", best_k_f)
        for c in range(best_k_f):
            mask_f = test_f["cluster"] == c
            plt.scatter(
                embedding_f[mask_f, 0],
                embedding_f[mask_f, 1],
                color=palette_f[c],
                s=30,
                alpha=0.7,
                label=f"Cluster {c} (avg risk = {cluster_risk_f[c]:.2f})",
            )
        plt.title("TSNE + KMeans Clustering with Average Risk (Female)")
        plt.xlabel("TSNE-1")
        plt.ylabel("TSNE-2")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Bootstrap
    B = 100
    accs_m = []
    aucs_m = []
    f1s_m = []
    sens_m = []
    spec_m = []
    accs_f = []
    aucs_f = []
    f1s_f = []
    sens_f = []
    spec_f = []

    for i in range(B):
        # Bootstrap the test df
        df_boot = resample(df, replace=True, n_samples=len(df), random_state=seed + i)

        # Male boot data
        m_boot = df_boot[df_boot["Female"] == 0]
        if not m_boot.empty:
            X_m_boot = m_boot[features]
            y_m_boot = m_boot[labels]
            prob_m_boot = best_male.predict_proba(X_m_boot)[:, 1]
            pred_m_boot = (prob_m_boot >= best_thr_m).astype(int)
            acc_m = accuracy_score(y_m_boot, pred_m_boot)
            auc_m = roc_auc_score(y_m_boot, prob_m_boot)
            f1_m = f1_score(y_m_boot, pred_m_boot)
            sen_m = recall_score(y_m_boot, pred_m_boot)
            spe_m = recall_score(y_m_boot, pred_m_boot, pos_label=0)
            accs_m.append(acc_m)
            aucs_m.append(auc_m)
            f1s_m.append(f1_m)
            sens_m.append(sen_m)
            spec_m.append(spe_m)

        # Female boot data
        f_boot = df_boot[df_boot["Female"] == 1]
        if not f_boot.empty:
            X_f_boot = f_boot[features]
            y_f_boot = f_boot[labels]
            prob_f_boot = best_female.predict_proba(X_f_boot)[:, 1]
            pred_f_boot = (prob_f_boot >= best_thr_f).astype(int)
            acc_f = accuracy_score(y_f_boot, pred_f_boot)
            auc_f = roc_auc_score(y_f_boot, prob_f_boot)
            f1_f = f1_score(y_f_boot, pred_f_boot)
            sen_f = recall_score(y_f_boot, pred_f_boot)
            spe_f = recall_score(y_f_boot, pred_f_boot, pos_label=0)
            accs_f.append(acc_f)
            aucs_f.append(auc_f)
            f1s_f.append(f1_f)
            sens_f.append(sen_f)
            spec_f.append(spe_f)

    # Report for Male
    if accs_m:
        mean_acc_m = np.mean(accs_m)
        ci_low_acc_m = np.percentile(accs_m, 2.5)
        ci_high_acc_m = np.percentile(accs_m, 97.5)
        mean_auc_m = np.mean(aucs_m)
        ci_low_auc_m = np.percentile(aucs_m, 2.5)
        ci_high_auc_m = np.percentile(aucs_m, 97.5)
        mean_f1_m = np.mean(f1s_m)
        ci_low_f1_m = np.percentile(f1s_m, 2.5)
        ci_high_f1_m = np.percentile(f1s_m, 97.5)
        mean_sen_m = np.mean(sens_m)
        ci_low_sen_m = np.percentile(sens_m, 2.5)
        ci_high_sen_m = np.percentile(sens_m, 97.5)
        mean_spe_m = np.mean(spec_m)
        ci_low_spe_m = np.percentile(spec_m, 2.5)
        ci_high_spe_m = np.percentile(spec_m, 97.5)
        print(
            f"Male Accuracy: {mean_acc_m:.4f} [95% CI: {ci_low_acc_m:.4f} - {ci_high_acc_m:.4f}]"
        )
        print(
            f"Male AUC: {mean_auc_m:.4f} [95% CI: {ci_low_auc_m:.4f} - {ci_high_auc_m:.4f}]"
        )
        print(
            f"Male F1: {mean_f1_m:.4f} [95% CI: {ci_low_f1_m:.4f} - {ci_high_f1_m:.4f}]"
        )
        print(
            f"Male Sensitivity: {mean_sen_m:.4f} [95% CI: {ci_low_sen_m:.4f} - {ci_high_sen_m:.4f}]"
        )
        print(
            f"Male Specificity: {mean_spe_m:.4f} [95% CI: {ci_low_spe_m:.4f} - {ci_high_spe_m:.4f}]"
        )

    # Report for Female
    if accs_f:
        mean_acc_f = np.mean(accs_f)
        ci_low_acc_f = np.percentile(accs_f, 2.5)
        ci_high_acc_f = np.percentile(accs_f, 97.5)
        mean_auc_f = np.mean(aucs_f)
        ci_low_auc_f = np.percentile(aucs_f, 2.5)
        ci_high_auc_f = np.percentile(aucs_f, 97.5)
        mean_f1_f = np.mean(f1s_f)
        ci_low_f1_f = np.percentile(f1s_f, 2.5)
        ci_high_f1_f = np.percentile(f1s_f, 97.5)
        mean_sen_f = np.mean(sens_f)
        ci_low_sen_f = np.percentile(sens_f, 2.5)
        ci_high_sen_f = np.percentile(sens_f, 97.5)
        mean_spe_f = np.mean(spec_f)
        ci_low_spe_f = np.percentile(spec_f, 2.5)
        ci_high_spe_f = np.percentile(spec_f, 97.5)
        print(
            f"Female Accuracy: {mean_acc_f:.4f} [95% CI: {ci_low_acc_f:.4f} - {ci_high_acc_f:.4f}]"
        )
        print(
            f"Female AUC: {mean_auc_f:.4f} [95% CI: {ci_low_auc_f:.4f} - {ci_high_auc_f:.4f}]"
        )
        print(
            f"Female F1: {mean_f1_f:.4f} [95% CI: {ci_low_f1_f:.4f} - {ci_high_f1_f:.4f}]"
        )
        print(
            f"Female Sensitivity: {mean_sen_f:.4f} [95% CI: {ci_low_sen_f:.4f} - {ci_high_sen_f:.4f}]"
        )
        print(
            f"Female Specificity: {mean_spe_f:.4f} [95% CI: {ci_low_spe_f:.4f} - {ci_high_spe_f:.4f}]"
        )

    return None

