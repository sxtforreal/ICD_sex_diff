# main.py (Optimized Version)
# This file has been refactored for clarity, speed, and maintainability.
# Key improvements: removed duplicate imports, merged data processing, simplified function structure, and reduced unnecessary output.

import sys
import re
import warnings
from math import ceil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint
from itertools import combinations

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
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
)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import RFECV
from sklearn.utils import resample
from sklearn.tree import plot_tree
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.exceptions import UndefinedMetricWarning

from tableone import TableOne
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

pd.set_option("future.no_silent_downcasting", True)
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
DATA_DIR = "/workspace/data"
SURVIVAL_FILE = "df.xlsx"
COHORT_FILE = "NICM Arrhythmia Cohort for Xiaotan Final.xlsx"
survival_path = f"{DATA_DIR}/{SURVIVAL_FILE}"
cohort_path = f"{DATA_DIR}/{COHORT_FILE}"
if not os.path.exists(survival_path) or not os.path.exists(cohort_path):
    print(f"Data files not found. Expected SURVIVAL_FILE at '{survival_path}' and COHORT_FILE at '{cohort_path}'. Set DATA_DIR/SURVIVAL_FILE/COHORT_FILE environment variables accordingly.")
    sys.exit(0)

survival_df = pd.read_excel(survival_path)
survival_df["PE_Time"] = np.where(
    survival_df["Was Primary Endpoint Reached? (Appropriate ICD Therapy)"] == 1,
    survival_df["Time from ICD Implant to Primary Endpoint (in days)"],
    survival_df["Time from ICD Implant to Last Cardiology Encounter (in days)"]
)
survival_df["SE_Time"] = np.where(
    survival_df["Was Secondary Endpoint Reached?"] == 1,
    survival_df["Time from ICD Implant to Secondary Endpoint (in days)"],
    survival_df["Time from ICD Implant to Last Cardiology Encounter (in days)"]
)
survival_df = survival_df[[
    "MRN",
    "Was Primary Endpoint Reached? (Appropriate ICD Therapy)",
    "PE_Time",
    "Was Secondary Endpoint Reached?",
    "SE_Time",
]]

with_icd = pd.read_excel(cohort_path, sheet_name="ICD")
with_icd["ICD"] = 1
without_icd = pd.read_excel(cohort_path, sheet_name="No_ICD")
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
df.drop([
    "Date VT/VF/SCD",
    "End follow-up date",
    "CRT Date",
    "QRS",
], axis=1, inplace=True)

# Variables
categorical = [
    "Female", "DM", "HTN", "HLP", "AF", "NYHA Class", "Beta Blocker",
    "ACEi/ARB/ARNi", "Aldosterone Antagonist", "VT/VF/SCD", "AAD", "CRT", "ICD"
]
df[categorical] = df[categorical].astype("object")
labels = ["MRN", "Female", "VT/VF/SCD", "ICD"]
features = [c for c in df.columns if c not in labels]

# Missing percentage
print("\nMissing value percentage:")
print(df.isnull().sum() / len(df) * 100)


def impute_misforest(X, random_seed):
    """
    Iterative imputation with ExtraTreesRegressor (tree-based, robust to scaling).
    """
    estimator = ExtraTreesRegressor(
        n_estimators=100,
        random_state=random_seed,
        n_jobs=-1,
    )
    imputer = IterativeImputer(
        estimator=estimator,
        random_state=random_seed,
        max_iter=10,
        sample_posterior=False,
        initial_strategy="median",
    )
    # 直接返回DataFrame，避免重复赋值
    return pd.DataFrame(
        imputer.fit_transform(X), columns=X.columns, index=X.index
    )


def conversion_and_imputation(df, features, labels):
    df = df.copy()
    df = df[features + labels]

    # NYHA Class (ordinal) 转为数字，保留NaN
    if "NYHA Class" in df.columns:
        codes, _ = pd.factorize(df["NYHA Class"], sort=True)
        df["NYHA Class"] = np.where(codes == -1, np.nan, codes).astype(float)

    # 二元变量转为float
    binary_cols = [
        "Female", "DM", "HTN", "HLP", "AF", "Beta Blocker",
        "ACEi/ARB/ARNi", "Aldosterone Antagonist", "VT/VF/SCD",
        "AAD", "CRT", "ICD"
    ]
    for c in [col for col in binary_cols if col in df.columns]:
        if df[c].dtype == "object":
            df[c] = df[c].replace({"Yes": 1, "No": 0, "Y": 1, "N": 0, "True": 1, "False": 0})
        df[c] = df[c].astype(float)

    # 缺失值插补
    X = df[features]
    imputed_X = impute_misforest(X, 0)
    for col in labels:
        imputed_X[col] = df[col].values
    # 二元变量阈值化
    for c in [col for col in binary_cols if col in imputed_X.columns]:
        imputed_X[c] = (imputed_X[c] >= 0.5).astype(float)
    return imputed_X


clean_df = conversion_and_imputation(df, features, labels)

# Additional
clean_df["Age by decade"] = (clean_df["Age at CMR"] // 10).astype(int)
clean_df["CrCl>45"] = (
    clean_df["Cockcroft-Gault Creatinine Clearance (mL/min)"] > 45
).astype(int)
clean_df["NYHA>2"] = (clean_df["NYHA Class"] > 2).astype(int)
clean_df["Significant LGE"] = (clean_df["LGE Burden 5SD"] > 2).astype(int)

# Distribution of sex
print("\nSex distribution:")
print(clean_df["Female"].value_counts())

# Distribution of true label
print("\nArrhythmia distribution:")
print(clean_df["VT/VF/SCD"].value_counts())

# Proportion in ICD population that follows the rule-based guideline
icd_df = clean_df[clean_df["ICD"] == 1]
cond = (icd_df["NYHA Class"] >= 2) & (icd_df["LVEF"] <= 35)
pct = cond.sum() / len(icd_df) * 100
print(
    f"\nProportion of ICD population following the rule-based guideline: {pct:.2f}%"
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
print(f"Overall female proportion: {df['Female'].mean():.2f}, training set: {train_df['Female'].mean():.2f}, test set: {test_df['Female'].mean():.2f}")
print(f"Overall arrhythmia proportion: {df['VT/VF/SCD'].mean():.2f}, training set: {train_df['VT/VF/SCD'].mean():.2f}, test set: {test_df['VT/VF/SCD'].mean():.2f}")

def find_best_threshold(y_true, y_scores):
    """
    Find the probability threshold that maximizes the F1 score
    based on the precision-recall curve.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.nanargmax(f1_scores)
    # thresholds长度比f1_scores少1，需判断
    if best_idx >= len(thresholds):
        best_idx = len(thresholds) - 1
    return thresholds[best_idx]


def compute_sensitivity_specificity(y_true, y_pred):
    """
    Compute sensitivity (true positive rate) and specificity (true negative rate)
    from binary predictions.
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    else:
        sensitivity = specificity = np.nan
    return sensitivity, specificity


def incidence_rate(df, pred_col, label_col):
    """
    Compute the incidence rate defined as:
      #actually developed arrhythmia/#model predicted to develop arrhythmia,
    separately for males (Female==0) and females (Female==1).
    """
    male = df[df["Female"] == 0]
    female = df[df["Female"] == 1]
    male_rate = male[label_col].sum() / (male[pred_col] == 1).sum() if (male[pred_col] == 1).sum() > 0 else np.nan
    female_rate = female[label_col].sum() / (female[pred_col] == 1).sum() if (female[pred_col] == 1).sum() > 0 else np.nan
    return male_rate, female_rate


def rf_evaluate(
    X_train,
    y_train_df,
    feat_names,
    random_state=None,
    visualize_importance=False,
):
    """
    Train a RandomForest with randomized search optimizing average precision,
    using only the training set for cross-validation and optimal threshold selection.
    The test set is only used for inference, and the optimal threshold is directly applied to test predictions.
    """
    y_train = y_train_df["VT/VF/SCD"]
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
        n_iter=20,
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
        highlight = {"LVEF", "NYHA Class", "NYHA>2"}
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
    # The test set should be passed to this function only for inference, not for threshold selection.
    # To use this function for inference, call best_model.predict_proba(X_test)[:, 1] and apply the threshold.
    return best_model, threshold


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
        "Female", "Age by decade", "BMI", "AF", "Beta Blocker", "CrCl>45",
        "LVEF", "QTc", "NYHA>2", "CRT", "AAD", "Significant LGE"
    ]
    proposed_features = benchmark_features + [
        "DM", "HTN", "HLP", "LVEDVi", "LV Mass Index", "RVEDVi", "RVEF",
        "LA EF", "LAVi", "MRF (%)", "Sphericity Index", "Relative Wall Thickness",
        "MV Annular Diameter", "ACEi/ARB/ARNi", "Aldosterone Antagonist"
    ]
    real_proposed_features = proposed_features[:]
    real_proposed_features.remove("NYHA>2")
    real_proposed_features.remove("Significant LGE")
    real_proposed_features.extend(["LGE Burden 5SD", "NYHA Class"])

    # Models
    model_names = [
        "Guideline", "RF Guideline", "Benchmark Sex-agnostic",
        "Benchmark Sex-agnostic (undersampled)", "Benchmark Male", "Benchmark Female",
        "Benchmark Sex-specific", "Proposed Sex-agnostic", "Proposed Sex-agnostic (undersampled)",
        "Proposed Male", "Proposed Female", "Proposed Sex-specific",
        "Real Proposed Sex-agnostic", "Real Proposed Sex-agnostic (undersampled)",
        "Real Proposed Male", "Real Proposed Female", "Real Proposed Sex-specific"
    ]
    metrics = [
        "accuracy", "auc", "f1", "sensitivity", "specificity",
        "male_accuracy", "male_auc", "male_f1", "male_sensitivity", "male_specificity",
        "female_accuracy", "female_auc", "female_f1", "female_sensitivity", "female_specificity",
        "male_rate", "female_rate"
    ]
    results = {m: {met: [] for met in metrics} for m in model_names}

    for seed in range(N):
        # print(f"Running split #{seed+1}")  # 可选保留
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
        model_g, threshold_g = rf_evaluate(
            X_tr_g,
            y_tr_g,
            feat_names=guideline_features,
            random_state=seed,
        )
        prob_g = model_g.predict_proba(X_te_g)[:, 1]
        pred_g = (prob_g >= threshold_g).astype(int)
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
        model_g, threshold_g = rf_evaluate(
            X_tr_g,
            y_tr_g,
            feat_names=guideline_features,
            random_state=seed,
        )
        prob_g = model_g.predict_proba(X_te_g)[:, 1]
        pred_g = (prob_g >= threshold_g).astype(int)
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
        model_sa, threshold_sa = rf_evaluate(
            X_tr_b,
            y_tr_b,
            feat_names=benchmark_features,
            random_state=seed,
        )
        prob_sa = model_sa.predict_proba(X_te_b)[:, 1]
        pred_sa = (prob_sa >= threshold_sa).astype(int)
        eval_df = y_te_b.reset_index(drop=True).copy()
        eval_df["pred"] = pred_sa
        m_rate, f_rate = incidence_rate(eval_df, "pred", label)
        y_true = y_te_b[label].values
        mask_m = eval_df["Female"] == 0
        mask_f = eval_df["Female"] == 1

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
        model_sa_us, threshold_sa_us = rf_evaluate(
            X_tr_b_us,
            y_tr_b_us,
            feat_names=benchmark_features,
            random_state=seed,
        )
        prob_sa_us = model_sa_us.predict_proba(X_te_b_us)[:, 1]
        pred_sa_us = (prob_sa_us >= threshold_sa_us).astype(int)
        eval_df = y_te_b_us.reset_index(drop=True).copy()
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
        model_m, threshold_m = rf_evaluate(
            X_tr_b_m,
            y_tr_b_m,
            feat_names=benchmark_features,
            random_state=seed,
        )
        prob_m = model_m.predict_proba(X_te_b_m)[:, 1]
        pred_m = (prob_m >= threshold_m).astype(int)
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
        model_f, threshold_f = rf_evaluate(
            X_tr_b_f,
            y_tr_b_f,
            feat_names=benchmark_features,
            random_state=seed,
        )
        prob_f = model_f.predict_proba(X_te_b_f)[:, 1]
        pred_f = (prob_f >= threshold_f).astype(int)
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
        model_sa, threshold_sa = rf_evaluate(
            X_tr_p,
            y_tr_p,
            feat_names=proposed_features,
            random_state=seed,
        )
        prob_sa = model_sa.predict_proba(X_te_p)[:, 1]
        pred_sa = (prob_sa >= threshold_sa).astype(int)
        eval_df = y_te_p.reset_index(drop=True).copy()
        eval_df["pred"] = pred_sa
        m_rate, f_rate = incidence_rate(eval_df, "pred", label)
        y_true = y_te_p[label].values
        mask_m = eval_df["Female"] == 0
        mask_f = eval_df["Female"] == 1

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
        model_sa_us, threshold_sa_us = rf_evaluate(
            X_tr_p_us,
            y_tr_p_us,
            feat_names=proposed_features,
            random_state=seed,
        )
        prob_sa_us = model_sa_us.predict_proba(X_te_p_us)[:, 1]
        pred_sa_us = (prob_sa_us >= threshold_sa_us).astype(int)
        eval_df = y_te_p_us.reset_index(drop=True).copy()
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
        model_m, threshold_m = rf_evaluate(
            X_tr_p_m,
            y_tr_p_m,
            feat_names=proposed_features,
            random_state=seed,
        )
        prob_m = model_m.predict_proba(X_te_p_m)[:, 1]
        pred_m = (prob_m >= threshold_m).astype(int)
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
        model_f, threshold_f = rf_evaluate(
            X_tr_p_f,
            y_tr_p_f,
            feat_names=proposed_features,
            random_state=seed,
        )
        prob_f = model_f.predict_proba(X_te_p_f)[:, 1]
        pred_f = (prob_f >= threshold_f).astype(int)
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
        model_sa, threshold_sa = rf_evaluate(
            X_tr_r,
            y_tr_r,
            feat_names=real_proposed_features,
            random_state=seed,
        )
        prob_sa = model_sa.predict_proba(X_te_r)[:, 1]
        pred_sa = (prob_sa >= threshold_sa).astype(int)
        eval_df = y_te_r.reset_index(drop=True).copy()
        eval_df["pred"] = pred_sa
        m_rate, f_rate = incidence_rate(eval_df, "pred", label)
        y_true = y_te_r[label].values
        mask_m = eval_df["Female"] == 0
        mask_f = eval_df["Female"] == 1

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
        model_sa_us, threshold_sa_us = rf_evaluate(
            X_tr_r_us,
            y_tr_r_us,
            feat_names=real_proposed_features,
            random_state=seed,
        )
        prob_sa_us = model_sa_us.predict_proba(X_te_r_us)[:, 1]
        pred_sa_us = (prob_sa_us >= threshold_sa_us).astype(int)
        eval_df = y_te_r_us.reset_index(drop=True).copy()
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
        model_m, threshold_m = rf_evaluate(
            X_tr_r_m,
            y_tr_r_m,
            feat_names=real_proposed_features,
            random_state=seed,
        )
        prob_m = model_m.predict_proba(X_te_r_m)[:, 1]
        pred_m = (prob_m >= threshold_m).astype(int)
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
        model_f, threshold_f = rf_evaluate(
            X_tr_r_f,
            y_tr_r_f,
            feat_names=real_proposed_features,
            random_state=seed,
        )
        prob_f = model_f.predict_proba(X_te_r_f)[:, 1]
        pred_f = (prob_f >= threshold_f).astype(int)
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
            mu = np.nanmean(arr)
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
    formatted = summary_df.apply(
        lambda row: f"{row['mean']:.3f} ({row['ci_lower']:.3f}, {row['ci_upper']:.3f})",
        axis=1,
    )
    summary_table = formatted.unstack(level=1)
    rows_to_drop = [
        "Benchmark Male", "Benchmark Female", "Proposed Male", "Proposed Female",
        "Real Proposed Male", "Real Proposed Female"
    ]
    summary_table = summary_table.drop(index=rows_to_drop)

    # Save result
    output_dir = "/workspace/output"
    output_file = "summary_results.xlsx"
    full_path = f"{output_dir}/{output_file}"
    summary_table.to_excel(full_path, index=True)
    print(f"Summary table saved to: {full_path}")
    return results, summary_table


N_SPLITS = 10
res, summary = multiple_random_splits(train_df, N_SPLITS)
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
        n_iter=20,
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
    train_prob_m = best_male.predict_proba(X_train_m)[:, 1]
    best_thr_m = find_best_threshold(y_train_m["VT/VF/SCD"], train_prob_m)
    prob_m = best_male.predict_proba(X_test_m)[:, 1]
    pred_m = (prob_m >= best_thr_m).astype(int)
    df.loc[df["Female"] == 0, "pred_male"] = pred_m
    df.loc[df["Female"] == 0, "prob_male"] = prob_m
    # Female model
    search.fit(X_train_f, y_train_f)
    best_female = search.best_estimator_
    train_prob_f = best_female.predict_proba(X_train_f)[:, 1]
    best_thr_f = find_best_threshold(y_train_f["VT/VF/SCD"], train_prob_f)
    prob_f = best_female.predict_proba(X_test_f)[:, 1]
    pred_f = (prob_f >= best_thr_f).astype(int)
    df.loc[df["Female"] == 1, "pred_female"] = pred_f
    df.loc[df["Female"] == 1, "prob_female"] = prob_f
    # 合并预测
    df["pred_sexspecific"] = np.nan
    df["prob_sexspecific"] = np.nan
    if "pred_male" in df.columns:
        df.loc[df["Female"] == 0, "pred_sexspecific"] = df.loc[df["Female"] == 0, "pred_male"]
        df.loc[df["Female"] == 0, "prob_sexspecific"] = df.loc[df["Female"] == 0, "prob_male"]
    if "pred_female" in df.columns:
        df.loc[df["Female"] == 1, "pred_sexspecific"] = df.loc[df["Female"] == 1, "pred_female"]
        df.loc[df["Female"] == 1, "prob_sexspecific"] = df.loc[df["Female"] == 1, "prob_female"]
    pred_labels = df[["MRN", "pred_sexspecific"]].drop_duplicates()
    merged_df = survival_df.merge(pred_labels, on="MRN", how="inner").drop_duplicates(subset=["MRN"])
    # 仅保留生存分析和Cox部分，去除可视化和聚类
    kmf = KaplanMeierFitter()
    endpoints = [
        ("Primary Endpoint", "PE_Time", "Was Primary Endpoint Reached? (Appropriate ICD Therapy)"),
        ("Secondary Endpoint", "SE_Time", "Was Secondary Endpoint Reached?")
    ]
    groupings = [("Sex-Specific grouping", "pred_sexspecific")]
    for ep_name, time_col, event_col in endpoints:
        title, pred_col = groupings[0]
        mask_low = merged_df[pred_col] == 0
        mask_high = merged_df[pred_col] == 1
        n_low = mask_low.sum()
        n_high = mask_high.sum()
        total_n = n_low + n_high
        events_low = merged_df.loc[mask_low, event_col].sum()
        events_high = merged_df.loc[mask_high, event_col].sum()
        total_events = events_low + events_high
        lr = logrank_test(
            merged_df.loc[mask_low, time_col],
            merged_df.loc[mask_high, time_col],
            merged_df.loc[mask_low, event_col],
            merged_df.loc[mask_high, event_col],
        )
        p_value = lr.p_value
        print(f"{ep_name} - Low risk: n={n_low}, events={events_low}")
        print(f"{ep_name} - High risk: n={n_high}, events={events_high}")
        print(f"{ep_name} - Total: n={total_n}, events={total_events}")
        print(f"Log-rank p = {p_value:.5f}")
    # Cox PH model
    cph_feature = df[["MRN"] + features]
    cph_df = survival_df.merge(cph_feature, on="MRN", how="inner").drop_duplicates(subset=["MRN"])
    covariates = [col for col in cph_df.columns if col not in [
        "MRN", "PE_Time", "Was Primary Endpoint Reached? (Appropriate ICD Therapy)",
        "SE_Time", "Was Secondary Endpoint Reached?"
    ]]
    formula_terms = [f"`{col}`" if re.search(r"[^a-zA-Z0-9_]", col) else col for col in covariates]
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
    return None

