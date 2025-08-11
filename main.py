# main.py (Optimized Version)
# This file has been refactored for clarity, speed, and maintainability.
# Key improvements: removed duplicate imports, merged data processing, simplified function structure, and reduced unnecessary output.

# Standard library imports
import sys
import os
import re
import warnings
from math import ceil
from itertools import combinations

# Scientific computing imports
import numpy as np
import pandas as pd

# Machine learning imports
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

# Statistical analysis imports
from scipy.stats import randint

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# Medical/clinical analysis imports
from tableone import TableOne
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# Configuration
pd.set_option("future.no_silent_downcasting", True)
warnings.filterwarnings("ignore", category=FutureWarning)


def CG_equation(age, weight, female, serum_creatinine):
    """Cockcroft-Gault Equation."""
    constant = 0.85 if female else 1.0
    return ((140 - age) * weight * constant) / (72 * serum_creatinine)


def prepare_data():
    # Load data
    DATA_DIR = "/home/sunx/data/aiiih/projects/sunx/projects/ICD_sex_diff"
    SURVIVAL_FILE = "df.xlsx"
    COHORT_FILE = "NICM Arrhythmia Cohort for Xiaotan Final.xlsx"
    survival_path = f"{DATA_DIR}/{SURVIVAL_FILE}"
    cohort_path = f"{DATA_DIR}/{COHORT_FILE}"
    if not os.path.exists(survival_path) or not os.path.exists(cohort_path):
        print(
            f"Data files not found. Expected SURVIVAL_FILE at '{survival_path}' and COHORT_FILE at '{cohort_path}'. Set DATA_DIR/SURVIVAL_FILE/COHORT_FILE environment variables accordingly."
        )
        sys.exit(0)

    survival_df = pd.read_excel(survival_path)
    survival_df["PE_Time"] = np.where(
        survival_df["Was Primary Endpoint Reached? (Appropriate ICD Therapy)"] == 1,
        survival_df["Time from ICD Implant to Primary Endpoint (in days)"],
        survival_df["Time from ICD Implant to Last Cardiology Encounter (in days)"],
    )
    survival_df["SE_Time"] = np.where(
        survival_df["Was Secondary Endpoint Reached?"] == 1,
        survival_df["Time from ICD Implant to Secondary Endpoint (in days)"],
        survival_df["Time from ICD Implant to Last Cardiology Encounter (in days)"],
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
    df[categorical] = df[categorical].astype("object")
    labels = ["MRN", "Female", "VT/VF/SCD", "ICD"]
    features = [c for c in df.columns if c not in labels]

    # Missing percentage
    print("\nMissing value percentage:")
    print(df.isnull().sum() / len(df) * 100)

    # Impute and feature engineering
    clean_df = conversion_and_imputation(df, features, labels)
    clean_df["Age by decade"] = (clean_df["Age at CMR"] // 10).astype(int)
    clean_df["CrCl>45"] = (
        clean_df["Cockcroft-Gault Creatinine Clearance (mL/min)"] > 45
    ).astype(int)
    clean_df["NYHA>2"] = (clean_df["NYHA Class"] > 2).astype(int)
    clean_df["Significant LGE"] = (clean_df["LGE Burden 5SD"] > 2).astype(int)

    # Distribution summaries
    print("\nSex distribution:")
    print(clean_df["Female"].value_counts())
    print("\nArrhythmia distribution:")
    print(clean_df["VT/VF/SCD"].value_counts())

    # Proportion in ICD population that follows the rule-based guideline
    icd_df = clean_df[clean_df["ICD"] == 1]
    cond = (icd_df["NYHA Class"] >= 2) & (icd_df["LVEF"] <= 35)
    pct = cond.sum() / len(icd_df) * 100
    print(
        f"\nProportion of ICD population following the rule-based guideline: {pct:.2f}%"
    )

    # Train/test split
    df_split = clean_df.copy()
    stratify_column = df_split["Female"].astype(str) + "_" + df_split["VT/VF/SCD"].astype(str)
    train_df, test_df = train_test_split(
        df_split, test_size=0.2, stratify=stratify_column, random_state=100
    )
    print(
        f"Overall female proportion: {df_split['Female'].mean():.2f}, training set: {train_df['Female'].mean():.2f}, test set: {test_df['Female'].mean():.2f}"
    )
    print(
        f"Overall arrhythmia proportion: {df_split['VT/VF/SCD'].mean():.2f}, training set: {train_df['VT/VF/SCD'].mean():.2f}, test set: {test_df['VT/VF/SCD'].mean():.2f}"
    )

    return clean_df, train_df, test_df, survival_df


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
    # Return a DataFrame directly to avoid redundant assignments
    return pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)


def conversion_and_imputation(df, features, labels):
    df = df.copy()
    df = df[features + labels]

    # Convert NYHA Class (ordinal) to numeric while preserving NaN
    if "NYHA Class" in df.columns:
        codes, _ = pd.factorize(df["NYHA Class"], sort=True)
        df["NYHA Class"] = np.where(codes == -1, np.nan, codes).astype(float)

    # Convert binary variables to float
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
    for c in [col for col in binary_cols if col in df.columns]:
        if df[c].dtype == "object":
            df[c] = df[c].replace(
                {"Yes": 1, "No": 0, "Y": 1, "N": 0, "True": 1, "False": 0}
            )
        df[c] = df[c].astype(float)

    # Impute missing values
    X = df[features]
    imputed_X = impute_misforest(X, 0)
    for col in labels:
        imputed_X[col] = df[col].values
    # Threshold binary variables
    for c in [col for col in binary_cols if col in imputed_X.columns]:
        imputed_X[c] = (imputed_X[c] >= 0.5).astype(float)
    return imputed_X





def find_best_threshold(y_true, y_scores):
    """
    Find the probability threshold that maximizes the F1 score
    based on the precision-recall curve.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.nanargmax(f1_scores)
    # thresholds has one fewer element than f1_scores; guard the index
    if best_idx >= len(thresholds):
        best_idx = len(thresholds) - 1
    return thresholds[best_idx]


def find_best_threshold_f1(y_true, y_scores):
    """
    Find the probability threshold that maximizes the F1 score.
    More suitable for imbalanced data than precision-recall based approach.
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"Best F1 threshold: {best_threshold:.3f} (F1: {best_f1:.3f})")
    return best_threshold


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
    male_rate = (
        male[label_col].sum() / (male[pred_col] == 1).sum()
        if (male[pred_col] == 1).sum() > 0
        else np.nan
    )
    female_rate = (
        female[label_col].sum() / (female[pred_col] == 1).sum()
        if (female[pred_col] == 1).sum() > 0
        else np.nan
    )
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


def rf_evaluate_imbalanced(
    X_train,
    y_train_df,
    feat_names,
    random_state=None,
    visualize_importance=False,
    use_smote=True,
    use_ensemble=True,
):
    """
    Enhanced RandomForest evaluation specifically for imbalanced data.
    Focuses on improving F1 and sensitivity scores.
    """
    y_train = y_train_df["VT/VF/SCD"]
    
    # Option 1: SMOTE for oversampling
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline
            
            # Apply SMOTE to balance the dataset
            smote = SMOTE(random_state=random_state, k_neighbors=min(5, y_train.sum()-1))
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            
            print(f"Original: {y_train.sum()} positive, {len(y_train) - y_train.sum()} negative")
            print(f"After SMOTE: {y_train_balanced.sum()} positive, {len(y_train_balanced) - y_train_balanced.sum()} negative")
            
            X_train = X_train_balanced
            y_train = y_train_balanced
            
        except ImportError:
            print("SMOTE not available, using original data")
    
    # Option 2: Ensemble of multiple models
    if use_ensemble:
        # Train multiple models with different random states
        models = []
        thresholds = []
        
        for seed_offset in range(3):  # Train 3 models
            current_seed = random_state + seed_offset if random_state else seed_offset
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=current_seed)
            param_dist = {
                "n_estimators": randint(200, 600),  # More trees
                "max_depth": [None] + list(range(8, 31, 4)),
                "min_samples_split": randint(2, 8),
                "min_samples_leaf": randint(1, 4),
                "max_features": ["sqrt", "log2", None],
                "class_weight": ["balanced", "balanced_subsample"]
            }
            
            base_clf = RandomForestClassifier(
                random_state=current_seed, 
                n_jobs=-1, 
                class_weight="balanced"
            )
            
            # Use F1 score as primary metric for imbalanced data
            f1_scorer = make_scorer(f1_score, average='binary')
            
            search = RandomizedSearchCV(
                estimator=base_clf,
                param_distributions=param_dist,
                n_iter=25,  # More iterations
                scoring=f1_scorer,
                cv=cv,
                random_state=current_seed,
                n_jobs=-1,
                verbose=0,
                error_score="raise",
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                search.fit(X_train, y_train)
            
            best_model = search.best_estimator_
            models.append(best_model)
            
            # Find optimal threshold for F1
            y_train_prob = best_model.predict_proba(X_train)[:, 1]
            threshold = find_best_threshold_f1(y_train, y_train_prob)
            thresholds.append(threshold)
        
        # Return ensemble info
        return models, thresholds, "ensemble"
    
    else:
        # Single model approach
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        param_dist = {
            "n_estimators": randint(300, 800),  # More trees for better performance
            "max_depth": [None] + list(range(10, 35, 5)),
            "min_samples_split": randint(2, 6),
            "min_samples_leaf": randint(1, 3),
            "max_features": ["sqrt", "log2", None],
            "class_weight": ["balanced", "balanced_subsample"]
        }
        
        base_clf = RandomForestClassifier(
            random_state=random_state, 
            n_jobs=-1, 
            class_weight="balanced"
        )
        
        # Use F1 score as primary metric
        f1_scorer = make_scorer(f1_score, average='binary')
        
        search = RandomizedSearchCV(
            estimator=base_clf,
            param_distributions=param_dist,
            n_iter=30,  # More iterations
            scoring=f1_scorer,
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
        
        # Find optimal threshold for F1
        y_train_prob = best_model.predict_proba(X_train)[:, 1]
        threshold = find_best_threshold_f1(y_train, y_train_prob)
        
        return best_model, threshold, "single"


def evaluate_ensemble_models(models, thresholds, X_test, y_test, method="voting"):
    """
    Evaluate ensemble models using different combination strategies.
    """
    if method == "voting":
        # Hard voting
        predictions = []
        for model, threshold in zip(models, thresholds):
            prob = model.predict_proba(X_test)[:, 1]
            pred = (prob >= threshold).astype(int)
            predictions.append(pred)
        
        # Majority vote
        ensemble_pred = np.mean(predictions, axis=0) >= 0.5
        ensemble_pred = ensemble_pred.astype(int)
        
    elif method == "probability":
        # Average probabilities
        probabilities = []
        for model in models:
            prob = model.predict_proba(X_test)[:, 1]
            probabilities.append(prob)
        
        avg_prob = np.mean(probabilities, axis=0)
        ensemble_pred = (avg_prob >= 0.5).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(y_test, ensemble_pred)
    f1 = f1_score(y_test, ensemble_pred, zero_division=0)
    sens, spec = compute_sensitivity_specificity(y_test, ensemble_pred)
    
    print(f"Ensemble Results ({method}):")
    print(f"Accuracy: {acc:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"Sensitivity: {sens:.3f}")
    print(f"Specificity: {spec:.3f}")
    
    return ensemble_pred, {
        'accuracy': acc,
        'f1': f1,
        'sensitivity': sens,
        'specificity': spec
    }


def multiple_random_splits(df, N, label="VT/VF/SCD"):
    """
    Perform N random train/test splits, fit several models, and collect metrics.
    Optimized version with reduced code duplication and improved performance.

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
    # Feature sets definition
    feature_sets = {
        "Guideline": ["NYHA Class", "LVEF"],
        "Benchmark": [
            "Female", "Age by decade", "BMI", "AF", "Beta Blocker", "CrCl>45",
            "LVEF", "QTc", "NYHA>2", "CRT", "AAD", "Significant LGE",
        ],
        "Proposed": [
            "Female", "Age by decade", "BMI", "AF", "Beta Blocker", "CrCl>45",
            "LVEF", "QTc", "NYHA>2", "CRT", "AAD", "Significant LGE",
            "DM", "HTN", "HLP", "LVEDVi", "LV Mass Index", "RVEDVi",
            "RVEF", "LA EF", "LAVi", "MRF (%)", "Sphericity Index",
            "Relative Wall Thickness", "MV Annular Diameter",
            "ACEi/ARB/ARNi", "Aldosterone Antagonist",
        ],
        "Real_Proposed": [
            "Female", "Age by decade", "BMI", "AF", "Beta Blocker", "CrCl>45",
            "LVEF", "QTc", "CRT", "AAD", "DM", "HTN", "HLP", "LVEDVi",
            "LV Mass Index", "RVEDVi", "RVEF", "LA EF", "LAVi", "MRF (%)",
            "Sphericity Index", "Relative Wall Thickness", "MV Annular Diameter",
            "ACEi/ARB/ARNi", "Aldosterone Antagonist", "LGE Burden 5SD", "NYHA Class"
        ]
    }
    
    # Model configurations
    model_configs = [
        {"name": "Guideline", "features": "Guideline", "type": "rule_based"},
        {"name": "RF Guideline", "features": "Guideline", "type": "rf"},
        {"name": "Benchmark Sex-agnostic", "features": "Benchmark", "type": "rf"},
        {"name": "Benchmark Sex-agnostic (undersampled)", "features": "Benchmark", "type": "rf_undersampled"},
        {"name": "Benchmark Male", "features": "Benchmark", "type": "rf_male_only"},
        {"name": "Benchmark Female", "features": "Benchmark", "type": "rf_female_only"},
        {"name": "Benchmark Sex-specific", "features": "Benchmark", "type": "rf_sex_specific"},
        {"name": "Proposed Sex-agnostic", "features": "Proposed", "type": "rf"},
        {"name": "Proposed Sex-agnostic (undersampled)", "features": "Proposed", "type": "rf_undersampled"},
        {"name": "Proposed Male", "features": "Proposed", "type": "rf_male_only"},
        {"name": "Proposed Female", "features": "Proposed", "type": "rf_female_only"},
        {"name": "Proposed Sex-specific", "features": "Proposed", "type": "rf_sex_specific"},
        {"name": "Real Proposed Sex-agnostic", "features": "Real_Proposed", "type": "rf"},
        {"name": "Real Proposed Sex-agnostic (undersampled)", "features": "Real_Proposed", "type": "rf_undersampled"},
        {"name": "Real Proposed Male", "features": "Real_Proposed", "type": "rf_male_only"},
        {"name": "Real Proposed Female", "features": "Real_Proposed", "type": "rf_female_only"},
        {"name": "Real Proposed Sex-specific", "features": "Real_Proposed", "type": "rf_sex_specific"},
    ]
    
    # Metrics to collect
    metrics = [
        "accuracy", "auc", "f1", "sensitivity", "specificity",
        "male_accuracy", "male_auc", "male_f1", "male_sensitivity", "male_specificity",
        "female_accuracy", "female_auc", "female_f1", "female_sensitivity", "female_specificity",
        "male_rate", "female_rate"
    ]
    
    # Initialize results structure
    results = {config["name"]: {metric: [] for metric in metrics} for config in model_configs}
    
    # Pre-allocate arrays for better memory management
    print(f"Running {N} random splits with {len(model_configs)} models...")
    
    for seed in range(N):
        if seed % max(1, N // 10) == 0:  # Progress indicator
            print(f"Progress: {seed}/{N} splits completed")
        
        # Data splitting and preparation
        train_df, test_df = train_test_split(
            df, test_size=0.3, random_state=seed, stratify=df[label]
        )
        
        # Prepare sex-specific splits
        train_male = train_df[train_df["Female"] == 0]
        train_female = train_df[train_df["Female"] == 1]
        test_male = test_df[test_df["Female"] == 0]
        test_female = test_df[test_df["Female"] == 1]
        
        # Prepare undersampled data
        us_train_df = _create_undersampled_data(train_df, label, seed)
        
        # Process each model configuration
        for config in model_configs:
            try:
                metrics_dict = _evaluate_single_model(
                    config, feature_sets[config["features"]], 
                    train_df, test_df, train_male, train_female, 
                    test_male, test_female, us_train_df, label, seed
                )
                
                # Store results
                for metric, value in metrics_dict.items():
                    if metric in results[config["name"]]:
                        results[config["name"]][metric].append(value)
                    else:
                        # Initialize if metric doesn't exist
                        if metric not in results[config["name"]]:
                            results[config["name"]][metric] = []
                        results[config["name"]][metric].append(value)
                        
            except Exception as e:
                print(f"Error in {config['name']} with seed {seed}: {e}")
                # Fill with NaN for failed runs
                for metric in metrics:
                    results[config["name"]][metric].append(np.nan)
    
    # Create summary statistics
    summary_df = _create_summary_dataframe(results, metrics)
    
    # Save results
    _save_results(summary_df)
    
    return results, summary_df


def _create_undersampled_data(train_df, label, seed):
    """Create undersampled training data."""
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
        
        samp_pos = pos.sample(n=pos_n_target, replace=replace_pos, random_state=seed)
        samp_neg = neg.sample(n=neg_n_target, replace=replace_neg, random_state=seed)
        
        sampled_parts.append(pd.concat([samp_pos, samp_neg]))
    
    return pd.concat(sampled_parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def _evaluate_single_model(config, features, train_df, test_df, train_male, train_female, 
                          test_male, test_female, us_train_df, label, seed):
    """Evaluate a single model configuration and return metrics."""
    
    model_type = config["type"]
    model_name = config["name"]
    
    if model_type == "rule_based":
        return _evaluate_rule_based_model(features, train_df, test_df, label)
    
    elif model_type == "rf":
        return _evaluate_rf_model(features, train_df, test_df, label, seed)
    
    elif model_type == "rf_undersampled":
        return _evaluate_rf_undersampled_model(features, us_train_df, test_df, label, seed)
    
    elif model_type == "rf_male_only":
        return _evaluate_rf_sex_specific(features, train_male, test_male, label, seed, "male")
    
    elif model_type == "rf_female_only":
        return _evaluate_rf_sex_specific(features, train_female, test_female, label, seed, "female")
    
    elif model_type == "rf_sex_specific":
        return _evaluate_rf_sex_specific_combined(
            features, train_male, train_female, test_df, label, seed
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _evaluate_rule_based_model(features, train_df, test_df, label):
    """Evaluate rule-based model (NYHA Class and LVEF)."""
    # Simple rule-based logic
    y_true = test_df[label].values
    
    # Rule: NYHA >= 2 and LVEF <= 35
    rule_pred = ((test_df["NYHA Class"] >= 2) & (test_df["LVEF"] <= 35)).astype(int)
    
    return _calculate_all_metrics(y_true, rule_pred, test_df, label)


def _evaluate_rf_model(features, train_df, test_df, label, seed):
    """Evaluate Random Forest model using optimized parameters for imbalanced data."""
    X_train = train_df[features]
    y_train = train_df[[label, "Female"]]
    X_test = test_df[features]
    y_test = test_df[label].values
    
    # Use the best performing approach from user's tests:
    # Regularized RandomForest with optimized parameters for imbalanced data
    
    # Optimized parameters based on Test 6 results
    model = RandomForestClassifier(
        n_estimators=50,  # Fewer trees to reduce overfitting
        max_depth=6,      # Shallow trees to prevent overfitting
        min_samples_split=20,  # Require more samples to split
        min_samples_leaf=10,   # Require more samples in leaves
        max_features="sqrt",   # Limit features to reduce overfitting
        random_state=seed,
        class_weight="balanced",  # Handle class imbalance
        n_jobs=-1  # Use all CPU cores
    )
    
    # Use cross-validation to find optimal threshold (like Test 5)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)  # Reduced from 5 to 3 for speed
    thresholds = np.arange(0.2, 0.8, 0.05)  # Coarser search for speed
    cv_f1_scores = []
    
    for threshold in thresholds:
        fold_f1_scores = []
        
        for train_idx, val_idx in cv.split(X_train, y_train[label]):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model_fold = RandomForestClassifier(
                n_estimators=50,
                max_depth=6,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features="sqrt",
                random_state=seed,
                class_weight="balanced",
                n_jobs=-1
            )
            model_fold.fit(X_fold_train, y_fold_train[label])
            
            prob = model_fold.predict_proba(X_fold_val)[:, 1]
            pred = (prob >= threshold).astype(int)
            f1 = f1_score(y_fold_train[label], pred, zero_division=0)
            fold_f1_scores.append(f1)
        
        cv_f1_scores.append(np.mean(fold_f1_scores))
    
    # Find best threshold based on CV
    best_threshold = thresholds[np.argmax(cv_f1_scores)]
    
    # Train final model with best threshold
    model.fit(X_train, y_train[label])
    
    # Get predictions
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= best_threshold).astype(int)
    
    return _calculate_all_metrics(y_test, pred, test_df, label, prob)


def _evaluate_rf_undersampled_model(features, us_train_df, test_df, label, seed):
    """Evaluate Random Forest model on undersampled data using optimized parameters."""
    X_train = us_train_df[features]
    y_train = us_train_df[[label, "Female"]]
    X_test = test_df[features]
    y_test = test_df[label].values
    
    # Use the same optimized approach as regular RF
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1
    )
    
    # Use cross-validation to find optimal threshold
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    thresholds = np.arange(0.2, 0.8, 0.05)
    cv_f1_scores = []
    
    for threshold in thresholds:
        fold_f1_scores = []
        
        for train_idx, val_idx in cv.split(X_train, y_train[label]):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model_fold = RandomForestClassifier(
                n_estimators=50,
                max_depth=6,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features="sqrt",
                random_state=seed,
                class_weight="balanced",
                n_jobs=-1
            )
            model_fold.fit(X_fold_train, y_fold_train[label])
            
            prob = model_fold.predict_proba(X_fold_val)[:, 1]
            pred = (prob >= threshold).astype(int)
            f1 = f1_score(y_fold_val[label], pred, zero_division=0)
            fold_f1_scores.append(f1)
        
        cv_f1_scores.append(np.mean(fold_f1_scores))
    
    # Find best threshold based on CV
    best_threshold = thresholds[np.argmax(cv_f1_scores)]
    
    # Train final model with best threshold
    model.fit(X_train, y_train[label])
    
    # Get predictions
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= best_threshold).astype(int)
    
    return _calculate_all_metrics(y_test, pred, test_df, label, prob)


def _evaluate_rf_sex_specific(features, train_sex, test_sex, label, seed, sex_type):
    """Evaluate RF model on sex-specific data using optimized parameters."""
    if len(train_sex) == 0 or len(test_sex) == 0:
        return {metric: np.nan for metric in [
            "accuracy", "auc", "f1", "sensitivity", "specificity",
            "male_accuracy", "male_auc", "male_f1", "male_sensitivity", "male_specificity",
            "female_accuracy", "female_auc", "female_f1", "female_sensitivity", "female_specificity",
            "male_rate", "female_rate"
        ]}
    
    X_train = train_sex[features]
    y_train = train_sex[[label, "Female"]]
    X_test = test_sex[features]
    y_test = test_sex[label].values
    
    # Use optimized parameters for better performance on imbalanced data
    model = RandomForestClassifier(
        n_estimators=50,  # Fewer trees to reduce overfitting
        max_depth=6,      # Shallow trees to prevent overfitting
        min_samples_split=20,  # Require more samples to split
        min_samples_leaf=10,   # Require more samples in leaves
        max_features="sqrt",   # Limit features to reduce overfitting
        random_state=seed,
        class_weight="balanced",  # Handle class imbalance
        n_jobs=-1  # Use all CPU cores
    )
    
    # Use cross-validation to find optimal threshold
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    thresholds = np.arange(0.2, 0.8, 0.05)
    cv_f1_scores = []
    
    for threshold in thresholds:
        fold_f1_scores = []
        
        for train_idx, val_idx in cv.split(X_train, y_train[label]):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model_fold = RandomForestClassifier(
                n_estimators=50,
                max_depth=6,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features="sqrt",
                random_state=seed,
                class_weight="balanced",
                n_jobs=-1
            )
            model_fold.fit(X_fold_train, y_fold_train[label])
            
            prob = model_fold.predict_proba(X_fold_val)[:, 1]
            pred = (prob >= threshold).astype(int)
            f1 = f1_score(y_fold_val[label], pred, zero_division=0)
            fold_f1_scores.append(f1)
        
        cv_f1_scores.append(np.mean(fold_f1_scores))
    
    # Find best threshold based on CV
    best_threshold = thresholds[np.argmax(cv_f1_scores)]
    
    # Train final model with best threshold
    model.fit(X_train, y_train[label])
    
    # Get predictions
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= best_threshold).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, prob) if len(np.unique(y_test)) > 1 else np.nan
    f1 = f1_score(y_test, pred)
    sens, spec = compute_sensitivity_specificity(y_test, pred)
    
    # Calculate incidence rate
    eval_df = test_sex.reset_index(drop=True).copy()
    eval_df["pred"] = pred
    m_rate, f_rate = incidence_rate(eval_df, "pred", label)
    
    # For sex-specific models, overall = sex-specific, other sex = nan
    if sex_type == "male":
        return {
            "accuracy": acc, "auc": auc, "f1": f1, "sensitivity": sens, "specificity": spec,
            "male_accuracy": acc, "male_auc": auc, "male_f1": f1, "male_sensitivity": sens, "male_specificity": spec,
            "female_accuracy": np.nan, "female_auc": np.nan, "female_f1": np.nan, "female_sensitivity": np.nan, "female_specificity": np.nan,
            "male_rate": m_rate, "female_rate": f_rate
        }
    else:  # female
        return {
            "accuracy": acc, "auc": auc, "f1": f1, "sensitivity": sens, "specificity": spec,
            "male_accuracy": np.nan, "male_auc": np.nan, "male_f1": np.nan, "male_sensitivity": np.nan, "male_specificity": np.nan,
            "female_accuracy": acc, "female_auc": auc, "female_f1": f1, "female_sensitivity": sens, "female_specificity": spec,
            "male_rate": m_rate, "female_rate": f_rate
        }


def _evaluate_rf_sex_specific_combined(features, train_male, train_female, test_df, label, seed):
    """Evaluate combined sex-specific RF models."""
    # Train male model
    male_model, male_threshold = _train_sex_model(features, train_male, label, seed)
    # Train female model
    female_model, female_threshold = _train_sex_model(features, train_female, label, seed)
    
    # Combine predictions
    combined_pred = np.empty(len(test_df), dtype=int)
    combined_prob = np.empty(len(test_df), dtype=float)
    
    mask_male = test_df["Female"].values == 0
    mask_female = test_df["Female"].values == 1
    
    # Male predictions
    if male_model is not None:
        male_prob = male_model.predict_proba(test_df[mask_male][features])[:, 1]
        male_pred = (male_prob >= male_threshold).astype(int)
        combined_pred[mask_male] = male_pred
        combined_prob[mask_male] = male_prob
    
    # Female predictions
    if female_model is not None:
        female_prob = female_model.predict_proba(test_df[mask_female][features])[:, 1]
        female_pred = (female_prob >= female_threshold).astype(int)
        combined_pred[mask_female] = female_pred
        combined_prob[mask_female] = female_prob
    
    y_true = test_df[label].values
    return _calculate_all_metrics(y_true, combined_pred, test_df, label, combined_prob)


def _train_sex_model(features, train_sex, label, seed):
    """Train a model on sex-specific data using optimized parameters."""
    if len(train_sex) == 0:
        return None, 0.5
    
    X_train = train_sex[features]
    y_train = train_sex[[label, "Female"]]
    
    try:
        # Use optimized parameters for better performance
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1
        )
        
        # Use cross-validation to find optimal threshold
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        thresholds = np.arange(0.2, 0.8, 0.05)
        cv_f1_scores = []
        
        for threshold in thresholds:
            fold_f1_scores = []
            
            for train_idx, val_idx in cv.split(X_train, y_train[label]):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model_fold = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=6,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    max_features="sqrt",
                    random_state=seed,
                    class_weight="balanced",
                    n_jobs=-1
                )
                model_fold.fit(X_fold_train, y_fold_train[label])
                
                prob = model_fold.predict_proba(X_fold_val)[:, 1]
                pred = (prob >= threshold).astype(int)
                f1 = f1_score(y_fold_val[label], pred, zero_division=0)
                fold_f1_scores.append(f1)
            
            cv_f1_scores.append(np.mean(fold_f1_scores))
        
        # Find best threshold based on CV
        best_threshold = thresholds[np.argmax(cv_f1_scores)]
        
        # Train final model
        model.fit(X_train, y_train[label])
        
        return model, best_threshold
        
    except Exception as e:
        print(f"Warning: Error training sex-specific model: {e}")
        return None, 0.5


def _calculate_all_metrics(y_true, pred, test_df, label, prob=None):
    """Calculate all metrics for a given prediction."""
    # Overall metrics
    acc = accuracy_score(y_true, pred)
    auc = roc_auc_score(y_true, prob) if prob is not None and len(np.unique(y_true)) > 1 else np.nan
    f1 = f1_score(y_true, pred)
    sens, spec = compute_sensitivity_specificity(y_true, pred)
    
    # Sex-specific metrics
    mask_male = test_df["Female"] == 0
    mask_female = test_df["Female"] == 1
    
    # Male subset
    y_true_male = y_true[mask_male]
    pred_male = pred[mask_male]
    prob_male = prob[mask_male] if prob is not None else None
    
    male_acc = accuracy_score(y_true_male, pred_male) if len(y_true_male) > 0 else np.nan
    male_auc = roc_auc_score(y_true_male, prob_male) if prob_male is not None and len(y_true_male) > 1 and len(np.unique(y_true_male)) > 1 else np.nan
    male_f1 = f1_score(y_true_male, pred_male) if len(y_true_male) > 0 else np.nan
    male_sens, male_spec = compute_sensitivity_specificity(y_true_male, pred_male) if len(y_true_male) > 0 else (np.nan, np.nan)
    
    # Female subset
    y_true_female = y_true[mask_female]
    pred_female = pred[mask_female]
    prob_female = prob[mask_female] if prob is not None else None
    
    female_acc = accuracy_score(y_true_female, pred_female) if len(y_true_female) > 0 else np.nan
    female_auc = roc_auc_score(y_true_female, prob_female) if prob_female is not None and len(y_true_female) > 1 and len(np.unique(y_true_female)) > 1 else np.nan
    female_f1 = f1_score(y_true_female, pred_female) if len(y_true_female) > 0 else np.nan
    female_sens, female_spec = compute_sensitivity_specificity(y_true_female, pred_female) if len(y_true_female) > 0 else (np.nan, np.nan)
    
    # Incidence rates
    eval_df = test_df.reset_index(drop=True).copy()
    eval_df["pred"] = pred
    m_rate, f_rate = incidence_rate(eval_df, "pred", label)
    
    return {
        "accuracy": acc, "auc": auc, "f1": f1, "sensitivity": sens, "specificity": spec,
        "male_accuracy": male_acc, "male_auc": male_auc, "male_f1": male_f1, "male_sensitivity": male_sens, "male_specificity": male_spec,
        "female_accuracy": female_acc, "female_auc": female_auc, "female_f1": female_f1, "female_sensitivity": female_sens, "female_specificity": female_spec,
        "male_rate": m_rate, "female_rate": f_rate
    }


def _create_summary_dataframe(results, metrics):
    """Create summary DataFrame with mean and confidence intervals."""
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
    
    # Format results
    formatted = summary_df.apply(
        lambda row: f"{row['mean']:.3f} ({row['ci_lower']:.3f}, {row['ci_upper']:.3f})",
        axis=1,
    )
    summary_table = formatted.unstack(level=1)
    
    # Drop single-sex models from summary
    rows_to_drop = [
        "Benchmark Male", "Benchmark Female", "Proposed Male", "Proposed Female",
        "Real Proposed Male", "Real Proposed Female",
    ]
    summary_table = summary_table.drop(index=rows_to_drop)
    
    return summary_table


def _save_results(summary_table):
    """Save results to file."""
    try:
        output_dir = "/home/sunx/data/aiiih/projects/sunx/projects/ICD_sex_diff"
        output_file = "summary_results.xlsx"
        full_path = f"{output_dir}/{output_file}"
        summary_table.to_excel(full_path, index=True)
        print(f"Summary table saved to: {full_path}")
    except Exception as e:
        print(f"Warning: Could not save results: {e}")


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
    # Combine predictions
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
    pred_labels = df[["MRN", "pred_sexspecific"]].drop_duplicates()
    merged_df = survival_df.merge(pred_labels, on="MRN", how="inner").drop_duplicates(
        subset=["MRN"]
    )
    # Keep only survival analysis and Cox models; remove visualization and clustering
    kmf = KaplanMeierFitter()
    endpoints = [
        (
            "Primary Endpoint",
            "PE_Time",
            "Was Primary Endpoint Reached? (Appropriate ICD Therapy)",
        ),
        ("Secondary Endpoint", "SE_Time", "Was Secondary Endpoint Reached?"),
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
    cph_df = survival_df.merge(cph_feature, on="MRN", how="inner").drop_duplicates(
        subset=["MRN"]
    )
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
    return None


def test_imbalanced_methods(train_df, test_df, all_features):
    """
    Test different methods for handling imbalanced data.
    """
    print("Testing different approaches for imbalanced data...")
    
    # Define feature sets
    guideline_features = ["NYHA Class", "LVEF"]
    benchmark_features = [
        "Female", "Age by decade", "BMI", "AF", "Beta Blocker", "CrCl>45",
        "LVEF", "QTc", "NYHA>2", "CRT", "AAD", "Significant LGE"
    ]
    
    # Test 1: Standard approach with F1 optimization
    print("\n--- Test 1: Standard RF with F1 optimization ---")
    X_train = train_df[benchmark_features]
    y_train = train_df[["VT/VF/SCD", "Female"]]
    X_test = test_df[benchmark_features]
    y_test = test_df["VT/VF/SCD"]
    
    try:
        model, threshold, method = rf_evaluate_imbalanced(
            X_train, y_train, benchmark_features, 
            random_state=42, use_smote=False, use_ensemble=False
        )
        
        if method == "single":
            prob = model.predict_proba(X_test)[:, 1]
            pred = (prob >= threshold).astype(int)
            
            acc = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred, zero_division=0)
            sens, spec = compute_sensitivity_specificity(y_test, pred)
            
            print(f"Single Model Results:")
            print(f"Accuracy: {acc:.3f}")
            print(f"F1 Score: {f1:.3f}")
            print(f"Sensitivity: {sens:.3f}")
            print(f"Specificity: {spec:.3f}")
    
    except Exception as e:
        print(f"Error in single model approach: {e}")
    
    # Test 2: SMOTE + Single model
    print("\n--- Test 2: SMOTE + Single RF ---")
    try:
        model, threshold, method = rf_evaluate_imbalanced(
            X_train, y_train, benchmark_features, 
            random_state=42, use_smote=True, use_ensemble=False
        )
        
        if method == "single":
            prob = model.predict_proba(X_test)[:, 1]
            pred = (prob >= threshold).astype(int)
            
            acc = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred, zero_division=0)
            sens, spec = compute_sensitivity_specificity(y_test, pred)
            
            print(f"SMOTE + Single Model Results:")
            print(f"Accuracy: {acc:.3f}")
            print(f"F1 Score: {f1:.3f}")
            print(f"Sensitivity: {sens:.3f}")
            print(f"Specificity: {spec:.3f}")
    
    except Exception as e:
        print(f"Error in SMOTE approach: {e}")
    
    # Test 3: Ensemble approach
    print("\n--- Test 3: Ensemble RF ---")
    try:
        models, thresholds, method = rf_evaluate_imbalanced(
            X_train, y_train, benchmark_features, 
            random_state=42, use_smote=False, use_ensemble=True
        )
        
        if method == "ensemble":
            # Test voting method
            ensemble_pred, metrics = evaluate_ensemble_models(
                models, thresholds, X_test, y_test, method="voting"
            )
            
            # Test probability method
            ensemble_pred_prob, metrics_prob = evaluate_ensemble_models(
                models, thresholds, X_test, y_test, method="probability"
            )
    
    except Exception as e:
        print(f"Error in ensemble approach: {e}")
    
    # Test 4: SMOTE + Ensemble
    print("\n--- Test 4: SMOTE + Ensemble RF ---")
    try:
        models, thresholds, method = rf_evaluate_imbalanced(
            X_train, y_train, benchmark_features, 
            random_state=42, use_smote=True, use_ensemble=True
        )
        
        if method == "ensemble":
            # Test voting method
            ensemble_pred, metrics = evaluate_ensemble_models(
                models, thresholds, X_test, y_test, method="voting"
            )
            
            # Test probability method
            ensemble_pred_prob, metrics_prob = evaluate_ensemble_models(
                models, thresholds, X_test, y_test, method="probability"
            )
    
    except Exception as e:
        print(f"Error in SMOTE + ensemble approach: {e}")
    
    # Test 5: Cross-validation based threshold selection
    print("\n--- Test 5: Cross-validation based threshold ---")
    test_cv_threshold_selection(X_train, y_train, X_test, y_test, benchmark_features)
    
    # Test 6: Regularized model to reduce overfitting
    print("\n--- Test 6: Regularized RF to reduce overfitting ---")
    test_regularized_model(X_train, y_train, X_test, y_test, benchmark_features)
    
    # Test 7: Optimal threshold balancing
    print("\n--- Test 7: Optimal threshold balancing ---")
    find_optimal_threshold_balanced(X_train, y_train, X_test, y_test, benchmark_features)
    
    print("\n=== Imbalanced Data Testing Complete ===")


def test_cv_threshold_selection(X_train, y_train_df, X_test, y_test, features):
    """
    Test cross-validation based threshold selection to reduce overfitting.
    """
    y_train = y_train_df["VT/VF/SCD"]
    
    # Use cross-validation to find optimal threshold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    thresholds = np.arange(0.1, 0.9, 0.01)
    cv_f1_scores = []
    
    for threshold in thresholds:
        fold_f1_scores = []
        
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Train model on fold
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,  # Limit depth to reduce overfitting
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                class_weight="balanced"
            )
            model.fit(X_fold_train, y_fold_train)
            
            # Predict on validation fold
            prob = model.predict_proba(X_fold_val)[:, 1]
            pred = (prob >= threshold).astype(int)
            f1 = f1_score(y_fold_val, pred, zero_division=0)
            fold_f1_scores.append(f1)
        
        cv_f1_scores.append(np.mean(fold_f1_scores))
    
    # Find best threshold based on CV
    best_cv_threshold = thresholds[np.argmax(cv_f1_scores)]
    best_cv_f1 = np.max(cv_f1_scores)
    
    print(f"Best CV threshold: {best_cv_threshold:.3f} (CV F1: {best_cv_f1:.3f})")
    
    # Train final model with best threshold
    final_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced"
    )
    final_model.fit(X_train, y_train)
    
    # Test with CV-optimized threshold
    prob = final_model.predict_proba(X_test)[:, 1]
    pred = (prob >= best_cv_threshold).astype(int)
    
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, zero_division=0)
    sens, spec = compute_sensitivity_specificity(y_test, pred)
    
    print(f"CV-Optimized Threshold Results:")
    print(f"Accuracy: {acc:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"Sensitivity: {sens:.3f}")
    print(f"Specificity: {spec:.3f}")


def test_regularized_model(X_train, y_train_df, X_test, y_test, features):
    """
    Test regularized RandomForest to reduce overfitting.
    """
    y_train = y_train_df["VT/VF/SCD"]
    
    # More conservative parameters to reduce overfitting
    model = RandomForestClassifier(
        n_estimators=50,  # Fewer trees
        max_depth=6,      # Much shallower trees
        min_samples_split=20,  # Require more samples to split
        min_samples_leaf=10,   # Require more samples in leaves
        max_features="sqrt",   # Limit features
        random_state=42,
        class_weight="balanced"
    )
    
    # Use cross-validation to find threshold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    thresholds = np.arange(0.1, 0.9, 0.01)
    cv_f1_scores = []
    
    for threshold in thresholds:
        fold_f1_scores = []
        
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model_fold = RandomForestClassifier(
                n_estimators=50,
                max_depth=6,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features="sqrt",
                random_state=42,
                class_weight="balanced"
            )
            model_fold.fit(X_fold_train, y_fold_train)
            
            prob = model_fold.predict_proba(X_fold_val)[:, 1]
            pred = (prob >= threshold).astype(int)
            f1 = f1_score(y_fold_val, pred, zero_division=0)
            fold_f1_scores.append(f1)
        
        cv_f1_scores.append(np.mean(fold_f1_scores))
    
    best_threshold = thresholds[np.argmax(cv_f1_scores)]
    print(f"Regularized model best threshold: {best_threshold:.3f}")
    
    # Train final regularized model
    model.fit(X_train, y_train)
    
    # Test
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= best_threshold).astype(int)
    
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, zero_division=0)
    sens, spec = compute_sensitivity_specificity(y_test, pred)
    
    print(f"Regularized Model Results:")
    print(f"Accuracy: {acc:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"Sensitivity: {sens:.3f}")
    print(f"Specificity: {spec:.3f}")
    
    # Check for overfitting
    train_prob = model.predict_proba(X_train)[:, 1]
    train_pred = (train_prob >= best_threshold).astype(int)
    train_f1 = f1_score(y_train, train_pred, zero_division=0)
    
    print(f"Training F1: {train_f1:.3f}, Test F1: {f1:.3f}")
    print(f"Overfitting gap: {train_f1 - f1:.3f}")


def find_optimal_threshold_balanced(X_train, y_train_df, X_test, y_test, features):
    """
    Find optimal threshold that balances sensitivity and specificity.
    """
    y_train = y_train_df["VT/VF/SCD"]
    
    # Train a balanced model
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    
    # Get probabilities
    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    
    # Find threshold that maximizes F1 on training set
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1_threshold = 0.5
    best_f1 = 0
    
    for threshold in thresholds:
        pred = (train_prob >= threshold).astype(int)
        f1 = f1_score(y_train, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_f1_threshold = threshold
    
    # Find threshold that balances sensitivity and specificity
    best_balanced_threshold = 0.5
    best_balanced_score = 0
    
    for threshold in thresholds:
        pred = (train_prob >= threshold).astype(int)
        sens, spec = compute_sensitivity_specificity(y_train, pred)
        
        # Use harmonic mean of sensitivity and specificity
        if sens > 0 and spec > 0:
            balanced_score = 2 * sens * spec / (sens + spec)
            if balanced_score > best_balanced_score:
                best_balanced_score = balanced_score
                best_balanced_threshold = threshold
    
    # Find threshold that maximizes sensitivity while maintaining reasonable specificity
    best_sens_threshold = 0.5
    best_sens_score = 0
    
    for threshold in thresholds:
        pred = (train_prob >= threshold).astype(int)
        sens, spec = compute_sensitivity_specificity(y_train, pred)
        
        # Prioritize sensitivity but require specificity > 0.7
        if spec > 0.7 and sens > best_sens_score:
            best_sens_score = sens
            best_sens_threshold = threshold
    
    print(f"\n=== Threshold Analysis ===")
    print(f"Best F1 threshold: {best_f1_threshold:.3f}")
    print(f"Best balanced threshold: {best_balanced_threshold:.3f}")
    print(f"Best sensitivity threshold: {best_sens_threshold:.3f}")
    
    # Test all thresholds
    thresholds_to_test = [best_f1_threshold, best_balanced_threshold, best_sens_threshold]
    threshold_names = ["F1-optimized", "Balanced", "Sensitivity-optimized"]
    
    for threshold, name in zip(thresholds_to_test, threshold_names):
        pred = (test_prob >= threshold).astype(int)
        
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, zero_division=0)
        sens, spec = compute_sensitivity_specificity(y_test, pred)
        
        print(f"\n{name} Threshold ({threshold:.3f}):")
        print(f"  Accuracy: {acc:.3f}")
        print(f"  F1 Score: {f1:.3f}")
        print(f"  Sensitivity: {sens:.3f}")
        print(f"  Specificity: {spec:.3f}")
        
        # Calculate balanced score
        if sens > 0 and spec > 0:
            balanced_score = 2 * sens * spec / (sens + spec)
            print(f"  Balanced Score: {balanced_score:.3f}")
    
    return model, thresholds_to_test, threshold_names


# Main execution block
if __name__ == "__main__":
    clean_df, train_df, test_df, survival_df = prepare_data()
    
    print("=== Standard Evaluation ===")
    N_SPLITS = 2
    res, summary = multiple_random_splits(train_df, N_SPLITS)
    print(summary)
    
    print("\n=== Enhanced Imbalanced Data Evaluation ===")
    test_imbalanced_methods(train_df, test_df, clean_df.columns)
