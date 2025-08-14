import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.metrics import (
    make_scorer, average_precision_score, accuracy_score, roc_auc_score, 
    f1_score, recall_score, precision_recall_curve, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from scipy.stats import randint
import warnings
from math import ceil


def find_best_threshold(y_true, y_scores):
    """Find the probability threshold that maximizes the F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.nanargmax(f1_scores[:-1])
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


def rf_evaluate(X_train, y_train_df, X_test, y_test_df, feat_names, random_state=None, visualize_importance=False):
    """Train RandomForest with randomized search and return predictions.

    Uses out-of-fold predictions on the training set to select a robust
    probability threshold (maximizing F1) without leaking test labels.
    """
    y_train = y_train_df["VT/VF/SCD"]
    y_test = y_test_df["VT/VF/SCD"]
    
    # Hyperparameter search
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
    
    # Best hyperparameters
    best_params = search.best_params_
    
    # Determine threshold using OOF probabilities on the training set
    oof_proba = np.zeros(len(y_train), dtype=float)
    for tr_idx, val_idx in cv.split(X_train, y_train):
        oof_model = RandomForestClassifier(
            **best_params, random_state=random_state, n_jobs=-1, class_weight="balanced"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            oof_model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        oof_proba[val_idx] = oof_model.predict_proba(X_train.iloc[val_idx])[:, 1]
    threshold = find_best_threshold(y_train.values, oof_proba)
    
    # Fit final model on the full training data
    final_model = RandomForestClassifier(
        **best_params, random_state=random_state, n_jobs=-1, class_weight="balanced"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model.fit(X_train, y_train)
    
    # Feature importance visualization
    if visualize_importance:
        importances = final_model.feature_importances_
        idx = np.argsort(importances)[::-1]
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
    
    # Predict on test set
    y_prob = final_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    return y_pred, y_prob


def evaluate_model_performance(y_true, y_pred, y_prob, mask_m, mask_f, overall_mask=None):
    """Evaluate model performance for overall, male, and female subsets.

    If overall_mask is provided, overall metrics are computed using only the
    entries where overall_mask is True.
    """
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


def create_undersampled_dataset(train_df, label_col, random_state):
    """Create balanced dataset using undersampling."""
    n_male = (train_df["Female"] == 0).sum()
    n_female = (train_df["Female"] == 1).sum()
    n_target = ceil((n_male + n_female) / 2)
    
    sampled_parts = []
    for sex_val in (0, 1):
        grp = train_df[train_df["Female"] == sex_val]
        pos = grp[grp[label_col] == 1]
        neg = grp[grp[label_col] == 0]
        
        pos_n_target = int(round(len(pos) / len(grp) * n_target))
        neg_n_target = n_target - pos_n_target
        
        replace_pos = pos_n_target > len(pos)
        replace_neg = neg_n_target > len(neg)
        
        samp_pos = pos.sample(n=pos_n_target, replace=replace_pos, random_state=random_state)
        samp_neg = neg.sample(n=neg_n_target, replace=replace_neg, random_state=random_state)
        
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


def run_sex_specific_models(train_m, train_f, test_m, test_f, features, label_col, random_state):
    """Train and evaluate sex-specific models."""
    results = {}
    
    # Train male model
    if not train_m.empty and not test_m.empty:
        pred_m, prob_m = rf_evaluate(
            train_m[features], train_m[[label_col, "Female"]], 
            test_m[features], test_m[[label_col, "Female"]], 
            features, random_state
        )
        results['male'] = {'pred': pred_m, 'prob': prob_m}
    
    # Train female model
    if not train_f.empty and not test_f.empty:
        pred_f, prob_f = rf_evaluate(
            train_f[features], train_f[[label_col, "Female"]], 
            test_f[features], test_f[[label_col, "Female"]], 
            features, random_state
        )
        results['female'] = {'pred': pred_f, 'prob': prob_f}
    
    return results


def multiple_random_splits_simplified(df, N, label="VT/VF/SCD"):
    """Simplified version of multiple random splits evaluation."""
    
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
    
    # Model configurations - now with all 17 models
    model_configs = [
        {'name': 'Guideline', 'features': 'guideline', 'type': 'rule_based'},
        {'name': 'RF Guideline', 'features': 'guideline', 'type': 'ml'},
        {'name': 'Benchmark Sex-agnostic', 'features': 'benchmark', 'type': 'ml'},
        {'name': 'Benchmark Sex-agnostic (undersampled)', 'features': 'benchmark', 'type': 'ml_undersampled'},
        {'name': 'Benchmark Male', 'features': 'benchmark', 'type': 'male_only'},
        {'name': 'Benchmark Female', 'features': 'benchmark', 'type': 'female_only'},
        {'name': 'Benchmark Sex-specific', 'features': 'benchmark', 'type': 'sex_specific'},
        {'name': 'Proposed Sex-agnostic', 'features': 'proposed', 'type': 'ml'},
        {'name': 'Proposed Sex-agnostic (undersampled)', 'features': 'proposed', 'type': 'ml_undersampled'},
        {'name': 'Proposed Male', 'features': 'proposed', 'type': 'male_only'},
        {'name': 'Proposed Female', 'features': 'proposed', 'type': 'female_only'},
        {'name': 'Proposed Sex-specific', 'features': 'proposed', 'type': 'sex_specific'},
        {'name': 'Real Proposed Sex-agnostic', 'features': 'real_proposed', 'type': 'ml'},
        {'name': 'Real Proposed Sex-agnostic (undersampled)', 'features': 'real_proposed', 'type': 'ml_undersampled'},
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
    
    for seed in range(N):
        print(f"Running split #{seed+1}")
        
        # Split data
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=seed, stratify=df[label])
        tr_m = train_df[train_df["Female"] == 0]
        tr_f = train_df[train_df["Female"] == 1]
        te_m = test_df[test_df["Female"] == 0]
        te_f = test_df[test_df["Female"] == 1]
        
        # Create undersampled dataset
        us_train_df = create_undersampled_dataset(train_df, label, seed)
        
        # Create masks for gender subsets
        mask_m = test_df["Female"].values == 0
        mask_f = test_df["Female"].values == 1
        y_true = test_df[label].values
        
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
                    test_df[feature_set], test_df[[label, "Female"]], label
                )
                
            elif model_type == 'ml':
                # Standard ML model
                pred, prob = rf_evaluate(
                    train_df[feature_set], train_df[[label, "Female"]],
                    test_df[feature_set], test_df[[label, "Female"]],
                    feature_set, seed
                )
                eval_df = test_df[[label, "Female"]].reset_index(drop=True).copy()
                eval_df["pred"] = pred
                m_rate, f_rate = incidence_rate(eval_df, "pred", label)
                
            elif model_type == 'ml_undersampled':
                # Undersampled ML model
                pred, prob = rf_evaluate(
                    us_train_df[feature_set], us_train_df[[label, "Female"]],
                    test_df[feature_set], test_df[[label, "Female"]],
                    feature_set, seed
                )
                eval_df = test_df[[label, "Female"]].reset_index(drop=True).copy()
                eval_df["pred"] = pred
                m_rate, f_rate = incidence_rate(eval_df, "pred", label)
                
            elif model_type == 'male_only':
                # Male-only model
                if not tr_m.empty and not te_m.empty:
                    pred, prob = rf_evaluate(
                        tr_m[feature_set], tr_m[[label, "Female"]],
                        te_m[feature_set], te_m[[label, "Female"]],
                        feature_set, seed
                    )
                    # Create full test set predictions (only male predictions)
                    full_pred = np.zeros(len(test_df), dtype=int)
                    full_prob = np.zeros(len(test_df), dtype=float)
                    full_pred[mask_m] = pred
                    full_prob[mask_m] = prob
                    pred, prob = full_pred, full_prob
                    eval_df = test_df[[label, "Female"]].reset_index(drop=True).copy()
                    eval_df["pred"] = pred
                    m_rate, f_rate = incidence_rate(eval_df, "pred", label)
                    # Evaluate overall metrics only on the male subset
                    mask_f_eval = np.zeros_like(mask_f, dtype=bool)
                    overall_mask_override = mask_m
                else:
                    # Handle case with no male data
                    pred = np.zeros(len(test_df), dtype=int)
                    prob = np.zeros(len(test_df), dtype=float)
                    m_rate, f_rate = 0.0, 0.0
                    mask_f_eval = np.zeros_like(mask_f, dtype=bool)
                    overall_mask_override = mask_m
                
            elif model_type == 'female_only':
                # Female-only model
                if not tr_f.empty and not te_f.empty:
                    pred, prob = rf_evaluate(
                        tr_f[feature_set], tr_f[[label, "Female"]],
                        te_f[feature_set], te_f[[label, "Female"]],
                        feature_set, seed
                    )
                    # Create full test set predictions (only female predictions)
                    full_pred = np.zeros(len(test_df), dtype=int)
                    full_prob = np.zeros(len(test_df), dtype=float)
                    full_pred[mask_f] = pred
                    full_prob[mask_f] = prob
                    pred, prob = full_pred, full_prob
                    eval_df = test_df[[label, "Female"]].reset_index(drop=True).copy()
                    eval_df["pred"] = pred
                    m_rate, f_rate = incidence_rate(eval_df, "pred", label)
                    # Evaluate overall metrics only on the female subset
                    mask_m_eval = np.zeros_like(mask_m, dtype=bool)
                    overall_mask_override = mask_f
                else:
                    # Handle case with no female data
                    pred = np.zeros(len(test_df), dtype=int)
                    prob = np.zeros(len(test_df), dtype=float)
                    m_rate, f_rate = 0.0, 0.0
                    mask_m_eval = np.zeros_like(mask_m, dtype=bool)
                    overall_mask_override = mask_f
                
            elif model_type == 'sex_specific':
                # Sex-specific models
                sex_results = run_sex_specific_models(
                    tr_m, tr_f, te_m, te_f, feature_set, label, seed
                )
                
                # Combine predictions
                combined_pred = np.empty(len(test_df), dtype=int)
                combined_prob = np.empty(len(test_df), dtype=float)
                
                if 'male' in sex_results:
                    combined_pred[mask_m] = sex_results['male']['pred']
                    combined_prob[mask_m] = sex_results['male']['prob']
                if 'female' in sex_results:
                    combined_pred[mask_f] = sex_results['female']['pred']
                    combined_prob[mask_f] = sex_results['female']['prob']
                
                pred, prob = combined_pred, combined_prob
                eval_df = test_df[[label, "Female"]].reset_index(drop=True).copy()
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


def rf_train_and_predict_no_test_labels(X_train, y_train_df, X_test, feat_names, random_state=None, visualize_importance=False):
    """Train RF on full training data and predict on X_test without needing test labels.

    - Uses randomized search optimizing average precision
    - Selects threshold from out-of-fold probabilities on training set
    - Fits final model on full training data and predicts probabilities on X_test
    - Returns (y_pred, y_prob, threshold)
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

    best_params = search.best_params_

    # Determine threshold using OOF probabilities on the training set
    oof_proba = np.zeros(len(y_train), dtype=float)
    for tr_idx, val_idx in cv.split(X_train, y_train):
        oof_model = RandomForestClassifier(
            **best_params, random_state=random_state, n_jobs=-1, class_weight="balanced"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            oof_model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        oof_proba[val_idx] = oof_model.predict_proba(X_train.iloc[val_idx])[:, 1]
    threshold = find_best_threshold(y_train.values, oof_proba)

    # Fit final model on the full training data
    final_model = RandomForestClassifier(
        **best_params, random_state=random_state, n_jobs=-1, class_weight="balanced"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model.fit(X_train, y_train)

    # Optional: visualize feature importance
    if visualize_importance:
        importances = final_model.feature_importances_
        idx = np.argsort(importances)[::-1]
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

    # Predict on inference set
    y_prob = final_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return y_pred, y_prob, threshold


def build_survival_based_label(survival_df, id_col, icd_indicator_col, appropriate_icd_shock_col, death_col, output_label_col="DerivedOutcome", icd_source_df=None):
    """Construct binary labels from survival data, optionally merging ICD indicator from another DataFrame.

    Logic:
      - If ICD implanted (icd_indicator_col == 1): label = appropriate_icd_shock_col
      - If no ICD (icd_indicator_col == 0):       label = death_col
    The resulting label is binarized to {0,1}.

    If icd_source_df is provided, the ICD indicator will be taken from icd_source_df (joined on id_col),
    otherwise it is expected to be present in survival_df.

    Returns a DataFrame with columns [id_col, output_label_col].
    """
    if icd_source_df is None:
        out = survival_df[[id_col, icd_indicator_col, appropriate_icd_shock_col, death_col]].copy()
    else:
        out = survival_df[[id_col, appropriate_icd_shock_col, death_col]].copy()
        out = out.merge(icd_source_df[[id_col, icd_indicator_col]], on=id_col, how="left")

    icd_values = pd.to_numeric(out[icd_indicator_col], errors="coerce").fillna(0).astype(int)
    label = np.where(icd_values == 1, out[appropriate_icd_shock_col], out[death_col])
    out[output_label_col] = (pd.to_numeric(label, errors="coerce").fillna(0).astype(float) > 0).astype(int)
    return out[[id_col, output_label_col]]


def run_full_train_sex_specific_inference(
    train_df,
    infer_df,
    survival_df,
    id_col,
    icd_indicator_col,
    appropriate_icd_shock_col,
    death_col,
    features,
    train_label_col="VT/VF/SCD",
    random_state=42,
    survival_label_col="DerivedOutcome",
    visualize_importance=False,
):
    """Train sex-specific RF models on ALL training data, run inference on infer_df, and
    evaluate against labels derived from survival_df (ICD→shock, No-ICD→death).

    Requirements:
      - train_df: contains columns features + train_label_col + "Female"
      - infer_df: contains columns features + "Female" + id_col + icd_indicator_col
      - survival_df: contains id_col, appropriate_icd_shock_col, death_col

    Returns:
      - eval_metrics: dict of overall/male/female metrics
      - incidence_rates: (male_rate, female_rate) using survival-derived label
      - merged_predictions: DataFrame with [id, Female, prob, pred, survival_label]
    """
    # Split training data by sex
    train_male_df = train_df[train_df["Female"] == 0]
    train_female_df = train_df[train_df["Female"] == 1]

    # Split inference data by sex
    infer_male_df = infer_df[infer_df["Female"] == 0]
    infer_female_df = infer_df[infer_df["Female"] == 1]

    combined_pred = np.zeros(len(infer_df), dtype=int)
    combined_prob = np.zeros(len(infer_df), dtype=float)

    # Train male model on all male training data and predict on male inference cohort
    if not train_male_df.empty and not infer_male_df.empty:
        pred_m, prob_m, _thr_m = rf_train_and_predict_no_test_labels(
            train_male_df[features],
            train_male_df[[train_label_col, "Female"]],
            infer_male_df[features],
            features,
            random_state=random_state,
            visualize_importance=visualize_importance,
        )
        combined_pred[infer_df["Female"].values == 0] = pred_m
        combined_prob[infer_df["Female"].values == 0] = prob_m

    # Train female model on all female training data and predict on female inference cohort
    if not train_female_df.empty and not infer_female_df.empty:
        pred_f, prob_f, _thr_f = rf_train_and_predict_no_test_labels(
            train_female_df[features],
            train_female_df[[train_label_col, "Female"]],
            infer_female_df[features],
            features,
            random_state=random_state,
            visualize_importance=visualize_importance,
        )
        combined_pred[infer_df["Female"].values == 1] = pred_f
        combined_prob[infer_df["Female"].values == 1] = prob_f

    # Derive validation labels from survival_df (no time, binary), using ICD from infer_df
    survival_labels = build_survival_based_label(
        survival_df,
        id_col=id_col,
        icd_indicator_col=icd_indicator_col,
        appropriate_icd_shock_col=appropriate_icd_shock_col,
        death_col=death_col,
        output_label_col=survival_label_col,
        icd_source_df=infer_df,
    )

    # Merge predictions with survival-derived labels by id
    merged = infer_df[[id_col, "Female"]].copy()
    merged["pred"] = combined_pred
    merged["prob"] = combined_prob
    merged = merged.merge(survival_labels, on=id_col, how="left")

    # Drop rows without survival labels (if any) for metric computation
    valid_mask = merged[survival_label_col].notna().values
    y_true = merged.loc[valid_mask, survival_label_col].astype(int).values
    y_pred = merged.loc[valid_mask, "pred"].astype(int).values
    y_prob = merged.loc[valid_mask, "prob"].astype(float).values

    # Gender masks aligned to the valid subset
    valid_female = merged.loc[valid_mask, "Female"].values
    mask_m = valid_female == 0
    mask_f = valid_female == 1

    # Compute performance metrics
    eval_metrics = evaluate_model_performance(y_true, y_pred, y_prob, mask_m, mask_f)

    # Incidence rates (events among predicted positives) by sex using survival-derived label
    eval_df = merged.loc[valid_mask, [survival_label_col, "Female", "pred"]].rename(columns={survival_label_col: "label"}).reset_index(drop=True)
    male_rate, female_rate = incidence_rate(eval_df, "pred", "label")

    return eval_metrics, (male_rate, female_rate), merged


# Example usage and testing
if __name__ == "__main__":
    # This would be your actual data loading
    print("Simplified multiple random splits function created successfully!")
    print("Key improvements:")
    print("1. Reduced code duplication")
    print("2. Better modularity with helper functions")
    print("3. Cleaner model configuration system")
    print("4. Easier to maintain and extend")
