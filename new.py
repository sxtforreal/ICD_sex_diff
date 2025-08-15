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
try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
except ImportError:
    KaplanMeierFitter = None
    logrank_test = None


def to_binary(series):
    """Convert a pandas Series with mixed types (numeric, Yes/No, True/False) to {0,1} with NaNs preserved.

    Rules:
      - Numeric values: > 0 -> 1, else 0 (NaN preserved)
      - Strings (case-insensitive, trimmed): {"yes","y","true","t","positive","pos","present"} -> 1,
        {"no","n","false","f","negative","neg","absent"} -> 0; others -> NaN
    """
    if series is None:
        return pd.Series(dtype="float64")
    s = pd.Series(series)
    # First, try numeric
    num = pd.to_numeric(s, errors="coerce")
    out = pd.Series(index=s.index, dtype="float64")
    is_num = num.notna()
    out.loc[is_num] = (num.loc[is_num].astype(float) > 0).astype(int)
    # For non-numeric entries, map common textual labels
    non = ~is_num
    if non.any():
        mapped = s.loc[non].astype(str).str.strip().str.lower()
        ones = {"yes", "y", "true", "t", "positive", "pos", "present"}
        zeros = {"no", "n", "false", "f", "negative", "neg", "absent"}
        out.loc[non] = mapped.map(lambda v: 1 if v in ones else (0 if v in zeros else np.nan))
    return out


def find_best_threshold(y_true, y_scores):
    """Find the probability threshold that maximizes the F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.nanargmax(f1_scores[:-1])
    return thresholds[best_idx]


def compute_sensitivity_specificity(y_true, y_pred):
    """Compute sensitivity and specificity from binary predictions."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
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


def build_survival_based_label(
    survival_df,
    id_col,
    icd_indicator_col,
    appropriate_icd_shock_col,
    death_col,
    output_label_col="DerivedOutcome",
    icd_source_df=None,
    master_df=None,
    master_label_col="Composite Outcome",
    master_id_col=None,
):
    """Construct binary labels for evaluation using ICD status and external master spreadsheet.

    Logic:
      - If ICD implanted (icd_indicator_col == 1): label = appropriate_icd_shock_col (from survival_df)
      - If no ICD (icd_indicator_col == 0):       label = master_df[master_label_col] by MRN join
        (falls back to death_col if master_df is not provided)
    The resulting label is binarized to {0,1} using robust conversion from mixed types.

    If icd_source_df is provided, the ICD indicator will be taken from icd_source_df (joined on id_col),
    otherwise it is expected to be present in survival_df.

    Returns a DataFrame with columns [id_col, output_label_col].
    """
    # Deduplicate survival_df for unique IDs
    survival_unique = survival_df.drop_duplicates(subset=[id_col], keep='first')

    if icd_source_df is None:
        # Keep icd indicator from survival_df when icd_source_df not provided
        cols = [id_col, icd_indicator_col, appropriate_icd_shock_col]
        available_cols = [c for c in cols if c in survival_unique.columns]
        out = survival_unique[available_cols].copy()
        # Include death_col only for backward-compatible fallback when master_df is None
        if death_col in survival_unique.columns:
            out[death_col] = survival_unique[death_col]
    else:
        # Deduplicate icd_source_df to avoid duplication in merge
        icd_source_unique = icd_source_df.drop_duplicates(subset=[id_col], keep='first')
        # Build a base ID frame from the union of IDs in survival_df and icd_source_df
        base_ids = pd.Index(survival_unique[id_col].dropna().unique()).union(
            pd.Index(icd_source_unique[id_col].dropna().unique())
        )
        out = pd.DataFrame({id_col: base_ids})
        # Bring in ICD indicator from icd_source_df
        out = out.merge(icd_source_unique[[id_col, icd_indicator_col]], on=id_col, how="left")
        # Bring in shock and optional death from survival_df
        merge_cols = [id_col, appropriate_icd_shock_col]
        if death_col in survival_unique.columns:
            merge_cols.append(death_col)
        out = out.merge(survival_unique[merge_cols], on=id_col, how="left")

    # Robustly binarize ICD indicator and possible label sources
    icd_values = to_binary(out[icd_indicator_col]).fillna(0).astype(int)

    # If a master DF is provided, merge and use its Composite Outcome for no-ICD rows
    if master_df is not None:
        if master_id_col is None:
            master_id_col = id_col
        # Deduplicate master_df to avoid duplication in merge
        master_unique = master_df.drop_duplicates(subset=[master_id_col], keep='first')
        out = out.merge(
            master_unique[[master_id_col, master_label_col]].rename(columns={master_id_col: id_col}),
            on=id_col,
            how="left",
        )
        label_source_no_icd = to_binary(out[master_label_col])
    else:
        # Backward-compatible fallback: use death for no-ICD
        label_source_no_icd = to_binary(out.get(death_col, pd.Series(index=out.index)))

    shock_labels = to_binary(out[appropriate_icd_shock_col])
    label = shock_labels.where(icd_values == 1, label_source_no_icd)
    out[output_label_col] = label
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
    master_df=None,
    master_label_col="Composite Outcome",
    master_id_col=None,
):
    """Train sex-specific RF models on ALL training data, run inference on infer_df, and
    evaluate against labels derived from survival_df/master_df (ICD→shock, No-ICD→Composite Outcome).

    Requirements:
      - train_df: contains columns features + train_label_col + "Female"
      - infer_df: contains columns features + "Female" + id_col + icd_indicator_col
      - survival_df: contains id_col and appropriate_icd_shock_col
      - master_df (optional): contains MRN (or id_col) and master_label_col (default "Composite Outcome")

    Notes:
      - survival_label_col is the name to give to the derived survival-based binary label
        produced by combining survival_df/master_df with the ICD indicator. Specifically, for rows
        with ICD implanted (icd_indicator_col == 1), the label comes from appropriate_icd_shock_col;
        for rows without ICD (icd_indicator_col == 0), the label comes from master_df[master_label_col]
        via MRN/id_col join. If master_df is not provided, it falls back to death_col for no-ICD rows.
        This label is used only for evaluation, not for training.

    Returns:
      - eval_metrics: dict of overall/male/female metrics
      - incidence_rates: (male_rate, female_rate) using survival-derived label
      - merged_predictions: DataFrame with [id, Female, prob, pred, survival_label]
    """
    # Remove 'Female' from features for sex-specific models to avoid constant feature
    features_clean = [f for f in features if f != "Female"]

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
            train_male_df[features_clean],
            train_male_df[[train_label_col, "Female"]],
            infer_male_df[features_clean],
            features_clean,
            random_state=random_state,
            visualize_importance=visualize_importance,
        )
        combined_pred[infer_df["Female"].values == 0] = pred_m
        combined_prob[infer_df["Female"].values == 0] = prob_m

    # Train female model on all female training data and predict on female inference cohort
    if not train_female_df.empty and not infer_female_df.empty:
        pred_f, prob_f, _thr_f = rf_train_and_predict_no_test_labels(
            train_female_df[features_clean],
            train_female_df[[train_label_col, "Female"]],
            infer_female_df[features_clean],
            features_clean,
            random_state=random_state,
            visualize_importance=visualize_importance,
        )
        combined_pred[infer_df["Female"].values == 1] = pred_f
        combined_prob[infer_df["Female"].values == 1] = prob_f

    # Derive validation labels using ICD from infer_df and Composite Outcome for no-ICD
    survival_labels = build_survival_based_label(
        survival_df,
        id_col=id_col,
        icd_indicator_col=icd_indicator_col,
        appropriate_icd_shock_col=appropriate_icd_shock_col,
        death_col=death_col,
        output_label_col=survival_label_col,
        icd_source_df=infer_df,
        master_df=master_df,
        master_label_col=master_label_col,
        master_id_col=master_id_col,
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


def plot_incidence_rates_by_group(eval_df, label_col, title):
    """Plot male/female incidence rates for a given label column.

    Incidence rate is defined as (# true events among predicted positives) within each sex.
    """
    m_rate, f_rate = incidence_rate(eval_df.rename(columns={label_col: "label"}), "pred", "label")
    rates = [m_rate, f_rate]
    sexes = ["Male", "Female"]

    plt.figure(figsize=(4, 4))
    sns.barplot(x=sexes, y=rates, palette=["#4C72B0", "#DD8452"], edgecolor="black")
    plt.ylim(0, 1)
    plt.ylabel("Incidence rate")
    plt.title(title)
    for i, v in enumerate(rates):
        if not np.isnan(v):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_km_by_sex_for_icd_group(
    test_df,
    survival_df,
    *,
    id_col,
    icd_col,
    female_col="Female",
    primary_time_col=None,
    primary_event_col=None,
    secondary_time_col=None,
    secondary_event_col=None,
    time_unit="days",
):
    """Plot KM curves and perform logrank tests for the ICD group split by sex.

    Parameters:
      - test_df: DataFrame with at least [id_col, icd_col, female_col]
      - survival_df: DataFrame with time-to-event and event indicator columns
      - id_col: join key present in both test_df and survival_df
      - icd_col: column in test_df indicating ICD implantation (1/0)
      - female_col: sex indicator (1=female, 0=male)
      - primary_time_col / primary_event_col: columns in survival_df for primary endpoint
      - secondary_time_col / secondary_event_col: columns in survival_df for secondary endpoint
      - time_unit: label for the x-axis

    Notes:
      - Requires lifelines. If not installed, this function will print a message and skip plotting.
      - Event columns are binarized with to_binary; time columns coerced to numeric and rows with missing values are dropped.
    """
    if KaplanMeierFitter is None or logrank_test is None:
        print("[KM] lifelines is not installed. Please `pip install lifelines` to enable KM plots.")
        return

    def _plot_one(name, time_col, event_col):
        if time_col is None or event_col is None:
            return
        if time_col not in survival_df.columns or event_col not in survival_df.columns:
            print(f"[KM] Columns for {name} not found in survival_df: {time_col}, {event_col}. Skipping.")
            return
        icd_mask = to_binary(test_df[icd_col]).fillna(0).astype(int).values == 1
        icd_ids = test_df.loc[icd_mask, [id_col, female_col]].dropna()
        if icd_ids.empty:
            print(f"[KM] No ICD subjects for {name}. Skipping.")
            return
        merged = icd_ids.merge(survival_df[[id_col, time_col, event_col]], on=id_col, how="left")
        durations = pd.to_numeric(merged[time_col], errors="coerce")
        events = to_binary(merged[event_col])
        keep = durations.notna() & events.notna()
        merged = merged.loc[keep].reset_index(drop=True)
        if merged.empty:
            print(f"[KM] No valid time/event data for {name}. Skipping.")
            return
        durations = pd.to_numeric(merged[time_col], errors="coerce").values
        events = to_binary(merged[event_col]).astype(int).values
        females = merged[female_col].astype(int).values
        mask_m = females == 0
        mask_f = females == 1
        if mask_m.sum() == 0 or mask_f.sum() == 0:
            print(f"[KM] Not enough subjects in one sex for {name}. Skipping logrank.")
        kmf_m = KaplanMeierFitter()
        kmf_f = KaplanMeierFitter()
        plt.figure(figsize=(5.5, 4))
        kmf_m.fit(durations[mask_m], events[mask_m], label=f"Male (n={mask_m.sum()})")
        ax = kmf_m.plot(ci_show=True)
        kmf_f.fit(durations[mask_f], events[mask_f], label=f"Female (n={mask_f.sum()})")
        kmf_f.plot(ax=ax, ci_show=True)
        plt.xlabel(f"Time ({time_unit})")
        plt.ylabel("Survival probability")
        plt.title(f"KM: {name} (ICD group)")
        pval = np.nan
        if mask_m.sum() > 0 and mask_f.sum() > 0:
            lr = logrank_test(durations[mask_m], durations[mask_f], events[mask_m], events[mask_f])
            pval = lr.p_value
            plt.text(0.7, 0.1, f"logrank p={pval:.3g}", transform=ax.transAxes)
            print(f"[KM] {name} logrank p-value: {pval:.3g}")
        plt.tight_layout()
        plt.show()
        return pval

    _plot_one("Primary endpoint", primary_time_col, primary_event_col)
    _plot_one("Secondary endpoint", secondary_time_col, secondary_event_col)


def plot_km_by_pred_within_sex(
    df_with_pred,
    survival_df,
    *,
    id_col,
    female_col="Female",
    pred_col="pred",
    primary_time_col=None,
    primary_event_col=None,
    secondary_time_col=None,
    secondary_event_col=None,
    time_unit="days",
    restrict_to_icd_col=None,
):
    """For each sex, plot KM curves comparing predicted positives vs negatives.

    Parameters:
      - df_with_pred: DataFrame with at least [id_col, female_col, pred_col] (e.g., the merged predictions from inference)
      - survival_df: DataFrame with time-to-event and event indicator columns
      - id_col: join key present in both df_with_pred and survival_df
      - female_col: sex indicator (1=female, 0=male)
      - pred_col: predicted binary label column (0/1)
      - primary_time_col / primary_event_col: columns in survival_df for primary endpoint
      - secondary_time_col / secondary_event_col: columns in survival_df for secondary endpoint
      - time_unit: label for the x-axis
      - restrict_to_icd_col: optional column in df_with_pred; if provided, filter to rows with ICD==1

    Notes:
      - Requires lifelines. If not installed, this function will print a message and skip plotting.
      - Event columns are binarized with to_binary; time columns coerced to numeric and rows with missing values are dropped.
    """
    if KaplanMeierFitter is None or logrank_test is None:
        print("[KM] lifelines is not installed. Please `pip install lifelines` to enable KM plots.")
        return

    def _plot_one(name, time_col, event_col):
        if time_col is None or event_col is None:
            return
        if time_col not in survival_df.columns or event_col not in survival_df.columns:
            print(f"[KM] Columns for {name} not found in survival_df: {time_col}, {event_col}. Skipping.")
            return

        data = df_with_pred.copy()
        if restrict_to_icd_col is not None:
            if restrict_to_icd_col not in data.columns:
                print(f"[KM] restrict_to_icd_col='{restrict_to_icd_col}' not found in df_with_pred. Skipping restriction.")
            else:
                icd_mask = to_binary(data[restrict_to_icd_col]).fillna(0).astype(int).values == 1
                data = data.loc[icd_mask]
        needed_cols = [id_col, female_col, pred_col]
        missing = [c for c in needed_cols if c not in data.columns]
        if missing:
            print(f"[KM] Required columns missing from df_with_pred: {missing}. Skipping.")
            return

        # Use only rows with non-missing id/sex/pred
        base = data[needed_cols].dropna(subset=[id_col, female_col, pred_col]).copy()
        if base.empty:
            print(f"[KM] No data available after filtering for {name}. Skipping.")
            return

        base["pred_bin"] = to_binary(base[pred_col]).astype(int)
        merged = base.merge(survival_df[[id_col, time_col, event_col]], on=id_col, how="left")

        durations = pd.to_numeric(merged[time_col], errors="coerce")
        events = to_binary(merged[event_col])
        keep = durations.notna() & events.notna()
        merged = merged.loc[keep].reset_index(drop=True)
        if merged.empty:
            print(f"[KM] No valid time/event data for {name}. Skipping.")
            return

        durations = pd.to_numeric(merged[time_col], errors="coerce").values
        events = to_binary(merged[event_col]).astype(int).values
        females = merged[female_col].astype(int).values
        pred_bin = merged["pred_bin"].astype(int).values

        def plot_for_sex(sex_label, sex_value):
            sex_mask = females == sex_value
            if sex_mask.sum() == 0:
                print(f"[KM] No subjects for {sex_label} in {name}. Skipping.")
                return
            group_pos = sex_mask & (pred_bin == 1)
            group_neg = sex_mask & (pred_bin == 0)
            if group_pos.sum() == 0 or group_neg.sum() == 0:
                print(f"[KM] Not enough subjects in one pred group for {sex_label} in {name}. Skipping logrank.")

            kmf_pos = KaplanMeierFitter()
            kmf_neg = KaplanMeierFitter()

            plt.figure(figsize=(5.5, 4))
            kmf_pos.fit(durations[group_pos], events[group_pos], label=f"pred=1 (n={group_pos.sum()})")
            ax = kmf_pos.plot(ci_show=True)
            kmf_neg.fit(durations[group_neg], events[group_neg], label=f"pred=0 (n={group_neg.sum()})")
            kmf_neg.plot(ax=ax, ci_show=True)
            plt.xlabel(f"Time ({time_unit})")
            plt.ylabel("Survival probability")
            plt.title(f"KM: {name} - {sex_label}")

            pval = np.nan
            if group_pos.sum() > 0 and group_neg.sum() > 0:
                lr = logrank_test(durations[group_pos], durations[group_neg], events[group_pos], events[group_neg])
                pval = lr.p_value
                plt.text(0.6, 0.1, f"logrank p={pval:.3g}", transform=ax.transAxes)
                print(f"[KM] {name} ({sex_label}) logrank p-value: {pval:.3g}")
            plt.tight_layout()
            plt.show()
            return pval

        plot_for_sex("Male", 0)
        plot_for_sex("Female", 1)

    _plot_one("Primary endpoint", primary_time_col, primary_event_col)
    _plot_one("Secondary endpoint", secondary_time_col, secondary_event_col)


def sex_specific_train_and_grouped_eval(
    train_df,
    test_df,
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
    plot_incidence=True,
    master_df=None,
    master_label_col="Composite Outcome",
    master_id_col=None,
    plot_km=False,
    km_primary_time_col=None,
    km_primary_event_col=None,
    km_secondary_time_col=None,
    km_secondary_event_col=None,
    km_time_unit="days",
):
    """Train sex-specific models on full training data, predict on test set, and
    evaluate by ICD vs no-ICD groups using survival/master-derived labels.

    For evaluation:
      - ICD group: ground truth = appropriate ICD shock from survival_df (by MRN/id join)
      - No-ICD group: ground truth = Composite Outcome from master_df (by MRN/id join)
      - Compare each group's metrics against metrics computed with VT/VF/SCD labels
      - Optionally: For ICD group only, plot KM curves and logrank tests by sex for
        specified primary and secondary endpoints present in survival_df.

    Notes:
      - survival_label_col is the output column name for the survival/master-derived binary label
        created by combining ICD shocks (for ICD patients) and master_df Composite Outcome (for non-ICD patients).
      - If a group's rows do not have survival/master-derived labels (e.g., missing joins),
        the group will still compute VT/VF/SCD metrics, and report survival-based evaluation as unavailable.
    """
    # 1) Train sex-specific models on all training data and predict on full test set
    eval_metrics_overall, _inc_rates_overall, merged_all = run_full_train_sex_specific_inference(
        train_df=train_df,
        infer_df=test_df,
        survival_df=survival_df,
        id_col=id_col,
        icd_indicator_col=icd_indicator_col,
        appropriate_icd_shock_col=appropriate_icd_shock_col,
        death_col=death_col,
        features=features,
        train_label_col=train_label_col,
        random_state=random_state,
        survival_label_col=survival_label_col,
        visualize_importance=visualize_importance,
        master_df=master_df,
        master_label_col=master_label_col,
        master_id_col=master_id_col,
    )

    # 2) Merge predictions with meta needed for grouped evaluation
    merged = test_df[[id_col, "Female", icd_indicator_col, train_label_col]].copy()
    merged = merged.merge(
        merged_all[[id_col, "pred", "prob", survival_label_col]], on=id_col, how="left"
    )

    # Do NOT drop rows without survival labels globally; handle per-group
    mask_icd = merged[icd_indicator_col].astype(int).values == 1
    mask_no_icd = merged[icd_indicator_col].astype(int).values == 0

    out = {}

    def eval_for_group(group_mask, group_name):
        sub_all = merged.loc[group_mask].reset_index(drop=True)
        if sub_all.empty:
            print(f"[{group_name}] No samples. Skipping.")
            return {
                "survival_metrics": None,
                "vtvfscd_metrics": None,
                "survival_incidence": (np.nan, np.nan),
                "vtvfscd_incidence": (np.nan, np.nan),
            }

        # All predictions for the group
        y_pred_all = sub_all["pred"].astype(int).values
        y_prob_all = sub_all["prob"].astype(float).values
        female_vals_all = sub_all["Female"].astype(int).values
        mask_m_all = female_vals_all == 0
        mask_f_all = female_vals_all == 1

        # Subset where survival/master-derived labels are available
        sub_surv = sub_all[sub_all[survival_label_col].notna()].reset_index(drop=True)
        has_surv = not sub_surv.empty

        if has_surv:
            y_true_surv = sub_surv[survival_label_col].astype(int).values
            y_pred_surv = sub_surv["pred"].astype(int).values
            y_prob_surv = sub_surv["prob"].astype(float).values
            female_vals_surv = sub_surv["Female"].astype(int).values
            mask_m_surv = female_vals_surv == 0
            mask_f_surv = female_vals_surv == 1
            surv_metrics = evaluate_model_performance(y_true_surv, y_pred_surv, y_prob_surv, mask_m_surv, mask_f_surv)
            eval_df_surv = sub_surv[["Female", "pred", survival_label_col]].rename(columns={survival_label_col: "label"})
            m_rate_surv, f_rate_surv = incidence_rate(eval_df_surv, "pred", "label")
        else:
            surv_metrics = None
            m_rate_surv, f_rate_surv = (np.nan, np.nan)

        # VT/VF/SCD ground truth on all rows in the group (always available from test_df)
        y_true_vt = sub_all[train_label_col].astype(int).values
        vtvfscd_metrics = evaluate_model_performance(y_true_vt, y_pred_all, y_prob_all, mask_m_all, mask_f_all)
        eval_df_vt = sub_all[["Female", "pred", train_label_col]].rename(columns={train_label_col: "label"})
        m_rate_vt, f_rate_vt = incidence_rate(eval_df_vt, "pred", "label")

        # Print comparison
        def fmt_metric(d, key):
            if d is None:
                return "nan"
            v = d.get(key, np.nan)
            return "nan" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"

        print(f"\n===== {group_name} =====")
        print(f"- Samples: total={len(sub_all)}, with_survival_label={len(sub_surv)}")
        print("- Using survival/master-derived labels:")
        if has_surv:
            print(
                f"  Acc={fmt_metric(surv_metrics, 'accuracy')}, AUC={fmt_metric(surv_metrics, 'auc')}, F1={fmt_metric(surv_metrics, 'f1')}, "
                f"Sens={fmt_metric(surv_metrics, 'sensitivity')}, Spec={fmt_metric(surv_metrics, 'specificity')}"
            )
            print(
                f"  Male: Acc={fmt_metric(surv_metrics, 'male_accuracy')}, AUC={fmt_metric(surv_metrics, 'male_auc')}, F1={fmt_metric(surv_metrics, 'male_f1')}, "
                f"Sens={fmt_metric(surv_metrics, 'male_sensitivity')}, Spec={fmt_metric(surv_metrics, 'male_specificity')}"
            )
            print(
                f"  Female: Acc={fmt_metric(surv_metrics, 'female_accuracy')}, AUC={fmt_metric(surv_metrics, 'female_auc')}, F1={fmt_metric(surv_metrics, 'female_f1')}, "
                f"Sens={fmt_metric(surv_metrics, 'female_sensitivity')}, Spec={fmt_metric(surv_metrics, 'female_specificity')}"
            )
            print(
                f"  Incidence (M/F) = ({'nan' if np.isnan(m_rate_surv) else f'{m_rate_surv:.3f}'}/{'nan' if np.isnan(f_rate_surv) else f'{f_rate_surv:.3f}'})"
            )
        else:
            print("  No survival/master-derived labels found for this group; skipping survival-based evaluation.")

        print("- Using VT/VF/SCD labels:")
        print(
            f"  Acc={fmt_metric(vtvfscd_metrics, 'accuracy')}, AUC={fmt_metric(vtvfscd_metrics, 'auc')}, F1={fmt_metric(vtvfscd_metrics, 'f1')}, "
            f"Sens={fmt_metric(vtvfscd_metrics, 'sensitivity')}, Spec={fmt_metric(vtvfscd_metrics, 'specificity')}"
        )
        print(
            f"  Male: Acc={fmt_metric(vtvfscd_metrics, 'male_accuracy')}, AUC={fmt_metric(vtvfscd_metrics, 'male_auc')}, F1={fmt_metric(vtvfscd_metrics, 'male_f1')}, "
            f"Sens={fmt_metric(vtvfscd_metrics, 'male_sensitivity')}, Spec={fmt_metric(vtvfscd_metrics, 'male_specificity')}"
        )
        print(
            f"  Female: Acc={fmt_metric(vtvfscd_metrics, 'female_accuracy')}, AUC={fmt_metric(vtvfscd_metrics, 'female_auc')}, F1={fmt_metric(vtvfscd_metrics, 'female_f1')}, "
            f"Sens={fmt_metric(vtvfscd_metrics, 'female_sensitivity')}, Spec={fmt_metric(vtvfscd_metrics, 'female_specificity')}"
        )
        print(
            f"  Incidence (M/F) = ({'nan' if np.isnan(m_rate_vt) else f'{m_rate_vt:.3f}'}/{'nan' if np.isnan(f_rate_vt) else f'{f_rate_vt:.3f}'})"
        )

        # Optional plots
        if plot_incidence and has_surv:
            plot_incidence_rates_by_group(
                eval_df_surv.rename(columns={"label": survival_label_col}),
                label_col=survival_label_col,
                title=f"Incidence rate ({group_name}) - Survival-derived",
            )
        if plot_incidence:
            plot_incidence_rates_by_group(
                eval_df_vt.rename(columns={"label": train_label_col}),
                label_col=train_label_col,
                title=f"Incidence rate ({group_name}) - VT/VF/SCD",
            )

        return {
            "survival_metrics": surv_metrics,
            "vtvfscd_metrics": vtvfscd_metrics,
            "survival_incidence": (m_rate_surv, f_rate_surv),
            "vtvfscd_incidence": (m_rate_vt, f_rate_vt),
        }

    out["ICD"] = eval_for_group(mask_icd, "ICD group")
    out["No-ICD"] = eval_for_group(mask_no_icd, "No-ICD group")

    # Optional: KM plots for ICD group by sex, for primary and secondary endpoints
    if plot_km:
        plot_km_by_sex_for_icd_group(
            test_df,
            survival_df,
            id_col=id_col,
            icd_col=icd_indicator_col,
            female_col="Female",
            primary_time_col=km_primary_time_col,
            primary_event_col=km_primary_event_col,
            secondary_time_col=km_secondary_time_col,
            secondary_event_col=km_secondary_event_col,
            time_unit=km_time_unit,
        )

    return out, merged


def train_and_eval_sex_specific_by_icd(
    train_df,
    test_df,
    survival_df,
    *,
    id_col,
    icd_col,
    shock_col,
    death_col,
    features,
    train_label_col="VT/VF/SCD",
    random_state=42,
    survival_label_col="DerivedOutcome",
    visualize_importance=False,
    master_df=None,
    master_label_col="Composite Outcome",
    master_id_col=None,
):
    """End-to-end: Train separate RF models for males and females on all train_df to predict VT/VF/SCD,
    apply to all rows in test_df by sex, then evaluate using ground-truth chosen by test_df's ICD:
      - ICD==1 -> compare to survival_df[shock_col]
      - ICD==0 -> compare to master_df[master_label_col]

    Returns:
      - eval_metrics: dict with overall/male/female metrics computed on survival/master-derived labels
      - incidence_rates: (male_rate, female_rate) among predicted positives using the same labels
      - merged_predictions: DataFrame with [id_col, Female, pred, prob, survival_label_col]
    """
    eval_metrics, incidence_rates, merged = run_full_train_sex_specific_inference(
        train_df=train_df,
        infer_df=test_df.rename(columns={icd_col: icd_col}),
        survival_df=survival_df,
        id_col=id_col,
        icd_indicator_col=icd_col,
        appropriate_icd_shock_col=shock_col,
        death_col=death_col,
        features=features,
        train_label_col=train_label_col,
        random_state=random_state,
        survival_label_col=survival_label_col,
        visualize_importance=visualize_importance,
        master_df=master_df,
        master_label_col=master_label_col,
        master_id_col=master_id_col,
    )
    return eval_metrics, incidence_rates, merged
