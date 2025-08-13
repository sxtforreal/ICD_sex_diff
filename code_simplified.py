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
from sklearn.utils.fixes import loguniform
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
    """Train RandomForest with randomized search and return predictions."""
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
    
    best_model = search.best_estimator_
    
    # Feature importance visualization
    if visualize_importance:
        importances = best_model.feature_importances_
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
    
    # Determine threshold on training set to avoid data leakage
    y_train_prob = best_model.predict_proba(X_train)[:, 1]
    threshold = find_best_threshold(y_train, y_train_prob)
    
    # Predict on test set
    y_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    return y_pred, y_prob


def evaluate_model_performance(y_true, y_pred, y_prob, mask_m, mask_f):
    """Evaluate model performance for overall, male, and female subsets."""
    # Overall performance
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    f1 = f1_score(y_true, y_pred)
    sens, spec = compute_sensitivity_specificity(y_true, y_pred)
    
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
                else:
                    # Handle case with no male data
                    pred = np.zeros(len(test_df), dtype=int)
                    prob = np.zeros(len(test_df), dtype=float)
                    m_rate, f_rate = 0.0, 0.0
                
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
                else:
                    # Handle case with no female data
                    pred = np.zeros(len(test_df), dtype=int)
                    prob = np.zeros(len(test_df), dtype=float)
                    m_rate, f_rate = 0.0, 0.0
                
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
            perf_metrics = evaluate_model_performance(y_true, pred, prob, mask_m, mask_f)
            
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


# Example usage and testing
if __name__ == "__main__":
    # This would be your actual data loading
    print("Simplified multiple random splits function created successfully!")
    print("Key improvements:")
    print("1. Reduced code duplication")
    print("2. Better modularity with helper functions")
    print("3. Cleaner model configuration system")
    print("4. Easier to maintain and extend")