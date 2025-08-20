import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    make_scorer, average_precision_score, accuracy_score, roc_auc_score, 
    f1_score, recall_score, precision_recall_curve, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, train_test_split, cross_val_predict
)
from scipy.stats import randint, chi2_contingency
import warnings
from math import ceil
try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
except ImportError:
    KaplanMeierFitter = None
    logrank_test = None


def train_sex_specific_model(X_train, y_train, features, seed):
    """
    Train a sex-specific model using cross-validation for threshold determination.
    """
    # RF params
    param_dist = {
        "n_estimators": randint(100, 500),
        "max_depth": [None] + list(range(5, 26, 5)),
        "min_samples_split": randint(2, 11),
        "min_samples_leaf": randint(1, 5),
        "max_features": ["sqrt", "log2", None],
    }
    
    base_clf = RandomForestClassifier(
        random_state=seed, n_jobs=1, class_weight="balanced"
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
        n_jobs=1,
        verbose=0,
        error_score="raise",
    )
    
    # Train model
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print("Best hyperparameters:", search.best_params_)
    
    # Use cross-validation to determine threshold
    cv_probs = cross_val_predict(best_model, X_train, y_train, cv=5, method='predict_proba')[:, 1]
    best_threshold = find_best_threshold(y_train, cv_probs)
    
    return best_model, best_threshold


def plot_feature_importances(model, features, title, seed):
    """Plot feature importances with consistent styling."""
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    sorted_features = [features[i] for i in idx]
    
    colors = [
        "red" if f in {"LVEF", "NYHA Class", "NYHA>2"}
        else "gold" if f in {"Significant LGE", "LGE Burden 5SD"} 
        else "lightgray"
        for f in sorted_features
    ]
    
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(sorted_features)), importances[idx], color=colors)
    plt.xticks(range(len(sorted_features)), sorted_features, rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_km_curves_four_groups(merged_df):
    """Plot Kaplan-Meier curves for 4 groups: Male-Pred0, Male-Pred1, Female-Pred0, Female-Pred1."""
    if KaplanMeierFitter is None or logrank_test is None:
        print("lifelines not available. Skipping KM plots.")
        return
    
    # Define the 4 groups
    groups = [
        (0, 0, "Male-Pred0", "blue"),
        (0, 1, "Male-Pred1", "red"), 
        (1, 0, "Female-Pred0", "lightblue"),
        (1, 1, "Female-Pred1", "pink")
    ]
    
    # Create plots for PE and SE
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    for ax, (ep_name, ep_time_col, ep_event_col) in zip(axes, [
        ("Primary Endpoint", "PE_Time", "PE"),
        ("Secondary Endpoint", "SE_Time", "SE")
    ]):
        kmf = KaplanMeierFitter()
        
        # Plot survival curves for each group
        for gender_val, pred_val, group_name, color in groups:
            mask = (merged_df["Female"] == gender_val) & (merged_df["pred_label"] == pred_val)
            group_data = merged_df[mask]
            
            if group_data.empty:
                continue
                
            n_samples = len(group_data)
            events = group_data[ep_event_col].sum()
            label = f"{group_name} (n={n_samples}, events={events})"
            
            kmf.fit(
                durations=group_data[ep_time_col],
                event_observed=group_data[ep_event_col],
                label=label
            )
            kmf.plot(ax=ax, color=color)
        
        # Perform pairwise log-rank tests
        print(f"\n=== {ep_name} Log-rank Tests ===")
        
        # Male Pred0 vs Male Pred1
        male_pred0 = merged_df[(merged_df["Female"] == 0) & (merged_df["pred_label"] == 0)]
        male_pred1 = merged_df[(merged_df["Female"] == 0) & (merged_df["pred_label"] == 1)]
        
        if not male_pred0.empty and not male_pred1.empty:
            lr_male = logrank_test(
                male_pred0[ep_time_col], male_pred1[ep_time_col],
                male_pred0[ep_event_col], male_pred1[ep_event_col]
            )
            print(f"Male Pred0 vs Male Pred1: p = {lr_male.p_value:.5f}")
        
        # Female Pred0 vs Female Pred1
        female_pred0 = merged_df[(merged_df["Female"] == 1) & (merged_df["pred_label"] == 0)]
        female_pred1 = merged_df[(merged_df["Female"] == 1) & (merged_df["pred_label"] == 1)]
        
        if not female_pred0.empty and not female_pred1.empty:
            lr_female = logrank_test(
                female_pred0[ep_time_col], female_pred1[ep_time_col],
                female_pred0[ep_event_col], female_pred1[ep_event_col]
            )
            print(f"Female Pred0 vs Female Pred1: p = {lr_female.p_value:.5f}")
        
        # Cross-gender comparisons (optional)
        if not male_pred0.empty and not female_pred0.empty:
            lr_pred0 = logrank_test(
                male_pred0[ep_time_col], female_pred0[ep_time_col],
                male_pred0[ep_event_col], female_pred0[ep_event_col]
            )
            print(f"Male Pred0 vs Female Pred0: p = {lr_pred0.p_value:.5f}")
            
        if not male_pred1.empty and not female_pred1.empty:
            lr_pred1 = logrank_test(
                male_pred1[ep_time_col], female_pred1[ep_time_col],
                male_pred1[ep_event_col], female_pred1[ep_event_col]
            )
            print(f"Male Pred1 vs Female Pred1: p = {lr_pred1.p_value:.5f}")
        
        ax.set_title(f"{ep_name} - Survival by Gender and Prediction")
        ax.set_xlabel("Time")
        ax.set_ylabel("Survival Probability")
        ax.legend()
    
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


def inference_with_features(train_df, test_df, features, labels, survival_df, seed):
    """
    Simplified inference function:
    1. Train separate Random Forest models for male and female using provided features
    2. Generate predictions for each sample
    3. Perform survival analysis by gender and predicted label (4 groups total)
    4. Generate KM plots and logrank tests
    5. Show feature importance and incidence rates
    """
    train = train_df.copy()
    test = test_df.copy()
    df = test.copy()

    # Separate training data by gender
    train_m = train[train["Female"] == 0].copy()
    train_f = train[train["Female"] == 1].copy()
    test_m = test[test["Female"] == 0].copy()
    test_f = test[test["Female"] == 1].copy()

    # Train gender-specific models
    print("Training Male Model...")
    best_male, best_thr_m = train_sex_specific_model(
        train_m[features], train_m[labels], features, seed
    )
    
    print("Training Female Model...")
    best_female, best_thr_f = train_sex_specific_model(
        train_f[features], train_f[labels], features, seed
    )

    # Make predictions on test set
    if not test_m.empty:
        prob_m = best_male.predict_proba(test_m[features])[:, 1]
        pred_m = (prob_m >= best_thr_m).astype(int)
        df.loc[df["Female"] == 0, "pred_label"] = pred_m
        df.loc[df["Female"] == 0, "pred_prob"] = prob_m
    
    if not test_f.empty:
        prob_f = best_female.predict_proba(test_f[features])[:, 1]
        pred_f = (prob_f >= best_thr_f).astype(int)
        df.loc[df["Female"] == 1, "pred_label"] = pred_f
        df.loc[df["Female"] == 1, "pred_prob"] = prob_f

    # Feature importance visualization
    plot_feature_importances(best_male, features, "Male Model Feature Importances", seed)
    plot_feature_importances(best_female, features, "Female Model Feature Importances", seed)

    # Merge with survival data
    pred_labels = df[["MRN", "pred_label", "Female"]].drop_duplicates()
    merged_df = survival_df.merge(pred_labels, on="MRN", how="inner").drop_duplicates(subset=["MRN"])

    print(f"\n=== Summary ===")
    print(f"Total test samples: {len(df)}")
    print(f"Samples with survival data: {len(merged_df)}")
    
    # Analyze and plot survival by the 4 groups: Male-Pred0, Male-Pred1, Female-Pred0, Female-Pred1
    analyze_survival_by_four_groups(merged_df)
    plot_km_curves_four_groups(merged_df)

    return merged_df


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


# Example usage:
# features = ['LVEF', 'NYHA Class', 'Age', 'BMI', 'Significant LGE']  # Your feature list
# result_df = inference_with_features(train_df, test_df, features, 'PE', survival_df, seed=42)
