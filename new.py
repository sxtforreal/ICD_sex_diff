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


def plot_km_curves_by_gender_and_risk(merged_df, title_prefix):
    """Plot Kaplan-Meier curves for all samples by gender and risk group."""
    if KaplanMeierFitter is None or logrank_test is None:
        print("lifelines not available. Skipping KM plots.")
        return
        
    genders = merged_df["Female"].unique()
    
    for gender in genders:
        gender_data = merged_df[merged_df["Female"] == gender]
        gender_label = "Female" if gender == 1 else "Male"
        
        if gender_data.empty:
            continue
            
        # Get risk groups for this gender
        risk_groups = gender_data["pred_sexspecific"].unique()
        
        if len(risk_groups) < 2:
            continue
            
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
        
        for ax, (ep_name, ep_time_col, ep_event_col) in zip(axes, [
            ("Primary Endpoint", "PE_Time", "PE"),
            ("Secondary Endpoint", "SE_Time", "SE")
        ]):
            kmf = KaplanMeierFitter()
            
            for risk_group in sorted(risk_groups):
                mask = (gender_data["Female"] == gender) & (gender_data["pred_sexspecific"] == risk_group)
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
                low_mask = (gender_data["Female"] == gender) & (gender_data["pred_sexspecific"] == 0)
                high_mask = (gender_data["Female"] == gender) & (gender_data["pred_sexspecific"] == 1)
                
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


def analyze_survival_by_risk_groups(merged_df):
    """Analyze survival outcomes by risk groups for all samples."""
    print("\n=== Survival Analysis for All Samples ===")
    
    # Overall analysis
    print(f"\n--- Overall Analysis (n={len(merged_df)}) ---")
    
    # High vs Low risk comparison
    mask_low = merged_df["pred_sexspecific"] == 0
    mask_high = merged_df["pred_sexspecific"] == 1
    
    n_low = mask_low.sum()
    n_high = mask_high.sum()
    
    if n_low > 0 and n_high > 0:
        # Primary endpoint analysis
        pe_events_low = merged_df.loc[mask_low, "PE"].sum()
        pe_events_high = merged_df.loc[mask_high, "PE"].sum()
        
        pe_low_rate = pe_events_low / n_low if n_low > 0 else 0
        pe_high_rate = pe_events_high / n_high if n_high > 0 else 0
        
        print(f"\nPrimary Endpoint Analysis:")
        print(f"Low Risk Group (n={n_low}): {pe_events_low} events, Incidence Rate: {pe_low_rate:.4f}")
        print(f"High Risk Group (n={n_high}): {pe_events_high} events, Incidence Rate: {pe_high_rate:.4f}")
        
        # Statistical test for primary endpoint
        pe_contingency = np.array([[pe_events_low, n_low - pe_events_low], 
                                  [pe_events_high, n_high - pe_events_high]])
        chi2_pe, p_value_pe, _, _ = chi2_contingency(pe_contingency)
        print(f"Primary Endpoint Chi-square test p-value: {p_value_pe:.5f}")
        
        # Risk ratio for primary endpoint
        if pe_low_rate > 0:
            pe_risk_ratio = pe_high_rate / pe_low_rate
            print(f"Primary Endpoint Risk Ratio (High/Low): {pe_risk_ratio:.3f}")
        else:
            print("Primary Endpoint Risk Ratio: Cannot calculate (no events in low risk group)")
        
        # Secondary endpoint analysis
        se_events_low = merged_df.loc[mask_low, "SE"].sum()
        se_events_high = merged_df.loc[mask_high, "SE"].sum()
        
        se_low_rate = se_events_low / n_low if n_low > 0 else 0
        se_high_rate = se_events_high / n_high if n_high > 0 else 0
        
        print(f"\nSecondary Endpoint Analysis:")
        print(f"Low Risk Group (n={n_low}): {se_events_low} events, Incidence Rate: {se_low_rate:.4f}")
        print(f"High Risk Group (n={n_high}): {se_events_high} events, Incidence Rate: {se_high_rate:.4f}")
        
        # Statistical test for secondary endpoint
        se_contingency = np.array([[se_events_low, n_low - se_events_low], 
                                  [se_events_high, n_high - se_events_high]])
        chi2_se, p_value_se, _, _ = chi2_contingency(se_contingency)
        print(f"Secondary Endpoint Chi-square test p-value: {p_value_se:.5f}")
        
        # Risk ratio for secondary endpoint
        if se_low_rate > 0:
            se_risk_ratio = se_high_rate / se_low_rate
            print(f"Secondary Endpoint Risk Ratio (High/Low): {se_risk_ratio:.3f}")
        else:
            print("Secondary Endpoint Risk Ratio: Cannot calculate (no events in low risk group)")
            
    else:
        print("Cannot perform analysis: insufficient samples in one or both risk groups")
    
    # Gender-stratified analysis
    print(f"\n--- Gender-Stratified Analysis ---")
    
    for gender_val, gender_name in [(0, "Male"), (1, "Female")]:
        gender_data = merged_df[merged_df["Female"] == gender_val]
        if gender_data.empty:
            continue
            
        print(f"\n{gender_name} Subgroup (n={len(gender_data)}):")
        
        mask_low_gender = gender_data["pred_sexspecific"] == 0
        mask_high_gender = gender_data["pred_sexspecific"] == 1
        
        n_low_gender = mask_low_gender.sum()
        n_high_gender = mask_high_gender.sum()
        
        if n_low_gender > 0 and n_high_gender > 0:
            # Primary endpoint
            pe_events_low_gender = gender_data.loc[mask_low_gender, "PE"].sum()
            pe_events_high_gender = gender_data.loc[mask_high_gender, "PE"].sum()
            
            pe_low_rate_gender = pe_events_low_gender / n_low_gender if n_low_gender > 0 else 0
            pe_high_rate_gender = pe_events_high_gender / n_high_gender if n_high_gender > 0 else 0
            
            print(f"  Primary Endpoint - Low Risk: {pe_events_low_gender}/{n_low_gender} ({pe_low_rate_gender:.4f})")
            print(f"  Primary Endpoint - High Risk: {pe_events_high_gender}/{n_high_gender} ({pe_high_rate_gender:.4f})")
            
            # Secondary endpoint
            se_events_low_gender = gender_data.loc[mask_low_gender, "SE"].sum()
            se_events_high_gender = gender_data.loc[mask_high_gender, "SE"].sum()
            
            se_low_rate_gender = se_events_low_gender / n_low_gender if n_low_gender > 0 else 0
            se_high_rate_gender = se_events_high_gender / n_high_gender if n_high_gender > 0 else 0
            
            print(f"  Secondary Endpoint - Low Risk: {se_events_low_gender}/{n_low_gender} ({se_low_rate_gender:.4f})")
            print(f"  Secondary Endpoint - High Risk: {se_events_high_gender}/{n_high_gender} ({se_high_rate_gender:.4f})")
        else:
            print(f"  Insufficient samples in risk groups (Low: {n_low_gender}, High: {n_high_gender})")


def full_model_inference(train_df, test_df, features, labels, survival_df, seed):
    """Main function with separate male/female models and survival analysis for all samples."""
    train = train_df.copy()
    test = test_df.copy()
    df = test.copy()

    # Separate data by gender
    train_m = train[train["Female"] == 0].copy()
    train_f = train[train["Female"] == 1].copy()
    test_m = test[test["Female"] == 0].copy()
    test_f = test[test["Female"] == 1].copy()

    # Train models and determine thresholds using cross-validation
    print("Training Male Model...")
    best_male, best_thr_m = train_sex_specific_model(
        train_m[features], train_m[labels], features, seed
    )
    
    print("Training Female Model...")
    best_female, best_thr_f = train_sex_specific_model(
        train_f[features], train_f[labels], features, seed
    )

    # Make predictions on test set using respective models
    if not test_m.empty:
        prob_m = best_male.predict_proba(test_m[features])[:, 1]
        pred_m = (prob_m >= best_thr_m).astype(int)
        df.loc[df["Female"] == 0, "pred_sexspecific"] = pred_m
        df.loc[df["Female"] == 0, "prob_sexspecific"] = prob_m
    
    if not test_f.empty:
        prob_f = best_female.predict_proba(test_f[features])[:, 1]
        pred_f = (prob_f >= best_thr_f).astype(int)
        df.loc[df["Female"] == 1, "pred_sexspecific"] = pred_f
        df.loc[df["Female"] == 1, "prob_sexspecific"] = prob_f

    # Plot feature importances
    plot_feature_importances(best_male, features, "Male Model Feature Importances", seed)
    plot_feature_importances(best_female, features, "Female Model Feature Importances", seed)

    # Merge with survival data for all samples
    pred_labels = df[["MRN", "pred_sexspecific", "Female"]].drop_duplicates()
    merged_df = survival_df.merge(pred_labels, on="MRN", how="inner").drop_duplicates(subset=["MRN"])

    print(f"\n=== Overall Summary ===")
    print(f"Total test samples: {len(df)}")
    print(f"Samples with survival data: {len(merged_df)}")
    
    # Risk group distribution
    if "pred_sexspecific" in merged_df.columns:
        risk_dist = merged_df["pred_sexspecific"].value_counts().sort_index()
        print(f"\nRisk Group Distribution:")
        for risk_level, count in risk_dist.items():
            risk_label = "Low Risk" if risk_level == 0 else "High Risk"
            print(f"  {risk_label}: {count}")
    
    # Perform survival analysis for all samples
    analyze_survival_by_risk_groups(merged_df)
    
    # Plot Kaplan-Meier curves
    plot_km_curves_by_gender_and_risk(merged_df, "Survival Analysis - All Samples")

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