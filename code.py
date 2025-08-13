import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.utils import resample
from sklearn.metrics import (
    make_scorer,
    average_precision_score,
    accuracy_score,
    roc_auc_score,
    f1_score,
    recall_score,
    precision_recall_curve,
    confusion_matrix
)
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.utils.fixes import loguniform
from scipy.stats import randint
from lifelines import KaplanMeierFitter, logrank_test
import re


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


def plot_decision_tree(model, features, title):
    """Plot a sample decision tree."""
    plt.figure(figsize=(12, 8))
    plot_tree(
        model.estimators_[0],
        feature_names=features,
        class_names=True,
        filled=True,
        rounded=True,
    )
    plt.title(title)
    plt.show()


def perform_clustering_analysis(X_test, prob_values, gender, seed, max_clusters=10):
    """Perform clustering analysis for a specific gender and return cluster risk scores."""
    if X_test.empty:
        return None, None
    
    # Elbow method
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=seed)
        kmeans.fit(X_test)
        inertias.append(kmeans.inertia_)
    
    diffs = np.diff(inertias)
    diff_ratios = diffs / inertias[:-1]
    best_k = np.argmin(diff_ratios) + 1
    if best_k < 2:
        best_k = 2
    
    print(f"Selected best number of clusters for {gender} using elbow method: {best_k}")
    
    # Final clustering
    kmeans = KMeans(n_clusters=best_k, random_state=seed)
    cluster_labels = kmeans.fit_predict(X_test)
    
    # Calculate average risk per cluster
    cluster_risk = pd.Series(prob_values, index=range(len(prob_values))).groupby(cluster_labels).mean().round(3)
    print(f"Average risk per cluster ({gender}):")
    print(cluster_risk)
    
    return cluster_labels, cluster_risk


def bootstrap_evaluation(df, features, labels, male_model, female_model, 
                        male_threshold, female_threshold, seed, B=100):
    """Perform bootstrap evaluation for both models."""
    metrics = {
        'male': {'accs': [], 'aucs': [], 'f1s': [], 'sens': [], 'spec': []},
        'female': {'accs': [], 'aucs': [], 'f1s': [], 'sens': [], 'spec': []}
    }
    
    for i in range(B):
        # Bootstrap the original data (not the predictions)
        df_boot = resample(df, replace=True, n_samples=len(df), random_state=seed + i)
        
        # Male bootstrap
        m_boot = df_boot[df_boot["Female"] == 0]
        if not m_boot.empty:
            X_m_boot = m_boot[features]
            y_m_boot = m_boot[labels]
            prob_m_boot = male_model.predict_proba(X_m_boot)[:, 1]
            pred_m_boot = (prob_m_boot >= male_threshold).astype(int)
            
            metrics['male']['accs'].append(accuracy_score(y_m_boot, pred_m_boot))
            metrics['male']['aucs'].append(roc_auc_score(y_m_boot, prob_m_boot))
            metrics['male']['f1s'].append(f1_score(y_m_boot, pred_m_boot))
            metrics['male']['sens'].append(recall_score(y_m_boot, pred_m_boot))
            metrics['male']['spec'].append(recall_score(y_m_boot, pred_m_boot, pos_label=0))
        
        # Female bootstrap
        f_boot = df_boot[df_boot["Female"] == 1]
        if not f_boot.empty:
            X_f_boot = f_boot[features]
            y_f_boot = f_boot[labels]
            prob_f_boot = female_model.predict_proba(X_f_boot)[:, 1]
            pred_f_boot = (prob_f_boot >= female_threshold).astype(int)
            
            metrics['female']['accs'].append(accuracy_score(y_f_boot, pred_f_boot))
            metrics['female']['aucs'].append(roc_auc_score(y_f_boot, prob_f_boot))
            metrics['female']['f1s'].append(f1_score(y_f_boot, pred_f_boot))
            metrics['female']['sens'].append(recall_score(y_f_boot, pred_f_boot))
            metrics['female']['spec'].append(recall_score(y_f_boot, pred_f_boot, pos_label=0))
    
    return metrics


def print_bootstrap_results(metrics, gender):
    """Print bootstrap results for a specific gender."""
    if not metrics[gender]['accs']:
        return
    
    for metric_name in ['accs', 'aucs', 'f1s', 'sens', 'spec']:
        values = metrics[gender][metric_name]
        mean_val = np.mean(values)
        ci_low = np.percentile(values, 2.5)
        ci_high = np.percentile(values, 97.5)
        metric_display = metric_name.replace('s', '').title()
        print(f"{gender.title()} {metric_display}: {mean_val:.4f} [95% CI: {ci_low:.4f} - {ci_high:.4f}]")


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
            ("Primary Endpoint", "PE_Time", "Was Primary Endpoint Reached? (Appropriate ICD Therapy)"),
            ("Secondary Endpoint", "SE_Time", "Was Secondary Endpoint Reached?")
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


def full_model_inference(train_df, test_df, features, labels, survival_df, seed):
    """Main function with separate male/female models and gender-specific KM plots."""
    import pandas as pd
    
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
        df.loc[df["Female"] == 1, "prob_sexspecific"] = pred_f

    # Plot feature importances and trees
    plot_feature_importances(best_male, features, "Male Model Feature Importances", seed)
    plot_decision_tree(best_male, features, "Visualization of One Decision Tree in Male Random Forest Model (Tree 0)")
    
    plot_feature_importances(best_female, features, "Female Model Feature Importances", seed)
    plot_decision_tree(best_female, features, "Visualization of One Decision Tree in Female Random Forest Model (Tree 0)")

    # Survival analysis with gender-specific KM plots
    pred_labels = df[["MRN", "pred_sexspecific", "Female"]].drop_duplicates()
    merged_df = survival_df.merge(pred_labels, on="MRN", how="inner").drop_duplicates(subset=["MRN"])

    # Plot KM curves separately for each gender and risk group
    plot_km_by_gender_and_risk(
        merged_df, 
        "Female", 
        "pred_sexspecific", 
        "PE_Time", 
        "Was Primary Endpoint Reached? (Appropriate ICD Therapy)",
        "Sex-Specific Model Predictions"
    )

    # Clustering analysis (without visualization, only risk score comparison)
    if not test_m.empty:
        print("\nMale Clustering Analysis:")
        cluster_labels_m, cluster_risk_m = perform_clustering_analysis(
            test_m[features], prob_m, "Male", seed
        )
        if cluster_labels_m is not None:
            df.loc[df["Female"] == 0, "cluster"] = cluster_labels_m
    
    if not test_f.empty:
        print("\nFemale Clustering Analysis:")
        cluster_labels_f, cluster_risk_f = perform_clustering_analysis(
            test_f[features], prob_f, "Female", seed
        )
        if cluster_labels_f is not None:
            df.loc[df["Female"] == 1, "cluster"] = cluster_labels_f

    # Bootstrap evaluation
    print("\nPerforming Bootstrap Evaluation...")
    bootstrap_metrics = bootstrap_evaluation(
        train, features, labels, best_male, best_female, 
        best_thr_m, best_thr_f, seed
    )
    
    print("\nBootstrap Results:")
    print_bootstrap_results(bootstrap_metrics, 'male')
    print_bootstrap_results(bootstrap_metrics, 'female')

    return None
