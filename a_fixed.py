# Fixed version of key functions to address sex-agnostic model prediction bias

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, average_precision_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import warnings

def find_best_threshold_improved(y_true, y_scores, method='youden'):
    """
    Find optimal threshold using multiple strategies to handle class imbalance.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities 
        method: 'youden', 'f1', 'balanced_accuracy', 'fixed_recall', or 'percentile'
    
    Returns:
        Optimal threshold
    """
    if method == 'youden':
        # Youden's J statistic (sensitivity + specificity - 1)
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return thresholds[best_idx]
    
    elif method == 'f1':
        # Original F1-based method (kept for comparison)
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        best_idx = np.nanargmax(f1_scores[:-1])
        return thresholds[best_idx]
    
    elif method == 'balanced_accuracy':
        # Maximize balanced accuracy (average of sensitivity and specificity)
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        specificity = 1 - fpr
        balanced_acc = (tpr + specificity) / 2
        best_idx = np.argmax(balanced_acc)
        return thresholds[best_idx]
    
    elif method == 'fixed_recall':
        # Target a specific recall (e.g., 0.8 to ensure we catch most positive cases)
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        target_recall = 0.8
        valid_indices = recalls[:-1] >= target_recall
        if not np.any(valid_indices):
            # If target recall not achievable, use minimum threshold
            return thresholds.min()
        valid_precisions = precisions[:-1][valid_indices]
        valid_thresholds = thresholds[valid_indices]
        best_idx = np.argmax(valid_precisions)
        return valid_thresholds[best_idx]
    
    elif method == 'percentile':
        # Use a percentile-based threshold (e.g., top 10% of predictions)
        percentile = 90  # Top 10%
        return np.percentile(y_scores, percentile)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def train_sex_agnostic_model_improved(train_df, features, label_col, seed, 
                                     use_undersampling=True, threshold_method='youden'):
    """
    Improved sex-agnostic model training with better threshold selection.
    
    Args:
        train_df: Training dataframe
        features: List of feature names
        label_col: Name of the target column
        seed: Random seed
        use_undersampling: Whether to use undersampling for balanced training
        threshold_method: Method for threshold selection ('youden', 'f1', 'balanced_accuracy', 'fixed_recall', 'percentile')
    
    Returns:
        Trained model, optimal threshold, and diagnostic information
    """
    if use_undersampling:
        # Create undersampled dataset for fair comparison
        train_data = create_undersampled_dataset(train_df, label_col, seed)
        print(f"Using undersampled training data: {len(train_data)} samples")
        print(f"Class distribution: {train_data[label_col].value_counts().to_dict()}")
    else:
        train_data = train_df.copy()
        print(f"Using full training data: {len(train_data)} samples")
        print(f"Class distribution: {train_data[label_col].value_counts().to_dict()}")
    
    X_train = train_data[features]
    y_train = train_data[label_col]
    
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    print("Best hyperparameters:", search.best_params_)
    
    # Use cross-validation to determine threshold with improved method
    cv_probs = []
    cv_true = []
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        fold_model = RandomForestClassifier(**search.best_params_, random_state=seed, 
                                          n_jobs=-1, class_weight="balanced")
        fold_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        fold_probs = fold_model.predict_proba(X_train.iloc[val_idx])[:, 1]
        cv_probs.extend(fold_probs)
        cv_true.extend(y_train.iloc[val_idx])
    
    cv_probs = np.array(cv_probs)
    cv_true = np.array(cv_true)
    
    # Try multiple threshold methods and report results
    threshold_results = {}
    for method in ['youden', 'f1', 'balanced_accuracy', 'fixed_recall', 'percentile']:
        try:
            threshold = find_best_threshold_improved(cv_true, cv_probs, method)
            cv_pred = (cv_probs >= threshold).astype(int)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            acc = accuracy_score(cv_true, cv_pred)
            prec = precision_score(cv_true, cv_pred, zero_division=0)
            rec = recall_score(cv_true, cv_pred, zero_division=0)
            f1 = f1_score(cv_true, cv_pred, zero_division=0)
            
            threshold_results[method] = {
                'threshold': threshold,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'n_predicted_positive': cv_pred.sum()
            }
        except Exception as e:
            print(f"Error with {method}: {e}")
            continue
    
    # Use the specified method
    best_threshold = find_best_threshold_improved(cv_true, cv_probs, threshold_method)
    
    # Print diagnostic information
    print(f"\n=== Threshold Selection Results ===")
    for method, results in threshold_results.items():
        print(f"{method:>15}: threshold={results['threshold']:.4f}, "
              f"acc={results['accuracy']:.3f}, prec={results['precision']:.3f}, "
              f"rec={results['recall']:.3f}, f1={results['f1']:.3f}, "
              f"pred_pos={results['n_predicted_positive']}")
    
    print(f"\nSelected method: {threshold_method}")
    print(f"Selected threshold: {best_threshold:.4f}")
    
    # Analyze probability distribution
    print(f"\n=== Probability Distribution Analysis ===")
    print(f"CV probabilities - min: {cv_probs.min():.4f}, max: {cv_probs.max():.4f}, mean: {cv_probs.mean():.4f}")
    print(f"Positive class probabilities - min: {cv_probs[cv_true==1].min():.4f}, max: {cv_probs[cv_true==1].max():.4f}, mean: {cv_probs[cv_true==1].mean():.4f}")
    print(f"Negative class probabilities - min: {cv_probs[cv_true==0].min():.4f}, max: {cv_probs[cv_true==0].max():.4f}, mean: {cv_probs[cv_true==0].mean():.4f}")
    
    return best_model, best_threshold, threshold_results


def sex_agnostic_model_inference_improved(train_df, test_df, features, label_col, survival_df, seed, 
                                        gray_features=None, red_features=None, use_undersampling=True,
                                        threshold_method='youden'):
    """
    Improved sex-agnostic model inference with better threshold selection and diagnostics.
    """
    test = test_df.copy()
    
    # Train sex-agnostic model with improved threshold selection
    print("Training Improved Sex-Agnostic Model...")
    best_model, best_threshold, threshold_results = train_sex_agnostic_model_improved(
        train_df, features, label_col, seed, use_undersampling, threshold_method
    )
    
    # Make predictions on test set
    test_probs = best_model.predict_proba(test[features])[:, 1]
    print(f"\n=== Test Set Probability Analysis ===")
    print(f"Test probabilities – min: {test_probs.min():.4f}, max: {test_probs.max():.4f}, mean: {test_probs.mean():.4f}")
    
    # Apply threshold
    test_preds = (test_probs >= best_threshold).astype(int)
    n_high_risk = test_preds.sum()
    n_low_risk = len(test_preds) - n_high_risk
    
    print(f"Using threshold {best_threshold:.4f}:")
    print(f"  High risk predictions: {n_high_risk} ({n_high_risk/len(test_preds)*100:.1f}%)")
    print(f"  Low risk predictions: {n_low_risk} ({n_low_risk/len(test_preds)*100:.1f}%)")
    
    # Try alternative thresholds for comparison
    print(f"\n=== Alternative Threshold Analysis ===")
    for method, results in threshold_results.items():
        if method != threshold_method:
            alt_threshold = results['threshold']
            alt_preds = (test_probs >= alt_threshold).astype(int)
            alt_high = alt_preds.sum()
            print(f"{method:>15} (t={alt_threshold:.4f}): {alt_high} high risk ({alt_high/len(test_preds)*100:.1f}%)")
    
    # Add predictions to test dataframe
    test["pred_label"] = test_preds
    test["pred_prob"] = test_probs
    
    # Feature importance visualization
    if 'plot_feature_importances' in globals():
        plot_feature_importances(
            best_model, features, 
            f"Improved Sex-Agnostic Model Feature Importances (method: {threshold_method})", 
            seed, gray_features, red_features
        )
    
    # Merge with survival data
    pred_labels = test[["MRN", "pred_label", "Female"]].drop_duplicates()
    merged_df = survival_df.merge(pred_labels, on="MRN", how="inner").drop_duplicates(subset=["MRN"])
    
    print(f"\n=== Improved Sex-Agnostic Model Summary ===")
    print(f"Total test samples: {len(test)}")
    print(f"Samples with survival data: {len(merged_df)}")
    
    # Calculate overall prediction statistics
    n_high_risk_merged = (merged_df["pred_label"] == 1).sum()
    n_low_risk_merged = (merged_df["pred_label"] == 0).sum()
    print(f"High risk predictions (with survival data): {n_high_risk_merged}")
    print(f"Low risk predictions (with survival data): {n_low_risk_merged}")
    
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
                else:
                    print(f"{gender_name}-{risk_name}: No samples")
    
    return merged_df, threshold_results


# Helper function to create undersampled dataset (assuming it exists in original code)
def create_undersampled_dataset(train_df, label_col, random_state):
    """Create balanced dataset using undersampling."""
    from math import ceil
    
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


if __name__ == "__main__":
    print("This is the improved version of the sex-agnostic model functions.")
    print("Key improvements:")
    print("1. Multiple threshold selection methods (Youden, balanced accuracy, fixed recall, percentile)")
    print("2. Detailed diagnostic information about probability distributions")
    print("3. Comparison of different threshold methods")
    print("4. Better handling of class imbalance")