import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    """Main function with separate male/female models and risk group analysis by ICD status."""
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

    # Plot feature importances
    plot_feature_importances(best_male, features, "Male Model Feature Importances", seed)
    plot_feature_importances(best_female, features, "Female Model Feature Importances", seed)

    # Merge with survival data and ICD information for risk group analysis
    pred_labels = df[["MRN", "pred_sexspecific", "ICD"]].drop_duplicates()
    merged_df = survival_df.merge(pred_labels, on="MRN", how="inner").drop_duplicates(subset=["MRN"])

    print("\n=== Risk Group Analysis by ICD Status ===")
    
    # Analyze ICD group (ICD=1) - Appropriate ICD Therapy endpoint
    icd_df = merged_df[merged_df["ICD"] == 1].copy()
    if not icd_df.empty:
        print(f"\n--- ICD Group Analysis (n={len(icd_df)}) ---")
        print("Endpoint: Appropriate ICD Therapy")
        
        # High vs Low risk comparison
        mask_low = icd_df["pred_sexspecific"] == 0
        mask_high = icd_df["pred_sexspecific"] == 1
        
        n_low = mask_low.sum()
        n_high = mask_high.sum()
        
        if n_low > 0 and n_high > 0:
            events_low = icd_df.loc[mask_low, "Was Primary Endpoint Reached? (Appropriate ICD Therapy)"].sum()
            events_high = icd_df.loc[mask_high, "Was Primary Endpoint Reached? (Appropriate ICD Therapy)"].sum()
            
            # Calculate incidence rates
            icd_low_rate = events_low / n_low if n_low > 0 else 0
            icd_high_rate = events_high / n_high if n_high > 0 else 0
            
            print(f"Low Risk Group (n={n_low}): {events_low} events, Incidence Rate: {icd_low_rate:.4f}")
            print(f"High Risk Group (n={n_high}): {events_high} events, Incidence Rate: {icd_high_rate:.4f}")
            
            # Statistical test for binary outcome
            from scipy.stats import chi2_contingency
            contingency_table = np.array([[events_low, n_low - events_low], 
                                        [events_high, n_high - events_high]])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            print(f"Chi-square test p-value: {p_value:.5f}")
            
            # Risk ratio
            if icd_low_rate > 0:
                risk_ratio = icd_high_rate / icd_low_rate
                print(f"Risk Ratio (High/Low): {risk_ratio:.3f}")
            else:
                print("Risk Ratio: Cannot calculate (no events in low risk group)")
        else:
            print("Cannot perform analysis: insufficient samples in one or both risk groups")
    
    # Analyze No-ICD group (ICD=0) - Mortality endpoint
    no_icd_df = merged_df[merged_df["ICD"] == 0].copy()
    if not no_icd_df.empty:
        print(f"\n--- No-ICD Group Analysis (n={len(no_icd_df)}) ---")
        print("Endpoint: Mortality (Secondary Endpoint)")
        
        # High vs Low risk comparison
        mask_low = no_icd_df["pred_sexspecific"] == 0
        mask_high = no_icd_df["pred_sexspecific"] == 1
        
        n_low = mask_low.sum()
        n_high = mask_high.sum()
        
        if n_low > 0 and n_high > 0:
            events_low = no_icd_df.loc[mask_low, "Was Secondary Endpoint Reached?"].sum()
            events_high = no_icd_df.loc[mask_high, "Was Secondary Endpoint Reached?"].sum()
            
            # Calculate incidence rates
            no_icd_low_rate = events_low / n_low if n_low > 0 else 0
            no_icd_high_rate = events_high / n_high if n_high > 0 else 0
            
            print(f"Low Risk Group (n={n_low}): {events_low} events, Incidence Rate: {no_icd_low_rate:.4f}")
            print(f"High Risk Group (n={n_high}): {events_high} events, Incidence Rate: {no_icd_high_rate:.4f}")
            
            # Statistical test for binary outcome
            from scipy.stats import chi2_contingency
            contingency_table = np.array([[events_low, n_low - events_low], 
                                        [events_high, n_high - events_high]])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            print(f"Chi-square test p-value: {p_value:.5f}")
            
            # Risk ratio
            if no_icd_low_rate > 0:
                risk_ratio = no_icd_high_rate / no_icd_low_rate
                print(f"Risk Ratio (High/Low): {risk_ratio:.3f}")
            else:
                print("Risk Ratio: Cannot calculate (no events in low risk group)")
        else:
            print("Cannot perform analysis: insufficient samples in one or both risk groups")
    
    # Overall summary
    print(f"\n=== Overall Summary ===")
    print(f"Total test samples: {len(df)}")
    print(f"ICD group: {len(merged_df[merged_df['ICD'] == 1])}")
    print(f"No-ICD group: {len(merged_df[merged_df['ICD'] == 0])}")
    
    # Risk group distribution
    risk_dist = merged_df["pred_sexspecific"].value_counts().sort_index()
    print(f"\nRisk Group Distribution:")
    for risk_level, count in risk_dist.items():
        risk_label = "Low Risk" if risk_level == 0 else "High Risk"
        print(f"  {risk_label}: {count}")

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
