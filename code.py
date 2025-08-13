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
        n_iter=50,
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
    print("Best hyperparameters (Male):", search.best_params_)
    train_prob_m = best_male.predict_proba(X_train_m)[:, 1]
    best_thr_m = find_best_threshold(y_train_m, train_prob_m)
    prob_m = best_male.predict_proba(X_test_m)[:, 1]
    pred_m = (prob_m >= best_thr_m).astype(int)
    df.loc[df["Female"] == 0, "pred_male"] = pred_m
    df.loc[df["Female"] == 0, "prob_male"] = prob_m

    # Plot feature importances
    importances = best_male.feature_importances_
    idx = np.argsort(importances)[::-1]
    sorted_features = [features[i] for i in idx]
    colors = [
        (
            "red"
            if f in {"LVEF", "NYHA Class", "NYHA>2"}
            else "gold" if f in {"Significant LGE", "LGE Burden 5SD"} else "lightgray"
        )
        for f in sorted_features
    ]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(sorted_features)), importances[idx], color=colors)
    plt.xticks(range(len(sorted_features)), sorted_features, rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Male Model Feature Importances")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    plot_tree(
        best_male.estimators_[0],
        feature_names=features,
        class_names=True,
        filled=True,
        rounded=True,
    )
    plt.title("Visualization of One Decision Tree in Male Random Forest Model (Tree 0)")
    plt.show()

    # Female model
    search.fit(X_train_f, y_train_f)
    best_female = search.best_estimator_
    print("Best hyperparameters (Female):", search.best_params_)
    train_prob_f = best_female.predict_proba(X_train_f)[:, 1]
    best_thr_f = find_best_threshold(y_train_f, train_prob_f)
    prob_f = best_female.predict_proba(X_test_f)[:, 1]
    pred_f = (prob_f >= best_thr_f).astype(int)
    df.loc[df["Female"] == 1, "pred_female"] = pred_f
    df.loc[df["Female"] == 1, "prob_female"] = prob_f
    # Plot feature importances
    importances = best_female.feature_importances_
    idx = np.argsort(importances)[::-1]
    sorted_features = [features[i] for i in idx]
    colors = [
        (
            "red"
            if f in {"LVEF", "NYHA Class", "NYHA>2"}
            else "gold" if f in {"Significant LGE", "LGE Burden 5SD"} else "lightgray"
        )
        for f in sorted_features
    ]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(sorted_features)), importances[idx], color=colors)
    plt.xticks(range(len(sorted_features)), sorted_features, rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Female Model Feature Importances")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    plot_tree(
        best_female.estimators_[0],
        feature_names=features,
        class_names=True,
        filled=True,
        rounded=True,
    )
    plt.title(
        "Visualization of One Decision Tree in Female Random Forest Model (Tree 0)"
    )
    plt.show()

    # Survival analysis
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

    pred_labels = df[["MRN", "pred_sexspecific"]].drop_duplicates()  # Remove duplicates

    # Merge with deduplication
    merged_df = survival_df.merge(pred_labels, on="MRN", how="inner")
    merged_df = merged_df.drop_duplicates(subset=["MRN"])

    kmf = KaplanMeierFitter()
    endpoints = [
        (
            "Primary Endpoint",
            "PE_Time",
            "Was Primary Endpoint Reached? (Appropriate ICD Therapy)",
        ),
        ("Secondary Endpoint", "SE_Time", "Was Secondary Endpoint Reached?"),
    ]
    groupings = [
        ("Sex-Specific grouping", "pred_sexspecific"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

    for ax, (ep_name, time_col, event_col) in zip(axes, endpoints):
        title, pred_col = groupings[0]
        mask_low = merged_df[pred_col] == 0
        mask_high = merged_df[pred_col] == 1

        # Calculate group sizes and number of events
        n_low = mask_low.sum()
        n_high = mask_high.sum()
        total_n = n_low + n_high
        events_low = merged_df.loc[mask_low, event_col].sum()
        events_high = merged_df.loc[mask_high, event_col].sum()
        total_events = events_low + events_high

        print(f"{ep_name} - Low risk: n={n_low}, events={events_low}")
        print(f"{ep_name} - High risk: n={n_high}, events={events_high}")
        print(f"{ep_name} - Total: n={total_n}, events={total_events}")

        lr = logrank_test(
            merged_df.loc[mask_low, time_col],
            merged_df.loc[mask_high, time_col],
            merged_df.loc[mask_low, event_col],
            merged_df.loc[mask_high, event_col],
        )
        p_value = lr.p_value

        for mask, lbl, count, events in [
            (mask_low, f"Low Risk (n={n_low}, events={events_low})", n_low, events_low),
            (
                mask_high,
                f"High Risk (n={n_high}, events={events_high})",
                n_high,
                events_high,
            ),
        ]:
            kmf.fit(
                durations=merged_df.loc[mask, time_col],
                event_observed=merged_df.loc[mask, event_col],
                label=lbl,  # Include both n and events in label
            )
            kmf.plot(ax=ax)

        ax.set_title(f"{ep_name} by {title} (Total n={total_n}, events={total_events})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Survival Probability")
        ax.text(
            0.95,
            0.05,
            f"Log-rank p = {p_value:.5f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
        )
        ax.legend()

    plt.tight_layout()
    plt.show()
    plt.close()

    # Fit Cox PH model
    cph_feature = df[["MRN"] + features]
    cph_df = survival_df.merge(cph_feature, on="MRN", how="inner")
    cph_df = cph_df.drop_duplicates(subset=["MRN"])
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

    # Clustering for Male
    if not X_test_m.empty:
        # Elbow method for male
        inertias_m = []
        max_clusters = 10
        for k in range(1, max_clusters + 1):
            kmeans_m = KMeans(n_clusters=k, random_state=seed)
            kmeans_m.fit(X_test_m)
            inertias_m.append(kmeans_m.inertia_)

        diffs_m = np.diff(inertias_m)
        diff_ratios_m = diffs_m / inertias_m[:-1]
        best_k_m = np.argmin(diff_ratios_m) + 1
        if best_k_m < 2:
            best_k_m = 2

        print(
            f"Selected best number of clusters for Male using elbow method: {best_k_m}"
        )

        # Final KMeans for male
        kmeans_m = KMeans(n_clusters=best_k_m, random_state=seed)
        cluster_labels_m = kmeans_m.fit_predict(X_test_m)  # Use X_test_m for consistency
        test_m["cluster"] = cluster_labels_m  # Add cluster labels to test_m
        df.loc[df["Female"] == 0, "cluster"] = (
            cluster_labels_m  # Update df with cluster labels
        )

        # TSNE for male using original features
        reducer_m = TSNE(n_components=2, random_state=seed)
        embedding_m = reducer_m.fit_transform(X_test_m)  # Use X_test_m for TSNE

        # Average risk per cluster for male
        test_m["prob_sexspecific"] = prob_m  # Add prob_sexspecific to test_m
        cluster_risk_m = test_m.groupby("cluster")["prob_sexspecific"].mean().round(3)
        print("Average risk per cluster (Male):")
        print(cluster_risk_m)

        # Visualization for male
        plt.figure(figsize=(8, 6))
        palette_m = sns.color_palette("Set2", best_k_m)
        for c in range(best_k_m):
            mask_m = test_m["cluster"] == c
            plt.scatter(
                embedding_m[mask_m, 0],
                embedding_m[mask_m, 1],
                color=palette_m[c],
                s=30,
                alpha=0.7,
                label=f"Cluster {c} (avg risk = {cluster_risk_m[c]:.2f})",
            )
        plt.title("TSNE + KMeans Clustering with Average Risk (Male)")
        plt.xlabel("TSNE-1")
        plt.ylabel("TSNE-2")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Clustering for Female
    if not X_test_f.empty:
        # Elbow method for female
        inertias_f = []
        for k in range(1, max_clusters + 1):
            kmeans_f = KMeans(n_clusters=k, random_state=seed)
            kmeans_f.fit(X_test_f)
            inertias_f.append(kmeans_f.inertia_)

        diffs_f = np.diff(inertias_f)
        diff_ratios_f = diffs_f / inertias_f[:-1]
        best_k_f = np.argmin(diff_ratios_f) + 1
        if best_k_f < 2:
            best_k_f = 2

        print(
            f"Selected best number of clusters for Female using elbow method: {best_k_f}"
        )

        # Final KMeans for female
        kmeans_f = KMeans(n_clusters=best_k_f, random_state=seed)
        cluster_labels_f = kmeans_f.fit_predict(X_test_f)  # Use X_test_f for consistency
        test_f["cluster"] = cluster_labels_f  # Add cluster labels to test_f
        df.loc[df["Female"] == 1, "cluster"] = (
            cluster_labels_f  # Update df with cluster labels
        )

        # TSNE for female using original features
        reducer_f = TSNE(n_components=2, random_state=seed)
        embedding_f = reducer_f.fit_transform(X_test_f)  # Use X_test_f for TSNE

        # Average risk per cluster for female
        test_f["prob_sexspecific"] = prob_f  # Add prob_sexspecific to test_f
        cluster_risk_f = test_f.groupby("cluster")["prob_sexspecific"].mean().round(3)
        print("Average risk per cluster (Female):")
        print(cluster_risk_f)

        # Visualization for female
        plt.figure(figsize=(8, 6))
        palette_f = sns.color_palette("Set2", best_k_f)
        for c in range(best_k_f):
            mask_f = test_f["cluster"] == c
            plt.scatter(
                embedding_f[mask_f, 0],
                embedding_f[mask_f, 1],
                color=palette_f[c],
                s=30,
                alpha=0.7,
                label=f"Cluster {c} (avg risk = {cluster_risk_f[c]:.2f})",
            )
        plt.title("TSNE + KMeans Clustering with Average Risk (Female)")
        plt.xlabel("TSNE-1")
        plt.ylabel("TSNE-2")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Bootstrap
    B = 100
    accs_m = []
    aucs_m = []
    f1s_m = []
    sens_m = []
    spec_m = []
    accs_f = []
    aucs_f = []
    f1s_f = []
    sens_f = []
    spec_f = []

    for i in range(B):
        # Bootstrap the test df
        df_boot = resample(df, replace=True, n_samples=len(df), random_state=seed + i)

        # Male boot data
        m_boot = df_boot[df_boot["Female"] == 0]
        if not m_boot.empty:
            X_m_boot = m_boot[features]
            y_m_boot = m_boot[labels]
            prob_m_boot = best_male.predict_proba(X_m_boot)[:, 1]
            pred_m_boot = (prob_m_boot >= best_thr_m).astype(int)
            acc_m = accuracy_score(y_m_boot, pred_m_boot)
            auc_m = roc_auc_score(y_m_boot, prob_m_boot)
            f1_m = f1_score(y_m_boot, pred_m_boot)
            sen_m = recall_score(y_m_boot, pred_m_boot)
            spe_m = recall_score(y_m_boot, pred_m_boot, pos_label=0)
            accs_m.append(acc_m)
            aucs_m.append(auc_m)
            f1s_m.append(f1_m)
            sens_m.append(sen_m)
            spec_m.append(spe_m)

        # Female boot data
        f_boot = df_boot[df_boot["Female"] == 1]
        if not f_boot.empty:
            X_f_boot = f_boot[features]
            y_f_boot = f_boot[labels]
            prob_f_boot = best_female.predict_proba(X_f_boot)[:, 1]
            pred_f_boot = (prob_f_boot >= best_thr_f).astype(int)
            acc_f = accuracy_score(y_f_boot, pred_f_boot)
            auc_f = roc_auc_score(y_f_boot, prob_f_boot)
            f1_f = f1_score(y_f_boot, pred_f_boot)
            sen_f = recall_score(y_f_boot, pred_f_boot)
            spe_f = recall_score(y_f_boot, pred_f_boot, pos_label=0)
            accs_f.append(acc_f)
            aucs_f.append(auc_f)
            f1s_f.append(f1_f)
            sens_f.append(sen_f)
            spec_f.append(spe_f)

    # Report for Male
    if accs_m:
        mean_acc_m = np.mean(accs_m)
        ci_low_acc_m = np.percentile(accs_m, 2.5)
        ci_high_acc_m = np.percentile(accs_m, 97.5)
        mean_auc_m = np.mean(aucs_m)
        ci_low_auc_m = np.percentile(aucs_m, 2.5)
        ci_high_auc_m = np.percentile(aucs_m, 97.5)
        mean_f1_m = np.mean(f1s_m)
        ci_low_f1_m = np.percentile(f1s_m, 2.5)
        ci_high_f1_m = np.percentile(f1s_m, 97.5)
        mean_sen_m = np.mean(sens_m)
        ci_low_sen_m = np.percentile(sens_m, 2.5)
        ci_high_sen_m = np.percentile(sens_m, 97.5)
        mean_spe_m = np.mean(spec_m)
        ci_low_spe_m = np.percentile(spec_m, 2.5)
        ci_high_spe_m = np.percentile(spec_m, 97.5)
        print(
            f"Male Accuracy: {mean_acc_m:.4f} [95% CI: {ci_low_acc_m:.4f} - {ci_high_acc_m:.4f}]"
        )
        print(
            f"Male AUC: {mean_auc_m:.4f} [95% CI: {ci_low_auc_m:.4f} - {ci_high_auc_m:.4f}]"
        )
        print(
            f"Male F1: {mean_f1_m:.4f} [95% CI: {ci_low_f1_m:.4f} - {ci_high_f1_m:.4f}]"
        )
        print(
            f"Male Sensitivity: {mean_sen_m:.4f} [95% CI: {ci_low_sen_m:.4f} - {ci_high_sen_m:.4f}]"
        )
        print(
            f"Male Specificity: {mean_spe_m:.4f} [95% CI: {ci_low_spe_m:.4f} - {ci_high_spe_m:.4f}]"
        )

    # Report for Female
    if accs_f:
        mean_acc_f = np.mean(accs_f)
        ci_low_acc_f = np.percentile(accs_f, 2.5)
        ci_high_acc_f = np.percentile(accs_f, 97.5)
        mean_auc_f = np.mean(aucs_f)
        ci_low_auc_f = np.percentile(aucs_f, 2.5)
        ci_high_auc_f = np.percentile(aucs_f, 97.5)
        mean_f1_f = np.mean(f1s_f)
        ci_low_f1_f = np.percentile(f1s_f, 2.5)
        ci_high_f1_f = np.percentile(f1s_f, 97.5)
        mean_sen_f = np.mean(sens_f)
        ci_low_sen_f = np.percentile(sens_f, 2.5)
        ci_high_sen_f = np.percentile(sens_f, 97.5)
        mean_spe_f = np.mean(spec_f)
        ci_low_spe_f = np.percentile(spec_f, 2.5)
        ci_high_spe_f = np.percentile(spec_f, 97.5)
        print(
            f"Female Accuracy: {mean_acc_f:.4f} [95% CI: {ci_low_acc_f:.4f} - {ci_high_acc_f:.4f}]"
        )
        print(
            f"Female AUC: {mean_auc_f:.4f} [95% CI: {ci_low_auc_f:.4f} - {ci_high_auc_f:.4f}]"
        )
        print(
            f"Female F1: {mean_f1_f:.4f} [95% CI: {ci_low_f1_f:.4f} - {ci_high_f1_f:.4f}]"
        )
        print(
            f"Female Sensitivity: {mean_sen_f:.4f} [95% CI: {ci_low_sen_f:.4f} - {ci_high_sen_f:.4f}]"
        )
        print(
            f"Female Specificity: {mean_spe_f:.4f} [95% CI: {ci_low_spe_f:.4f} - {ci_high_spe_f:.4f}]"
        )

    return None
