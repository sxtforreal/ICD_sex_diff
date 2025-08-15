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
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
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

        # highlight specific features by color: red for NYHA>2/NYHA Class, yellow for LGE features
        try:
            from new import feature_color_for_importance
            colors = [feature_color_for_importance(feat_names[i]) for i in idx]
        except Exception:
            # Fallback to original behavior if helper unavailable
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

    y_prob = best_model.predict_proba(X_test)[:, 1]
    threshold = find_best_threshold(y_test, y_prob)
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
      6) RF sex-specific: combine male/female RF predictions on full test set.

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

    # Models
    model_names = [
        "Guideline",
        "RF Guideline",
        "Benchmark Sex-agnostic",
        "Benchmark Sex-agnostic (undersampled)",
        "Benchmark Sex-specific",
        "Benchmark Male",
        "Benchmark Female",
        "Proposed Sex-agnostic",
        "Proposed Sex-agnostic (undersampled)",
        "Proposed Male",
        "Proposed Female",
        "Proposed Sex-specific",
    ]
    # Metrics
    metrics = [
        "accuracy",
        "auc",
        "f1",
        "sensitivity",
        "specificity",
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

        # --- Guideline --- #
        pred_g = (
            ((X_te_g["NYHA Class"] >= 2) & (X_te_g["LVEF"] <= 35)).astype(int).values
        )
        y_true = y_te_g[label].values
        eval_df = y_te_g.reset_index(drop=True).copy()
        eval_df["pred"] = pred_g
        m_rate, f_rate = incidence_rate(eval_df, "pred", label)

        acc = accuracy_score(y_true, pred_g)
        auc = np.nan
        f1 = f1_score(y_true, pred_g)
        sens, spec = compute_sensitivity_specificity(y_true, pred_g)

        for met, val in zip(metrics, [acc, auc, f1, sens, spec, m_rate, f_rate]):
            results["Guideline"][met].append(val)

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

        acc = accuracy_score(y_true, pred_g)
        auc = roc_auc_score(y_true, prob_g)
        f1 = f1_score(y_true, pred_g)
        sens, spec = compute_sensitivity_specificity(y_true, pred_g)

        for met, val in zip(metrics, [acc, auc, f1, sens, spec, m_rate, f_rate]):
            results["RF Guideline"][met].append(val)

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

        acc = accuracy_score(y_true, pred_sa)
        auc = roc_auc_score(y_true, prob_sa)
        f1 = f1_score(y_true, pred_sa)
        sens, spec = compute_sensitivity_specificity(y_true, pred_sa)

        for met, val in zip(metrics, [acc, auc, f1, sens, spec, m_rate, f_rate]):
            results["Benchmark Sex-agnostic"][met].append(val)

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

        acc_us = accuracy_score(y_true, pred_sa_us)
        auc_us = roc_auc_score(y_true, prob_sa_us)
        f1_us = f1_score(y_true, pred_sa_us)
        sens_us, spec_us = compute_sensitivity_specificity(y_true, pred_sa_us)

        for met, val in zip(
            metrics, [acc_us, auc_us, f1_us, sens_us, spec_us, m_rate, f_rate]
        ):
            results["Benchmark Sex-agnostic (undersampled)"][met].append(val)

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

        for met, val in zip(metrics, [acc, auc, f1, sens, spec, m_rate_m, f_rate_m]):
            results["Benchmark Male"][met].append(val)

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

        for met, val in zip(metrics, [acc, auc, f1, sens, spec, m_rate_f, f_rate_f]):
            results["Benchmark Female"][met].append(val)

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

        acc = accuracy_score(y_true, combined_pred)
        auc = roc_auc_score(y_true, combined_prob)
        f1 = f1_score(y_true, combined_pred)
        sens, spec = compute_sensitivity_specificity(y_true, combined_pred)

        for met, val in zip(metrics, [acc, auc, f1, sens, spec, m_rate_c, f_rate_c]):
            results["Benchmark Sex-specific"][met].append(val)

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

        acc = accuracy_score(y_true, pred_sa)
        auc = roc_auc_score(y_true, prob_sa)
        f1 = f1_score(y_true, pred_sa)
        sens, spec = compute_sensitivity_specificity(y_true, pred_sa)

        for met, val in zip(metrics, [acc, auc, f1, sens, spec, m_rate, f_rate]):
            results["Proposed Sex-agnostic"][met].append(val)

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

        acc_us = accuracy_score(y_true, pred_sa_us)
        auc_us = roc_auc_score(y_true, prob_sa_us)
        f1_us = f1_score(y_true, pred_sa_us)
        sens_us, spec_us = compute_sensitivity_specificity(y_true, pred_sa_us)

        for met, val in zip(
            metrics, [acc_us, auc_us, f1_us, sens_us, spec_us, m_rate, f_rate]
        ):
            results["Proposed Sex-agnostic (undersampled)"][met].append(val)

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

        for met, val in zip(metrics, [acc, auc, f1, sens, spec, m_rate_m, f_rate_m]):
            results["Proposed Male"][met].append(val)

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

        for met, val in zip(metrics, [acc, auc, f1, sens, spec, m_rate_f, f_rate_f]):
            results["Proposed Female"][met].append(val)

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

        acc = accuracy_score(y_true, combined_pred)
        auc = roc_auc_score(y_true, combined_prob)
        f1 = f1_score(y_true, combined_pred)
        sens, spec = compute_sensitivity_specificity(y_true, combined_pred)

        for met, val in zip(metrics, [acc, auc, f1, sens, spec, m_rate_c, f_rate_c]):
            results["Proposed Sex-specific"][met].append(val)

    # After all seeds, compute mean and 95% CI
    summary = {}
    for model, mets in results.items():
        summary[model] = {}
        for met, vals in mets.items():
            arr = np.array(vals, dtype=float)
            mu = arr.mean()
            se = arr.std(ddof=1) / np.sqrt(N)
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
    ]
    summary_table = summary_table.drop(index=rows_to_drop)

    # Save result
    output_dir = "/home/sunx/data/aiiih/projects/sunx/clinical_projects/Diane's"
    output_file = "summary_results.xlsx"
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, output_file)
    summary_table.to_excel(full_path, index=True)
    print(f"Summary table saved to: {full_path}")

    return results, summary_table
