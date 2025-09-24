import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd

from lifelines.utils import concordance_index

# Reuse modeling and preprocessing utilities from cox.py
import ACC
from ACC import (
    fit_cox_model,
    predict_risk,
    drop_rows_with_missing_local_features,
    select_features_max_cindex_forward,
    stability_select_features,
    FEATURE_SETS,
)

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore

    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


@dataclass
class GroupModel:
    features: List[str]
    threshold: float
    model: object  # CoxPHFitter


class BenefitClassifier:
    """Binary classifier predicting whether a sample benefits from Plus over Base.

    - Uses ONLY Base features for training and inference
    - Internally uses L1 logistic regression with class_weight='balanced'
    - Excludes uncertain training points near zero-benefit margin if specified
    """

    def __init__(
        self,
        base_features: List[str],
        margin_std: float = 0.1,
        random_state: int = 0,
    ) -> None:
        self.base_features = list(base_features)
        self.margin_std = float(margin_std)
        self.random_state = int(random_state)
        self.model: Optional[LogisticRegression] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        z_benefit: np.ndarray,
    ) -> None:
        X = train_df[self.base_features].copy()
        z = np.asarray(z_benefit, dtype=float)
        if not np.isfinite(z).any():
            # Degenerate
            self.model = None
            return

        # Build labels with margin exclusion
        z_clean = z[np.isfinite(z)]
        if z_clean.size == 0:
            self.model = None
            return
        std_z = float(np.nanstd(z_clean))
        margin = self.margin_std * std_z if std_z > 0 else 0.0
        # Keep only confident points
        keep_mask = (z > margin) | (z < -margin)
        if not np.any(keep_mask):
            # No confident supervision
            self.model = None
            return
        y = (z > margin).astype(int)[keep_mask]
        X_keep = X.loc[keep_mask]

        # If only one class after margining, cannot train a classifier
        unique_labels = np.unique(y)
        if unique_labels.size < 2:
            self.model = None
            return

        # Simple hyperparameter sweep over C
        best_auc = -np.inf
        best_model: Optional[LogisticRegression] = None
        Cs = [0.1, 0.5, 1.0]
        skf = StratifiedKFold(
            n_splits=min(3, int(np.bincount(y).min())),
            shuffle=True,
            random_state=self.random_state,
        )
        for C in Cs:
            try:
                lr = LogisticRegression(
                    penalty="l1",
                    C=C,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=self.random_state,
                    max_iter=200,
                )
                aucs: List[float] = []
                for tr_idx, val_idx in skf.split(X_keep, y):
                    X_tr = X_keep.iloc[tr_idx]
                    y_tr = y[tr_idx]
                    X_va = X_keep.iloc[val_idx]
                    y_va = y[val_idx]
                    lr.fit(X_tr, y_tr)
                    p = lr.predict_proba(X_va)[:, 1]
                    # Use ROC AUC for binary benefit classification; safe if only one class in fold
                    try:
                        if np.unique(y_va).size < 2:
                            auc = np.nan
                        else:
                            auc = roc_auc_score(y_va, p)
                    except Exception:
                        auc = np.nan
                    aucs.append(auc)
                # Avoid RuntimeWarning when all values are NaN
                finite_aucs = np.asarray(aucs, dtype=float)
                finite_aucs = finite_aucs[np.isfinite(finite_aucs)]
                mean_auc = float(np.mean(finite_aucs)) if finite_aucs.size > 0 else -np.inf
                if mean_auc > best_auc:
                    best_auc = mean_auc
                    best_model = lr
            except Exception:
                continue

        self.model = best_model

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            # Default to predicting all non-benefit
            return np.zeros(len(df), dtype=float)
        X = df[self.base_features].copy()
        try:
            return self.model.predict_proba(X)[:, 1]
        except Exception:
            return np.zeros(len(df), dtype=float)

    def predict_label(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        p = self.predict_proba(df)
        return (p >= float(threshold)).astype(int)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Return feature importance for the logistic classifier as a DataFrame.

        Columns: feature, coef, abs_coef, odds_ratio
        """
        if self.model is None:
            return None
        try:
            coefs = self.model.coef_.reshape(-1)
            feats = list(self.base_features)
            # Align in case model dropped features internally
            if len(coefs) != len(feats):
                # Try to get from model classes if available; otherwise fallback with NaNs
                data = {
                    "feature": feats,
                    "coef": [np.nan] * len(feats),
                }
                df_imp = pd.DataFrame(data)
            else:
                df_imp = pd.DataFrame({"feature": feats, "coef": coefs})
            df_imp["abs_coef"] = df_imp["coef"].abs()
            with np.errstate(over="ignore"):
                df_imp["odds_ratio"] = np.exp(df_imp["coef"])  # may overflow -> inf acceptable
            df_imp = df_imp.sort_values("abs_coef", ascending=False).reset_index(drop=True)
            return df_imp
        except Exception:
            return None


def _standardize_by_train(risk_tr: np.ndarray, risk_va: np.ndarray) -> np.ndarray:
    mu = float(np.nanmean(risk_tr))
    sd = float(np.nanstd(risk_tr, ddof=1))
    if not np.isfinite(sd) or sd == 0.0:
        return risk_va
    return (risk_va - mu) / sd


def compute_oof_risks(
    train_df: pd.DataFrame,
    base_features: List[str],
    plus_features: List[str],
    time_col: str,
    event_col: str,
    k_splits: int = 5,
    random_state: int = 0,
    enforce_fair_subset: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute out-of-fold risks for Base and Plus Cox models on training data.

    - Standardizes each fold's validation risks using that fold's training-risk mean/std
    - If enforce_fair_subset=True, restricts to rows with complete local categorical features
    - Returns risks aligned with train_df's index; missing entries filled with NaN
    """
    if enforce_fair_subset:
        tr_local = drop_rows_with_missing_local_features(train_df)
    else:
        tr_local = train_df.copy()

    # Work arrays
    risk_base_oof = np.full(len(tr_local), np.nan, dtype=float)
    risk_plus_oof = np.full(len(tr_local), np.nan, dtype=float)

    # Indices for alignment
    idx_array = tr_local.index.to_numpy()

    # Stratify by events if possible
    y = tr_local[event_col].values
    try:
        skf = StratifiedKFold(
            n_splits=k_splits, shuffle=True, random_state=random_state
        )
        splits = list(skf.split(np.zeros(len(tr_local)), y))
    except Exception:
        # Fallback to simple KFold-like partition
        rng = np.random.RandomState(random_state)
        order = rng.permutation(len(tr_local))
        folds = np.array_split(order, k_splits)
        splits = [
            (
                np.setdiff1d(np.arange(len(tr_local)), f, assume_unique=False),
                f,
            )
            for f in folds
        ]

    # Optional inner progress over OOF folds
    iter_splits = (
        tqdm(splits, desc="[Benefit] OOF folds", leave=False)
        if _HAS_TQDM
        else splits
    )
    for tr_idx_local, va_idx_local in iter_splits:
        tr_part = tr_local.iloc[tr_idx_local]
        va_part = tr_local.iloc[va_idx_local]

        # Base model
        try:
            cph_b = fit_cox_model(tr_part, base_features, time_col, event_col)
            risk_tr_b = predict_risk(cph_b, tr_part, base_features)
            risk_va_b = predict_risk(cph_b, va_part, base_features)
            risk_va_b_std = _standardize_by_train(risk_tr_b, risk_va_b)
            risk_base_oof[va_idx_local] = risk_va_b_std
        except Exception:
            pass

        # Plus model
        try:
            cph_p = fit_cox_model(tr_part, plus_features, time_col, event_col)
            risk_tr_p = predict_risk(cph_p, tr_part, plus_features)
            risk_va_p = predict_risk(cph_p, va_part, plus_features)
            risk_va_p_std = _standardize_by_train(risk_tr_p, risk_va_p)
            risk_plus_oof[va_idx_local] = risk_va_p_std
        except Exception:
            pass

    # Rebuild arrays aligned with original train_df order; fill others as NaN
    full_base = np.full(len(train_df), np.nan, dtype=float)
    full_plus = np.full(len(train_df), np.nan, dtype=float)
    pos = train_df.index.get_indexer(idx_array)
    full_base[pos] = risk_base_oof
    full_plus[pos] = risk_plus_oof
    return full_base, full_plus


def _train_group_model(
    df_group: pd.DataFrame,
    candidate_features: List[str],
    time_col: str,
    event_col: str,
    stability_seeds: List[int],
    verbose: bool = False,
) -> Optional[GroupModel]:
    if df_group is None or df_group.empty:
        return None
    # No feature selection: use full candidate features; rely on fit_cox_model sanitization
    kept = [f for f in candidate_features if f in df_group.columns]
    if not kept:
        return None
    try:
        cph = fit_cox_model(df_group, kept, time_col, event_col)
        # Use model's learned features to ensure consistency
        model_feats = list(getattr(cph, "params_", pd.Series()).index)
        risk_tr = predict_risk(cph, df_group, model_feats if model_feats else kept)
        thr = float(np.nanmedian(risk_tr))
        return GroupModel(features=(model_feats if model_feats else list(kept)), threshold=thr, model=cph)
    except Exception:
        return None


def evaluate_benefit_specific_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    time_col: str,
    event_col: str,
    base_pool: List[str],
    plus_pool: List[str],
    random_state: int,
    k_splits: int = 5,
    enforce_fair_subset: bool = True,
    min_group_frac: float = 0.15,
    min_group_n: int = 100,
    min_events: int = 30,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Train benefit-specific models on train_df and evaluate on test_df.

    Returns (pred_label, risk_scores, metrics) on test_df order.
    """
    # Optional fairness: require complete local features throughout
    if enforce_fair_subset:
        train_df = drop_rows_with_missing_local_features(train_df)
        test_df = drop_rows_with_missing_local_features(test_df)

    # 1) OOF risks on training data
    risk_base_oof, risk_plus_oof = compute_oof_risks(
        train_df,
        base_pool,
        plus_pool,
        time_col,
        event_col,
        k_splits=k_splits,
        random_state=random_state,
        enforce_fair_subset=False,  # already enforced above if requested
    )
    # Construct benefit signal z
    y_tr = train_df[event_col].values
    y_signed = (2 * y_tr - 1).astype(float)
    z = (risk_plus_oof - risk_base_oof) * y_signed

    # 2) Train benefit classifier using ONLY Base features
    benefit_clf = BenefitClassifier(
        base_features=base_pool, margin_std=0.1, random_state=random_state
    )
    benefit_clf.fit(train_df, z)
    # Capture classifier importance if available
    clf_importance = benefit_clf.get_feature_importance()

    # 3) Split train by predicted benefit, enforce minimum sizes
    train_pred = benefit_clf.predict_label(train_df, threshold=0.5)
    benefit_mask = train_pred == 1
    non_benefit_mask = ~benefit_mask
    n_total = len(train_df)

    def _num_events(df: pd.DataFrame) -> int:
        try:
            return int(df[event_col].sum())
        except Exception:
            return 0

    group_ok = True
    df_benefit = train_df.loc[benefit_mask]
    df_non_benefit = train_df.loc[non_benefit_mask]
    # Enforce minimums
    if (
        len(df_benefit) < max(int(np.ceil(min_group_frac * n_total)), int(min_group_n))
        or _num_events(df_benefit) < int(min_events)
        or len(df_non_benefit)
        < max(int(np.ceil(min_group_frac * n_total)), int(min_group_n))
        or _num_events(df_non_benefit) < int(min_events)
    ):
        group_ok = False

    # 4) Train models
    stability_seeds = list(range(0, min(20, n_total)))
    base_model: Optional[GroupModel] = None
    plus_model: Optional[GroupModel] = None

    if group_ok:
        # Benefit group uses Plus pool
        plus_model = _train_group_model(
            df_group=df_benefit,
            candidate_features=plus_pool,
            time_col=time_col,
            event_col=event_col,
            stability_seeds=stability_seeds,
            verbose=verbose,
        )
        # Non-benefit group uses Base pool
        base_model = _train_group_model(
            df_group=df_non_benefit,
            candidate_features=base_pool,
            time_col=time_col,
            event_col=event_col,
            stability_seeds=stability_seeds,
            verbose=verbose,
        )
        if plus_model is None or base_model is None:
            group_ok = False

    # Fallback: single-model selection using OOF c-index on training
    single_model: Optional[GroupModel] = None
    single_model_is_plus = False
    if not group_ok:
        # Evaluate OOF c-index
        try:
            cidx_base = concordance_index(
                train_df[time_col], -risk_base_oof, train_df[event_col]
            )
        except Exception:
            cidx_base = np.nan
        try:
            cidx_plus = concordance_index(
                train_df[time_col], -risk_plus_oof, train_df[event_col]
            )
        except Exception:
            cidx_plus = np.nan
        use_plus = bool(
            np.nan_to_num(cidx_plus, nan=-np.inf)
            > np.nan_to_num(cidx_base, nan=-np.inf)
        )
        pool = plus_pool if use_plus else base_pool
        try:
            single_model = _train_group_model(
                df_group=train_df,
                candidate_features=pool,
                time_col=time_col,
                event_col=event_col,
                stability_seeds=stability_seeds,
                verbose=verbose,
            )
            single_model_is_plus = use_plus
        except Exception:
            single_model = None

    # 5) Inference on test
    risk_scores = np.zeros(len(test_df), dtype=float)
    pred_labels = np.zeros(len(test_df), dtype=int)

    if single_model is not None:
        feats = single_model.features
        try:
            r = predict_risk(single_model.model, test_df, feats)
            risk_scores = r
            pred_labels = (r >= single_model.threshold).astype(int)
        except Exception:
            risk_scores = np.zeros(len(test_df))
            pred_labels = np.zeros(len(test_df), dtype=int)
        try:
            cidx = concordance_index(
                test_df[time_col], -risk_scores, test_df[event_col]
            )
        except Exception:
            cidx = np.nan
        # Even in fallback, expose benefit mask predicted by classifier for downstream subgroup analysis
        try:
            benefit_mask = benefit_clf.predict_label(test_df, threshold=0.5).astype(bool)
        except Exception:
            benefit_mask = np.zeros(len(test_df), dtype=bool)
        return (
            pred_labels,
            risk_scores,
            {
                "c_index": cidx,
                "grouping": 0,
                "single_is_plus": float(single_model_is_plus),
                "benefit_mask": benefit_mask,
                "benefit_importance": clf_importance,
            },
        )

    # Grouped inference
    test_pred = benefit_clf.predict_label(test_df, threshold=0.5)
    mask_benefit = test_pred == 1
    mask_non_benefit = ~mask_benefit

    # Non-benefit group
    if base_model is not None and np.any(mask_non_benefit):
        te_nb = test_df.loc[mask_non_benefit]
        try:
            r_nb = predict_risk(base_model.model, te_nb, base_model.features)
            risk_scores[mask_non_benefit] = r_nb
            pred_labels[mask_non_benefit] = (r_nb >= base_model.threshold).astype(int)
        except Exception:
            pass

    # Benefit group
    if plus_model is not None and np.any(mask_benefit):
        te_b = test_df.loc[mask_benefit]
        try:
            r_b = predict_risk(plus_model.model, te_b, plus_model.features)
            risk_scores[mask_benefit] = r_b
            pred_labels[mask_benefit] = (r_b >= plus_model.threshold).astype(int)
        except Exception:
            # If prediction fails for any reason, leave zeros
            pass

    try:
        cidx_all = concordance_index(
            test_df[time_col], -risk_scores, test_df[event_col]
        )
    except Exception:
        cidx_all = np.nan
    return (
        pred_labels,
        risk_scores,
        {
            "c_index": cidx_all,
            "grouping": 1,
            "benefit_mask": mask_benefit.astype(bool),
            "benefit_importance": clf_importance,
        },
    )


def _train_two_models_on_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    time_col: str,
    event_col: str,
    base_pool: List[str],
    plus_pool: List[str],
    verbose: bool = False,
) -> Tuple[
    Optional[GroupModel],
    Optional[GroupModel],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Train Base and Plus Cox models on train_df and apply to test_df.

    Returns (base_model, plus_model, risk_base_test, risk_plus_test, choose_plus_mask)
    where choose_plus_mask indicates samples where Plus model yields lower predicted risk.
    """
    base_model = _train_group_model(
        df_group=train_df,
        candidate_features=base_pool,
        time_col=time_col,
        event_col=event_col,
        stability_seeds=list(range(5)),
        verbose=verbose,
    )
    plus_model = _train_group_model(
        df_group=train_df,
        candidate_features=plus_pool,
        time_col=time_col,
        event_col=event_col,
        stability_seeds=list(range(5)),
        verbose=verbose,
    )

    risk_base_test = np.zeros(len(test_df), dtype=float)
    risk_plus_test = np.zeros(len(test_df), dtype=float)
    choose_plus_mask = np.zeros(len(test_df), dtype=bool)

    try:
        if base_model is not None:
            risk_base_test = predict_risk(base_model.model, test_df, base_model.features)
    except Exception:
        risk_base_test = np.zeros(len(test_df), dtype=float)
    try:
        if plus_model is not None:
            risk_plus_test = predict_risk(plus_model.model, test_df, plus_model.features)
    except Exception:
        risk_plus_test = np.zeros(len(test_df), dtype=float)

    # Choose model with lower predicted risk per sample
    try:
        choose_plus_mask = np.asarray(risk_plus_test) < np.asarray(risk_base_test)
    except Exception:
        choose_plus_mask = np.zeros(len(test_df), dtype=bool)

    return base_model, plus_model, risk_base_test, risk_plus_test, choose_plus_mask


def _majority_features(feature_lists: List[List[str]], min_frac: float = 0.5) -> List[str]:
    if not feature_lists:
        return []
    total = len(feature_lists)
    counter: Counter = Counter()
    for feats in feature_lists:
        counter.update(set(f for f in feats))
    keep = [f for f, c in counter.items() if c / total >= float(min_frac)]
    return sorted(keep)


def run_stabilized_two_model_pipeline(
    df: pd.DataFrame,
    N: int = 50,
    time_col: str = "PE_Time",
    event_col: str = "VT/VF/SCD",
    k_splits: int = 5,
    enforce_fair_subset: bool = True,
    tableone_excel_path: Optional[str] = None,
    clf_importance_excel_path: Optional[str] = None,
) -> Dict[str, object]:
    """Run N random splits to stabilize Base and Plus models, then finalize and evaluate.

    Steps:
    - For each split, train Base and Plus on train, apply to test, record used features
    - Aggregate majority features across runs for Base and Plus
    - Train final Base/Plus on full dataset using majority features (fallback to pools)
    - Compute risks on full dataset for three scenarios: all-base, all-plus, per-sample-best
    - Define benefit group as samples where Plus risk < Base risk; generate TableOne
    - Train logistic classifier (Base features only) to predict benefit group; export importance
    """
    base_pool = list(FEATURE_SETS.get("Proposed", []))
    plus_pool = list(FEATURE_SETS.get("Proposed Plus", []))

    # Exclude specified features from both pools
    exclude_features = {"Age by decade", "CrCl>45", "NYHA>2", "Significant LGE"}
    base_pool = [f for f in base_pool if f not in exclude_features]
    plus_pool = [f for f in plus_pool if f not in exclude_features]

    # Exclude specified features from both pools
    exclude_features = {"Age by decade", "CrCl>45", "NYHA>2", "Significant LGE"}
    base_pool = [f for f in base_pool if f not in exclude_features]
    plus_pool = [f for f in plus_pool if f not in exclude_features]

    # Print feature pools used in experiments pipeline
    try:
        print("==== Feature Pools (Experiments) ====")
        print(f"Base (Proposed) features ({len(base_pool)}): {base_pool}")
        print(f"Plus (Proposed Plus) features ({len(plus_pool)}): {plus_pool}")
    except Exception:
        pass

    # Print feature pools used in stabilized pipeline
    try:
        print("==== Feature Pools (Stabilized Pipeline) ====")
        print(f"Base (Proposed) features ({len(base_pool)}): {base_pool}")
        print(f"Plus (Proposed Plus) features ({len(plus_pool)}): {plus_pool}")
    except Exception:
        pass

    if enforce_fair_subset:
        df_use = drop_rows_with_missing_local_features(df)
    else:
        df_use = df.copy()

    base_feat_runs: List[List[str]] = []
    plus_feat_runs: List[List[str]] = []

    iterator = tqdm(range(N), desc="[Stabilize] Splits", leave=True) if _HAS_TQDM else range(N)
    for seed in iterator:
        try:
            tr, te = train_test_split(
                df_use.dropna(subset=[time_col, event_col]).copy(),
                test_size=0.3,
                random_state=seed,
                stratify=df_use[event_col] if df_use[event_col].nunique() > 1 else None,
            )
        except Exception:
            continue

        base_model, plus_model, _, _, _ = _train_two_models_on_split(
            tr,
            te,
            time_col,
            event_col,
            base_pool,
            plus_pool,
            verbose=False,
        )
        if base_model is not None:
            base_feat_runs.append(list(base_model.features))
        if plus_model is not None:
            plus_feat_runs.append(list(plus_model.features))

    # Majority features across runs; fallback to pools if empty
    base_major = _majority_features(base_feat_runs, min_frac=0.5)
    if not base_major:
        base_major = [f for f in base_pool if f in df_use.columns]
    plus_major = _majority_features(plus_feat_runs, min_frac=0.5)
    if not plus_major:
        plus_major = [f for f in plus_pool if f in df_use.columns]

    # Print final selected feature sets
    try:
        print("==== Final Selected Features (Stabilized Pipeline) ====")
        print(f"Final Base features ({len(base_major)}): {base_major}")
        print(f"Final Plus features ({len(plus_major)}): {plus_major}")
    except Exception:
        pass

    # Train final models on full dataset
    final_base = _train_group_model(
        df_group=df_use,
        candidate_features=base_major,
        time_col=time_col,
        event_col=event_col,
        stability_seeds=list(range(10)),
        verbose=False,
    )
    final_plus = _train_group_model(
        df_group=df_use,
        candidate_features=plus_major,
        time_col=time_col,
        event_col=event_col,
        stability_seeds=list(range(10)),
        verbose=False,
    )

    # Compute risks on all data
    risk_base_all = np.zeros(len(df_use), dtype=float)
    risk_plus_all = np.zeros(len(df_use), dtype=float)
    try:
        if final_base is not None:
            risk_base_all = predict_risk(final_base.model, df_use, final_base.features)
    except Exception:
        pass
    try:
        if final_plus is not None:
            risk_plus_all = predict_risk(final_plus.model, df_use, final_plus.features)
    except Exception:
        pass

    # Three scenarios
    risk_best_all = np.minimum(risk_base_all, risk_plus_all)
    benefit_mask = risk_plus_all < risk_base_all

    # Metrics
    def _cidx_safe(risk: np.ndarray) -> float:
        try:
            return float(concordance_index(df_use[time_col], -np.asarray(risk), df_use[event_col]))
        except Exception:
            return np.nan

    metrics = {
        "c_index_all_base": _cidx_safe(risk_base_all),
        "c_index_all_plus": _cidx_safe(risk_plus_all),
        "c_index_per_sample_best": _cidx_safe(risk_best_all),
    }

    # TableOne for benefit vs non-benefit groups
    try:
        df_tab = df_use.copy()
        df_tab["BenefitGroup"] = np.where(benefit_mask, "Benefit", "Non-Benefit")
        try:
            ACC.generate_tableone_by_group(
                df_tab,
                group_col="BenefitGroup",
                output_excel_path=tableone_excel_path,
            )
        except Exception:
            pass
    except Exception:
        pass

    # Train a classifier (Base features only) to predict benefit group
    clf_model: Optional[LogisticRegression] = None
    clf_importance: Optional[pd.DataFrame] = None
    try:
        X = df_use[[f for f in base_pool if f in df_use.columns]].copy()
        y = benefit_mask.astype(int)
        if X.shape[1] > 0 and len(np.unique(y)) == 2:
            best_auc = -np.inf
            best_lr: Optional[LogisticRegression] = None
            Cs = [0.1, 0.5, 1.0]
            skf = StratifiedKFold(
                n_splits=min(5, int(np.bincount(y).min())) if np.bincount(y).size > 1 else 3,
                shuffle=True,
                random_state=0,
            )
            for C in Cs:
                try:
                    lr = LogisticRegression(
                        penalty="l1",
                        C=C,
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=0,
                        max_iter=200,
                    )
                    aucs: List[float] = []
                    for tr_idx, va_idx in skf.split(X, y):
                        X_tr = X.iloc[tr_idx]
                        y_tr = y[tr_idx]
                        X_va = X.iloc[va_idx]
                        y_va = y[va_idx]
                        lr.fit(X_tr, y_tr)
                        p = lr.predict_proba(X_va)[:, 1]
                        try:
                            if np.unique(y_va).size < 2:
                                auc = np.nan
                            else:
                                auc = roc_auc_score(y_va, p)
                        except Exception:
                            auc = np.nan
                        aucs.append(auc)
                    finite_aucs = np.asarray(aucs, dtype=float)
                    finite_aucs = finite_aucs[np.isfinite(finite_aucs)]
                    mean_auc = float(np.mean(finite_aucs)) if finite_aucs.size > 0 else -np.inf
                    if mean_auc > best_auc:
                        best_auc = mean_auc
                        best_lr = lr
                except Exception:
                    continue
            clf_model = best_lr
            if clf_model is not None:
                clf_model.fit(X, y)
                try:
                    coefs = clf_model.coef_.reshape(-1)
                    feats = list(X.columns)
                    df_imp = pd.DataFrame({"feature": feats, "coef": coefs})
                    df_imp["abs_coef"] = df_imp["coef"].abs()
                    with np.errstate(over="ignore"):
                        df_imp["odds_ratio"] = np.exp(df_imp["coef"])  # may overflow
                    clf_importance = df_imp.sort_values("abs_coef", ascending=False).reset_index(drop=True)
                    if clf_importance_excel_path is not None:
                        try:
                            os.makedirs(os.path.dirname(clf_importance_excel_path), exist_ok=True)
                            clf_importance.to_excel(clf_importance_excel_path, index=False)
                        except Exception:
                            pass
                except Exception:
                    clf_importance = None
    except Exception:
        clf_model = None
        clf_importance = None

    return {
        "final_base_features": base_major,
        "final_plus_features": plus_major,
        "risk_base_all": risk_base_all,
        "risk_plus_all": risk_plus_all,
        "benefit_mask": benefit_mask,
        "metrics": metrics,
        "classifier_importance": clf_importance,
    }


def run_benefit_specific_experiments(
    df: pd.DataFrame,
    N: int = 50,
    time_col: str = "PE_Time",
    event_col: str = "VT/VF/SCD",
    k_splits: int = 5,
    enforce_fair_subset: bool = True,
    export_excel_path: Optional[str] = None,
    print_first_split_preview: bool = False,
) -> Tuple[Dict[str, Dict[str, List[float]]], pd.DataFrame]:
    """Run outer-loop evaluation for benefit-specific vs Proposed sex-specific baselines.

    Returns (results_dict, summary_table)

    If print_first_split_preview=True, print classifier importance and TableOne for the first split.
    """
    base_pool = list(FEATURE_SETS.get("Proposed", []))
    plus_pool = list(FEATURE_SETS.get("Proposed Plus", []))

    model_configs = [
        {"name": "Proposed Plus (benefit-specific)", "mode": "benefit_specific"},
        {"name": "Proposed (sex-specific)", "mode": "sex_specific_baseline"},
    ]
    metrics = [
        "c_index_all",
        "c_index_male",
        "c_index_female",
        "c_index_benefit",
        "c_index_non_benefit",
    ]
    results: Dict[str, Dict[str, List[float]]] = {
        cfg["name"]: {m: [] for m in metrics} for cfg in model_configs
    }

    iterator = tqdm(range(N), desc="[Benefit] Splits", leave=True) if _HAS_TQDM else range(N)
    for seed in iterator:
        tr, te = train_test_split(
            df.dropna(subset=[time_col, event_col]).copy(),
            test_size=0.3,
            random_state=seed,
            stratify=df[event_col] if df[event_col].nunique() > 1 else None,
        )

        # Benefit-specific
        pred_b, risk_b, met_b = evaluate_benefit_specific_split(
            tr,
            te,
            time_col,
            event_col,
            base_pool,
            plus_pool,
            random_state=seed,
            k_splits=k_splits,
            enforce_fair_subset=enforce_fair_subset,
        )

        # Sex-specific baseline on Proposed features (reuse cox.evaluate_split)
        pred_s, risk_s, met_s = ACC.evaluate_split(
            tr,
            te,
            feature_cols=base_pool,
            time_col=time_col,
            event_col=event_col,
            mode="sex_specific",
            seed=seed,
            use_undersampling=False,
            disable_within_split_feature_selection=False,
        )

        # Collect metrics consistently (per-sex c-index computed from risk arrays)
        def _safe_cidx(mask: np.ndarray, risk: np.ndarray, te_local: pd.DataFrame) -> float:
            try:
                if mask.dtype != bool:
                    mask_bool = mask.astype(bool)
                else:
                    mask_bool = mask
                t = te_local.loc[mask_bool, time_col].values
                e = te_local.loc[mask_bool, event_col].values
                r = np.asarray(risk)[mask]
                if len(t) < 2 or np.all(~np.isfinite(r)) or np.allclose(r, r[0]):
                    return np.nan
                return float(concordance_index(t, -r, e))
            except Exception:
                return np.nan

        # Build aligned evaluation frames for subgroup metrics
        te_eval_b = (
            ACC.drop_rows_with_missing_local_features(te)
            if enforce_fair_subset
            else te.copy()
        )
        te_eval_s = ACC.drop_rows_with_missing_local_features(te)
        mask_m = (
            te_eval_s["Female"].values == 0
            if "Female" in te_eval_s.columns
            else np.zeros(len(te_eval_s), dtype=bool)
        )
        mask_f = (
            te_eval_s["Female"].values == 1
            if "Female" in te_eval_s.columns
            else np.zeros(len(te_eval_s), dtype=bool)
        )

        # Benefit-specific
        results["Proposed Plus (benefit-specific)"]["c_index_all"].append(
            met_b.get("c_index", np.nan)
        )
        # Benefit vs Non-benefit subgroup C-index
        benefit_mask = met_b.get("benefit_mask", np.zeros(len(te_eval_b), dtype=bool))
        if benefit_mask.shape[0] != len(te_eval_b):
            # Best-effort alignment fallback: truncate or pad
            min_len = min(benefit_mask.shape[0], len(te_eval_b))
            benefit_mask = np.asarray(benefit_mask).astype(bool)[:min_len]
            risk_b = np.asarray(risk_b)[:min_len]
            te_eval_b = te_eval_b.iloc[:min_len]
        non_benefit_mask = ~benefit_mask
        results["Proposed Plus (benefit-specific)"]["c_index_benefit"].append(
            _safe_cidx(benefit_mask, risk_b, te_eval_b)
        )
        results["Proposed Plus (benefit-specific)"]["c_index_non_benefit"].append(
            _safe_cidx(non_benefit_mask, risk_b, te_eval_b)
        )

        # Sex-specific baseline
        results["Proposed (sex-specific)"]["c_index_all"].append(
            met_s.get("c_index", np.nan)
        )
        results["Proposed (sex-specific)"]["c_index_male"].append(
            _safe_cidx(mask_m, risk_s, te_eval_s)
        )
        results["Proposed (sex-specific)"]["c_index_female"].append(
            _safe_cidx(mask_f, risk_s, te_eval_s)
        )
        # On the first split, optionally generate TableOne and print classifier importance
        if print_first_split_preview and seed == 0:
            try:
                benefit_importance = met_b.get("benefit_importance", None)
                if benefit_importance is not None:
                    print("==== Benefit Classifier Feature Importance (top 20 by |coef|) ====")
                    print(benefit_importance.head(20))
            except Exception:
                pass
            try:
                te_b_tab = te_eval_b.copy()
                te_b_tab["BenefitGroup"] = np.where(benefit_mask, "Benefit", "Non-Benefit")
                # Generate TableOne for BenefitGroup
                try:
                    ACC.generate_tableone_by_group(
                        te_b_tab,
                        group_col="BenefitGroup",
                        output_excel_path=None,
                    )
                except Exception:
                    # Silent fail if grouping utility missing
                    pass
            except Exception:
                pass

    # Summaries
    summary = {}
    for model, mvals in results.items():
        summary[model] = {}
        for metric, values in mvals.items():
            arr = np.array(values, dtype=float)
            finite = np.isfinite(arr)
            n = int(finite.sum())
            if n == 0:
                mu = np.nan
                ci_lower = np.nan
                ci_upper = np.nan
            else:
                vals = arr[finite]
                mu = float(np.mean(vals))
                if n > 1:
                    se = float(np.std(vals, ddof=1)) / np.sqrt(n)
                    ci = 1.96 * se
                else:
                    ci = 0.0
                ci_lower = mu - ci
                ci_upper = mu + ci
            summary[model][metric] = (mu, ci_lower, ci_upper)

    summary_df = pd.concat(
        {
            model: pd.DataFrame.from_dict(
                mvals, orient="index", columns=["mean", "ci_lower", "ci_upper"]
            )
            for model, mvals in summary.items()
        },
        axis=0,
    )

    formatted = summary_df.apply(
        lambda row: f"{row['mean']:.3f} ({row['ci_lower']:.3f}, {row['ci_upper']:.3f})",
        axis=1,
    )
    summary_table = formatted.unstack(level=1)
    summary_table = summary_table.rename(
        columns={
            "c_index_all": "all",
            "c_index_male": "male",
            "c_index_female": "female",
            "c_index_benefit": "benefit",
            "c_index_non_benefit": "non_benefit",
        }
    )

    if export_excel_path is not None:
        os.makedirs(os.path.dirname(export_excel_path), exist_ok=True)
        summary_table.to_excel(export_excel_path, index=True, index_label="RowName")

    return results, summary_table


if __name__ == "__main__":
    # Load and prepare data using cox utilities
    try:
        df = ACC.load_dataframes()
    except Exception as e:
        raise SystemExit(f"Failed to load dataframes: {e}")

    # Optional: original experiments (can be skipped if focusing on stabilized pipeline)
    try:
        export_path = None  # set to an absolute path to export Excel
        results, summary = run_benefit_specific_experiments(
            df=df,
            N=10,
            time_col="PE_Time",
            event_col="VT/VF/SCD",
            k_splits=5,
            enforce_fair_subset=True,
            export_excel_path=export_path,
        )
        print(summary)
    except Exception:
        pass

    # New stabilized two-model pipeline
    try:
        tableone_path = None  # e.g., "/workspace/outputs/benefit_tableone.xlsx"
        clf_imp_path = None  # e.g., "/workspace/outputs/benefit_classifier_importance.xlsx"
        stabilized = run_stabilized_two_model_pipeline(
            df=df,
            N=10,
            time_col="PE_Time",
            event_col="VT/VF/SCD",
            k_splits=5,
            enforce_fair_subset=True,
            tableone_excel_path=tableone_path,
            clf_importance_excel_path=clf_imp_path,
        )
        print("==== Stabilized Two-Model Metrics ====")
        print(stabilized.get("metrics", {}))
        if stabilized.get("classifier_importance", None) is not None:
            print("==== Benefit Classifier (Base features only) - Top 20 ====")
            try:
                print(stabilized["classifier_importance"].head(20))
            except Exception:
                pass
    except Exception:
        pass
