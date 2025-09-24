import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from lifelines.utils import concordance_index

# Reuse modeling and preprocessing utilities from cox.py
import cox
from cox import (
    fit_cox_model,
    predict_risk,
    drop_rows_with_missing_local_features,
    select_features_max_cindex_forward,
    stability_select_features,
    FEATURE_SETS,
)

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression


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
        skf = StratifiedKFold(n_splits=min(3, int(np.bincount(y).min())), shuffle=True, random_state=self.random_state)
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
                    # Use Rank-based metric robust to threshold; fall back if degenerate
                    try:
                        auc = concordance_index(y_va, p, y_va)
                    except Exception:
                        auc = np.nan
                    aucs.append(auc)
                mean_auc = float(np.nanmean(aucs))
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
        skf = StratifiedKFold(n_splits=k_splits, shuffle=True, random_state=random_state)
        splits = list(skf.split(np.zeros(len(tr_local)), y))
    except Exception:
        # Fallback to simple KFold-like partition
        rng = np.random.RandomState(random_state)
        order = rng.permutation(len(tr_local))
        folds = np.array_split(order, k_splits)
        splits = [(
            np.setdiff1d(np.arange(len(tr_local)), f, assume_unique=False),
            f,
        ) for f in folds]

    for tr_idx_local, va_idx_local in splits:
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
    # Stability selection first
    try:
        kept = stability_select_features(
            df=df_group,
            candidate_features=list(candidate_features),
            time_col=time_col,
            event_col=event_col,
            seeds=stability_seeds,
            max_features=None,
            threshold=0.4,
            min_features=None,
            verbose=verbose,
        )
    except Exception:
        kept = []
    if not kept:
        try:
            kept = select_features_max_cindex_forward(
                df_group,
                list(candidate_features),
                time_col,
                event_col,
                random_state=42,
                verbose=verbose,
            )
        except Exception:
            kept = [f for f in candidate_features if f in df_group.columns]
    if not kept:
        return None
    try:
        cph = fit_cox_model(df_group, kept, time_col, event_col)
        risk_tr = predict_risk(cph, df_group, kept)
        thr = float(np.nanmedian(risk_tr))
        return GroupModel(features=list(kept), threshold=thr, model=cph)
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
    benefit_clf = BenefitClassifier(base_features=base_pool, margin_std=0.1, random_state=random_state)
    benefit_clf.fit(train_df, z)

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
        or len(df_non_benefit) < max(int(np.ceil(min_group_frac * n_total)), int(min_group_n))
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
            cidx_base = concordance_index(train_df[time_col], -risk_base_oof, train_df[event_col])
        except Exception:
            cidx_base = np.nan
        try:
            cidx_plus = concordance_index(train_df[time_col], -risk_plus_oof, train_df[event_col])
        except Exception:
            cidx_plus = np.nan
        use_plus = bool(np.nan_to_num(cidx_plus, nan=-np.inf) > np.nan_to_num(cidx_base, nan=-np.inf))
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
            cidx = concordance_index(test_df[time_col], -risk_scores, test_df[event_col])
        except Exception:
            cidx = np.nan
        return pred_labels, risk_scores, {"c_index": cidx, "grouping": 0, "single_is_plus": float(single_model_is_plus)}

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
        cidx_all = concordance_index(test_df[time_col], -risk_scores, test_df[event_col])
    except Exception:
        cidx_all = np.nan
    return pred_labels, risk_scores, {"c_index": cidx_all, "grouping": 1}


def run_benefit_specific_experiments(
    df: pd.DataFrame,
    N: int = 50,
    time_col: str = "PE_Time",
    event_col: str = "VT/VF/SCD",
    k_splits: int = 5,
    enforce_fair_subset: bool = True,
    export_excel_path: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, List[float]]], pd.DataFrame]:
    """Run outer-loop evaluation for benefit-specific vs Proposed sex-specific baselines.

    Returns (results_dict, summary_table)
    """
    base_pool = list(FEATURE_SETS.get("Proposed", []))
    plus_pool = list(FEATURE_SETS.get("Proposed Plus", []))

    model_configs = [
        {"name": "Proposed Plus (benefit-specific)", "mode": "benefit_specific"},
        {"name": "Proposed (sex-specific)", "mode": "sex_specific_baseline"},
    ]
    metrics = ["c_index_all", "c_index_male", "c_index_female"]
    results: Dict[str, Dict[str, List[float]]] = {
        cfg["name"]: {m: [] for m in metrics} for cfg in model_configs
    }

    for seed in range(N):
        tr, te = train_test_split(
            df.dropna(subset=[time_col, event_col]).copy(),
            test_size=0.3,
            random_state=seed,
            stratify=df[event_col]
            if df[event_col].nunique() > 1
            else None,
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
        pred_s, risk_s, met_s = cox.evaluate_split(
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
        def _safe_cidx(mask: np.ndarray, risk: np.ndarray) -> float:
            try:
                t = te.loc[mask, time_col].values
                e = te.loc[mask, event_col].values
                r = np.asarray(risk)[mask]
                if len(t) < 2 or np.all(~np.isfinite(r)) or np.allclose(r, r[0]):
                    return np.nan
                return float(concordance_index(t, -r, e))
            except Exception:
                return np.nan

        mask_m = te["Female"].values == 0 if "Female" in te.columns else np.zeros(len(te), dtype=bool)
        mask_f = te["Female"].values == 1 if "Female" in te.columns else np.zeros(len(te), dtype=bool)

        # Benefit-specific
        results["Proposed Plus (benefit-specific)"]["c_index_all"].append(met_b.get("c_index", np.nan))
        results["Proposed Plus (benefit-specific)"]["c_index_male"].append(_safe_cidx(mask_m, risk_b))
        results["Proposed Plus (benefit-specific)"]["c_index_female"].append(_safe_cidx(mask_f, risk_b))

        # Sex-specific baseline
        results["Proposed (sex-specific)"]["c_index_all"].append(met_s.get("c_index", np.nan))
        results["Proposed (sex-specific)"]["c_index_male"].append(_safe_cidx(mask_m, risk_s))
        results["Proposed (sex-specific)"]["c_index_female"].append(_safe_cidx(mask_f, risk_s))

    # Summaries
    summary = {}
    for model, mvals in results.items():
        summary[model] = {}
        for metric, values in mvals.items():
            arr = np.array(values, dtype=float)
            mu = np.nanmean(arr)
            se = np.nanstd(arr, ddof=1) / np.sqrt(np.sum(~np.isnan(arr)))
            ci = 1.96 * se
            summary[model][metric] = (mu, mu - ci, mu + ci)

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
        }
    )

    if export_excel_path is not None:
        os.makedirs(os.path.dirname(export_excel_path), exist_ok=True)
        summary_table.to_excel(export_excel_path, index=True, index_label="RowName")

    return results, summary_table


if __name__ == "__main__":
    # Load and prepare data using cox utilities
    try:
        df = cox.load_dataframes()
    except Exception as e:
        raise SystemExit(f"Failed to load dataframes: {e}")

    # Optional: apply same conversion/imputation as in cox.load_dataframes already does
    # Run experiments
    export_path = None  # set to an absolute path to export Excel
    results, summary = run_benefit_specific_experiments(
        df=df,
        N=50,
        time_col="PE_Time",
        event_col="VT/VF/SCD",
        k_splits=5,
        enforce_fair_subset=True,
        export_excel_path=export_path,
    )
    print(summary)

