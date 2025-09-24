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
    predict_absolute_risk_at_time,
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
                mean_auc = (
                    float(np.mean(finite_aucs)) if finite_aucs.size > 0 else -np.inf
                )
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
                df_imp["odds_ratio"] = np.exp(
                    df_imp["coef"]
                )  # may overflow -> inf acceptable
            df_imp = df_imp.sort_values("abs_coef", ascending=False).reset_index(
                drop=True
            )
            return df_imp
        except Exception:
            return None


def _standardize_by_train(risk_tr: np.ndarray, risk_va: np.ndarray) -> np.ndarray:
    mu = float(np.nanmean(risk_tr))
    sd = float(np.nanstd(risk_tr, ddof=1))
    if not np.isfinite(sd) or sd == 0.0:
        return risk_va
    return (risk_va - mu) / sd


def _make_joint_stratify_labels(
    df: pd.DataFrame, cols: List[str]
) -> Optional[pd.Series]:
    """Build a joint stratification label from multiple discrete columns.

    Returns a Series of string labels like "ICD|Female|Event" or None if any column missing.
    Robust to numeric/boolean/categorical types; missing values are labeled as "NA".
    """
    try:
        for c in cols:
            if c not in df.columns:
                return None
        parts: List[pd.Series] = []
        for c in cols:
            s = df[c]
            try:
                s_num = pd.to_numeric(s, errors="coerce")
                s_int = s_num.round().astype("Int64")
                parts.append(s_int.astype(str).fillna("NA"))
            except Exception:
                try:
                    s_cat = s.astype("category").cat.codes.astype("Int64")
                    parts.append(s_cat.astype(str).fillna("NA"))
                except Exception:
                    parts.append(s.astype("string").fillna("NA"))
        label = parts[0]
        for p in parts[1:]:
            label = label.str.cat(p, sep="|")
        return label
    except Exception:
        return None


def compute_oof_risks(
    train_df: pd.DataFrame,
    base_features: List[str],
    plus_features: List[str],
    time_col: str,
    event_col: str,
    k_splits: int = 5,
    random_state: int = 0,
    enforce_fair_subset: bool = True,
    horizon_days: float = 1825.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute out-of-fold 5-year absolute risks for Base and Plus Cox models.

    - Uses Cox baseline cumulative hazard to convert to absolute risk at horizon_days
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
        tqdm(splits, desc="[Benefit] OOF folds", leave=False) if _HAS_TQDM else splits
    )
    for tr_idx_local, va_idx_local in iter_splits:
        tr_part = tr_local.iloc[tr_idx_local]
        va_part = tr_local.iloc[va_idx_local]

        # Base model -> absolute risk at horizon
        try:
            cph_b = fit_cox_model(tr_part, base_features, time_col, event_col)
            risk_va_b = predict_absolute_risk_at_time(
                cph_b, va_part, base_features, horizon_days=horizon_days
            )
            risk_base_oof[va_idx_local] = risk_va_b
        except Exception:
            pass

        # Plus model -> absolute risk at horizon
        try:
            cph_p = fit_cox_model(tr_part, plus_features, time_col, event_col)
            risk_va_p = predict_absolute_risk_at_time(
                cph_p, va_part, plus_features, horizon_days=horizon_days
            )
            risk_plus_oof[va_idx_local] = risk_va_p
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
        return GroupModel(
            features=(model_feats if model_feats else list(kept)),
            threshold=thr,
            model=cph,
        )
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
    horizon_days: float = 1825.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Unified two-model pipeline with 5-year absolute risk and triage classifier.

    - Trains Base and Plus Cox models on the train split
    - Computes OOF absolute risks on train to derive benefit labels (sign only)
    - Trains a Base-features-only logistic triage classifier (no subgrouping)
    - On test, outputs triaged risk (Plus when classifier>=0.5 else Base) and metrics
    """
    # Optional fairness: require complete local features throughout
    if enforce_fair_subset:
        train_df = drop_rows_with_missing_local_features(train_df)
        test_df = drop_rows_with_missing_local_features(test_df)

    # 1) OOF absolute risks on training data to construct benefit labels
    p_base_oof, p_plus_oof = compute_oof_risks(
        train_df,
        base_pool,
        plus_pool,
        time_col,
        event_col,
        k_splits=k_splits,
        random_state=random_state,
        enforce_fair_subset=False,
        horizon_days=horizon_days,
    )
    y_tr = train_df[event_col].values.astype(int)
    y_signed = (2 * y_tr - 1).astype(float)
    z = (p_plus_oof - p_base_oof) * y_signed

    # 2) Train triage classifier using ONLY Base features; no thresholding/margin
    benefit_clf = BenefitClassifier(
        base_features=base_pool, margin_std=0.0, random_state=random_state
    )
    benefit_clf.fit(train_df, z)
    clf_importance = benefit_clf.get_feature_importance()

    # 3) Train final Base and Plus Cox models on full train
    cph_base = fit_cox_model(train_df, base_pool, time_col, event_col)
    cph_plus = fit_cox_model(train_df, plus_pool, time_col, event_col)

    # 4) Predict absolute risks on test for both models
    p_base_test = predict_absolute_risk_at_time(
        cph_base, test_df, base_pool, horizon_days=horizon_days
    )
    p_plus_test = predict_absolute_risk_at_time(
        cph_plus, test_df, plus_pool, horizon_days=horizon_days
    )

    # 5) Triage decision on test
    triage_label = benefit_clf.predict_label(test_df, threshold=0.5)
    p_triage = np.where(triage_label == 1, p_plus_test, p_base_test)

    # 6) Metrics
    try:
        cidx_base = concordance_index(test_df[time_col], -p_base_test, test_df[event_col])
    except Exception:
        cidx_base = np.nan
    try:
        cidx_plus = concordance_index(test_df[time_col], -p_plus_test, test_df[event_col])
    except Exception:
        cidx_plus = np.nan
    try:
        cidx_triage = concordance_index(
            test_df[time_col], -p_triage, test_df[event_col]
        )
    except Exception:
        cidx_triage = np.nan

    # Benefit ground-truth on test for classifier evaluation (sign only, no threshold)
    try:
        z_test = (p_plus_test - p_base_test) * (2 * test_df[event_col].values.astype(int) - 1)
        y_ben_test = (z_test > 0).astype(int)
        try:
            auc = roc_auc_score(y_ben_test, benefit_clf.predict_proba(test_df))
        except Exception:
            auc = np.nan
    except Exception:
        auc = np.nan

    return (
        triage_label,
        p_triage,
        {
            "c_index_base": cidx_base,
            "c_index_plus": cidx_plus,
            "c_index_triage": cidx_triage,
            "benefit_auc": auc,
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
            risk_base_test = predict_risk(
                base_model.model, test_df, base_model.features
            )
    except Exception:
        risk_base_test = np.zeros(len(test_df), dtype=float)
    try:
        if plus_model is not None:
            risk_plus_test = predict_risk(
                plus_model.model, test_df, plus_model.features
            )
    except Exception:
        risk_plus_test = np.zeros(len(test_df), dtype=float)

    # Choose better model per sample based on true label:
    # If event==1 -> higher risk is better; if event==0 -> lower risk is better.
    try:
        diff = np.asarray(risk_plus_test) - np.asarray(risk_base_test)
        events = np.asarray(test_df[event_col]).astype(int)
        choose_plus_mask = np.zeros(len(test_df), dtype=bool)
        finite_idx = np.isfinite(diff) & np.isfinite(events.astype(float))
        # For events==1, choose plus when diff>0; for events==0, choose plus when diff<0
        choose_plus_mask[finite_idx] = np.where(
            events[finite_idx] == 1, diff[finite_idx] > 0.0, diff[finite_idx] < 0.0
        )
    except Exception:
        choose_plus_mask = np.zeros(len(test_df), dtype=bool)

    return base_model, plus_model, risk_base_test, risk_plus_test, choose_plus_mask


def _majority_features(
    feature_lists: List[List[str]], min_frac: float = 0.5
) -> List[str]:
    if not feature_lists:
        return []
    total = len(feature_lists)
    counter: Counter = Counter()
    for feats in feature_lists:
        counter.update(set(f for f in feats))
    keep = [f for f, c in counter.items() if c / total >= float(min_frac)]
    return sorted(keep)


def run_unified_holdout(
    df: pd.DataFrame,
    time_col: str = "PE_Time",
    event_col: str = "VT/VF/SCD",
    horizon_days: float = 1825.0,
    enforce_fair_subset: bool = True,
    random_state: int = 0,
) -> Dict[str, object]:
    """Single holdout evaluation following the unified plan.

    - Models: Base (Proposed) and Plus (Proposed Plus)
    - Predictions: 5-year absolute risk via Cox baseline cumulative hazard
    - Benefit: sign of Δp only (no threshold)
    - Triage: Base-features logistic classifier; no subgroup analysis
    """
    base_pool = list(FEATURE_SETS.get("Proposed", []))
    plus_pool = list(FEATURE_SETS.get("Proposed Plus", []))

    data_use = df.dropna(subset=[time_col, event_col]).copy()
    # Stratify by joint labels if available
    strat_labels = _make_joint_stratify_labels(data_use, ["ICD", "Female", event_col])
    stratify_arg = (
        strat_labels
        if (strat_labels is not None and strat_labels.nunique() > 1)
        else (data_use[event_col] if data_use[event_col].nunique() > 1 else None)
    )
    tr, te = train_test_split(
        data_use,
        test_size=0.3,
        random_state=random_state,
        stratify=stratify_arg,
    )

    pred, risk, met = evaluate_benefit_specific_split(
        tr,
        te,
        time_col,
        event_col,
        base_pool,
        plus_pool,
        random_state=random_state,
        k_splits=5,
        enforce_fair_subset=enforce_fair_subset,
        horizon_days=horizon_days,
    )

    out = {
        "triage_pred": pred,
        "triage_risk": risk,
        "metrics": met,
        "base_features": base_pool,
        "plus_features": plus_pool,
    }
    return out


def run_benefit_specific_experiments(
    df: pd.DataFrame,
    N: int = 1,
    time_col: str = "PE_Time",
    event_col: str = "VT/VF/SCD",
    k_splits: int = 5,
    enforce_fair_subset: bool = True,
    export_excel_path: Optional[str] = None,
    print_first_split_preview: bool = False,
) -> Tuple[Dict[str, Dict[str, List[float]]], pd.DataFrame]:
    """Kept for API compatibility: run 1 unified holdout and summarize metrics.

    Returns (results_dict, summary_table) with columns: base, plus, triage, auc.
    """
    iterator = (
        tqdm(range(N), desc="[Unified] Splits", leave=True) if _HAS_TQDM else range(N)
    )
    results: Dict[str, Dict[str, List[float]]] = {
        "Unified": {m: [] for m in ["c_index_base", "c_index_plus", "c_index_triage", "benefit_auc"]}
    }
    for seed in iterator:
        out = run_unified_holdout(
            df,
            time_col=time_col,
            event_col=event_col,
            horizon_days=1825.0,
            enforce_fair_subset=enforce_fair_subset,
            random_state=seed,
        )
        met = out.get("metrics", {})
        for k in results["Unified"].keys():
            results["Unified"][k].append(met.get(k, np.nan))

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
            "c_index_base": "base",
            "c_index_plus": "plus",
            "c_index_triage": "triage",
            "benefit_auc": "benefit_auc",
        }
    )

    if export_excel_path is not None:
        os.makedirs(os.path.dirname(export_excel_path), exist_ok=True)
        summary_table.to_excel(export_excel_path, index=True, index_label="RowName")

    return results, summary_table


if __name__ == "__main__":
    # Load and prepare data
    try:
        df = ACC.load_dataframes()
    except Exception as e:
        raise SystemExit(f"Failed to load dataframes: {e}")

    # Single unified run
    out = run_unified_holdout(
        df,
        time_col="PE_Time",
        event_col="VT/VF/SCD",
        horizon_days=1825.0,
        enforce_fair_subset=True,
        random_state=0,
    )
    print("==== Unified holdout metrics ====")
    print(out.get("metrics", {}))
    # Print top-20 feature importance for the triage classifier if available
    imp = out.get("metrics", {}).get("benefit_importance", None)
    if isinstance(imp, pd.DataFrame):
        try:
            print(imp.head(20))
        except Exception:
            pass
