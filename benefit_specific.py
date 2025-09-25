import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd

from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter  # NEW

# Reuse modeling and preprocessing utilities from ACC (your cox utils)
import ACC
from ACC import (
    fit_cox_model,
    predict_risk,
    drop_rows_with_missing_local_features,
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

# ===== 5-year configuration =====
T_STAR_YEARS = 5
T_STAR_DAYS = int(round(365.25 * T_STAR_YEARS))
EPS = 1e-9


@dataclass
class GroupModel:
    features: List[str]
    threshold: float  # threshold on 5y probability
    model: object  # CoxPHFitter or pipeline wrapping it


class BenefitClassifier:
    """Binary classifier predicting whether a sample benefits from Plus over Base (at 5y).

    - Uses ONLY Base features for training and inference
    - Internally uses L1 logistic regression with class_weight='balanced'
    - Trains on ΔLL@5y signal with margin-based exclusion
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

    def fit(self, train_df: pd.DataFrame, z_benefit: np.ndarray) -> None:
        X = train_df[self.base_features].copy()
        z = np.asarray(z_benefit, dtype=float)
        if not np.isfinite(z).any():
            self.model = None
            return

        z_clean = z[np.isfinite(z)]
        if z_clean.size == 0:
            self.model = None
            return
        std_z = float(np.nanstd(z_clean))
        margin = self.margin_std * std_z if std_z > 0 else 0.0
        keep_mask = (z > margin) | (z < -margin)
        if not np.any(keep_mask):
            self.model = None
            return
        y = (z > margin).astype(int)[keep_mask]
        X_keep = X.loc[keep_mask]

        unique_labels = np.unique(y)
        if unique_labels.size < 2:
            self.model = None
            return

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
                    try:
                        auc = (
                            roc_auc_score(y_va, p)
                            if np.unique(y_va).size > 1
                            else np.nan
                        )
                    except Exception:
                        auc = np.nan
                    aucs.append(auc)
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
        if self.model is None:
            return None
        try:
            coefs = self.model.coef_.reshape(-1)
            feats = list(self.base_features)
            if len(coefs) != len(feats):
                data = {"feature": feats, "coef": [np.nan] * len(feats)}
                df_imp = pd.DataFrame(data)
            else:
                df_imp = pd.DataFrame({"feature": feats, "coef": coefs})
            df_imp["abs_coef"] = df_imp["coef"].abs()
            with np.errstate(over="ignore"):
                df_imp["odds_ratio"] = np.exp(df_imp["coef"])
            df_imp = df_imp.sort_values("abs_coef", ascending=False).reset_index(
                drop=True
            )
            return df_imp
        except Exception:
            return None


def _choose_theta(
    p: np.ndarray, method: str = "budget", coverage: float = 0.4
) -> float:
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return 0.5
    if method == "budget":
        q = 1.0 - float(coverage)
        return float(np.quantile(p, q))
    return 0.5


def _choose_theta_by_cindex(
    p: np.ndarray,
    p5y_base: np.ndarray,
    p5y_plus: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    q_grid: Optional[np.ndarray] = None,
    min_coverage: float = 0.0,
    max_coverage: float = 1.0,
) -> Tuple[float, float, float]:
    """Select triage threshold by maximizing C-index of the mixture.

    Returns (theta, coverage, c_index).
    """
    try:
        p = np.asarray(p, float)
        p5y_base = np.asarray(p5y_base, float)
        p5y_plus = np.asarray(p5y_plus, float)
        time = np.asarray(time, float)
        event = np.asarray(event, int)
        if q_grid is None:
            q_grid = np.linspace(float(min_coverage), float(max_coverage), 101)
        best_c = -np.inf
        best_theta = float("nan")
        best_cov = float("nan")
        # Ensure quantiles are computed on finite probs
        p_finite = p[np.isfinite(p)]
        if p_finite.size == 0:
            return 0.5, 0.0, float("nan")
        for cov in q_grid:
            cov = float(np.clip(cov, 0.0, 1.0))
            # route cov fraction to plus -> threshold at (1 - cov) quantile
            q = 1.0 - cov
            theta = float(np.quantile(p_finite, q))
            route_plus = p >= theta
            mix = np.where(route_plus, p5y_plus, p5y_base)
            try:
                cidx = float(concordance_index(time, -np.asarray(mix), event))
            except Exception:
                cidx = np.nan
            if np.isfinite(cidx) and (cidx > best_c):
                best_c = cidx
                best_theta = theta
                best_cov = float(np.mean(route_plus))
        return best_theta, best_cov, best_c
    except Exception:
        return 0.5, float("nan"), float("nan")


def _make_joint_stratify_labels(
    df: pd.DataFrame, cols: List[str]
) -> Optional[pd.Series]:
    """Build a joint stratification label from multiple discrete columns."""
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


# ===== 5y survival probability & IPCW utilities =====


def _predict_surv_prob_at_t_cox(
    model, df_part: pd.DataFrame, feats: List[str], t_star_days: int
) -> np.ndarray:
    """Return 1 - S_i(t*) from a lifelines CoxPHFitter-like model/pipeline."""
    try:
        cph = (
            getattr(model, "named_steps", {}).get("cox", None)
            if hasattr(model, "named_steps")
            else None
        )
        if cph is None:
            cph = getattr(model, "cox", None) or getattr(model, "model", None) or model
        if hasattr(cph, "predict_survival_function"):
            X = df_part[feats]
            sf = cph.predict_survival_function(X, times=[t_star_days])
            s_t = np.asarray(sf.iloc[0, :], dtype=float)
            return np.clip(1.0 - s_t, 0.0, 1.0)
    except Exception:
        pass
    # Fallback: map risk to (0,1) via logistic (not preferred; only for robustness)
    try:
        from scipy.special import expit

        r = predict_risk(model, df_part, feats)
        z = (r - np.nanmean(r)) / (np.nanstd(r) + 1e-12)
        return expit(z)
    except Exception:
        return np.zeros(len(df_part), dtype=float)


def _km_ipcw_weights(
    time: np.ndarray, event: np.ndarray, t_star_days: int, clip: float = 0.05
) -> np.ndarray:
    """IPCW weights w = 1 / G(min(T, t*)) with G from KM of 'not censored'."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    censor = 1 - event  # 1=censored
    # Fit KM for "not censored" survival (event_observed = 1 - censor)
    km = KaplanMeierFitter()
    km.fit(durations=time, event_observed=1 - censor)
    eval_t = np.minimum(time, float(t_star_days))
    G = km.predict(eval_t).to_numpy(dtype=float)
    G = np.clip(G, clip, 1.0)
    return 1.0 / G


def _delta_loglik_ipcw_binary_at_t(
    p_base: np.ndarray, p_plus: np.ndarray, Z: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """Δ(LL) at t*: IPCW-weighted binary log-likelihood difference (plus - base)."""
    p_base = np.clip(np.asarray(p_base, float), EPS, 1.0 - EPS)
    p_plus = np.clip(np.asarray(p_plus, float), EPS, 1.0 - EPS)
    Z = np.asarray(Z, int)
    w = np.asarray(w, float)
    ll_base = w * (Z * np.log(p_base) + (1 - Z) * np.log(1.0 - p_base))
    ll_plus = w * (Z * np.log(p_plus) + (1 - Z) * np.log(1.0 - p_plus))
    return ll_plus - ll_base


def compute_oof_probs_5y(
    train_df: pd.DataFrame,
    base_features: List[str],
    plus_features: List[str],
    time_col: str,
    event_col: str,
    k_splits: int = 5,
    random_state: int = 0,
    enforce_fair_subset: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute out-of-fold 5y event probabilities for Base and Plus Cox models."""
    if enforce_fair_subset:
        tr_local = drop_rows_with_missing_local_features(train_df)
    else:
        tr_local = train_df.copy()

    p_base_oof = np.full(len(tr_local), np.nan, dtype=float)
    p_plus_oof = np.full(len(tr_local), np.nan, dtype=float)
    idx_array = tr_local.index.to_numpy()

    y = tr_local[event_col].values
    try:
        skf = StratifiedKFold(
            n_splits=k_splits, shuffle=True, random_state=random_state
        )
        splits = list(skf.split(np.zeros(len(tr_local)), y))
    except Exception:
        rng = np.random.RandomState(random_state)
        order = rng.permutation(len(tr_local))
        folds = np.array_split(order, k_splits)
        splits = [
            (np.setdiff1d(np.arange(len(tr_local)), f, assume_unique=False), f)
            for f in folds
        ]

    it = tqdm(splits, desc="[Benefit] OOF folds", leave=False) if _HAS_TQDM else splits
    for tr_idx_local, va_idx_local in it:
        tr_part = tr_local.iloc[tr_idx_local]
        va_part = tr_local.iloc[va_idx_local]

        # Base
        try:
            cph_b = fit_cox_model(tr_part, base_features, time_col, event_col)
            pB = _predict_surv_prob_at_t_cox(cph_b, va_part, base_features, T_STAR_DAYS)
            p_base_oof[va_idx_local] = pB
        except Exception:
            pass

        # Plus
        try:
            cph_p = fit_cox_model(tr_part, plus_features, time_col, event_col)
            pP = _predict_surv_prob_at_t_cox(cph_p, va_part, plus_features, T_STAR_DAYS)
            p_plus_oof[va_idx_local] = pP
        except Exception:
            pass

    full_base = np.full(len(train_df), np.nan, dtype=float)
    full_plus = np.full(len(train_df), np.nan, dtype=float)
    pos = train_df.index.get_indexer(idx_array)
    full_base[pos] = p_base_oof
    full_plus[pos] = p_plus_oof
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
    kept = [f for f in candidate_features if f in df_group.columns]
    if not kept:
        return None
    try:
        cph = fit_cox_model(df_group, kept, time_col, event_col)
        p5y_tr = _predict_surv_prob_at_t_cox(cph, df_group, kept, T_STAR_DAYS)
        thr = float(np.nanmedian(p5y_tr))  # threshold on 5y prob
        model_feats = list(getattr(getattr(cph, "params_", pd.Series()), "index", kept))
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
    min_group_frac: float = 0.15,
    min_group_n: int = 100,
    min_events: int = 30,
    verbose: bool = False,
    use_base_features_only_for_group_models: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Train benefit-specific models on train_df and evaluate on test_df (5y version).

    Returns (pred_label_highrisk, p5y_scores, metrics_dict) on test_df order.
    """
    if enforce_fair_subset:
        train_df = drop_rows_with_missing_local_features(train_df)
        test_df = drop_rows_with_missing_local_features(test_df)

    # 1) OOF 5y probabilities on training data
    p5y_base_oof, p5y_plus_oof = compute_oof_probs_5y(
        train_df,
        base_pool,
        plus_pool,
        time_col,
        event_col,
        k_splits=k_splits,
        random_state=random_state,
        enforce_fair_subset=False,  # already handled above
    )

    # 5y binary label & IPCW weight on training
    Z_tr = (
        (train_df[event_col].values.astype(int) == 1)
        & (train_df[time_col].values.astype(float) <= T_STAR_DAYS)
    ).astype(int)
    w_tr = _km_ipcw_weights(
        train_df[time_col].values.astype(float),
        train_df[event_col].values.astype(int),
        T_STAR_DAYS,
    )

    # 2) ΔLL@5y (plus - base), used as training signal z
    z = _delta_loglik_ipcw_binary_at_t(p5y_base_oof, p5y_plus_oof, Z_tr, w_tr)

    # 3) Train benefit classifier ONLY with Base features
    benefit_clf = BenefitClassifier(
        base_features=base_pool, margin_std=0.1, random_state=random_state
    )
    benefit_clf.fit(train_df, z)
    clf_importance = benefit_clf.get_feature_importance()

    # 4) Split train by predicted benefit, enforce minimum sizes
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
    if (
        len(df_benefit) < max(int(np.ceil(min_group_frac * n_total)), int(min_group_n))
        or _num_events(df_benefit) < int(min_events)
        or len(df_non_benefit)
        < max(int(np.ceil(min_group_frac * n_total)), int(min_group_n))
        or _num_events(df_non_benefit) < int(min_events)
    ):
        group_ok = False

    # 5) Train group models (benefit -> Plus; non-benefit -> Base)
    stability_seeds = list(range(0, min(20, n_total)))
    base_model: Optional[GroupModel] = None
    plus_model: Optional[GroupModel] = None

    if group_ok:
        plus_candidate_pool = (
            base_pool if use_base_features_only_for_group_models else plus_pool
        )
        plus_model = _train_group_model(
            df_group=df_benefit,
            candidate_features=plus_candidate_pool,
            time_col=time_col,
            event_col=event_col,
            stability_seeds=stability_seeds,
            verbose=verbose,
        )
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

    # 6) Fallback: single model if grouping invalid (compare OOF 5y AUC proxy using C-index on -p)
    single_model: Optional[GroupModel] = None
    single_model_is_plus = False
    if not group_ok:
        # crude proxy: higher C-index when ranking by -p5y (lower prob => longer survival)
        try:
            cidx_base = concordance_index(
                train_df[time_col], -np.nan_to_num(p5y_base_oof), train_df[event_col]
            )
        except Exception:
            cidx_base = np.nan
        try:
            cidx_plus = concordance_index(
                train_df[time_col], -np.nan_to_num(p5y_plus_oof), train_df[event_col]
            )
        except Exception:
            cidx_plus = np.nan
        use_plus = bool(
            np.nan_to_num(cidx_plus, nan=-np.inf)
            > np.nan_to_num(cidx_base, nan=-np.inf)
        )
        pool = (
            base_pool
            if use_base_features_only_for_group_models
            else (plus_pool if use_plus else base_pool)
        )
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

    # 7) Inference on test (5y probabilities)
    p5y_scores = np.zeros(len(test_df), dtype=float)
    pred_labels = np.zeros(len(test_df), dtype=int)

    if single_model is not None:
        feats = single_model.features
        try:
            p = _predict_surv_prob_at_t_cox(
                single_model.model, test_df, feats, T_STAR_DAYS
            )
            p5y_scores = p
            pred_labels = (p >= single_model.threshold).astype(int)
        except Exception:
            p5y_scores = np.zeros(len(test_df))
            pred_labels = np.zeros(len(test_df), dtype=int)
        try:
            cidx = concordance_index(test_df[time_col], -p5y_scores, test_df[event_col])
        except Exception:
            cidx = np.nan
        try:
            benefit_mask_pred = benefit_clf.predict_label(
                test_df, threshold=0.5
            ).astype(bool)
        except Exception:
            benefit_mask_pred = np.zeros(len(test_df), dtype=bool)
        return (
            pred_labels,
            p5y_scores,
            {
                "c_index": float(cidx),
                "grouping": 0,
                "single_is_plus": float(single_model_is_plus),
                "benefit_mask": benefit_mask_pred,
                "benefit_importance": clf_importance,
            },
        )

    # Grouped inference (benefit -> Plus; non-benefit -> Base)
    test_pred = benefit_clf.predict_label(test_df, threshold=0.5)
    mask_benefit = test_pred == 1
    mask_non_benefit = ~mask_benefit

    if base_model is not None and np.any(mask_non_benefit):
        te_nb = test_df.loc[mask_non_benefit]
        try:
            p_nb = _predict_surv_prob_at_t_cox(
                base_model.model, te_nb, base_model.features, T_STAR_DAYS
            )
            p5y_scores[mask_non_benefit] = p_nb
            pred_labels[mask_non_benefit] = (p_nb >= base_model.threshold).astype(int)
        except Exception:
            pass

    if plus_model is not None and np.any(mask_benefit):
        te_b = test_df.loc[mask_benefit]
        try:
            p_b = _predict_surv_prob_at_t_cox(
                plus_model.model, te_b, plus_model.features, T_STAR_DAYS
            )
            p5y_scores[mask_benefit] = p_b
            pred_labels[mask_benefit] = (p_b >= plus_model.threshold).astype(int)
        except Exception:
            pass

    try:
        cidx_all = concordance_index(test_df[time_col], -p5y_scores, test_df[event_col])
    except Exception:
        cidx_all = np.nan

    return (
        pred_labels,
        p5y_scores,
        {
            "c_index": float(cidx_all),
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
    """Train Base and Plus Cox models on train_df and apply to test_df (5y prob)."""
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

    p5y_base_test = np.zeros(len(test_df), dtype=float)
    p5y_plus_test = np.zeros(len(test_df), dtype=float)
    choose_plus_mask = np.zeros(len(test_df), dtype=bool)

    try:
        if base_model is not None:
            p5y_base_test = _predict_surv_prob_at_t_cox(
                base_model.model, test_df, base_model.features, T_STAR_DAYS
            )
    except Exception:
        p5y_base_test = np.zeros(len(test_df), dtype=float)
    try:
        if plus_model is not None:
            p5y_plus_test = _predict_surv_prob_at_t_cox(
                plus_model.model, test_df, plus_model.features, T_STAR_DAYS
            )
    except Exception:
        p5y_plus_test = np.zeros(len(test_df), dtype=float)

    # For preview/observation only: choose plus when its 5y prob is lower
    try:
        diff = np.asarray(p5y_plus_test) - np.asarray(p5y_base_test)
        finite_idx = np.isfinite(diff)
        choose_plus_mask = np.zeros(len(test_df), dtype=bool)
        choose_plus_mask[finite_idx] = diff[finite_idx] < 0.0
    except Exception:
        choose_plus_mask = np.zeros(len(test_df), dtype=bool)

    return base_model, plus_model, p5y_base_test, p5y_plus_test, choose_plus_mask


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


def run_stabilized_two_model_pipeline(
    df: pd.DataFrame,
    N: int = 50,
    time_col: str = "PE_Time",
    event_col: str = "VT/VF/SCD",
    k_splits: int = 5,
    enforce_fair_subset: bool = True,
    tableone_excel_path: Optional[str] = None,
    clf_importance_excel_path: Optional[str] = None,
    train_benefit_classifier: bool = False,
) -> Dict[str, object]:
    """Run N random splits to stabilize Base and Plus models, then finalize and evaluate (5y)."""

    base_pool = list(FEATURE_SETS.get("Proposed", []))
    plus_pool = list(FEATURE_SETS.get("Proposed Plus", []))

    # Exclude certain features globally if needed
    exclude_features = {"Age by decade", "CrCl>45", "NYHA>2", "Significant LGE"}
    base_pool = [f for f in base_pool if f not in exclude_features]
    plus_pool = [f for f in plus_pool if f not in exclude_features]

    # Print pools (optional)
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

    iterator = (
        tqdm(range(N), desc="[Stabilize] Splits", leave=True) if _HAS_TQDM else range(N)
    )
    for seed in iterator:
        try:
            data_use = df_use.dropna(subset=[time_col, event_col]).copy()
            strat_labels = _make_joint_stratify_labels(
                data_use, ["ICD", "Female", event_col]
            )
            if strat_labels is not None and strat_labels.nunique() > 1:
                stratify_arg = strat_labels
            else:
                stratify_arg = (
                    data_use[event_col] if data_use[event_col].nunique() > 1 else None
                )
            tr, te = train_test_split(
                data_use,
                test_size=0.3,
                random_state=seed,
                stratify=stratify_arg,
            )
        except Exception:
            continue

        base_model, plus_model, _, _, _ = _train_two_models_on_split(
            tr, te, time_col, event_col, base_pool, plus_pool, verbose=False
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

    try:
        print("==== Final Selected Features (Stabilized Pipeline) ====")
        print(f"Final Base features ({len(base_major)}): {base_major}")
        print(f"Final Plus features ({len(plus_major)}): {plus_major}")
    except Exception:
        pass

    # === Final models on full dataset ===
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

    # 5y probabilities on all data
    p5y_base_all = np.zeros(len(df_use), dtype=float)
    p5y_plus_all = np.zeros(len(df_use), dtype=float)
    try:
        if final_base is not None:
            p5y_base_all = _predict_surv_prob_at_t_cox(
                final_base.model, df_use, final_base.features, T_STAR_DAYS
            )
    except Exception:
        pass
    try:
        if final_plus is not None:
            p5y_plus_all = _predict_surv_prob_at_t_cox(
                final_plus.model, df_use, final_plus.features, T_STAR_DAYS
            )
    except Exception:
        pass

    # ΔLL@5y and benefit mask (tau can be tuned by CV; use 0 as conservative default)
    Z_all = (
        (df_use[event_col].values.astype(int) == 1)
        & (df_use[time_col].values.astype(float) <= T_STAR_DAYS)
    ).astype(int)
    w_all = _km_ipcw_weights(
        df_use[time_col].values.astype(float),
        df_use[event_col].values.astype(int),
        T_STAR_DAYS,
    )
    dLL_all = _delta_loglik_ipcw_binary_at_t(p5y_base_all, p5y_plus_all, Z_all, w_all)
    tau = 0.0
    benefit_mask = dLL_all > tau

    # Three scenarios on 5y prob
    p5y_best_all = np.where(benefit_mask, p5y_plus_all, p5y_base_all)

    def _cidx_safe_from_prob(p: np.ndarray) -> float:
        try:
            return float(
                concordance_index(df_use[time_col], -np.asarray(p), df_use[event_col])
            )
        except Exception:
            return np.nan

    c_index_triage = np.nan
    coverage_triage = np.nan
    theta_used = np.nan
    triage_probs = None

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score

        X_tri = df_use[[f for f in base_pool if f in df_use.columns]].copy()
        y_tri = benefit_mask.astype(int)

        if X_tri.shape[1] > 0 and len(np.unique(y_tri)) == 2:
            lr_tri = LogisticRegression(
                penalty="l1",
                C=0.5,
                solver="liblinear",
                class_weight="balanced",
                random_state=0,
                max_iter=300,
            ).fit(X_tri, y_tri)

            triage_probs = lr_tri.predict_proba(X_tri)[:, 1]
            # Optimize threshold by maximizing mixture C-index over coverage grid
            theta_opt, cov_opt, cidx_opt = _choose_theta_by_cindex(
                p=triage_probs,
                p5y_base=p5y_base_all,
                p5y_plus=p5y_plus_all,
                time=df_use[time_col].values,
                event=df_use[event_col].values,
                q_grid=np.linspace(0.0, 1.0, 101),
            )
            # Fallback to budget rule if optimization failed
            if not np.isfinite(theta_opt):
                theta_opt = _choose_theta(triage_probs, method="budget", coverage=0.4)
                route_plus = triage_probs >= theta_opt
                p5y_triage_all = np.where(route_plus, p5y_plus_all, p5y_base_all)
                try:
                    cidx_opt = float(
                        concordance_index(
                            df_use[time_col], -np.asarray(p5y_triage_all), df_use[event_col]
                        )
                    )
                except Exception:
                    cidx_opt = np.nan
                cov_opt = float(np.mean(route_plus))
            else:
                route_plus = triage_probs >= theta_opt
                p5y_triage_all = np.where(route_plus, p5y_plus_all, p5y_base_all)
            # Ensure triage is not worse than the better single model at extremes
            try:
                cidx_base_only = float(
                    concordance_index(
                        df_use[time_col].values, -np.asarray(p5y_base_all), df_use[event_col].values
                    )
                )
            except Exception:
                cidx_base_only = np.nan
            try:
                cidx_plus_only = float(
                    concordance_index(
                        df_use[time_col].values, -np.asarray(p5y_plus_all), df_use[event_col].values
                    )
                )
            except Exception:
                cidx_plus_only = np.nan

            theta_used = float(theta_opt)
            coverage_triage = float(cov_opt)
            c_index_triage = float(cidx_opt)

            best_single = np.nanmax([cidx_base_only, cidx_plus_only])
            if not np.isfinite(c_index_triage) or (np.isfinite(best_single) and c_index_triage < best_single - 1e-12):
                # Override with the better single model routing
                if np.nan_to_num(cidx_plus_only, nan=-np.inf) >= np.nan_to_num(cidx_base_only, nan=-np.inf):
                    route_plus = np.ones(len(df_use), dtype=bool)
                    p5y_triage_all = p5y_plus_all
                    coverage_triage = 1.0
                    c_index_triage = float(cidx_plus_only)
                    theta_used = float(np.nanmin(triage_probs) - 1e-9)
                else:
                    route_plus = np.zeros(len(df_use), dtype=bool)
                    p5y_triage_all = p5y_base_all
                    coverage_triage = 0.0
                    c_index_triage = float(cidx_base_only)
                    theta_used = float(np.nanmax(triage_probs) + 1e-9)
    except Exception:
        pass

    metrics = {
        "c_index_all_base": _cidx_safe_from_prob(p5y_base_all),
        "c_index_all_plus": _cidx_safe_from_prob(p5y_plus_all),
        "c_index_per_sample_best": _cidx_safe_from_prob(p5y_best_all),
        "c_index_triage": c_index_triage,
        "coverage_triage": coverage_triage,
        "theta_triage": theta_used,
    }

    # TableOne on a held-out test split for observational comparison
    try:
        data_tab = df_use.dropna(subset=[time_col, event_col]).copy()
        strat_labels = _make_joint_stratify_labels(
            data_tab, ["ICD", "Female", event_col]
        )
        stratify_arg = (
            strat_labels
            if (strat_labels is not None and strat_labels.nunique() > 1)
            else (data_tab[event_col] if data_tab[event_col].nunique() > 1 else None)
        )
        tr_tab, te_tab = train_test_split(
            data_tab, test_size=0.3, random_state=0, stratify=stratify_arg
        )

        tab_base_model = _train_group_model(
            tr_tab,
            base_major,
            time_col,
            event_col,
            stability_seeds=list(range(10)),
            verbose=False,
        )
        tab_plus_model = _train_group_model(
            tr_tab,
            plus_major,
            time_col,
            event_col,
            stability_seeds=list(range(10)),
            verbose=False,
        )

        p_base_tab = np.zeros(len(te_tab), dtype=float)
        p_plus_tab = np.zeros(len(te_tab), dtype=float)
        if tab_base_model is not None:
            p_base_tab = _predict_surv_prob_at_t_cox(
                tab_base_model.model, te_tab, tab_base_model.features, T_STAR_DAYS
            )
        if tab_plus_model is not None:
            p_plus_tab = _predict_surv_prob_at_t_cox(
                tab_plus_model.model, te_tab, tab_plus_model.features, T_STAR_DAYS
            )

        choose_plus_tab = p_plus_tab < p_base_tab
        df_tab = te_tab.copy()
        df_tab["BenefitGroup"] = np.where(choose_plus_tab, "Benefit", "Non-Benefit")
        try:
            ACC.generate_tableone_by_group(
                df_tab, group_col="BenefitGroup", output_excel_path=tableone_excel_path
            )
        except Exception:
            pass
    except Exception:
        pass

    # Optional: benefit classifier trained on entire df_use (base features only)
    clf_importance: Optional[pd.DataFrame] = None
    if train_benefit_classifier:
        try:
            X = df_use[[f for f in base_pool if f in df_use.columns]].copy()
            y = benefit_mask.astype(int)
            if X.shape[1] > 0 and len(np.unique(y)) == 2:
                best_auc = -np.inf
                best_lr: Optional[LogisticRegression] = None
                Cs = [0.1, 0.5, 1.0]
                skf = StratifiedKFold(
                    n_splits=(
                        min(5, int(np.bincount(y).min()))
                        if np.bincount(y).size > 1
                        else 3
                    ),
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
                            aucs.append(
                                roc_auc_score(y_va, p)
                                if np.unique(y_va).size > 1
                                else np.nan
                            )
                        finite = np.asarray(aucs, float)
                        finite = finite[np.isfinite(finite)]
                        mean_auc = (
                            float(np.mean(finite)) if finite.size > 0 else -np.inf
                        )
                        if mean_auc > best_auc:
                            best_auc, best_lr = mean_auc, lr
                    except Exception:
                        continue
                if best_lr is not None:
                    best_lr.fit(X, y)
                    try:
                        coefs = best_lr.coef_.reshape(-1)
                        feats = list(X.columns)
                        df_imp = pd.DataFrame({"feature": feats, "coef": coefs})
                        df_imp["abs_coef"] = df_imp["coef"].abs()
                        with np.errstate(over="ignore"):
                            df_imp["odds_ratio"] = np.exp(df_imp["coef"])
                        clf_importance = df_imp.sort_values(
                            "abs_coef", ascending=False
                        ).reset_index(drop=True)
                        if clf_importance_excel_path is not None:
                            os.makedirs(
                                os.path.dirname(clf_importance_excel_path),
                                exist_ok=True,
                            )
                            clf_importance.to_excel(
                                clf_importance_excel_path, index=False
                            )
                    except Exception:
                        clf_importance = None
        except Exception:
            clf_importance = None

    return {
        "final_base_features": base_major,
        "final_plus_features": plus_major,
        "p5y_base_all": p5y_base_all,
        "p5y_plus_all": p5y_plus_all,
        "delta_ll_5y": dLL_all,
        "tau": tau,
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
    """Outer-loop evaluation for benefit-specific approach only (no sex-specific baseline)."""

    base_pool = list(FEATURE_SETS.get("Proposed", []))
    plus_pool = list(FEATURE_SETS.get("Proposed Plus", []))

    model_configs = [
        {"name": "Proposed Plus (benefit-specific)", "mode": "benefit_specific"},
        {
            "name": "Proposed Plus (benefit-specific, base-only)",
            "mode": "benefit_specific_base_only",
        },
    ]
    metrics = ["c_index_all"]
    results: Dict[str, Dict[str, List[float]]] = {
        cfg["name"]: {m: [] for m in metrics} for cfg in model_configs
    }

    iterator = (
        tqdm(range(N), desc="[Benefit] Splits", leave=True) if _HAS_TQDM else range(N)
    )
    for seed in iterator:
        data_use = df.dropna(subset=[time_col, event_col]).copy()
        strat_labels = _make_joint_stratify_labels(
            data_use, ["ICD", "Female", event_col]
        )
        stratify_arg = (
            strat_labels
            if (strat_labels is not None and strat_labels.nunique() > 1)
            else (data_use[event_col] if data_use[event_col].nunique() > 1 else None)
        )
        tr, te = train_test_split(
            data_use,
            test_size=0.3,
            random_state=seed,
            stratify=stratify_arg,
        )

        # Benefit-specific (full Plus features)
        pred_b, p5y_b, met_b = evaluate_benefit_specific_split(
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
        results["Proposed Plus (benefit-specific)"]["c_index_all"].append(
            met_b.get("c_index", np.nan)
        )

        # Benefit-specific (group models restricted to Base features)
        pred_bb, p5y_bb, met_bb = evaluate_benefit_specific_split(
            tr,
            te,
            time_col,
            event_col,
            base_pool,
            plus_pool,
            random_state=seed,
            k_splits=k_splits,
            enforce_fair_subset=enforce_fair_subset,
            use_base_features_only_for_group_models=True,
        )
        results["Proposed Plus (benefit-specific, base-only)"]["c_index_all"].append(
            met_bb.get("c_index", np.nan)
        )

        # Optional TableOne preview on first split
        if print_first_split_preview and seed == 0:
            try:
                te_eval_b = (
                    ACC.drop_rows_with_missing_local_features(te)
                    if enforce_fair_subset
                    else te.copy()
                )
                benefit_mask = met_b.get(
                    "benefit_mask", np.zeros(len(te_eval_b), dtype=bool)
                )
                if len(benefit_mask) != len(te_eval_b):
                    min_len = min(len(benefit_mask), len(te_eval_b))
                    benefit_mask = np.asarray(benefit_mask).astype(bool)[:min_len]
                    te_b_tab = te_eval_b.iloc[:min_len].copy()
                else:
                    te_b_tab = te_eval_b.copy()
                te_b_tab["BenefitGroup"] = np.where(
                    benefit_mask, "Benefit", "Non-Benefit"
                )
                try:
                    ACC.generate_tableone_by_group(
                        te_b_tab, group_col="BenefitGroup", output_excel_path=None
                    )
                except Exception:
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
    summary_table = summary_table.rename(columns={"c_index_all": "all"})

    if export_excel_path is not None:
        os.makedirs(os.path.dirname(export_excel_path), exist_ok=True)
        summary_table.to_excel(export_excel_path, index=True, index_label="RowName")

    return results, summary_table


if __name__ == "__main__":
    # Load and prepare data using ACC utilities
    try:
        df = ACC.load_dataframes()
    except Exception as e:
        raise SystemExit(f"Failed to load dataframes: {e}")

    # (Optional) Outer-loop experiments for benefit-specific only
    try:
        export_path = None  # set a path if you want Excel export
        results, summary = run_benefit_specific_experiments(
            df=df,
            N=10,
            time_col="PE_Time",
            event_col="VT/VF/SCD",
            k_splits=5,
            enforce_fair_subset=True,
            export_excel_path=export_path,
            print_first_split_preview=False,
        )
        print(summary)
    except Exception:
        pass

    # Stabilized two-model pipeline with 5y ΔLL-based benefit + TableOne export
    try:
        tableone_path = (
            "/home/sunx/data/aiiih/projects/sunx/projects/ICD/benefit_tableone.xlsx"
        )
        clf_imp_path = "/home/sunx/data/aiiih/projects/sunx/projects/ICD/benefit_classifier_importance.xlsx"
        stabilized = run_stabilized_two_model_pipeline(
            df=df,
            N=10,
            time_col="PE_Time",
            event_col="VT/VF/SCD",
            k_splits=5,
            enforce_fair_subset=True,
            tableone_excel_path=tableone_path,
            clf_importance_excel_path=clf_imp_path,
            train_benefit_classifier=True,  # set False if you don't need it
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
