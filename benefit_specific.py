import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter

import matplotlib.pyplot as plt
import ACC
from ACC import (
    fit_cox_model,
    predict_risk,
    drop_rows_with_missing_local_features,
    FEATURE_SETS,
    stability_select_features,
    select_features_max_cindex_forward,
    plot_cox_coefficients,
)

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression

try:
    from tqdm import tqdm  # type: ignore

    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

try:
    from xgboost import XGBClassifier

    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

T_STAR_YEARS = 5
T_STAR_DAYS = int(round(365.25 * T_STAR_YEARS))
EPS = 1e-9


@dataclass
class GroupModel:
    features: List[str]
    threshold: float
    model: object


def _choose_theta(
    p: np.ndarray,
    method: str = "budget",
    coverage: float = 0.4,
    y_benefit: Optional[np.ndarray] = None,
    harm_ratio: float = 1.0,
) -> float:
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return 0.5
    if method == "budget":
        q = 1.0 - float(coverage)
        return float(np.quantile(p, q))
    if y_benefit is None or len(y_benefit) != len(p):
        return 0.5
    thetas = np.quantile(p, np.linspace(0.2, 0.8, 13))
    best_nb, best_theta = -1e9, 0.5
    for th in thetas:
        pred = (p >= th).astype(int)
        TP = int(((pred == 1) & (y_benefit == 1)).sum())
        FP = int(((pred == 1) & (y_benefit == 0)).sum())
        N = len(y_benefit)
        nb = (TP / N) - harm_ratio * (FP / N)
        if nb > best_nb:
            best_nb, best_theta = nb, float(th)
    return best_theta


def _make_joint_stratify_labels(
    df: pd.DataFrame, cols: List[str]
) -> Optional[pd.Series]:
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


def _predict_surv_prob_at_t_cox(
    model, df_part: pd.DataFrame, feats: List[str], t_star_days: int
) -> np.ndarray:
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
    try:
        from scipy.special import expit

        r = predict_risk(model, df_part, feats)
        z = (r - np.nanmean(r)) / (np.nanstd(r) + 1e-12)
        return expit(z)
    except Exception:
        return np.zeros(len(df_part), dtype=float)


def _km_ipcw_weights(time, event, t_star_days, clip: float = 0.1) -> np.ndarray:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    censor = 1 - event
    km = KaplanMeierFitter()
    km.fit(durations=time, event_observed=censor)
    eval_t = np.minimum(time, float(t_star_days))
    G = km.predict(eval_t).to_numpy(dtype=float)
    G = np.clip(G, clip, 1.0)
    return 1.0 / G


def _winsorize(x: np.ndarray, low_q: float = 0.01, high_q: float = 0.99) -> np.ndarray:
    x = np.asarray(x, float)
    lo, hi = np.nanquantile(x, low_q), np.nanquantile(x, high_q)
    return np.clip(x, lo, hi)


def _delta_loglik_ipcw_binary_at_t(
    p_base: np.ndarray, p_plus: np.ndarray, Z: np.ndarray, w: np.ndarray
) -> np.ndarray:
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
        try:
            cph_b = fit_cox_model(tr_part, base_features, time_col, event_col)
            pB = _predict_surv_prob_at_t_cox(cph_b, va_part, base_features, T_STAR_DAYS)
            p_base_oof[va_idx_local] = pB
        except Exception:
            pass
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


def make_benefit_labels_crossfit(
    train_df: pd.DataFrame,
    base_features: List[str],
    plus_features: List[str],
    time_col: str,
    event_col: str,
    k_splits: int = 5,
    random_state: int = 0,
    tau_rule: str = "q65",
    margin_std: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, float]:
    pB_oof, pP_oof = compute_oof_probs_5y(
        train_df,
        base_features,
        plus_features,
        time_col=time_col,
        event_col=event_col,
        k_splits=k_splits,
        random_state=random_state,
        enforce_fair_subset=False,
    )
    Z = (
        (train_df[event_col].values.astype(int) == 1)
        & (train_df[time_col].values.astype(float) <= T_STAR_DAYS)
    ).astype(int)
    w = _km_ipcw_weights(
        train_df[time_col].values.astype(float),
        train_df[event_col].values.astype(int),
        T_STAR_DAYS,
    )
    z_raw = _delta_loglik_ipcw_binary_at_t(pB_oof, pP_oof, Z, w)
    z = _winsorize(z_raw, 0.01, 0.99)
    z_clean = z[np.isfinite(z)]
    if z_clean.size == 0:
        return z, np.zeros(len(train_df), int), 0.0
    if tau_rule == "zero":
        tau = 0.0
    elif tau_rule == "q60":
        tau = float(np.quantile(z_clean, 0.60))
    elif tau_rule == "q70":
        tau = float(np.quantile(z_clean, 0.70))
    else:
        tau = float(np.quantile(z_clean, 0.65))
    std_z = float(np.nanstd(z_clean))
    margin = margin_std * std_z if std_z > 0 else 0.0
    keep = (z > margin) | (z < -margin)
    B = np.full(len(z), -1, dtype=int)
    B[keep] = (z[keep] > tau).astype(int)
    return z, B, tau


def fit_calibrated_triage(
    train_df: pd.DataFrame,
    base_features: list,
    B: np.ndarray,
    random_state: int = 0,
    calibrate_cv: int = 5,
    backend: str = "sgd",
    calibration: str = "sigmoid",
    monotone_constraints: Optional[Dict[str, int]] = None,
):
    y = np.asarray(B, int)
    mask = (y == 0) | (y == 1)
    X = train_df.loc[mask, base_features].copy()
    y = y[mask]
    if X.shape[1] == 0 or len(np.unique(y)) < 2:
        return None

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    penalty="l2",
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipe.fit(X, y)

    clf = CalibratedClassifierCV(pipe, method=calibration, cv=calibrate_cv)
    clf.fit(X, y)
    return clf


def _train_group_model(
    df_group: pd.DataFrame,
    candidate_features: List[str],
    time_col: str,
    event_col: str,
    stability_seeds: List[int],
    verbose: bool = False,
    do_feature_selection: bool = True,
    fs_max_features: Optional[int] = None,
    stability_threshold: float = 0.4,
    plot_importance: bool = False,
) -> Optional[GroupModel]:
    if df_group is None or df_group.empty:
        return None
    kept = [f for f in candidate_features if f in df_group.columns]
    if not kept:
        return None
    try:
        feats_for_fit = list(kept)
        if do_feature_selection:
            try:
                rs = int(stability_seeds[0]) if stability_seeds else 0
                selected = select_features_max_cindex_forward(
                    train_df=df_group.dropna(subset=[time_col, event_col]).copy(),
                    candidate_features=feats_for_fit,
                    time_col=time_col,
                    event_col=event_col,
                    random_state=rs,
                    max_features=fs_max_features,
                    verbose=verbose,
                )
                if selected:
                    feats_for_fit = list(selected)
                    if verbose:
                        try:
                            print(f"[FS] Selected {len(feats_for_fit)} features: {feats_for_fit}")
                        except Exception:
                            pass
            except Exception:
                pass

        cph = fit_cox_model(df_group, feats_for_fit, time_col, event_col)
        p5y_tr = _predict_surv_prob_at_t_cox(cph, df_group, feats_for_fit, T_STAR_DAYS)
        thr = float(np.nanmedian(p5y_tr))
        model_feats = list(getattr(getattr(cph, "params_", pd.Series()), "index", feats_for_fit))
        if plot_importance:
            try:
                plot_cox_coefficients(
                    cph,
                    title="Cox Coefficients (Selected Features)",
                    reference_df=df_group,
                    effect_scale="per_sd",
                )
            except Exception:
                pass
        return GroupModel(
            features=(model_feats if model_feats else list(feats_for_fit)),
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
    if enforce_fair_subset:
        train_df = drop_rows_with_missing_local_features(train_df)
        test_df = drop_rows_with_missing_local_features(test_df)
    z, B, tau = make_benefit_labels_crossfit(
        train_df=train_df,
        base_features=base_pool,
        plus_features=plus_pool,
        time_col=time_col,
        event_col=event_col,
        k_splits=k_splits,
        random_state=random_state,
        tau_rule="q65",
        margin_std=0.10,
    )
    triage_clf = fit_calibrated_triage(
        train_df=train_df,
        base_features=base_pool,
        B=B,
        random_state=random_state,
        backend="sgd",
        calibration="sigmoid",
    )
    if triage_clf is None:
        triage_probs_tr = np.zeros(len(train_df))
    else:
        triage_probs_tr = triage_clf.predict_proba(train_df[base_pool])[:, 1]
    mask_lbl = (B == 0) | (B == 1)
    X_theta = train_df.loc[mask_lbl, base_pool]
    y_theta = B[mask_lbl].astype(int)
    X_tr_theta, X_va_theta, y_tr_theta, y_va_theta = train_test_split(
        X_theta, y_theta, test_size=0.2, random_state=random_state, stratify=y_theta
    )
    p_va = (
        triage_clf.predict_proba(X_va_theta)[:, 1]
        if triage_clf is not None
        else np.zeros(len(X_va_theta))
    )
    theta = _choose_theta(
        p_va, method="max_net_benefit", y_benefit=y_va_theta, harm_ratio=1.0
    )
    triage_probs_te = (
        triage_clf.predict_proba(test_df[base_pool])[:, 1]
        if triage_clf is not None
        else np.zeros(len(test_df))
    )
    test_pred_benefit = (triage_probs_te >= theta).astype(int)
    benefit_mask_tr = triage_probs_tr >= theta
    non_benefit_mask_tr = ~benefit_mask_tr
    n_total = len(train_df)

    def _num_events(df: pd.DataFrame) -> int:
        try:
            return int(df[event_col].sum())
        except Exception:
            return 0

    group_ok = True
    df_benefit = train_df.loc[benefit_mask_tr]
    df_non_benefit = train_df.loc[non_benefit_mask_tr]
    if (
        len(df_benefit) < max(int(np.ceil(min_group_frac * n_total)), int(min_group_n))
        or _num_events(df_benefit) < int(min_events)
        or len(df_non_benefit)
        < max(int(np.ceil(min_group_frac * n_total)), int(min_group_n))
        or _num_events(df_non_benefit) < int(min_events)
    ):
        group_ok = False
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
    single_model: Optional[GroupModel] = None
    single_model_is_plus = False
    if not group_ok:
        p5y_base_oof, p5y_plus_oof = compute_oof_probs_5y(
            train_df,
            base_pool,
            plus_pool,
            time_col,
            event_col,
            k_splits=k_splits,
            random_state=random_state,
            enforce_fair_subset=False,
        )
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
        return (
            pred_labels,
            p5y_scores,
            {
                "c_index": float(cidx),
                "grouping": 0,
                "single_is_plus": float(single_model_is_plus),
                "benefit_mask": (triage_probs_te >= theta),
                "benefit_theta": float(theta),
                "benefit_tau": float(tau),
            },
        )
    mask_benefit_te = test_pred_benefit == 1
    mask_non_benefit_te = ~mask_benefit_te
    if base_model is not None and np.any(mask_non_benefit_te):
        te_nb = test_df.loc[mask_non_benefit_te]
        try:
            p_nb = _predict_surv_prob_at_t_cox(
                base_model.model, te_nb, base_model.features, T_STAR_DAYS
            )
            p5y_scores[mask_non_benefit_te] = p_nb
            pred_labels[mask_non_benefit_te] = (p_nb >= base_model.threshold).astype(
                int
            )
        except Exception:
            pass
    if plus_model is not None and np.any(mask_benefit_te):
        te_b = test_df.loc[mask_benefit_te]
        try:
            p_b = _predict_surv_prob_at_t_cox(
                plus_model.model, te_b, plus_model.features, T_STAR_DAYS
            )
            p5y_scores[mask_benefit_te] = p_b
            pred_labels[mask_benefit_te] = (p_b >= plus_model.threshold).astype(int)
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
            "benefit_mask": mask_benefit_te.astype(bool),
            "benefit_theta": float(theta),
            "benefit_tau": float(tau),
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
    try:
        diff = np.asarray(p5y_plus_test) - np.asarray(p5y_base_test)
        finite_idx = np.isfinite(diff)
        choose_plus_mask = np.zeros(len(test_df), dtype=bool)
        choose_plus_mask[finite_idx] = diff[finite_idx] < 0.0
    except Exception:
        choose_plus_mask = np.zeros(len(test_df), dtype=bool)
    return base_model, plus_model, p5y_base_test, p5y_plus_test, choose_plus_mask


def _majority_features(
    feature_lists: List[List[str]], min_frac: float = 0.6
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
    train_benefit_classifier: bool = True,
    triage_tableone_excel_path: Optional[str] = None,
    triage_km_plot_path: Optional[str] = None,
    triage_featimp_plot_path: Optional[str] = None,
    best_per_sample_tableone_excel_path: Optional[str] = None,
    plot_model_featimp: bool = True,
) -> Dict[str, object]:
    base_pool = list(FEATURE_SETS.get("Proposed", []))
    plus_pool = list(FEATURE_SETS.get("Proposed Plus", []))
    exclude_features = {"Age by decade", "CrCl>45", "NYHA>2", "Significant LGE"}
    base_pool = [f for f in base_pool if f not in exclude_features]
    plus_pool = [f for f in plus_pool if f not in exclude_features]
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
            stratify_arg = (
                strat_labels
                if (strat_labels is not None and strat_labels.nunique() > 1)
                else (
                    data_use[event_col] if data_use[event_col].nunique() > 1 else None
                )
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
    base_major = _majority_features(base_feat_runs, min_frac=0.6)
    if not base_major:
        base_major = [f for f in base_pool if f in df_use.columns]
    plus_major = _majority_features(plus_feat_runs, min_frac=0.6)
    if not plus_major:
        plus_major = [f for f in plus_pool if f in df_use.columns]
    try:
        print("==== Final Selected Features (Stabilized Pipeline) ====")
        print(f"Final Base features ({len(base_major)}): {base_major}")
        print(f"Final Plus features ({len(plus_major)}): {plus_major}")
    except Exception:
        pass
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
        plot_importance=False,
    )
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
    Z_all = (
        (df_use[event_col].values.astype(int) == 1)
        & (df_use[time_col].values.astype(float) <= T_STAR_DAYS)
    ).astype(int)
    w_all = _km_ipcw_weights(
        df_use[time_col].values.astype(float),
        df_use[event_col].values.astype(int),
        T_STAR_DAYS,
    )
    dLL_all = _winsorize(
        _delta_loglik_ipcw_binary_at_t(p5y_base_all, p5y_plus_all, Z_all, w_all)
    )
    tau = (
        float(np.quantile(dLL_all[np.isfinite(dLL_all)], 0.65))
        if np.isfinite(dLL_all).any()
        else 0.0
    )
    benefit_mask = dLL_all > tau
    p5y_best_all = np.where(benefit_mask, p5y_plus_all, p5y_base_all)

    # Generate TableOne for the real best-per-sample grouping (Plus better vs Base better)
    try:
        best_route_plus_all = np.asarray(p5y_plus_all) < np.asarray(p5y_base_all)
        df_best = df_use.copy()
        df_best["BenefitGroup"] = np.where(best_route_plus_all, "Benefit", "Non-Benefit")
        ACC.generate_tableone_by_group(
            df_best,
            group_col="BenefitGroup",
            output_excel_path=best_per_sample_tableone_excel_path,
        )
    except Exception:
        pass

    # Plot feature importance (Cox coefficients) for final base/plus models trained on the full training set
    if plot_model_featimp:
        try:
            if final_base is not None:
                plot_cox_coefficients(
                    final_base.model,
                    title="Final Base Model - Cox Coefficients (per SD)",
                    reference_df=df_use,
                    effect_scale="per_sd",
                )
        except Exception:
            pass
        try:
            if final_plus is not None:
                plot_cox_coefficients(
                    final_plus.model,
                    title="Final Plus Model - Cox Coefficients (per SD)",
                    reference_df=df_use,
                    effect_scale="per_sd",
                )
        except Exception:
            pass

    def _cidx_safe_from_prob(p: np.ndarray) -> float:
        try:
            return float(
                concordance_index(df_use[time_col], -np.asarray(p), df_use[event_col])
            )
        except Exception:
            return np.nan

    triage_clf = fit_calibrated_triage(
        train_df=df_use,
        base_features=base_pool,
        B=(dLL_all > tau).astype(int),
        random_state=0,
        backend="sgd",
        calibration="sigmoid",
    )
    if triage_clf is not None:
        triage_probs_all = triage_clf.predict_proba(df_use[base_pool])[:, 1]
        labels_all = (dLL_all > tau).astype(int)
        mask_lbl = np.isfinite(triage_probs_all) & np.isfinite(labels_all)
        p_all = triage_probs_all[mask_lbl]
        y_all = labels_all[mask_lbl]
        p_tr, p_va, y_tr, y_va = train_test_split(
            p_all, y_all, test_size=0.2, random_state=0, stratify=y_all
        )
        theta_used = _choose_theta(
            p_va, method="max_net_benefit", y_benefit=y_va, harm_ratio=1.0
        )
        route_plus = triage_probs_all >= theta_used
        coverage_triage = float(np.mean(route_plus))
        p5y_triage_all = np.where(route_plus, p5y_plus_all, p5y_base_all)
        c_index_triage = _cidx_safe_from_prob(p5y_triage_all)
    else:
        theta_used = np.nan
        coverage_triage = np.nan
        c_index_triage = np.nan
    metrics = {
        "c_index_all_base": _cidx_safe_from_prob(p5y_base_all),
        "c_index_all_plus": _cidx_safe_from_prob(p5y_plus_all),
        "c_index_per_sample_best": _cidx_safe_from_prob(p5y_best_all),
        "c_index_triage": c_index_triage,
        "coverage_triage": coverage_triage,
        "theta_triage": float(theta_used) if np.isfinite(theta_used) else np.nan,
        "tau": float(tau),
    }
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
        df_tab = te_tab.copy()
        route_plus_tab = None
        # Prefer grouping by triage classifier prediction when available
        if triage_clf is not None and np.isfinite(theta_used):
            try:
                triage_probs_te_tab = triage_clf.predict_proba(te_tab[base_pool])[:, 1]
                route_plus_tab = triage_probs_te_tab >= theta_used
            except Exception:
                route_plus_tab = None
        # Fallback: group by which model yields lower predicted 5-year probability
        if route_plus_tab is None:
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
            route_plus_tab = p_plus_tab < p_base_tab
        df_tab["BenefitGroup"] = np.where(route_plus_tab, "Benefit", "Non-Benefit")
        try:
            ACC.generate_tableone_by_group(
                df_tab, group_col="BenefitGroup", output_excel_path=tableone_excel_path
            )
        except Exception:
            pass
        # Additionally, save triage TableOne and KM for TEST split only
        try:
            ACC.generate_tableone_by_group(
                df_tab,
                group_col="BenefitGroup",
                output_excel_path=triage_tableone_excel_path,
            )
        except Exception:
            pass
        try:
            ACC.plot_km_by_group(
                df_tab,
                group_col="BenefitGroup",
                time_col=time_col,
                event_col=event_col,
                output_path=triage_km_plot_path,
                t_star_days=T_STAR_DAYS,
            )
        except Exception:
            pass
    except Exception:
        pass
    clf_importance: Optional[pd.DataFrame] = None
    if train_benefit_classifier and triage_clf is not None:
        try:
            fitted_base = None
            if hasattr(triage_clf, "calibrated_classifiers_") and getattr(
                triage_clf, "calibrated_classifiers_", None
            ):
                fitted_base = triage_clf.calibrated_classifiers_[0].base_estimator
            else:
                fitted_base = getattr(triage_clf, "base_estimator", None)
            if fitted_base is not None and hasattr(fitted_base, "named_steps"):
                inner = fitted_base.named_steps.get("clf", fitted_base)
            else:
                inner = fitted_base
            if inner is not None and hasattr(inner, "coef_"):
                coefs = np.asarray(inner.coef_).reshape(-1)
                feats = list(df_use[base_pool].columns)
                if len(coefs) == len(feats):
                    df_imp = pd.DataFrame({"feature": feats, "coef": coefs})
                    df_imp["abs_coef"] = df_imp["coef"].abs()
                    with np.errstate(over="ignore"):
                        df_imp["odds_ratio"] = np.exp(df_imp["coef"])
                    clf_importance = df_imp.sort_values(
                        "abs_coef", ascending=False
                    ).reset_index(drop=True)
                    if clf_importance_excel_path is not None:
                        os.makedirs(
                            os.path.dirname(clf_importance_excel_path), exist_ok=True
                        )
                        clf_importance.to_excel(clf_importance_excel_path, index=False)
                    # Print top-k importance here so it shows up even when imported
                    try:
                        top_show = int(min(20, len(clf_importance)))
                        print("==== Triage Classifier Feature Importance (|coef|) - Top 20 ====")
                        print(clf_importance.head(top_show))
                    except Exception:
                        pass
                    # Plot and save feature importance if requested
                    try:
                        if (
                            triage_featimp_plot_path is not None
                            and clf_importance is not None
                            and len(clf_importance) > 0
                        ):
                            os.makedirs(
                                os.path.dirname(triage_featimp_plot_path), exist_ok=True
                            )
                            top_k = int(min(30, len(clf_importance)))
                            df_plot = clf_importance.head(top_k).iloc[::-1]
                            height = max(4.0, 0.4 * top_k)
                            plt.figure(figsize=(10, height))
                            colors = [
                                "#d62728" if v > 0 else "#1f77b4"
                                for v in df_plot["coef"].tolist()
                            ]
                            plt.barh(
                                df_plot["feature"],
                                df_plot["abs_coef"],
                                color=colors,
                                alpha=0.85,
                            )
                            plt.xlabel("Absolute Coefficient |beta|")
                            plt.ylabel("Feature")
                            plt.title("Triage Classifier Feature Importance (|coef|)")
                            plt.tight_layout()
                            plt.savefig(
                                triage_featimp_plot_path, dpi=200, bbox_inches="tight"
                            )
                            plt.close()
                    except Exception:
                        pass
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


def run_single_split_two_model_pipeline(
    df: pd.DataFrame,
    time_col: str = "PE_Time",
    event_col: str = "VT/VF/SCD",
    test_size: float = 0.3,
    random_state: int = 0,
    enforce_fair_subset: bool = True,
    plot_model_featimp: bool = True,
    tableone_excel_path: Optional[str] = None,
) -> Dict[str, object]:
    base_pool = list(FEATURE_SETS.get("Proposed", []))
    plus_pool = list(FEATURE_SETS.get("Proposed Plus", []))
    exclude_features = {"Age by decade", "CrCl>45", "NYHA>2", "Significant LGE"}
    base_pool = [f for f in base_pool if f not in exclude_features]
    plus_pool = [f for f in plus_pool if f not in exclude_features]
    try:
        print("==== Feature Pools (Single Split) ====")
        print(f"Base (Proposed) features ({len(base_pool)}): {base_pool}")
        print(f"Plus (Proposed Plus) features ({len(plus_pool)}): {plus_pool}")
    except Exception:
        pass

    df_use = drop_rows_with_missing_local_features(df) if enforce_fair_subset else df.copy()
    data_use = df_use.dropna(subset=[time_col, event_col]).copy()

    strat_labels = _make_joint_stratify_labels(data_use, ["ICD", "Female", event_col])
    stratify_arg = (
        strat_labels if (strat_labels is not None and strat_labels.nunique() > 1) else (
            data_use[event_col] if data_use[event_col].nunique() > 1 else None
        )
    )
    tr, te = train_test_split(
        data_use,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=stratify_arg,
    )

    # Train one base model and one plus model on the training set with single-run forward selection
    final_base = _train_group_model(
        df_group=tr,
        candidate_features=[f for f in base_pool if f in tr.columns],
        time_col=time_col,
        event_col=event_col,
        stability_seeds=[int(random_state)],
        verbose=False,
        do_feature_selection=True,
        fs_max_features=None,
        plot_importance=plot_model_featimp,
    )
    final_plus = _train_group_model(
        df_group=tr,
        candidate_features=[f for f in plus_pool if f in tr.columns],
        time_col=time_col,
        event_col=event_col,
        stability_seeds=[int(random_state)],
        verbose=False,
        do_feature_selection=True,
        fs_max_features=None,
        plot_importance=plot_model_featimp,
    )

    p5y_base_te = np.zeros(len(te), dtype=float)
    p5y_plus_te = np.zeros(len(te), dtype=float)
    try:
        if final_base is not None:
            p5y_base_te = _predict_surv_prob_at_t_cox(final_base.model, te, final_base.features, T_STAR_DAYS)
    except Exception:
        pass
    try:
        if final_plus is not None:
            p5y_plus_te = _predict_surv_prob_at_t_cox(final_plus.model, te, final_plus.features, T_STAR_DAYS)
    except Exception:
        pass

    # Oracle best-of-two by true label: if event=1 pick higher risk; if event=0 pick lower risk
    event_te = te[event_col].astype(int).to_numpy()
    route_oracle_plus = np.where(event_te == 1, p5y_plus_te > p5y_base_te, p5y_plus_te < p5y_base_te)
    p5y_best_te = np.where(route_oracle_plus, p5y_plus_te, p5y_base_te)
    try:
        cidx_base = float(concordance_index(te[time_col], -p5y_base_te, te[event_col]))
    except Exception:
        cidx_base = np.nan
    try:
        cidx_plus = float(concordance_index(te[time_col], -p5y_plus_te, te[event_col]))
    except Exception:
        cidx_plus = np.nan
    try:
        cidx_best = float(concordance_index(te[time_col], -p5y_best_te, te[event_col]))
    except Exception:
        cidx_best = np.nan

    # Train triage classifier on training set using the same method as before (crossfit OOF labels)
    triage_clf = None
    theta_used = np.nan
    coverage_triage = np.nan
    p5y_triage_te = np.zeros(len(te), dtype=float)
    clf_importance: Optional[pd.DataFrame] = None
    try:
        # Crossfit benefit labels on training set
        _, B, tau = make_benefit_labels_crossfit(
            train_df=tr,
            base_features=[f for f in base_pool if f in tr.columns],
            plus_features=[f for f in plus_pool if f in tr.columns],
            time_col=time_col,
            event_col=event_col,
            k_splits=5,
            random_state=int(random_state),
            tau_rule="q65",
            margin_std=0.10,
        )
        triage_clf = fit_calibrated_triage(
            train_df=tr,
            base_features=[f for f in base_pool if f in tr.columns],
            B=B,
            random_state=int(random_state),
            backend="sgd",
            calibration="sigmoid",
        )
        # Pick threshold theta on a small validation split from train where labels exist
        mask_lbl = (B == 0) | (B == 1)
        X_theta = tr.loc[mask_lbl, [f for f in base_pool if f in tr.columns]]
        y_theta = B[mask_lbl].astype(int)
        if len(X_theta) > 1 and triage_clf is not None:
            X_tr_theta, X_va_theta, y_tr_theta, y_va_theta = train_test_split(
                X_theta, y_theta, test_size=0.2, random_state=int(random_state), stratify=y_theta
            )
            p_va = triage_clf.predict_proba(X_va_theta)[:, 1]
            theta_used = _choose_theta(p_va, method="max_net_benefit", y_benefit=y_va_theta, harm_ratio=1.0)
        # Extract and display triage classifier feature importance (logistic coefficients)
        try:
            fitted_base = None
            if hasattr(triage_clf, "calibrated_classifiers_") and getattr(
                triage_clf, "calibrated_classifiers_", None
            ):
                fitted_base = triage_clf.calibrated_classifiers_[0].base_estimator
            else:
                fitted_base = getattr(triage_clf, "base_estimator", None)
            if fitted_base is not None and hasattr(fitted_base, "named_steps"):
                inner = fitted_base.named_steps.get("clf", fitted_base)
            else:
                inner = fitted_base
            if inner is not None and hasattr(inner, "coef_"):
                coefs = np.asarray(inner.coef_).reshape(-1)
                feats = [f for f in base_pool if f in tr.columns]
                if len(coefs) == len(feats):
                    df_imp = pd.DataFrame({"feature": feats, "coef": coefs})
                    df_imp["abs_coef"] = df_imp["coef"].abs()
                    with np.errstate(over="ignore"):
                        df_imp["odds_ratio"] = np.exp(df_imp["coef"])
                    clf_importance = df_imp.sort_values("abs_coef", ascending=False).reset_index(drop=True)
                    try:
                        print("==== Triage Classifier Feature Importance (|coef|) - Top 20 ====")
                        print(clf_importance.head(int(min(20, len(clf_importance)))))
                    except Exception:
                        pass
                    if plot_model_featimp and len(clf_importance) > 0:
                        try:
                            top_k = int(min(30, len(clf_importance)))
                            df_plot = clf_importance.head(top_k).iloc[::-1]
                            height = max(4.0, 0.4 * top_k)
                            plt.figure(figsize=(10, height))
                            colors = ["#d62728" if v > 0 else "#1f77b4" for v in df_plot["coef"].tolist()]
                            plt.barh(df_plot["feature"], df_plot["abs_coef"], color=colors, alpha=0.85)
                            plt.xlabel("Absolute Coefficient |beta|")
                            plt.ylabel("Feature")
                            plt.title("Triage Classifier Feature Importance (|coef|)")
                            plt.tight_layout()
                            plt.show()
                        except Exception:
                            pass
        except Exception:
            clf_importance = None
    except Exception:
        triage_clf = None

    # Triage routing on test using the globally trained base/plus models
    try:
        if triage_clf is not None and np.isfinite(theta_used):
            triage_probs_te = triage_clf.predict_proba(te[[f for f in base_pool if f in te.columns]])[:, 1]
            route_plus = triage_probs_te >= float(theta_used)
            coverage_triage = float(np.mean(route_plus))
            p5y_triage_te = np.where(route_plus, p5y_plus_te, p5y_base_te)
            cidx_triage = float(concordance_index(te[time_col], -p5y_triage_te, te[event_col]))
        else:
            cidx_triage = np.nan
    except Exception:
        cidx_triage = np.nan

    # Optional descriptive table by oracle best-per-sample routing (Benefit vs Non-Benefit)
    try:
        route_plus = route_oracle_plus
        te_tab = te.copy()
        te_tab["BenefitGroup"] = np.where(route_plus, "Benefit", "Non-Benefit")
        ACC.generate_tableone_by_group(te_tab, group_col="BenefitGroup", output_excel_path=tableone_excel_path)
    except Exception:
        pass

    return {
        "final_base_features": (final_base.features if final_base is not None else []),
        "final_plus_features": (final_plus.features if final_plus is not None else []),
        "p5y_base_test": p5y_base_te,
        "p5y_plus_test": p5y_plus_te,
        "p5y_triage_test": p5y_triage_te,
        "p5y_best_test": p5y_best_te,
        "best_route_plus": route_oracle_plus,
        "triage_route_plus": (triage_clf is not None and np.isfinite(theta_used) and (triage_probs_te >= float(theta_used))) if 'triage_probs_te' in locals() else np.zeros(len(te), dtype=bool),
        "metrics": {
            "c_index_base": cidx_base,
            "c_index_plus": cidx_plus,
            "c_index_best": cidx_best,
            "c_index_triage": cidx_triage,
            "coverage_triage": coverage_triage,
        },
        "triage_classifier_importance": clf_importance,
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
    # Single-run only: load data, split once, train with feature selection on train, evaluate on test
    try:
        df = ACC.load_dataframes()
    except Exception as e:
        raise SystemExit(f"Failed to load dataframes: {e}")

    try:
        outdir = "/home/sunx/data/aiiih/projects/sunx/projects/ICD"
        os.makedirs(outdir, exist_ok=True)
    except Exception:
        outdir = None

    try:
        table_path = (
            os.path.join(outdir, "tableone_best_per_sample.xlsx") if outdir else None
        )
    except Exception:
        table_path = None

    try:
        result = run_single_split_two_model_pipeline(
            df=df,
            time_col="PE_Time",
            event_col="VT/VF/SCD",
            test_size=0.3,
            random_state=0,
            enforce_fair_subset=True,
            plot_model_featimp=True,
            tableone_excel_path=table_path,
        )
        try:
            print("==== Single-split Benefit-Specific Metrics ====")
            print(result.get("metrics", {}))
            if table_path is not None:
                print("Saved best-per-sample TableOne:", table_path)
        except Exception:
            pass
    except Exception as e:
        print(f"Single-split pipeline failed: {e}")
