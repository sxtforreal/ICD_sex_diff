import os
import math
import warnings
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional

# ==========================================
# Global store for Proposed selected features
# ==========================================
SELECTED_FEATURES_STORE: Dict[str, Any] = {
    "proposed_sex_agnostic": None,  # type: List[str] | None
    "proposed_sex_specific": {  # type: Dict[str, List[str]] | None
        "male": None,
        "female": None,
    },
}

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from lifelines.exceptions import ConvergenceError

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_dataframes() -> pd.DataFrame:
    base = "/home/sunx/data/aiiih/projects/sunx/projects/ICD"
    icd = pd.read_excel(os.path.join(base, "LGE granularity.xlsx"), sheet_name="ICD")
    noicd = pd.read_excel(
        os.path.join(base, "LGE granularity.xlsx"), sheet_name="No_ICD"
    )

    # columns
    cols = [
        "LGE_LGE Burden 5SD",
        "LGE_Extent (1; subendocardial, 2; mid mural, 3; epicardial, 4; transmural)",
        "LGE_Circumural",
        "LGE_Ring-Like",
        "LGE_Basal anterior  (0; No, 1; yes)",
        "LGE_Basal anterior septum",
        "LGE_Basal inferoseptum",
        "LGE_Basal inferio",
        "LGE_Basal inferolateral",
        "LGE_Basal anterolateral ",
        "LGE_mid anterior",
        "LGE_mid anterior septum",
        "LGE_mid inferoseptum",
        "LGE_mid inferior",
        "LGE_mid inferolateral ",
        "LGE_mid anterolateral ",
        "LGE_apical anterior",
        "LGE_apical septum",
        "LGE_apical inferior",
        "LGE_apical lateral",
        "LGE_Apical cap",
        "LGE_RV insertion site (1 superior, 2 inferior. 3 both)",
        "LGE_Score",
        "Female",
        "VT/VF/SCD",
        "MRI Date",
        "Date VT/VF/SCD",
        "End follow-up date",
    ]

    # Stack dfs
    icd_common = icd[cols].copy()
    noicd_common = noicd[cols].copy()
    icd_common["ICD"] = 1
    noicd_common["ICD"] = 0
    nicm = pd.concat([icd_common, noicd_common], ignore_index=True)

    # PE time
    nicm["MRI Date"] = pd.to_datetime(nicm["MRI Date"])
    nicm["Date VT/VF/SCD"] = pd.to_datetime(nicm["Date VT/VF/SCD"])
    nicm["End follow-up date"] = pd.to_datetime(nicm["End follow-up date"])
    nicm["PE_Time"] = nicm.apply(
        lambda row: (
            (row["Date VT/VF/SCD"] - row["MRI Date"]).days
            if row["VT/VF/SCD"] == 1
            and pd.notna(row["Date VT/VF/SCD"])
            and pd.notna(row["MRI Date"])
            else (
                (row["End follow-up date"] - row["MRI Date"]).days
                if pd.notna(row["End follow-up date"]) and pd.notna(row["MRI Date"])
                else None
            )
        ),
        axis=1,
    )
    nicm.drop(
        ["MRI Date", "Date VT/VF/SCD", "End follow-up date"],
        axis=1,
        inplace=True,
        errors="ignore",
    )

    # Variables
    c = nicm.columns.to_list()
    categorical = [item for item in c if item not in ["LGE_LGE Burden 5SD", "PE_Time"]]
    nicm[categorical] = nicm[categorical].astype("object")
    labels = ["Female", "VT/VF/SCD", "PE_Time", "ICD"]
    features = [v for v in c if v not in labels]
    areas = [
        item
        for item in features
        if item
        not in [
            "LGE_LGE Burden 5SD",
            "LGE_Extent (1; subendocardial, 2; mid mural, 3; epicardial, 4; transmural)",
            "LGE_Circumural",
            "LGE_Ring-Like",
            "LGE_RV insertion site (1 superior, 2 inferior. 3 both)",
            "LGE_Score",
        ]
    ]

    # Imputation
    nicm = nicm[~nicm[areas].isna().all(axis=1)]
    nicm[areas] = nicm[areas].fillna(0)

    return nicm


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def _prep(df, time_col, event_col):
    d = df.copy()
    d[time_col] = _to_num(d[time_col])
    d[event_col] = _to_num(d[event_col]).fillna(0).clip(0, 1)
    d = d.dropna(subset=[time_col, event_col])
    d = d[d[time_col] > 0]
    return d


def _safe_fit(d, feats, time_col, event_col, penalizer=0.5):
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=0.0)
    cph.fit(
        d[[time_col, event_col] + feats], duration_col=time_col, event_col=event_col
    )
    return cph


def _score_model(cph, criterion, n, k):
    if criterion == "pll":
        return cph.log_likelihood_
    if criterion == "aic":
        return -2 * cph.log_likelihood_ + 2 * k
    if criterion == "bic":
        return -2 * cph.log_likelihood_ + k * np.log(max(n, 1))
    if criterion == "concordance":
        return float(getattr(cph, "concordance_index_", np.nan))
    return cph.log_likelihood_


def forward_select_once(
    df: pd.DataFrame,
    candidate_cols: List[str],
    time_col="PE_Time",
    event_col="VT/VF/SCD",
    penalizer=0.5,
    criterion="bic",
    min_improve=1e-4,
    max_features=None,
    random_state=None,
    epv: int = 10,
) -> List[str]:
    rng = np.random.default_rng(random_state)
    d = _prep(df, time_col, event_col)
    n_events = int(np.nan_to_num(d[event_col].sum(), nan=0.0))
    if epv is not None and n_events > 0:
        max_by_epv = max(1, n_events // epv)
        max_features = (
            max_by_epv if max_features is None else min(max_features, max_by_epv)
        )
    X = d[candidate_cols].apply(_to_num).replace([np.inf, -np.inf], np.nan).fillna(0)
    d = pd.concat([d[[time_col, event_col]], X], axis=1)
    chosen, remaining = [], list(candidate_cols)
    higher_better = criterion in ("pll", "concordance")
    best_score = -np.inf if higher_better else np.inf
    best_model = None
    while remaining and (max_features is None or len(chosen) < max_features):
        rng.shuffle(remaining)
        trial_scores, trial_models = [], {}
        for f in remaining:
            feats = chosen + [f]
            try:
                m = _safe_fit(d, feats, time_col, event_col, penalizer)
                s = _score_model(m, criterion, len(d), len(feats))
                trial_scores.append((f, s))
                trial_models[f] = m
            except Exception:
                continue
        if not trial_scores:
            break
        if higher_better:
            f_new, s_new = max(trial_scores, key=lambda x: x[1])
            improve = s_new - best_score
            if improve <= min_improve and best_model is not None:
                break
            best_score = s_new
            best_model = trial_models[f_new]
            chosen.append(f_new)
            remaining.remove(f_new)
        else:
            f_new, s_new = min(trial_scores, key=lambda x: x[1])
            improve = best_score - s_new
            if improve <= min_improve and best_model is not None:
                break
            best_score = s_new
            best_model = trial_models[f_new]
            chosen.append(f_new)
            remaining.remove(f_new)
    return chosen


def stability_select_features(
    df: pd.DataFrame,
    candidate_features: List[str],
    time_col="PE_Time",
    event_col="VT/VF/SCD",
    seeds: List[int] = list(range(20)),
    penalizer=0.5,
    criterion="bic",
    min_improve=1e-4,
    max_features=None,
    threshold=0.6,
    epv: int = 10,
) -> List[str]:
    counts = {f: 0 for f in candidate_features}
    for sd in seeds:
        sel = forward_select_once(
            df=df,
            candidate_cols=candidate_features,
            time_col=time_col,
            event_col=event_col,
            penalizer=penalizer,
            criterion=criterion,
            min_improve=min_improve,
            max_features=max_features,
            random_state=sd,
            epv=epv,
        )
        for f in sel:
            counts[f] += 1
    freq = {f: counts[f] / max(1, len(seeds)) for f in candidate_features}
    selected = [f for f in candidate_features if freq[f] >= threshold]
    return selected


def fit_cox_four_groups_select_plot(
    df: pd.DataFrame,
    feature_cols: List[str],
    time_col="PE_Time",
    event_col="VT/VF/SCD",
    alpha=0.05,
    penalizer=0.5,
    criterion="bic",
    seeds: List[int] = list(range(20)),
    threshold=0.6,
    max_features=None,
    epv: int = 10,
) -> Dict[Tuple[str, str], Dict]:
    exclude = {time_col, event_col, "Female", "ICD"}
    feature_cols = [c for c in feature_cols if c not in exclude]
    groups = [
        ((0, "Male"), (0, "No ICD")),
        ((0, "Male"), (1, "ICD")),
        ((1, "Female"), (0, "No ICD")),
        ((1, "Female"), (1, "ICD")),
    ]
    fits = []
    for (sx_val, sx_name), (icd_val, icd_name) in groups:
        sub = df[(df["Female"] == sx_val) & (df["ICD"] == icd_val)].copy()
        if sub.empty:
            fits.append(
                (
                    (sx_name, icd_name),
                    {"model": None, "features": [], "n": 0, "events": 0.0},
                )
            )
            continue
        sel = stability_select_features(
            df=sub,
            candidate_features=feature_cols,
            time_col=time_col,
            event_col=event_col,
            seeds=seeds,
            penalizer=penalizer,
            criterion=criterion,
            min_improve=1e-4,
            max_features=max_features,
            threshold=threshold,
            epv=epv,
        )
        if not sel:
            sel = forward_select_once(
                sub, feature_cols, time_col, event_col, penalizer, criterion, epv=epv
            )
        sel = [f for f in feature_cols if f in set(sel)]
        if sel:
            d = _prep(sub, time_col, event_col)
            X = d[sel].apply(_to_num).replace([np.inf, -np.inf], np.nan).fillna(0)
            d = pd.concat([d[[time_col, event_col]], X], axis=1)
            model = _safe_fit(d, sel, time_col, event_col, penalizer)
        else:
            model = None
        fits.append(
            (
                (sx_name, icd_name),
                {
                    "model": model,
                    "features": sel,
                    "n": len(sub),
                    "events": float(sub[event_col].sum()),
                },
            )
        )

    max_len = max(1, max(len(v["features"]) for _, v in fits))
    fig, axes = plt.subplots(2, 2, figsize=(14, max(6, 0.9 * max_len)))
    axes = axes.flatten()
    for ax, ((sx_name, icd_name), res) in zip(axes, fits):
        m, feats = res["model"], res["features"]
        if m is None or not feats:
            ax.set_axis_off()
            ax.set_title(f"{sx_name}, {icd_name} (no selected features)")
            continue
        coef = m.params_.reindex(feats).fillna(0)
        sig = set()
        if getattr(m, "summary", None) is not None:
            sig = set(m.summary.loc[m.summary["p"] < alpha].index)
        labels = [f + ("*" if f in sig else "") for f in feats]
        pd.DataFrame({f"{sx_name}-{icd_name}": coef}).plot(
            kind="barh", legend=False, ax=ax
        )
        ax.axvline(0, lw=1)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("log hazard ratio")
        ax.set_title(f"{sx_name}, {icd_name}")
    plt.suptitle(f"Cox forward+stability by Sex × ICD  (* = p<{alpha})", y=1.02)
    plt.tight_layout()
    plt.show()

    out = {}
    for (sx_name, icd_name), res in fits:
        out[(sx_name, icd_name)] = {
            "selected_features": res.get("features", []),
            "n": res.get("n", None),
            "events": res.get("events", None),
            "summary": None if res.get("model") is None else res["model"].summary,
        }
    return out


###############################################
# Random Survival Forest hierarchical importance
###############################################

def _try_import_rsf_backend():
    """Return available RSF backend and class.

    Preference: scikit-survival -> lifelines. Returns (backend_name, cls)
    backend_name in {"sksurv", "lifelines"}
    """
    try:
        from sksurv.ensemble import RandomSurvivalForest as SKSURV_RSF  # type: ignore
        return "sksurv", SKSURV_RSF
    except Exception:
        try:
            from lifelines import RandomSurvivalForest as LL_RSF  # type: ignore
            return "lifelines", LL_RSF
        except Exception:
            raise ImportError(
                "Random Survival Forest backend not found. Install scikit-survival or lifelines."
            )


class RSFModel:
    """A small wrapper to unify sksurv and lifelines RSF APIs."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 3,
        max_features: Optional[Any] = "sqrt",
        random_state: Optional[int] = 42,
        n_jobs: int = -1,
    ) -> None:
        backend, cls = _try_import_rsf_backend()
        self.backend = backend
        self.feature_names_: List[str] = []
        self.eval_time_: Optional[float] = None
        if backend == "sksurv":
            # sksurv RSF parameters
            self.model = cls(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                n_jobs=n_jobs,
                random_state=random_state,
            )
        else:
            # lifelines RSF parameters (subset; n_jobs supported in recent versions)
            try:
                self.model = cls(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    n_jobs=n_jobs,
                    random_state=random_state,
                )
            except TypeError:
                # Fallback without n_jobs if older lifelines
                self.model = cls(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=random_state,
                )

    def fit(self, X: pd.DataFrame, durations: np.ndarray, events: np.ndarray) -> "RSFModel":
        self.feature_names_ = list(X.columns)
        if self.backend == "sksurv":
            # y needs structured array with (event, time)
            y = np.array(
                [(bool(e), float(t)) for e, t in zip(events.astype(int), durations)],
                dtype=[("event", "?"), ("time", "<f8")],
            )
            self.model.fit(X.values, y)
        else:
            df_fit = pd.concat(
                [
                    pd.DataFrame({"duration": durations, "event": events}),
                    X.reset_index(drop=True),
                ],
                axis=1,
            )
            self.model.fit(df_fit, duration_col="duration", event_col="event")
        return self

    def predict_survival_at(self, X: pd.DataFrame, t: float) -> np.ndarray:
        if self.backend == "sksurv":
            surv_funcs = self.model.predict_survival_function(X.values)
            # surv_funcs is a list of step functions
            out = np.array([float(sf(t)) for sf in surv_funcs], dtype=float)
            return out
        else:
            # lifelines returns list of pd.Series or a DataFrame over times
            try:
                sf_list = self.model.predict_survival_function(X, times=[t])
                # sf_list may be a list of length n with Series indexed by times
                if isinstance(sf_list, list):
                    return np.array([float(s.iloc[0]) for s in sf_list], dtype=float)
                # or a DataFrame (times x n)
                if hasattr(sf_list, "values"):
                    arr = np.asarray(sf_list)
                    # shape (1, n)
                    return arr[0].astype(float)
            except Exception:
                # Fallback: use cumulative hazard then S = exp(-H)
                ch = self.model.predict_cumulative_hazard_function(X, times=[t])
                if isinstance(ch, list):
                    return np.array([math.exp(-float(s.iloc[0])) for s in ch], dtype=float)
                if hasattr(ch, "values"):
                    arr = np.asarray(ch)
                    return np.exp(-arr[0].astype(float))
            raise RuntimeError("Unexpected survival function output format")

    def predict_risk_at(self, X: pd.DataFrame, t: float) -> np.ndarray:
        # Risk = 1 - S(t)
        s = self.predict_survival_at(X, t)
        return 1.0 - s

    @property
    def impurity_importances_(self) -> pd.Series:
        if self.backend == "sksurv":
            if hasattr(self.model, "feature_importances_"):
                return pd.Series(self.model.feature_importances_, index=self.feature_names_)
        else:
            # lifelines may expose variable_importances_ as dict or pandas Series
            if hasattr(self.model, "variable_importances_"):
                imp = self.model.variable_importances_
                if isinstance(imp, dict):
                    return pd.Series(imp)
                try:
                    return pd.Series(imp).reindex(self.feature_names_)
                except Exception:
                    return pd.Series(imp)
            if hasattr(self.model, "feature_importances_"):
                return pd.Series(self.model.feature_importances_, index=self.feature_names_)
        # fallback zeros
        return pd.Series(np.zeros(len(self.feature_names_), dtype=float), index=self.feature_names_)

    def _get_sklearn_estimators(self) -> List[Any]:
        # Try common attributes to get sklearn trees for structure analysis
        cand_attrs = ["estimators_", "trees_", "_forest", "survival_forest"]
        obj = self.model
        estimators: List[Any] = []
        for attr in cand_attrs:
            if hasattr(obj, attr):
                val = getattr(obj, attr)
                if isinstance(val, list):
                    estimators = val
                    break
                # Some backends might have a RandomForest-like object
                if hasattr(val, "estimators_"):
                    estimators = list(getattr(val, "estimators_"))
                    break
        return estimators


def _compute_min_depth_and_frequencies(model: RSFModel, high_level_depth: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute minimal depth, root frequency, and high-level frequency for features.

    Returns: (min_depth_mean, root_freq, highlevel_freq)
    """
    estimators = model._get_sklearn_estimators()
    feature_names = model.feature_names_
    if not estimators:
        # Graceful fallback: no tree access
        zeros = pd.Series(np.zeros(len(feature_names)), index=feature_names)
        return zeros.replace(0, np.nan), zeros, zeros

    min_depths_per_tree: List[Dict[str, int]] = []
    root_counts = Counter()
    highlevel_counts = Counter()

    for est in estimators:
        if not hasattr(est, "tree_"):
            continue
        tr = est.tree_
        features = tr.feature  # array of feature indices, -2 for leaves
        children_left = tr.children_left
        children_right = tr.children_right

        # BFS from root to compute first depth of each feature
        first_depth: Dict[int, int] = {}
        q = [(0, 0)]  # (node_index, depth)
        while q:
            node, d = q.pop(0)
            f_idx = int(features[node])
            if f_idx >= 0 and f_idx not in first_depth:
                first_depth[f_idx] = d
            left = int(children_left[node])
            right = int(children_right[node])
            if left != -1:
                q.append((left, d + 1))
            if right != -1:
                q.append((right, d + 1))

        # Record per-tree depths
        md_map = {}
        for f_idx, d in first_depth.items():
            if 0 <= f_idx < len(feature_names):
                md_map[feature_names[f_idx]] = d
        min_depths_per_tree.append(md_map)

        # Root and high-level counts
        root_f_idx = int(features[0])
        if root_f_idx >= 0 and root_f_idx < len(feature_names):
            root_counts[feature_names[root_f_idx]] += 1

        # Count any feature that appears first at depth <= high_level_depth
        for f_idx, d in first_depth.items():
            if 0 <= f_idx < len(feature_names) and d <= high_level_depth:
                highlevel_counts[feature_names[f_idx]] += 1

    # Aggregate minimal depth across trees: use mean depth, NaN if never used
    all_features = feature_names
    depth_values = {}
    for f in all_features:
        vals = [md[f] for md in min_depths_per_tree if f in md]
        depth_values[f] = float(np.mean(vals)) if len(vals) > 0 else np.nan

    n_trees = max(1, len(estimators))
    min_depth_mean = pd.Series(depth_values, index=all_features)
    root_freq = pd.Series({f: root_counts.get(f, 0) / n_trees for f in all_features}, index=all_features)
    highlevel_freq = pd.Series({f: highlevel_counts.get(f, 0) / n_trees for f in all_features}, index=all_features)
    return min_depth_mean, root_freq, highlevel_freq


def _compute_cooccurrence_on_paths(model: RSFModel, level_depth: int = 2) -> pd.DataFrame:
    """Compute feature co-occurrence counts within high-level paths (depth <= level_depth).

    Returns a symmetric DataFrame of co-occurrence frequencies normalized by number of trees.
    """
    estimators = model._get_sklearn_estimators()
    feature_names = model.feature_names_
    if not estimators:
        return pd.DataFrame(0.0, index=feature_names, columns=feature_names)

    pair_counts = Counter()
    single_counts = Counter()

    for est in estimators:
        if not hasattr(est, "tree_"):
            continue
        tr = est.tree_
        features = tr.feature
        children_left = tr.children_left
        children_right = tr.children_right

        # Traverse all root-to-leaf paths; collect features up to level_depth
        stack = [(0, 0, [])]  # node, depth, path_feature_indices
        while stack:
            node, d, path = stack.pop()
            f_idx = int(features[node])
            path2 = list(path)
            if f_idx >= 0 and d <= level_depth:
                path2.append(f_idx)
            left = int(children_left[node])
            right = int(children_right[node])
            if left == -1 and right == -1:
                # leaf: update co-occurrence for this path
                # deduplicate features within this path
                uniq = sorted(set([i for i in path2 if 0 <= i < len(feature_names)]))
                for i in uniq:
                    single_counts[feature_names[i]] += 1
                for i, j in combinations(uniq, 2):
                    pair = tuple(sorted((feature_names[i], feature_names[j])))
                    pair_counts[pair] += 1
            else:
                if left != -1:
                    stack.append((left, d + 1, path2))
                if right != -1:
                    stack.append((right, d + 1, path2))

    n_trees = max(1, len(estimators))
    # Build symmetric matrix normalized by number of trees
    co_mat = pd.DataFrame(0.0, index=feature_names, columns=feature_names)
    for (fi, fj), cnt in pair_counts.items():
        co_mat.loc[fi, fj] = cnt / n_trees
        co_mat.loc[fj, fi] = cnt / n_trees
    for f, cnt in single_counts.items():
        co_mat.loc[f, f] = cnt / n_trees
    return co_mat


def _permutation_importance_manual(
    model: RSFModel,
    X: pd.DataFrame,
    durations: np.ndarray,
    events: np.ndarray,
    eval_time: float,
    n_repeats: int = 20,
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """Permutation importance measured by drop in C-index."""
    rng = np.random.default_rng(random_state)
    base_risk = model.predict_risk_at(X, eval_time)
    base_c = concordance_index(durations, base_risk, events)
    cols = list(X.columns)
    drops = []
    for c in cols:
        vals = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[c] = rng.permutation(Xp[c].values)
            risk_p = model.predict_risk_at(Xp, eval_time)
            cidx = concordance_index(durations, risk_p, events)
            vals.append(base_c - cidx)
        drops.append((c, float(np.mean(vals)), float(np.std(vals))))
    return pd.DataFrame(drops, columns=["feature", "perm_drop_mean", "perm_drop_std"]).set_index("feature")


def _choose_eval_time(durations: np.ndarray, quantile: float = 0.5) -> float:
    arr = np.asarray(durations, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return 1.0
    return float(np.quantile(arr, quantile))


def _prep_xy(
    df: pd.DataFrame,
    feature_cols: List[str],
    time_col: str,
    event_col: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    d = _prep(df, time_col, event_col)
    X = d[feature_cols].apply(_to_num).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    durations = d[time_col].values.astype(float)
    events = d[event_col].values.astype(int)
    return X, durations, events


def rsf_hierarchical_importance(
    df: pd.DataFrame,
    feature_cols: List[str],
    time_col: str = "PE_Time",
    event_col: str = "VT/VF/SCD",
    n_estimators: int = 500,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 3,
    max_features: Any = "sqrt",
    high_level_depth: int = 2,
    n_perm_repeats: int = 30,
    eval_time: Optional[float] = None,
    random_state: Optional[int] = 42,
) -> Dict[str, Any]:
    # Train RSF
    X, durations, events = _prep_xy(df, feature_cols, time_col, event_col)
    if eval_time is None:
        eval_time = _choose_eval_time(durations, 0.5)
    rsf = RSFModel(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
    ).fit(X, durations, events)

    # Metrics
    impurity = rsf.impurity_importances_.rename("impurity_importance").reindex(feature_cols)
    perm = _permutation_importance_manual(
        rsf, X, durations, events, eval_time, n_repeats=n_perm_repeats, random_state=random_state
    )
    min_depth, root_freq, highlevel_freq = _compute_min_depth_and_frequencies(rsf, high_level_depth)

    summary = pd.concat(
        [impurity, perm, min_depth.rename("mean_min_depth"), root_freq.rename("root_freq"), highlevel_freq.rename("highlevel_freq")],
        axis=1,
    )
    # Sort by permutation drop, then impurity
    summary = summary.sort_values(by=["perm_drop_mean", "impurity_importance"], ascending=[False, False])

    # Interaction/co-occurrence
    co_mat = _compute_cooccurrence_on_paths(rsf, level_depth=high_level_depth)

    return {
        "rsf": rsf,
        "eval_time": eval_time,
        "summary": summary,
        "cooccurrence": co_mat,
    }


def _km_event_prob_by_time(durations: np.ndarray, events: np.ndarray, t: float) -> float:
    # 1 - KM survival at t
    df_tmp = pd.DataFrame({"time": durations, "event": events})
    df_tmp = df_tmp.dropna()
    if len(df_tmp) == 0:
        return float("nan")
    kmf = KaplanMeierFitter()
    kmf.fit(df_tmp["time"], event_observed=df_tmp["event"])
    s = float(kmf.survival_function_at_times(t).iloc[0])
    return max(0.0, min(1.0, 1.0 - s))


def plot_calibration_by_bins(
    model: RSFModel,
    X: pd.DataFrame,
    durations: np.ndarray,
    events: np.ndarray,
    eval_time: float,
    n_bins: int = 10,
    title: str = "Calibration (KM vs Predicted)",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    pred = model.predict_risk_at(X, eval_time)
    bins = pd.qcut(pd.Series(pred), q=n_bins, duplicates="drop")
    dfb = pd.DataFrame({"pred": pred, "time": durations, "event": events, "bin": bins})
    obs, est = [], []
    for b, sub in dfb.groupby("bin"):
        if len(sub) == 0:
            continue
        obs.append(_km_event_prob_by_time(sub["time"].values, sub["event"].values, eval_time))
        est.append(float(np.mean(sub["pred"].values)))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.scatter(est, obs, c="C0")
    ax.set_xlabel(f"Predicted risk @ t={eval_time:.1f}")
    ax.set_ylabel("Observed (KM)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def plot_partial_dependence(
    model: RSFModel,
    X: pd.DataFrame,
    feature: str,
    eval_time: float,
    grid: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Simple PDP for survival risk at a fixed time horizon."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    if grid is None:
        x = X[feature].values
        lo, hi = np.nanpercentile(x, 5), np.nanpercentile(x, 95)
        grid = np.linspace(lo, hi, 25)
    preds = []
    X_copy = X.copy()
    for v in grid:
        X_copy[feature] = v
        preds.append(np.mean(model.predict_risk_at(X_copy, eval_time)))
    ax.plot(grid, preds, color="C1")
    ax.set_xlabel(feature)
    ax.set_ylabel(f"Avg risk @ t={eval_time:.1f}")
    ax.set_title(title or f"PDP: {feature}")
    ax.grid(alpha=0.3)
    return ax


def run_rsf_full_pipeline(
    df: pd.DataFrame,
    feature_cols: List[str],
    time_col: str = "PE_Time",
    event_col: str = "VT/VF/SCD",
    out_dir: str = "rsf_outputs",
    n_estimators: int = 500,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 3,
    max_features: Any = "sqrt",
    high_level_depth: int = 2,
    n_perm_repeats: int = 30,
    eval_time: Optional[float] = None,
    random_state: Optional[int] = 42,
    do_subgroups: bool = True,
    top_k_plot: int = 10,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)

    # Global
    global_res = rsf_hierarchical_importance(
        df,
        feature_cols,
        time_col,
        event_col,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        high_level_depth=high_level_depth,
        n_perm_repeats=n_perm_repeats,
        eval_time=eval_time,
        random_state=random_state,
    )
    global_summary = global_res["summary"]
    global_summary.to_csv(os.path.join(out_dir, "global_importance.csv"))
    global_res["cooccurrence"].to_csv(os.path.join(out_dir, "global_cooccurrence.csv"))

    # Calibration plot
    Xg, Tg, Eg = _prep_xy(df, feature_cols, time_col, event_col)
    et = float(global_res["eval_time"])  # type: ignore
    ax = plot_calibration_by_bins(global_res["rsf"], Xg, Tg, Eg, et, title="Global calibration")  # type: ignore
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "global_calibration.png"), dpi=160)
    plt.close()

    # PDPs for top features by permutation drop
    top_feats = list(global_summary.sort_values("perm_drop_mean", ascending=False).index[:max(1, top_k_plot)])
    ncols = 3
    nrows = int(math.ceil(len(top_feats) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axs = np.array(axs).reshape(-1)
    i = -1
    for i, f in enumerate(top_feats):
        plot_partial_dependence(global_res["rsf"], Xg, f, et, ax=axs[i], title=f)  # type: ignore
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")
    fig.suptitle("Global PDP for top features", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "global_pdp.png"), dpi=160)
    plt.close()

    outputs: Dict[str, Any] = {"global": global_res}

    # Subgroups: (Male/Female) x (ICD/No ICD)
    if do_subgroups:
        groups = [
            ((0, "Male"), (0, "No ICD")),
            ((0, "Male"), (1, "ICD")),
            ((1, "Female"), (0, "No ICD")),
            ((1, "Female"), (1, "ICD")),
        ]
        for (sx_val, sx_name), (icd_val, icd_name) in groups:
            sub = df[(df["Female"] == sx_val) & (df["ICD"] == icd_val)].copy()
            if sub.empty or sub[event_col].sum() == 0:
                continue
            sub_res = rsf_hierarchical_importance(
                sub,
                feature_cols,
                time_col,
                event_col,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                high_level_depth=high_level_depth,
                n_perm_repeats=max(10, n_perm_repeats // 2),
                eval_time=et,
                random_state=random_state,
            )
            key = f"{sx_name}_{icd_name}"
            outputs[key] = sub_res
            sub_dir = os.path.join(out_dir, f"subgroup_{sx_name}_{icd_name}")
            os.makedirs(sub_dir, exist_ok=True)
            sub_res["summary"].to_csv(os.path.join(sub_dir, "importance.csv"))
            sub_res["cooccurrence"].to_csv(os.path.join(sub_dir, "cooccurrence.csv"))
            # Calibration
            Xs, Ts, Es = _prep_xy(sub, feature_cols, time_col, event_col)
            ax = plot_calibration_by_bins(sub_res["rsf"], Xs, Ts, Es, et, title=f"Calibration: {sx_name}, {icd_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(sub_dir, "calibration.png"), dpi=160)
            plt.close()
            # PDP
            sub_top = list(sub_res["summary"].sort_values("perm_drop_mean", ascending=False).index[:max(1, top_k_plot)])
            ncols = 3
            nrows = int(math.ceil(len(sub_top) / ncols))
            fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
            axs = np.array(axs).reshape(-1)
            Xs_df, _, _ = _prep_xy(sub, feature_cols, time_col, event_col)
            i = -1
            for i, f in enumerate(sub_top):
                plot_partial_dependence(sub_res["rsf"], Xs_df, f, et, ax=axs[i], title=f)
            for j in range(i + 1, len(axs)):
                axs[j].axis("off")
            fig.suptitle(f"PDP: {sx_name}, {icd_name}", y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(sub_dir, "pdp.png"), dpi=160)
            plt.close()

    return outputs


def bootstrap_rsf_stability(
    df: pd.DataFrame,
    feature_cols: List[str],
    time_col: str = "PE_Time",
    event_col: str = "VT/VF/SCD",
    n_bootstrap: int = 50,
    random_state: Optional[int] = 42,
    **rsf_kwargs: Any,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    ranks: List[pd.Series] = []
    for b in range(n_bootstrap):
        idx = rng.integers(0, len(df), size=len(df))
        boot = df.iloc[idx].reset_index(drop=True)
        # vary RSF seed per bootstrap for robustness
        rsf_kwargs_b = dict(rsf_kwargs)
        rsf_kwargs_b["random_state"] = (None if random_state is None else int(random_state) + b)
        res = rsf_hierarchical_importance(boot, feature_cols, time_col, event_col, **rsf_kwargs_b)
        imp = res["summary"]["perm_drop_mean"].fillna(0.0)
        rank = (-imp).rank(method="average")  # lower rank = more important
        ranks.append(rank.rename(f"boot_{b}"))
    R = pd.concat(ranks, axis=1)
    # Stability metrics
    freq_top5 = (R.apply(lambda s: s <= 5)).mean(axis=1)
    mean_rank = R.mean(axis=1)
    std_rank = R.std(axis=1, ddof=1)
    out = pd.DataFrame({"freq_top5": freq_top5, "mean_rank": mean_rank, "std_rank": std_rank})
    return out.sort_values(["freq_top5", "mean_rank"], ascending=[False, True])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cox selection and RSF hierarchical analysis")
    parser.add_argument("--mode", choices=["cox", "rsf"], default="rsf", help="Which analysis to run")
    parser.add_argument("--out", default="rsf_outputs", help="Output directory for RSF mode")
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--min_samples_leaf", type=int, default=3)
    parser.add_argument("--max_features", default="sqrt")
    parser.add_argument("--high_level_depth", type=int, default=2)
    parser.add_argument("--n_perm", type=int, default=30)
    parser.add_argument("--eval_time", type=float, default=None, help="Fixed time horizon; default = median duration")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_subgroups", action="store_true", help="Disable subgroup RSF")
    parser.add_argument("--topk", type=int, default=10, help="Top-K features to plot PDP")
    parser.add_argument("--bootstrap", type=int, default=0, help="Number of bootstrap runs for stability (0 to skip)")
    args = parser.parse_args()

    # Load data
    df = load_dataframes()
    feature_cols = [
        c for c in df.columns if c not in {"PE_Time", "VT/VF/SCD", "Female", "ICD"}
    ]

    if args.mode == "cox":
        fit_cox_four_groups_select_plot(
            df,
            feature_cols=feature_cols,
            alpha=0.05,
            penalizer=0.2,
            criterion="aic",
            seeds=list(range(40)),
            threshold=0.35,
            max_features=10,
            epv=None,
        )
    else:
        outputs = run_rsf_full_pipeline(
            df,
            feature_cols=feature_cols,
            out_dir=args.out,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            high_level_depth=args.high_level_depth,
            n_perm_repeats=args.n_perm,
            eval_time=args.eval_time,
            random_state=args.seed,
            do_subgroups=not args.no_subgroups,
            top_k_plot=args.topk,
        )
        if args.bootstrap and args.bootstrap > 0:
            stab = bootstrap_rsf_stability(
                df,
                feature_cols,
                n_bootstrap=args.bootstrap,
                random_state=args.seed,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
                max_features=args.max_features,
                high_level_depth=args.high_level_depth,
                n_perm_repeats=max(10, args.n_perm // 2),
                eval_time=outputs["global"]["eval_time"],  # type: ignore
            )
            os.makedirs(args.out, exist_ok=True)
            stab.to_csv(os.path.join(args.out, "stability_bootstrap.csv"))
