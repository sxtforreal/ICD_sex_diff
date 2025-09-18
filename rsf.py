import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

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


if __name__ == "__main__":
    # Load and prepare data
    df = load_dataframes()
    # Fit cox models for male and female
    res = fit_cox_four_groups_select_plot(
        df,
        feature_cols=[
            c for c in df.columns if c not in {"PE_Time", "VT/VF/SCD", "Female", "ICD"}
        ],
        alpha=0.05,
        penalizer=0.2,
        criterion="aic",
        seeds=list(range(40)),
        threshold=0.35,
        max_features=10,
        epv=None,
    )
