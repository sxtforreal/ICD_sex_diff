# === rcs_cox_sex_thresholds.py ===
# 功能：
# 1) Cox + RCS 拟合 LGE->VT/VF 非线性 + 性别交互（可选协变量）
# 2) 绘制男女 HR–LGE 曲线（含95%CI）
# 3) 给出两种性别特异的“阈值”候选：曲线法 & 三分法（AIC最佳）

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from patsy import dmatrix

from cox import load_dataframes


# ---------------------- 数据准备 ----------------------
def prepare_df_for_model(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) NYHA Class -> 标准 1–4，0->1，5->4；转 float64
    if "NYHA Class" in df.columns:
        nyha = df["NYHA Class"].astype("string").str.strip().str.upper()
        MAP = {
            "0": 1,
            "I": 1,
            "1": 1,
            "II": 2,
            "2": 2,
            "III": 3,
            "3": 3,
            "IV": 4,
            "4": 4,
            "5": 4,
        }
        df["NYHA Class"] = nyha.map(MAP)
        df["NYHA Class"] = pd.to_numeric(df["NYHA Class"], errors="coerce").astype(
            "float64"
        )

    # 2) 可空扩展数值列统一为 float64（消灭 pd.NA）
    nullable_mask = df.dtypes.astype(str).str.contains(
        r"^(Int64|Float64|boolean)$", case=False
    )
    nullable_cols = df.columns[nullable_mask].tolist()
    if nullable_cols:
        df[nullable_cols] = df[nullable_cols].apply(
            lambda s: pd.to_numeric(s, errors="coerce").astype("float64")
        )

    return df


# ---------------------- 基础构件 ----------------------
def build_rcs_basis(x, df=4, knots=None):
    if knots is not None:
        knots = np.sort(np.array(knots, dtype=float))
        dm = dmatrix(
            "cr(x, knots=knots, constraints='center') - 1",
            {"x": np.asarray(x, dtype=float), "knots": knots},
            return_type="dataframe",
        )
    else:
        dm = dmatrix(
            f"cr(x, df={df}, constraints='center') - 1",
            {"x": np.asarray(x, dtype=float)},
            return_type="dataframe",
        )
    dm.columns = [f"rcs_{i+1}" for i in range(dm.shape[1])]
    return dm


def add_interactions(X_rcs, sex_binary):
    X_int = X_rcs.mul(sex_binary.values.reshape(-1, 1))
    X_int.columns = [c + ":sex" for c in X_rcs.columns]
    return X_int


def _numeric_covariates(df_cov: pd.DataFrame) -> pd.DataFrame:
    if df_cov is None or df_cov.empty:
        return pd.DataFrame(index=df_cov.index if df_cov is not None else None)
    num = df_cov.select_dtypes(include=["number"])
    boo = df_cov.select_dtypes(include=["bool"]).astype(int)
    other = df_cov.drop(columns=list(num.columns) + list(boo.columns), errors="ignore")
    if not other.empty:
        other = pd.get_dummies(other, drop_first=True, dummy_na=False)
    out = pd.concat([num, boo, other], axis=1)
    out = out.apply(pd.to_numeric, errors="coerce")
    return out


def _clean_design_matrix(dfX, verbose=True):
    X = dfX.copy().apply(pd.to_numeric, errors="coerce")
    arr = X.to_numpy(dtype="float64", na_value=np.nan)

    bad_rows = ~np.isfinite(arr).all(axis=1)
    if verbose and bad_rows.sum() > 0:
        print(f"[clean] drop rows with non-finite values: {bad_rows.sum()}")
        bad_cols = X.columns[~np.isfinite(arr).all(axis=0)].tolist()
        print("[clean] columns with non-finite after coercion:", bad_cols[:20])
    X = X.loc[~bad_rows]
    arr = arr[~bad_rows]

    nunq = X.nunique(dropna=False)
    const_cols = nunq[nunq <= 1].index.tolist()
    if const_cols:
        print(f"[clean] drop constant/near-constant cols: {const_cols}")
        X = X.drop(columns=const_cols)
        arr = X.to_numpy(dtype="float64", na_value=np.nan)

    dup_cols, seen = [], {}
    for j, c in enumerate(X.columns):
        key = tuple(np.round(arr[:, j], 12))
        if key in seen:
            dup_cols.append(c)
        else:
            seen[key] = c
    if dup_cols:
        print(f"[clean] drop duplicate cols: {dup_cols}")
        X = X.drop(columns=dup_cols)

    return X


# ---------------------- 拟合 & 预测 ----------------------
def fit_cox_rcs_interaction(
    df,
    time_col,
    event_col,
    lge_col,
    sex_col,
    covariates=None,
    df_rcs=4,
    knots=None,
    use_penalizer=True,
    penalizer=0.1,
):
    dat = df[[time_col, event_col, lge_col, sex_col] + (covariates or [])].copy()
    dat = dat.replace([np.inf, -np.inf], np.nan)
    dat = dat.dropna(subset=[time_col, event_col, lge_col, sex_col])
    dat = dat[dat[time_col] > 0].copy()
    dat[event_col] = (dat[event_col].astype(int) > 0).astype(int)

    # Female=1, Male=0
    dat["Sex_bin"] = dat[sex_col].astype(int)

    print("\n=== Sanity check: counts by sex (Sex_bin: 0=Male, 1=Female) ===")
    cnt = dat.groupby("Sex_bin")[event_col].agg(n="count", events="sum")
    print(cnt)

    X_rcs = build_rcs_basis(dat[lge_col].values, df=df_rcs, knots=knots)
    X_int = add_interactions(X_rcs, dat["Sex_bin"])
    X = pd.concat([X_rcs, dat[["Sex_bin"]], X_int], axis=1)

    if covariates:
        cov_num = _numeric_covariates(dat[covariates])
        X = pd.concat([X, cov_num], axis=1)

    X_clean = _clean_design_matrix(X)
    fit_df = pd.concat([dat[[time_col, event_col]].loc[X_clean.index], X_clean], axis=1)

    cph = CoxPHFitter(penalizer=(penalizer if use_penalizer else 0.0))
    cph.fit(
        fit_df,
        duration_col=time_col,
        event_col=event_col,
        show_progress=False,
        robust=True,
    )
    return cph, X_clean.columns.tolist(), fit_df


def make_lge_grid(x, n=200):
    x = np.asarray(pd.Series(x).dropna().astype(float).values)
    if x.size == 0:
        raise ValueError("No valid LGE values to build grid.")
    q1, q99 = np.percentile(x, [1, 99])
    if not (np.isfinite(q1) and np.isfinite(q99)):
        q1, q99 = np.nanmin(x), np.nanmax(x)
    lo, hi = float(q1), float(q99)
    if lo >= hi:
        span = max(1.0, abs(lo) * 0.1, 0.1)
        lo, hi = lo - span / 2, hi + span / 2
    return np.linspace(lo, hi, n)


def hr_curve_with_ci(
    cph,
    feature_names,
    lge_grid,
    sex_value,
    ref_value=None,
    df_rcs=4,
    knots=None,
    covariates_means=None,
):
    sex_bin = int(sex_value)

    basis_all = build_rcs_basis(
        np.asarray(lge_grid, dtype=float), df=df_rcs, knots=knots
    )
    inter_all = basis_all.mul(sex_bin)
    inter_all.columns = [c + ":sex" for c in basis_all.columns]

    Xg = pd.concat(
        [basis_all, pd.Series([sex_bin] * len(basis_all), name="Sex_bin"), inter_all],
        axis=1,
    )

    extra_feats = [f for f in feature_names if f not in Xg.columns]
    if extra_feats:
        if covariates_means is None:
            covariates_means = {f: 0.0 for f in extra_feats}
        for f in extra_feats:
            Xg[f] = covariates_means.get(f, 0.0)

    beta = cph.params_.reindex(feature_names).values.reshape(-1, 1)
    cov = cph.variance_matrix_.loc[feature_names, feature_names].values

    M = Xg[feature_names].to_numpy()
    eta = M @ beta
    var_eta = np.einsum("ij,jk,ik->i", M, cov, M)

    if ref_value is None:
        ref_value = float(np.median(lge_grid))
    ref_idx = int(np.argmin(np.abs(np.asarray(lge_grid, float) - ref_value)))
    Xref = M[ref_idx : ref_idx + 1, :]

    eta_ref = float(Xref @ beta)
    var_eta_ref = float(Xref @ cov @ Xref.T)
    cov_eta_ref = (M @ cov @ Xref.T).reshape(-1)

    eta_rel = eta.reshape(-1) - eta_ref
    var_eta_rel = var_eta + var_eta_ref - 2 * cov_eta_ref
    var_eta_rel = np.maximum(var_eta_rel, 1e-12)

    hr = np.exp(eta_rel)
    se = np.sqrt(var_eta_rel)
    ci_lo = np.exp(eta_rel - 1.96 * se)
    ci_hi = np.exp(eta_rel + 1.96 * se)
    return hr, ci_lo, ci_hi, ref_value


# ---------------------- 阈值与绘图 ----------------------
def suggest_thresholds_from_curve(lge_grid, hr, ci_lo, ci_hi, min_hr=1.2, smooth=7):
    def moving_avg(a, w):
        if w <= 1:
            return a
        return np.convolve(a, np.ones(w) / w, mode="same")

    hr_s = moving_avg(hr, smooth)
    lo_s = moving_avg(ci_lo, smooth)
    d1 = np.gradient(hr_s, lge_grid)
    d2 = np.gradient(d1, lge_grid)
    valid = (hr_s > min_hr) & (lo_s > 1.0)
    candidates = []
    if np.any(valid):
        slope_thr = np.nanpercentile(d1[valid], 75)
        curve_thr = np.nanpercentile(d2[valid], 75)
        idxs = np.where(valid & (d1 >= slope_thr) & (d2 >= curve_thr))[0]
        for idx in idxs:
            if (
                not candidates
                or abs(lge_grid[idx] - candidates[-1])
                > (lge_grid.max() - lge_grid.min()) * 0.05
            ):
                candidates.append(lge_grid[idx])
            if len(candidates) >= 2:
                break
    return candidates


def aic_of_three_groups(df, time_col, event_col, cut1, cut2, lge_col, covariates=None):
    tmp = df[[time_col, event_col, lge_col] + (covariates or [])].dropna().copy()
    tmp["LGE_3g"] = pd.cut(
        tmp[lge_col], bins=[-np.inf, cut1, cut2, np.inf], labels=["low", "mid", "high"]
    )
    grp = pd.get_dummies(tmp["LGE_3g"], drop_first=True)  # mid, high
    X = grp
    if covariates:
        X = pd.concat([X, tmp[covariates]], axis=1)
    cph = CoxPHFitter()
    cph.fit(
        pd.concat([tmp[[time_col, event_col]], X], axis=1),
        duration_col=time_col,
        event_col=event_col,
        show_progress=False,
    )
    return cph.AIC_


def three_cut_grid_search_by_sex(
    df,
    time_col,
    event_col,
    lge_col,
    sex_col,
    covariates=None,
    candidate_q=(20, 30, 40, 50, 60, 70, 80),
):
    out = {}
    dat = (
        df[[time_col, event_col, lge_col, sex_col] + (covariates or [])].dropna().copy()
    )
    dat[event_col] = (dat[event_col].astype(int) > 0).astype(int)
    dat["Sex_bin"] = dat[sex_col].astype(int)

    for sex in [0, 1]:
        sub = dat[dat["Sex_bin"] == sex]
        if sub.empty or sub[event_col].sum() == 0:
            out[sex] = (np.nan, np.nan, np.nan)
            continue
        qs = np.percentile(sub[lge_col].values, candidate_q)
        best = (np.nan, np.nan, np.inf)
        for i in range(len(qs)):
            for j in range(i + 1, len(qs)):
                c1, c2 = qs[i], qs[j]
                try:
                    aic = aic_of_three_groups(
                        sub, time_col, event_col, c1, c2, lge_col, covariates
                    )
                    if aic < best[2]:
                        best = (float(c1), float(c2), float(aic))
                except Exception:
                    continue
        out[sex] = best
    return out


def plot_hr_curves(
    lge_grid,
    male_curve,
    female_curve,
    male_ci,
    female_ci,
    male_thresh=None,
    female_thresh=None,
    title="HR vs LGE (RCS + Sex Interaction)",
):
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(lge_grid, male_curve, label="Male HR")
    plt.fill_between(lge_grid, male_ci[0], male_ci[1], alpha=0.25)
    plt.plot(lge_grid, female_curve, label="Female HR")
    plt.fill_between(lge_grid, female_ci[0], female_ci[1], alpha=0.25)
    plt.axhline(1.0, linestyle="--", linewidth=1)

    def add_vlines(ths, lab_prefix):
        if not ths:
            return
        for k, t in enumerate(ths, 1):
            plt.axvline(t, linestyle=":", linewidth=1.5)
            plt.plot([], [], label=f"{lab_prefix} thr{k}: {t:.2f}")

    add_vlines(male_thresh, "Male")
    add_vlines(female_thresh, "Female")
    plt.xlabel("LGE Burden")
    plt.ylabel("Hazard Ratio (relative to reference)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_dataframes()
    df.drop(
        columns=["Age by decade", "CrCl>45", "NYHA>2", "Significant LGE"], inplace=True
    )
    df = prepare_df_for_model(df)
    time_col = "PE_Time"
    event_col = "VT/VF/SCD"
    lge_col = "LGE Burden 5SD"
    sex_col = "Female"

    covariates = [
        "Age at CMR",
        "BMI",
        "DM",
        "HTN",
        "HLP",
        "AF",
        "NYHA Class",
        "LVEDVi",
        "LVEF",
        "LV Mass Index",
        "RVEDVi",
        "RVEF",
        "LA EF",
        "LAVi",
        "MRF (%)",
        "Sphericity Index",
        "Relative Wall Thickness",
        "MV Annular Diameter",
        "Beta Blocker",
        "ACEi/ARB/ARNi",
        "Aldosterone Antagonist",
        "AAD",
        "CRT",
        "QRS",
        "QTc",
        "Cockcroft-Gault Creatinine Clearance (mL/min)",
        "ICD",
    ]

    # 2) 拟合 Cox + RCS + 性别交互
    #    你可以调 df_rcs 或指定 knots（例如按 5/35/65/95 分位）
    df_rcs = 4
    knots = None  # 例如：np.percentile(df[lge_col].dropna(), [5, 35, 65, 95])[1:-1]  # patsy::cr 的 knots 是“内部”结点

    cph, feat_names, fit_df = fit_cox_rcs_interaction(
        df,
        time_col,
        event_col,
        lge_col,
        sex_col,
        covariates=covariates,
        df_rcs=df_rcs,
        knots=knots,
    )
    print("\n=== Cox model summary ===")
    print(cph.summary)

    # 3) 画 HR–LGE 曲线（男/女各一条）
    #    网格范围建议覆盖 LGE 的 1~99 分位
    q1, q99 = np.percentile(df[lge_col].dropna(), [1, 99])
    lge_grid = np.linspace(q1, q99, 200)

    # 协变量在绘图时的固定值（平均/众数）；若不指定，默认 0
    cov_means = None
    if covariates:
        cov_means = {}
        for c in covariates:
            if df[c].dtype.kind in "ifu":
                cov_means[c] = float(np.nanmean(df[c]))
            else:
                # 类别/布尔，取众数或0
                try:
                    cov_means[c] = df[c].mode(dropna=True).iat[0]
                except Exception:
                    cov_means[c] = 0.0

    # 男性曲线
    m_hr, m_lo, m_hi, ref_val_m = hr_curve_with_ci(
        cph,
        feat_names,
        lge_grid,
        sex_value=1,
        ref_value=None,
        df_rcs=df_rcs,
        knots=knots,
        covariates_means=cov_means,
    )
    # 女性曲线
    f_hr, f_lo, f_hi, ref_val_f = hr_curve_with_ci(
        cph,
        feat_names,
        lge_grid,
        sex_value=0,
        ref_value=None,
        df_rcs=df_rcs,
        knots=knots,
        covariates_means=cov_means,
    )

    # 4) 曲线法：给出候选“阈值”（最多两个）
    male_thr_curve = suggest_thresholds_from_curve(
        lge_grid, m_hr, m_lo, m_hi, min_hr=1.2, smooth=7
    )
    female_thr_curve = suggest_thresholds_from_curve(
        lge_grid, f_hr, f_lo, f_hi, min_hr=1.2, smooth=7
    )

    print("\n=== Curve-based candidate thresholds ===")
    print(
        f"Male   (ref={ref_val_m:.2f}): {['{:.2f}'.format(t) for t in male_thr_curve]}"
    )
    print(
        f"Female (ref={ref_val_f:.2f}): {['{:.2f}'.format(t) for t in female_thr_curve]}"
    )

    # 5) 三分法：按性别做两切点网格搜索（AIC最优）
    #    默认候选分位：20,30,40,50,60,70,80（可按需加密）
    tri_best = three_cut_grid_search_by_sex(
        df,
        time_col,
        event_col,
        lge_col,
        sex_col,
        covariates=covariates,
        candidate_q=(20, 30, 40, 50, 60, 70, 80),
    )
    print("\n=== Three-group AIC-best thresholds ===")
    for sex, (c1, c2, aic) in tri_best.items():
        tag = "Male" if sex == 1 else "Female"
        if np.isfinite(aic):
            print(f"{tag}: cut1={c1:.2f}, cut2={c2:.2f}, AIC={aic:.2f}")
        else:
            print(f"{tag}: insufficient data for grid search.")

    # 6) 绘图（带阈值竖线，图例保留两位小数）
    male_lines = male_thr_curve or [
        tri_best.get(1, (np.nan, np.nan, np.nan))[0],
        tri_best.get(1, (np.nan, np.nan, np.nan))[1],
    ]
    male_lines = [t for t in male_lines if t is not None and np.isfinite(t)]
    female_lines = female_thr_curve or [
        tri_best.get(0, (np.nan, np.nan, np.nan))[0],
        tri_best.get(0, (np.nan, np.nan, np.nan))[1],
    ]
    female_lines = [t for t in female_lines if t is not None and np.isfinite(t)]

    plot_hr_curves(
        lge_grid,
        male_curve=m_hr,
        female_curve=f_hr,
        male_ci=(m_lo, m_hi),
        female_ci=(f_lo, f_hi),
        male_thresh=male_lines[:2],
        female_thresh=female_lines[:2],
        title="HR vs LGE by Sex (RCS interaction)",
    )

    print("\nDone.")
