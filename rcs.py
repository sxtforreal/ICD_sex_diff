import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from patsy import dmatrix
from scipy.stats import chi2

from cox import load_dataframes


# ---------------------- 数据准备 ----------------------
def prepare_df_for_model(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 统一 NYHA
    if "NYHA Class" in df.columns:
        nyha = (
            df["NYHA Class"]
            .astype("string")
            .str.replace(r"\s+", "", regex=True)
            .str.upper()
        )
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
        }
        df["NYHA Class"] = pd.to_numeric(nyha.map(MAP), errors="coerce").astype(
            "float64"
        )

    # 可空整型/浮点/布尔 -> float64
    nullable_mask = df.dtypes.astype(str).str.contains(
        r"^(Int64|Float64|boolean)$", case=False
    )
    nullable_cols = df.columns[nullable_mask].tolist()
    if nullable_cols:
        df[nullable_cols] = df[nullable_cols].apply(
            lambda s: pd.to_numeric(s, errors="coerce").astype("float64")
        )
    return df


def _dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 保留第一次出现的同名列
    return df.loc[:, ~df.columns.duplicated()].copy()


# ---------------------- RCS 相关 ----------------------
def build_rcs_basis(x, df_rcs=4, knots=None, lower_bound=None, upper_bound=None):
    """
    自然三次样条基函数（不含截距）。注意：df_rcs 是样条的自由度（基函数数量），
    若提供 knots（仅 interior knots），patsy 会忽略 df 参数。
    """
    x = np.asarray(x, dtype=float)
    if knots is not None:
        knots = np.sort(np.array(knots, dtype=float))
        dm = dmatrix(
            "cr(x, knots=knots, lower_bound=lb, upper_bound=ub) - 1",
            {"x": x, "knots": knots, "lb": lower_bound, "ub": upper_bound},
            return_type="dataframe",
        )
    else:
        dm = dmatrix(
            "cr(x, df=df_, lower_bound=lb, upper_bound=ub) - 1",
            {"x": x, "df_": int(df_rcs), "lb": lower_bound, "ub": upper_bound},
            return_type="dataframe",
        )
    dm.columns = [f"rcs_{i+1}" for i in range(dm.shape[1])]
    return dm


def choose_rcs_spec_from_quantiles(
    series, df_rcs=4, lb_q=0.0, ub_q=0.95, interior_q=None
):
    """
    用分位数选择 interior knots 和边界（Harrell 建议的思路）：
    - 边界默认 5% / 95% 分位
    - interior knots：在 [lb, ub] 上按分位“等距”布点（数量 = df_rcs - 2）
      若提供 interior_q（列表，取值在 (0,1)），则按给定分位数取。
      例如 df_rcs=4 -> 2 个 interior knots -> 建议在 35% 与 65%。
    """
    s = pd.to_numeric(pd.Series(series).dropna(), errors="coerce")
    lb = float(s.quantile(lb_q))
    ub = float(s.quantile(ub_q))
    if not np.isfinite(lb) or not np.isfinite(ub) or ub <= lb:
        # 兜底：给一个最小跨度
        m = float(np.nanmean(s))
        span = max(1.0, abs(m) * 0.1, 0.1)
        lb, ub = m - span, m + span

    n_interior = max(0, int(df_rcs) - 2)
    if interior_q is not None:
        qs = list(interior_q)
    else:
        if n_interior == 0:
            qs = []
        else:
            # 在 [lb_q, ub_q] 之间“等距”取 n_interior 个分位（不含两端）
            qs = np.linspace(lb_q, ub_q, n_interior + 2)[1:-1].tolist()
            # 特例修正：df=4 时更贴近惯例 35%/65%
            if df_rcs == 4:
                qs = [0.35, 0.65]

    ks = [float(s.quantile(q)) for q in qs]
    # 保证 knots 落在边界内部
    eps = 1e-10
    ks = [min(max(k, lb + eps), ub - eps) for k in ks]
    return ks, lb, ub


def add_interactions(X_rcs: pd.DataFrame, sex_series, sex_col="Female"):
    # 保证性别是 0/1
    sex_bin = pd.Series(sex_series).astype(float)
    # 若不是 0/1，尝试阈值化：>0 视为 1
    if not set(pd.unique(sex_bin.dropna())).issubset({0.0, 1.0}):
        sex_bin = (sex_bin > 0).astype(float)
    X_int = X_rcs.mul(sex_bin.values.reshape(-1, 1))
    X_int.columns = [c + f":{sex_col}" for c in X_rcs.columns]
    return X_int, sex_bin.astype(int)


# ---------------------- 数据清理 ----------------------
def _numeric_covariates(df_cov: pd.DataFrame) -> pd.DataFrame:
    if df_cov is None or df_cov.empty:
        return pd.DataFrame(index=df_cov.index if df_cov is not None else None)
    num = df_cov.select_dtypes(include=["number"])
    boo = df_cov.select_dtypes(include=["bool"]).astype(int)
    other = df_cov.drop(columns=list(num.columns) + list(boo.columns), errors="ignore")
    if not other.empty:
        other = pd.get_dummies(other, drop_first=True, dummy_na=False)
    X = pd.concat([num, boo, other], axis=1).apply(pd.to_numeric, errors="coerce")
    # 去除常量列
    nunique = X.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        X = X.drop(columns=const_cols)
    return X


def _clean_design_matrix(dfX, verbose=True):
    X = dfX.copy().apply(pd.to_numeric, errors="coerce")
    arr = X.to_numpy(dtype="float64", na_value=np.nan)
    bad_rows = ~np.isfinite(arr).all(axis=1)
    if verbose and bad_rows.sum() > 0:
        print(f"[clean] drop rows with non-finite values: {bad_rows.sum()}")
    X = X.loc[~bad_rows]
    # 再次去掉常量列
    nunique = X.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        if verbose:
            print(f"[clean] drop constant cols: {const_cols}")
        X = X.drop(columns=const_cols)
    return X


def _drop_separation_features(X: pd.DataFrame, y: pd.Series, verbose=True):
    yy = y.astype(bool).values
    to_drop = []
    for col in X.columns:
        v1 = float(np.nanvar(X.loc[yy, col].values)) if yy.any() else 0.0
        v0 = float(np.nanvar(X.loc[~yy, col].values)) if (~yy).any() else 0.0
        if v1 < 1e-10 or v0 < 1e-10:
            to_drop.append(col)
    if to_drop and verbose:
        print(f"[clean] drop potential-separation cols: {to_drop}")
    X2 = X.drop(columns=to_drop, errors="ignore")
    if X2.empty:
        raise ValueError(
            "All features dropped by separation filter; relax threshold or add covariates."
        )
    return X2


# ---------------------- 拟合 & 诊断 ----------------------
def fit_cox_rcs_interaction(
    df,
    time_col,
    event_col,
    x_col,
    sex_col,
    covariates=None,
    df_rcs=4,
    knots=None,
    lower_bound=None,
    upper_bound=None,
    use_penalizer=True,
    penalizer=0.1,
):
    dat = df[[time_col, event_col, x_col, sex_col] + (covariates or [])].copy()
    dat = dat.dropna(subset=[time_col, event_col, x_col, sex_col])
    dat = dat[dat[time_col] > 0].copy()
    dat[event_col] = dat[event_col].astype(int)

    print("\n=== Sanity check: counts by sex ===")
    print(dat.groupby(sex_col)[event_col].agg(n="count", events="sum"))

    # knots/bounds
    if knots is None or lower_bound is None or upper_bound is None:
        ks, lb, ub = choose_rcs_spec_from_quantiles(dat[x_col], df_rcs=df_rcs)
    else:
        ks, lb, ub = knots, lower_bound, upper_bound
    print(f"[rcs] knots={ks}, bounds=({lb:.2f},{ub:.2f})")

    # 设计矩阵
    X_rcs = build_rcs_basis(
        dat[x_col].values, df_rcs=df_rcs, knots=ks, lower_bound=lb, upper_bound=ub
    )
    X_int, sex_bin = add_interactions(X_rcs, dat[sex_col], sex_col)
    X = pd.concat([X_rcs, sex_bin.rename(sex_col), X_int], axis=1)
    if covariates:
        X = pd.concat([X, _numeric_covariates(dat[covariates])], axis=1)

    X_tmp = _clean_design_matrix(X)
    y_tmp = dat.loc[X_tmp.index, event_col]
    X_clean = _drop_separation_features(X_tmp, y_tmp)

    fit_df = pd.concat([dat[[time_col, event_col]].loc[X_clean.index], X_clean], axis=1)

    last_err = None
    for pen in [penalizer, 1.0, 5.0] if use_penalizer else [0.0]:
        try:
            cph = CoxPHFitter(penalizer=pen)
            cph.fit(
                fit_df,
                duration_col=time_col,
                event_col=event_col,
                robust=True,
                show_progress=False,
            )
            return (
                cph,
                X_clean.columns.tolist(),
                fit_df,
                {"knots": ks, "lb": lb, "ub": ub},
            )
        except Exception as e:
            last_err = e
            continue
    raise last_err


def test_nonlinearity_LRT(
    df, time_col, event_col, x_col, covariates, df_rcs, ks, lb, ub, penalizer=0.1
):
    # 基础表
    base = df[[time_col, event_col, x_col]].copy()
    if covariates:
        base = pd.concat([base, _numeric_covariates(df[covariates])], axis=1)
    base = base.dropna()
    base = _dedup_columns(base)

    # 线性
    lin_df = base.rename(columns={x_col: "x"})
    lin_df[event_col] = lin_df[event_col].astype(int)
    cph_lin = CoxPHFitter(penalizer=penalizer)
    cph_lin.fit(lin_df, duration_col=time_col, event_col=event_col, robust=True)

    # RCS —— 关键：对齐索引
    rcs = build_rcs_basis(
        lin_df["x"].values, df_rcs=df_rcs, knots=ks, lower_bound=lb, upper_bound=ub
    )
    rcs.index = lin_df.index  # ★★★ 对齐到 lin_df 的 index

    others = lin_df.drop(columns=["x", time_col, event_col])
    rcs_df = pd.concat([lin_df[[time_col, event_col]], rcs, others], axis=1)
    rcs_df = _dedup_columns(rcs_df)  # 保险

    # 再跑 RCS 模型
    cph_rcs = CoxPHFitter(penalizer=penalizer)
    cph_rcs.fit(rcs_df, duration_col=time_col, event_col=event_col, robust=True)

    # LRT
    from scipy.stats import chi2

    LL_lin, LL_rcs = cph_lin.log_likelihood_, cph_rcs.log_likelihood_
    df_nl = rcs.shape[1] - 1
    stat = 2 * (LL_rcs - LL_lin)
    p = 1 - chi2.cdf(stat, df=df_nl)
    print(f"[LRT nonlinearity] chi2={stat:.2f}, df={df_nl}, p={p:.4g}")
    return stat, p


# ---------------------- 预测曲线 ----------------------
def hr_curve_with_ci(
    cph,
    feature_names,
    x_grid,
    sex_value,
    ref_value=None,
    df_rcs=4,
    knots=None,
    lower_bound=None,
    upper_bound=None,
    covariates_means=None,
    sex_col="Female",
):
    basis_all = build_rcs_basis(
        x_grid,
        df_rcs=df_rcs,
        knots=knots,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    inter_all = basis_all.mul(float(sex_value))
    inter_all.columns = [c + f":{sex_col}" for c in basis_all.columns]

    Xg = pd.concat(
        [basis_all, pd.Series([sex_value] * len(basis_all), name=sex_col), inter_all],
        axis=1,
    )
    # 对于模型里存在、此处没给值的协变量，填均值（或 0）
    extra_feats = [f for f in feature_names if f not in Xg.columns]
    means = covariates_means or {f: 0.0 for f in extra_feats}
    for f in extra_feats:
        Xg[f] = means.get(f, 0.0)

    valid_features = [f for f in feature_names if f in cph.params_.index]
    beta = cph.params_.reindex(valid_features).values.reshape(-1, 1)
    cov = cph.variance_matrix_.loc[valid_features, valid_features].values
    M = Xg[valid_features].to_numpy()

    eta = (M @ beta).reshape(-1)
    var_eta = np.maximum(np.einsum("ij,jk,ik->i", M, cov, M), 0.0)

    if ref_value is None:
        ref_value = float(np.median(x_grid))
    ref_idx = int(np.argmin(np.abs(x_grid - ref_value)))
    Xref = M[ref_idx : ref_idx + 1, :]
    eta_ref = float(Xref @ beta)
    var_eta_ref = float(Xref @ cov @ Xref.T)
    cov_eta_ref = (M @ cov @ Xref.T).reshape(-1)

    eta_rel = eta - eta_ref
    var_eta_rel = np.maximum(var_eta + var_eta_ref - 2 * cov_eta_ref, 1e-12)
    se = np.sqrt(var_eta_rel)

    hr = np.exp(eta_rel)
    lo = np.exp(eta_rel - 1.96 * se)
    hi = np.exp(eta_rel + 1.96 * se)
    return hr, lo, hi, ref_value


# ---------------------- 阈值与绘图 ----------------------
def suggest_thresholds_from_curve(
    x_grid, hr, ci_lo, ci_hi, min_hr=1.2, smooth=7, max_n=2
):
    """
    返回最多两个“高风险起点”阈值：HR 平滑后 > min_hr 且下界 > 1 的区间的起点。
    这便于 High/Med/Low 的三段式解释。
    """

    def moving_avg(a, w):
        return np.convolve(a, np.ones(w) / w, mode="same") if w > 1 else a

    hr_s = moving_avg(hr, smooth)
    lo_s = moving_avg(ci_lo, smooth)
    valid = (hr_s > min_hr) & (lo_s > 1.0)

    # 找到 valid 的“岛屿”起点
    starts = []
    prev = False
    for i, v in enumerate(valid):
        if v and not prev:
            starts.append(i)
        prev = v
    ths = [x_grid[i] for i in starts][:max_n]
    return ths


def plot_hr_curves(
    x_grid,
    male_curve,
    female_curve,
    male_ci,
    female_ci,
    male_thresh=None,
    female_thresh=None,
    title="HR vs Marker (RCS + Female interaction)",
    xlabel="Marker",
):
    plt.figure(figsize=(7.8, 5.0))
    plt.plot(x_grid, male_curve, label="Male HR")
    plt.fill_between(x_grid, male_ci[0], male_ci[1], alpha=0.25)
    plt.plot(x_grid, female_curve, label="Female HR")
    plt.fill_between(x_grid, female_ci[0], female_ci[1], alpha=0.25)
    plt.axhline(1.0, linestyle="--", linewidth=1)

    def add_vlines(ths, lab):
        for t in ths or []:
            plt.axvline(t, ls=":", lw=1.5)
            plt.plot([], [], label=f"{lab} thr: {t:.2f}")  # 2 位小数（你的偏好）

    add_vlines(male_thresh, "Male")
    add_vlines(female_thresh, "Female")
    plt.xlabel(xlabel)
    plt.ylabel("Hazard Ratio")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------- main ----------------------
if __name__ == "__main__":
    df = load_dataframes()
    df = prepare_df_for_model(df)

    time_col = "PE_Time"
    event_col = "VT/VF/SCD"
    x_col = "LGE Burden 5SD"
    sex_col = "Female"
    covariates = [
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
    ]

    # knots / bounds：默认 5%~95%，df=4 -> interior 在 35%/65%
    ks, lb, ub = choose_rcs_spec_from_quantiles(df[x_col], df_rcs=4)

    cph, feat_names, fit_df, spec = fit_cox_rcs_interaction(
        df,
        time_col,
        event_col,
        x_col,
        sex_col,
        covariates,
        df_rcs=4,
        knots=ks,
        lower_bound=lb,
        upper_bound=ub,
    )
    print(cph.summary)

    # 非线性检验（可选）
    _ = test_nonlinearity_LRT(
        df.dropna(subset=[time_col, event_col, x_col]),
        time_col,
        event_col,
        x_col,
        covariates,
        df_rcs=4,
        ks=spec["knots"],
        lb=spec["lb"],
        ub=spec["ub"],
    )

    grid = np.linspace(spec["lb"], spec["ub"], 200)

    # 用训练样本均值填充其他协变量（若有）
    cov_means = (
        fit_df[
            [
                c
                for c in feat_names
                if c not in (sex_col,)
                and not c.startswith("rcs_")
                and f"{c}:{sex_col}" not in feat_names
            ]
        ]
        .mean()
        .to_dict()
    )

    m_hr, m_lo, m_hi, ref_m = hr_curve_with_ci(
        cph,
        feat_names,
        grid,
        sex_value=0,
        df_rcs=4,
        knots=spec["knots"],
        lower_bound=spec["lb"],
        upper_bound=spec["ub"],
        covariates_means=cov_means,
        sex_col=sex_col,
    )
    f_hr, f_lo, f_hi, ref_f = hr_curve_with_ci(
        cph,
        feat_names,
        grid,
        sex_value=1,
        df_rcs=4,
        knots=spec["knots"],
        lower_bound=spec["lb"],
        upper_bound=spec["ub"],
        covariates_means=cov_means,
        sex_col=sex_col,
    )

    male_thr = suggest_thresholds_from_curve(grid, m_hr, m_lo, m_hi, max_n=2)
    female_thr = suggest_thresholds_from_curve(grid, f_hr, f_lo, f_hi, max_n=2)

    plot_hr_curves(
        grid,
        m_hr,
        f_hr,
        (m_lo, m_hi),
        (f_lo, f_hi),
        male_thresh=male_thr,
        female_thresh=female_thr,
        title="HR vs LGE (RCS + Female interaction)",
        xlabel="LGE Burden",
    )
