import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from patsy import dmatrix

from cox import load_dataframes


# ---------------------- 数据准备 ----------------------
def prepare_df_for_model(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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
        df["NYHA Class"] = pd.to_numeric(nyha.map(MAP), errors="coerce").astype(
            "float64"
        )
    # 可空整型/布尔转 float64
    nullable_mask = df.dtypes.astype(str).str.contains(
        r"^(Int64|Float64|boolean)$", case=False
    )
    nullable_cols = df.columns[nullable_mask].tolist()
    if nullable_cols:
        df[nullable_cols] = df[nullable_cols].apply(
            lambda s: pd.to_numeric(s, errors="coerce").astype("float64")
        )
    return df


# ---------------------- RCS 相关 ----------------------
def build_rcs_basis(x, df=4, knots=None, lower_bound=None, upper_bound=None):
    """自然样条 (Restricted Cubic Spline)，带边界。"""
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
            {"x": x, "df_": int(df), "lb": lower_bound, "ub": upper_bound},
            return_type="dataframe",
        )
    dm.columns = [f"rcs_{i+1}" for i in range(dm.shape[1])]
    return dm


def choose_rcs_spec_from_quantiles(
    series, df_rcs=4, knot_q=(0.35, 0.65), lb_q=0.01, ub_q=0.97
):
    """用分位数选择 knots 和边界。"""
    s = pd.to_numeric(pd.Series(series).dropna(), errors="coerce")
    lb = float(s.quantile(lb_q))
    ub = float(s.quantile(ub_q))
    if ub <= lb:
        span = max(1.0, abs(lb) * 0.1, 0.1)
        lb, ub = lb - span / 2, ub + span / 2
    if df_rcs <= 3:
        ks = [float(s.quantile(0.5))]
    else:
        qs = [0.35, 0.65][: max(1, df_rcs - 2)]
        ks = [float(s.quantile(q)) for q in qs]
    ks = [min(max(k, lb + 1e-8), ub - 1e-8) for k in ks]
    return ks, lb, ub


def add_interactions(X_rcs, sex_series, sex_col="Female"):
    X_int = X_rcs.mul(sex_series.values.reshape(-1, 1))
    X_int.columns = [c + f":{sex_col}" for c in X_rcs.columns]
    return X_int


# ---------------------- 数据清理 ----------------------
def _numeric_covariates(df_cov: pd.DataFrame) -> pd.DataFrame:
    if df_cov is None or df_cov.empty:
        return pd.DataFrame(index=df_cov.index if df_cov is not None else None)
    num = df_cov.select_dtypes(include=["number"])
    boo = df_cov.select_dtypes(include=["bool"]).astype(int)
    other = df_cov.drop(columns=list(num.columns) + list(boo.columns), errors="ignore")
    if not other.empty:
        other = pd.get_dummies(other, drop_first=True, dummy_na=False)
    return pd.concat([num, boo, other], axis=1).apply(pd.to_numeric, errors="coerce")


def _clean_design_matrix(dfX, verbose=True):
    X = dfX.copy().apply(pd.to_numeric, errors="coerce")
    arr = X.to_numpy(dtype="float64", na_value=np.nan)
    bad_rows = ~np.isfinite(arr).all(axis=1)
    if verbose and bad_rows.sum() > 0:
        print(f"[clean] drop rows with non-finite values: {bad_rows.sum()}")
    X = X.loc[~bad_rows]
    return X


def _drop_separation_features(X: pd.DataFrame, y: pd.Series, verbose=True):
    yy = y.astype(bool).values
    to_drop = []
    for col in X.columns:
        v1 = float(np.nanvar(X.loc[yy, col].values))
        v0 = float(np.nanvar(X.loc[~yy, col].values))
        if v1 < 1e-10 or v0 < 1e-10:
            to_drop.append(col)
    if to_drop and verbose:
        print(f"[clean] drop potential-separation cols: {to_drop}")
    return X.drop(columns=to_drop, errors="ignore")


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
    lower_bound=None,
    upper_bound=None,
    use_penalizer=True,
    penalizer=0.1,
):
    dat = df[[time_col, event_col, lge_col, sex_col] + (covariates or [])].copy()
    dat = dat.dropna(subset=[time_col, event_col, lge_col, sex_col])
    dat = dat[dat[time_col] > 0].copy()
    dat[event_col] = dat[event_col].astype(int)

    print("\n=== Sanity check: counts by sex ===")
    print(dat.groupby(sex_col)[event_col].agg(n="count", events="sum"))

    if knots is None or lower_bound is None or upper_bound is None:
        ks, lb, ub = choose_rcs_spec_from_quantiles(dat[lge_col], df_rcs=df_rcs)
    else:
        ks, lb, ub = knots, lower_bound, upper_bound
    print(f"[rcs] knots={ks}, bounds=({lb:.2f},{ub:.2f})")

    X_rcs = build_rcs_basis(
        dat[lge_col].values, df=df_rcs, knots=ks, lower_bound=lb, upper_bound=ub
    )
    X_int = add_interactions(X_rcs, dat[sex_col], sex_col)
    X = pd.concat([X_rcs, dat[[sex_col]], X_int], axis=1)
    if covariates:
        X = pd.concat([X, _numeric_covariates(dat[covariates])], axis=1)

    X_tmp = _clean_design_matrix(X)
    y_tmp = dat.loc[X_tmp.index, event_col]
    X_clean = _drop_separation_features(X_tmp, y_tmp)

    fit_df = pd.concat([dat[[time_col, event_col]].loc[X_clean.index], X_clean], axis=1)

    for pen in [penalizer, 1.0, 5.0] if use_penalizer else [0.0]:
        try:
            cph = CoxPHFitter(penalizer=pen)
            cph.fit(
                fit_df,
                duration_col=time_col,
                event_col=event_col,
                show_progress=False,
                robust=True,
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


def make_lge_grid_from_bounds(lb, ub, n=200):
    return np.linspace(lb, ub, n)


def hr_curve_with_ci(
    cph,
    feature_names,
    lge_grid,
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
        lge_grid,
        df=df_rcs,
        knots=knots,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    inter_all = basis_all.mul(sex_value)
    inter_all.columns = [c + f":{sex_col}" for c in basis_all.columns]

    Xg = pd.concat(
        [basis_all, pd.Series([sex_value] * len(basis_all), name=sex_col), inter_all],
        axis=1,
    )

    extra_feats = [f for f in feature_names if f not in Xg.columns]
    for f in extra_feats:
        Xg[f] = 0.0 if covariates_means is None else covariates_means.get(f, 0.0)

    valid_features = [f for f in feature_names if f in cph.params_.index]
    beta = cph.params_.reindex(valid_features).values.reshape(-1, 1)
    cov = cph.variance_matrix_.loc[valid_features, valid_features].values
    M = Xg[valid_features].to_numpy()
    eta = M @ beta
    var_eta = np.einsum("ij,jk,ik->i", M, cov, M)

    if ref_value is None:
        ref_value = float(np.median(lge_grid))
    ref_idx = int(np.argmin(np.abs(lge_grid - ref_value)))
    Xref = M[ref_idx : ref_idx + 1, :]
    eta_ref = float(Xref @ beta)
    var_eta_ref = float(Xref @ cov @ Xref.T)
    cov_eta_ref = (M @ cov @ Xref.T).reshape(-1)

    eta_rel = eta.reshape(-1) - eta_ref
    var_eta_rel = np.maximum(var_eta + var_eta_ref - 2 * cov_eta_ref, 1e-12)
    hr = np.exp(eta_rel)
    se = np.sqrt(var_eta_rel)
    return hr, np.exp(eta_rel - 1.96 * se), np.exp(eta_rel + 1.96 * se), ref_value


# ---------------------- 阈值与绘图 ----------------------
def suggest_thresholds_from_curve(lge_grid, hr, ci_lo, ci_hi, min_hr=1.2, smooth=7):
    def moving_avg(a, w):
        return np.convolve(a, np.ones(w) / w, mode="same") if w > 1 else a

    hr_s = moving_avg(hr, smooth)
    lo_s = moving_avg(ci_lo, smooth)
    valid = (hr_s > min_hr) & (lo_s > 1.0)
    if np.any(valid):
        return [lge_grid[np.where(valid)[0][0]]]
    return []


def plot_hr_curves(
    lge_grid,
    male_curve,
    female_curve,
    male_ci,
    female_ci,
    male_thresh=None,
    female_thresh=None,
    title="HR vs LGE (RCS + Female interaction)",
):
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(lge_grid, male_curve, label="Male HR")
    plt.fill_between(lge_grid, male_ci[0], male_ci[1], alpha=0.25)
    plt.plot(lge_grid, female_curve, label="Female HR")
    plt.fill_between(lge_grid, female_ci[0], female_ci[1], alpha=0.25)
    plt.axhline(1.0, linestyle="--", linewidth=1)

    def add_vlines(ths, lab):
        for t in ths or []:
            plt.axvline(t, ls=":", lw=1.5)
            plt.plot([], [], label=f"{lab} thr: {t:.2f}")

    add_vlines(male_thresh, "Male")
    add_vlines(female_thresh, "Female")
    plt.xlabel("LGE Burden")
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
    lge_col = "LGE Burden 5SD"
    sex_col = "Female"
    covariates = []

    ks, lb, ub = choose_rcs_spec_from_quantiles(df[lge_col], df_rcs=4)
    cph, feat_names, fit_df, spec = fit_cox_rcs_interaction(
        df,
        time_col,
        event_col,
        lge_col,
        sex_col,
        covariates,
        df_rcs=4,
        knots=ks,
        lower_bound=lb,
        upper_bound=ub,
    )
    print(cph.summary)

    grid = make_lge_grid_from_bounds(spec["lb"], spec["ub"], n=200)
    m_hr, m_lo, m_hi, ref_m = hr_curve_with_ci(
        cph,
        feat_names,
        grid,
        sex_value=0,
        df_rcs=4,
        knots=spec["knots"],
        lower_bound=spec["lb"],
        upper_bound=spec["ub"],
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
        sex_col=sex_col,
    )

    male_thr = suggest_thresholds_from_curve(grid, m_hr, m_lo, m_hi)
    female_thr = suggest_thresholds_from_curve(grid, f_hr, f_lo, f_hi)

    plot_hr_curves(
        grid,
        m_hr,
        f_hr,
        (m_lo, m_hi),
        (f_lo, f_hi),
        male_thresh=male_thr,
        female_thresh=female_thr,
    )
