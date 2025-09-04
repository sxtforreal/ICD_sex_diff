import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from patsy import dmatrix
from scipy.stats import chi2

from cox import load_dataframes


# ---------------------- Data preparation ----------------------
def prepare_df_for_model(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardize NYHA
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

    # Nullable integer/float/boolean -> float64
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
    # Keep the first occurrence of duplicated column names
    return df.loc[:, ~df.columns.duplicated()].copy()


# ---------------------- RCS related ----------------------
def build_rcs_basis(x, df_rcs=4, knots=None, lower_bound=None, upper_bound=None):
    """
    Natural cubic spline basis (without intercept). Note: df_rcs is the
    degrees of freedom (number of basis functions). If knots (interior knots
    only) are provided, patsy will ignore the df parameter.
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
    Choose interior knots and bounds by quantiles (Harrell's recommendation):
    - Bounds default to the 5% / 95% quantiles
    - Interior knots: equally spaced by quantiles over [lb, ub]
      (number = df_rcs - 2). If interior_q (list in (0,1)) is provided,
      use it. For example, df_rcs=4 -> 2 interior knots -> suggest 35% and 65%.
    """
    s = pd.to_numeric(pd.Series(series).dropna(), errors="coerce")
    lb = float(s.quantile(lb_q))
    ub = float(s.quantile(ub_q))
    if not np.isfinite(lb) or not np.isfinite(ub) or ub <= lb:
        # Fallback: enforce a minimal span
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
            # Take n_interior quantiles equally spaced in [lb_q, ub_q] (excluding endpoints)
            qs = np.linspace(lb_q, ub_q, n_interior + 2)[1:-1].tolist()
            # Special-case tweak: for df=4, use the conventional 35%/65%
            if df_rcs == 4:
                qs = [0.35, 0.65]

    ks = [float(s.quantile(q)) for q in qs]
    # Ensure knots lie within bounds
    eps = 1e-10
    ks = [min(max(k, lb + eps), ub - eps) for k in ks]
    return ks, lb, ub


def add_interactions(X_rcs: pd.DataFrame, sex_series, sex_col="Female"):
    # Ensure sex is 0/1
    sex_bin = pd.Series(sex_series).astype(float)
    # If not 0/1, thresholdize: values > 0 are treated as 1
    if not set(pd.unique(sex_bin.dropna())).issubset({0.0, 1.0}):
        sex_bin = (sex_bin > 0).astype(float)
    X_int = X_rcs.mul(sex_bin.values.reshape(-1, 1))
    X_int.columns = [c + f":{sex_col}" for c in X_rcs.columns]
    return X_int, sex_bin.astype(int)


# ---------------------- Data cleaning ----------------------
def _numeric_covariates(df_cov: pd.DataFrame) -> pd.DataFrame:
    if df_cov is None or df_cov.empty:
        return pd.DataFrame(index=df_cov.index if df_cov is not None else None)
    num = df_cov.select_dtypes(include=["number"])
    boo = df_cov.select_dtypes(include=["bool"]).astype(int)
    other = df_cov.drop(columns=list(num.columns) + list(boo.columns), errors="ignore")
    if not other.empty:
        other = pd.get_dummies(other, drop_first=True, dummy_na=False)
    X = pd.concat([num, boo, other], axis=1).apply(pd.to_numeric, errors="coerce")
    # Drop constant columns
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
    # Drop constant columns again
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


# ---------------------- Fitting & diagnostics ----------------------
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

    # Design matrix
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
    # Base table
    base = df[[time_col, event_col, x_col]].copy()
    if covariates:
        base = pd.concat([base, _numeric_covariates(df[covariates])], axis=1)
    base = base.dropna()
    base = _dedup_columns(base)

    # Linear model
    lin_df = base.rename(columns={x_col: "x"})
    lin_df[event_col] = lin_df[event_col].astype(int)
    cph_lin = CoxPHFitter(penalizer=penalizer)
    cph_lin.fit(lin_df, duration_col=time_col, event_col=event_col, robust=True)

    # RCS - key: align indices
    rcs = build_rcs_basis(
        lin_df["x"].values, df_rcs=df_rcs, knots=ks, lower_bound=lb, upper_bound=ub
    )
    rcs.index = lin_df.index  # Align to the index of lin_df

    others = lin_df.drop(columns=["x", time_col, event_col])
    rcs_df = pd.concat([lin_df[[time_col, event_col]], rcs, others], axis=1)
    rcs_df = _dedup_columns(rcs_df)  # Safety: ensure unique columns

    # Fit the RCS model
    cph_rcs = CoxPHFitter(penalizer=penalizer)
    cph_rcs.fit(rcs_df, duration_col=time_col, event_col=event_col, robust=True)

    # Likelihood ratio test (nonlinearity)
    from scipy.stats import chi2

    LL_lin, LL_rcs = cph_lin.log_likelihood_, cph_rcs.log_likelihood_
    df_nl = rcs.shape[1] - 1
    stat = 2 * (LL_rcs - LL_lin)
    p = 1 - chi2.cdf(stat, df=df_nl)
    print(f"[LRT nonlinearity] chi2={stat:.2f}, df={df_nl}, p={p:.4g}")
    return stat, p


# ---------------------- Predicted curves ----------------------
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
    # For model covariates not provided here, fill with means (or 0)
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


# ---------------------- Thresholds and plotting ----------------------
def suggest_thresholds_from_curve(
    x_grid, hr, ci_lo, ci_hi, min_hr=1.2, smooth=7, max_n=2
):
    """
    Return up to two "high-risk start" thresholds: the starting indices of
    intervals where the smoothed HR > min_hr AND the lower CI > 1. This
    supports a three-segment interpretation (High/Med/Low).
    """

    def moving_avg(a, w):
        return np.convolve(a, np.ones(w) / w, mode="same") if w > 1 else a

    hr_s = moving_avg(hr, smooth)
    lo_s = moving_avg(ci_lo, smooth)
    valid = (hr_s > min_hr) & (lo_s > 1.0)

    # Find the start indices of contiguous "islands" of validity
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
            plt.plot([], [], label=f"{lab} thr: {t:.2f}")  # 2 decimals

    add_vlines(male_thresh, "Male")
    add_vlines(female_thresh, "Female")
    plt.xlabel(xlabel)
    plt.ylabel("Hazard Ratio")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def km_plot_and_logrank(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    x_col: str,
    sex_col: str,
    male_thresh=None,
    female_thresh=None,
    title_prefix: str = "KM by threshold",
):
    """
    Plot Kaplan-Meier curves and perform logrank tests by sex using the
    recommended thresholds. If a sex-specific threshold is missing or a group
    is empty, that sex will be skipped.
    """
    dat = df[[time_col, event_col, x_col, sex_col]].copy()
    dat = dat.dropna(subset=[time_col, event_col, x_col, sex_col])
    dat = dat[dat[time_col] > 0].copy()
    dat[event_col] = dat[event_col].astype(int)

    def _first_thr(th):
        if th is None:
            return None
        if isinstance(th, (list, tuple, np.ndarray)):
            if len(th) == 0:
                return None
            return float(th[0])
        try:
            return float(th)
        except Exception:
            return None

    thr_m = _first_thr(male_thresh)
    thr_f = _first_thr(female_thresh)

    sexes = []
    if thr_m is not None:
        sexes.append((0, thr_m, "Male"))
    if thr_f is not None:
        sexes.append((1, thr_f, "Female"))

    if not sexes:
        print("[KM] No thresholds provided; skipping KM plots.")
        return

    n = len(sexes)
    fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 5.2), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (sex_value, thr, lab) in zip(axes, sexes):
        sub = dat[dat[sex_col] == sex_value]
        if sub.empty:
            ax.set_title(f"{lab}: no data")
            continue
        low_mask = sub[x_col] < thr
        high_mask = sub[x_col] >= thr

        if low_mask.sum() == 0 or high_mask.sum() == 0:
            ax.set_title(f"{lab}: group empty at thr={thr:.2f}")
            continue

        kmf_low = KaplanMeierFitter()
        kmf_high = KaplanMeierFitter()

        kmf_low.fit(sub[time_col][low_mask], event_observed=sub[event_col][low_mask], label=f"{lab} Low (<{thr:.2f})")
        kmf_high.fit(sub[time_col][high_mask], event_observed=sub[event_col][high_mask], label=f"{lab} High (>={thr:.2f})")

        kmf_low.plot(ax=ax, ci_show=True)
        kmf_high.plot(ax=ax, ci_show=True)
        ax.set_xlabel("Time")
        ax.set_ylabel("Survival probability")

        res = logrank_test(
            sub[time_col][low_mask],
            sub[time_col][high_mask],
            event_observed_A=sub[event_col][low_mask],
            event_observed_B=sub[event_col][high_mask],
        )
        pval = res.p_value
        ax.set_title(f"{title_prefix} - {lab} (thr={thr:.2f}, p={pval:.3g})")
        ax.grid(True, alpha=0.3, linestyle=":")

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

    # knots / bounds: default 5%-95%; for df=4, interior at 35%/65%
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

    # Nonlinearity test (optional)
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

    # Fill other covariates with training means (if any)
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

    # KM plots and logrank tests by sex using recommended thresholds
    km_plot_and_logrank(
        df,
        time_col=time_col,
        event_col=event_col,
        x_col=x_col,
        sex_col=sex_col,
        male_thresh=male_thr,
        female_thresh=female_thr,
        title_prefix="KM by LGE threshold",
    )
