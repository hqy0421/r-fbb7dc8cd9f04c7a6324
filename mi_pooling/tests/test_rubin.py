import numpy as np
import pandas as pd
import pytest

from patsy import build_design_matrices

from mi_pooling import PoolingError, RubinPooledOLS, pool_ols_rubin


RNG_SEED = 20261116


def make_data(n=60, m=3, seed=RNG_SEED, drop_response_ids=(), nan_ids=None):
    """Build m imputed feature tables + one observed response table.

    The imputations differ only in the imputed x1 values (and an imputed
    categorical feature), so between-imputation variance is non-zero.
    """
    rng = np.random.default_rng(seed)
    ids = np.arange(1, n + 1)
    x1_true = rng.normal(0.0, 1.0, size=n)
    x2 = rng.normal(5.0, 2.0, size=n)
    # x3 never enters a spline, so its centered main effect is independent
    # of every bs() basis column.
    x3 = rng.normal(-2.0, 1.5, size=n)
    groups = rng.choice(["a", "b", "c"], size=n)
    true_beta = np.array([1.0, 2.0, -0.5, 0.3, 0.0, 0.0])  # intercept,x1,x2,Cb,Cc,interact

    response = pd.DataFrame({"case_id": ids})
    # Structural mean in *raw* coordinates (design used only to build y).
    g = pd.Series(groups)
    y = (
        1.0
        + 2.0 * x1_true
        - 0.5 * (x2 - x2.mean())
        + 0.3 * (g == "b").astype(float)
        + 0.0 * (g == "c").astype(float)
        + 0.0 * x1_true * x2
        + rng.normal(0.0, 0.4, size=n)
    )
    response["y"] = y
    response = response[~response["case_id"].isin(drop_response_ids)].reset_index(
        drop=True
    )

    tables = []
    for k in range(m):
        # Each imputation adds its own noise to the missing-mechanism values.
        x1_imp = x1_true + rng.normal(0.0, 0.5, size=n)
        frame = pd.DataFrame(
            {
                "case_id": ids,
                "x1": x1_imp,
                "x2": x2,
                "x3": x3,
                "grp": groups,
            }
        )
        if nan_ids:
            # Simulate an incomplete imputation: these rows must be excluded
            # from *every* version (listwise across all tables).
            frame.loc[frame["case_id"].isin(nan_ids), "x1"] = np.nan
        tables.append(frame)
    return tables, response


# Full-rank design that still exercises every required native mechanism:
# named-column addition, center, standardize, bs, C(), and an interaction.
# bs() acts on x1 while center() acts on the independent x3 so the spline
# basis cannot span another continuous main effect; the interaction reuses
# the standardized main effect's exact factor spelling so Patsy removes the
# reference-group interaction column itself.
FORMULA = (
    "y ~ center(x3) + standardize(x2) + bs(x1, df=4) + C(grp) "
    "+ standardize(x2):grp"
)


def test_basic_pooling_structure_and_rubin_components():
    tables, response = make_data()
    result = pool_ols_rubin(FORMULA, tables, response, id_col="case_id")
    assert isinstance(result, RubinPooledOLS)
    assert result.m == 3
    assert result.n == 60
    assert result.p == len(result.column_names)
    assert result.coef.shape == (result.p,)
    assert result.within_cov.shape == (result.p, result.p)
    assert result.between_cov.shape == (result.p, result.p)
    assert result.total_cov.shape == (result.p, result.p)

    # T = Ubar + (1 + 1/m) B exactly.
    expected_total = result.within_cov + (1 + 1 / result.m) * result.between_cov
    np.testing.assert_allclose(result.total_cov, expected_total, atol=1e-12)
    # Covariance matrices symmetric.
    for cov in (result.within_cov, result.between_cov, result.total_cov):
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)
    # Total variance is no smaller than within variance (PSD difference).
    eig = np.linalg.eigvalsh(result.total_cov - result.within_cov)
    assert np.all(eig > -1e-10)
    # FMI and lambda live in [0, 1] and the df are positive.
    assert np.all(result.fraction_missing_info >= -1e-12)
    assert np.all(result.fraction_missing_info <= 1.0 + 1e-12)
    assert np.all(result.relative_increase_variance >= -1e-12)
    assert np.all(result.df_barnard_rubin > 0)
    assert np.all(result.df_rubin > 0)


def test_design_learned_once_and_reused_identically():
    tables, response = make_data()
    result = pool_ols_rubin(FORMULA, tables, response, id_col="case_id")
    included = result.cases.included_ids

    # Every stateful factor carries memorized state in the single frozen
    # DesignInfo (documented FactorInfo.state attribute).
    states = [
        fi.state
        for fi in result.design_info.factor_infos.values()
        if fi.state is not None
    ]
    assert len(states) >= 3  # center, standardize, bs at minimum

    # Rebuilding from the frozen DesignInfo must reproduce, column for column,
    # the matrices used at fit time, including frozen knots/means/scales.
    for frame in tables:
        sub = frame[frame["case_id"].isin(included)].sort_values("case_id")
        (x,) = build_design_matrices([result.design_info], sub)
        assert list(x.design_info.column_names) == result.column_names
        assert x.shape[1] == result.p

    # Functional freeze check: bs(x1) memorizes its interior knots from the
    # *pooled* complete cases (all m versions stacked). x1 differs across
    # imputations, so those pooled knots are not what learning on version 0
    # alone would produce.
    pooled_x1 = np.concatenate(
        [
            t[t["case_id"].isin(included)].sort_values("case_id")["x1"].to_numpy()
            for t in tables
        ]
    )
    bs_state = None
    for fi in result.design_info.factor_infos.values():
        if fi.factor.code == "bs(x1, df=4)":
            bs_state = next(iter(fi.state["transforms"].values()))
    assert bs_state is not None
    frozen_knots = np.asarray(bs_state._all_knots, dtype=float)

    # Reference knots learned independently on version 0 only.
    from patsy import dmatrices as _dmatrices
    from patsy.splines import _R_compat_quantile

    v0 = (
        tables[0][tables[0]["case_id"].isin(included)]
        .sort_values("case_id")
        .copy()
    )
    v0["y"] = (
        response[response["case_id"].isin(included)]
        .sort_values("case_id")["y"]
        .to_numpy()
    )
    _, x_v0 = _dmatrices(FORMULA, v0)
    v0_bs = next(
        iter(
            next(iter(fi.state["transforms"].values()))
            for fi in x_v0.design_info.factor_infos.values()
            if fi.factor.code == "bs(x1, df=4)"
        )
    )
    v0_knots = np.asarray(v0_bs._all_knots, dtype=float)
    # Pooled knots differ from the single-version ones...
    assert np.max(np.abs(frozen_knots - v0_knots)) > 1e-8
    # ...and equal the knots computed on the vertically pooled x1. With
    # degree=3, df=4, include_intercept=False: n_inner = df - order + 1 = 1,
    # a single median (R type-7) knot.
    inner = _R_compat_quantile(pooled_x1, np.array([0.5]))
    expected = np.sort(
        np.concatenate(
            [[pooled_x1.min(), pooled_x1.max()] * 4, np.atleast_1d(inner)]
        )
    )
    np.testing.assert_allclose(frozen_knots, expected, rtol=1e-10)


def test_per_version_fit_details():
    tables, response = make_data()
    result = pool_ols_rubin(FORMULA, tables, response, id_col="case_id")
    assert len(result.fits) == 3
    for fit in result.fits:
        assert fit.n == result.n
        assert fit.rank == fit.p
        assert fit.residual_df == fit.n - fit.p
        assert fit.sigma2 > 0
        # cov = sigma2 * (X'X)^-1, positive definite.
        assert np.linalg.eigvalsh(fit.cov).min() > 0

    # Pooled coefficient equals the simple mean of version coefficients.
    mean_coef = np.mean([f.coef for f in result.fits], axis=0)
    np.testing.assert_allclose(result.coef, mean_coef, atol=1e-12)
    # Within covariance equals the mean of per-version covariances.
    mean_cov = np.mean([f.cov for f in result.fits], axis=0)
    np.testing.assert_allclose(result.within_cov, mean_cov, atol=1e-12)


def test_rubin_math_against_manual_computation():
    tables, response = make_data(m=4, seed=7)
    result = pool_ols_rubin("y ~ x1 + x2", tables, response, id_col="case_id")
    m = result.m
    q = np.vstack([f.coef for f in result.fits])
    u = np.stack([f.cov for f in result.fits])
    qbar = q.mean(axis=0)
    ubar = u.mean(axis=0)
    b = ((q - qbar).T @ (q - qbar)) / (m - 1)
    t = ubar + (1 + 1 / m) * b
    np.testing.assert_allclose(result.coef, qbar)
    np.testing.assert_allclose(result.within_cov, ubar)
    np.testing.assert_allclose(result.between_cov, b)
    np.testing.assert_allclose(result.total_cov, t)

    # Rubin df formula, coefficient 0.
    se_within = np.diag(ubar)[0]
    b_diag = np.diag(b)[0]
    expected_df = (m - 1) * (1 + se_within / ((1 + 1 / m) * b_diag)) ** 2
    np.testing.assert_allclose(result.df_rubin[0], expected_df, rtol=1e-10)


def test_case_alignment_exclusions_are_reported():
    # Response is missing cases 5 and 6; imputation 2 has NaN for cases 10,11.
    tables, response = make_data(
        drop_response_ids=(5, 6), nan_ids=(10, 11)
    )
    result = pool_ols_rubin("y ~ x1 + x2", tables, response, id_col="case_id")
    included = set(result.cases.included_ids)
    excluded = set(result.cases.excluded_ids)
    assert result.n == 60 - 4
    assert {5, 6, 10, 11} <= excluded
    assert included.isdisjoint({5, 6, 10, 11})
    # Every included case is present, and the versions all share the same n.
    assert all(fit.n == result.n for fit in result.fits)
    # Reasons are populated and mention the offending table.
    reasons = result.cases.excluded_reasons
    assert any("response table" in r for r in reasons[5])
    assert any("imputation" in r for r in reasons[10])
    report = result.cases.exclusion_report()
    assert set(report["id"]) == excluded


def test_rank_deficient_version_raises_and_does_not_drop_columns():
    # A perfectly collinear column in one imputation must abort the whole
    # pooling rather than silently dropping a column.
    tables, response = make_data()
    tables[1] = tables[1].copy()
    tables[1]["x1_dup"] = tables[1]["x1"] * 3.0
    formula = "y ~ x1 + x1_dup + x2"
    # All versions carry the column (it is constant-zero / missing elsewhere),
    # make it present but collinear only in version 1.
    for k, t in enumerate(tables):
        if k != 1:
            t["x1_dup"] = np.linspace(0, 1, len(t)) + k  # independent elsewhere
    with pytest.raises(PoolingError, match="rank-deficient"):
        pool_ols_rubin(formula, tables, response, id_col="case_id")


def test_no_residual_df_raises():
    tables, response = make_data(n=4, m=2)
    # Intercept + 3 predictors on 4 cases -> residual df = 0.
    with pytest.raises(PoolingError, match="residual degrees of freedom"):
        pool_ols_rubin("y ~ x1 + x2 + grp", tables, response, id_col="case_id")


def test_duplicate_ids_rejected():
    tables, response = make_data()
    dup = tables[0].copy()
    dup = pd.concat([dup, dup.iloc[[0]]], ignore_index=True)
    tables[0] = dup
    with pytest.raises(PoolingError, match="duplicate case IDs"):
        pool_ols_rubin("y ~ x1", tables, response, id_col="case_id")


def test_single_imputation_rejected():
    tables, response = make_data(m=3)
    with pytest.raises(PoolingError, match="at least two imputed"):
        pool_ols_rubin("y ~ x1", tables[:1], response, id_col="case_id")


def test_rhs_only_formula_uses_single_outcome():
    tables, response = make_data()
    result_lhs = pool_ols_rubin("y ~ x1 + x2", tables, response, id_col="case_id")
    result_rhs = pool_ols_rubin("x1 + x2", tables, response, id_col="case_id")
    np.testing.assert_allclose(result_lhs.coef, result_rhs.coef, atol=1e-12)
    assert result_rhs.outcome_name == "y"


def test_predict_uses_frozen_design_and_matches_fit():
    tables, response = make_data()
    result = pool_ols_rubin(FORMULA, tables, response, id_col="case_id")
    new_data = tables[0].sort_values("case_id").reset_index(drop=True)
    preds = result.predict(new_data)
    # Manual prediction through the frozen design must agree.
    (x_new,) = build_design_matrices([result.design_info], new_data)
    manual = np.asarray(x_new) @ result.coef
    np.testing.assert_allclose(np.asarray(preds), manual, atol=1e-12)
    # Stateful transform freeze check: standardize(x2) constants are the
    # fitted ones, so a shifted raw x2 shifts predictions deterministically.
    shifted = new_data.copy()
    shifted["x2"] = shifted["x2"] + 1.0
    # No exception: the same frozen knots/means are applied to new values.
    shifted_pred = result.predict(shifted)
    assert shifted_pred.shape == preds.shape


def test_predict_raises_on_missing_value_by_default():
    tables, response = make_data()
    result = pool_ols_rubin("y ~ x1 + x2", tables, response, id_col="case_id")
    new_data = tables[0].copy()
    new_data.loc[0, "x1"] = np.nan
    with pytest.raises(PoolingError):
        result.predict(new_data)


def test_coef_table_and_summary():
    tables, response = make_data()
    result = pool_ols_rubin("y ~ x1 + x2", tables, response, id_col="case_id")
    table = result.coef_table()
    assert list(table["term"]) == result.column_names
    assert len(table) == result.p
    assert np.all(np.isfinite(table["p_value"]))
    ci = result.confidence_intervals()
    assert np.all(ci[:, 0] <= result.coef + 1e-9)
    assert np.all(ci[:, 1] >= result.coef - 1e-9)
    text = result.summary()
    assert "Rubin-pooled OLS" in text and "x1" in text


def test_equal_weight_ols_recovers_known_coefficients():
    # With tiny imputation noise and residual noise, the pooled x1 coefficient
    # should be close to the structural value of 2.
    rng = np.random.default_rng(123)
    n = 400
    ids = np.arange(n)
    x1 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 + rng.normal(0, 0.1, size=n)
    response = pd.DataFrame({"case_id": ids, "y": y})
    tables = [
        pd.DataFrame(
            {"case_id": ids, "x1": x1 + rng.normal(0, 0.05, size=n)}
        )
        for _ in range(3)
    ]
    result = pool_ols_rubin("y ~ x1", tables, response, id_col="case_id")
    x1_idx = result.column_names.index("x1")
    assert abs(result.coef[x1_idx] - 2.0) < 0.1
