# Rubin-pooled OLS over multiply-imputed feature tables

`mi_pooling.pool_ols_rubin` combines equal-weight OLS fits across several
completed imputations of the same feature table, against a single observed
response table. It is built entirely on Patsy's **public** formula API and
keeps Patsy's formula interface unchanged.

## Guarantees

1. **Case-ID alignment, listwise across *all* tables.** A case enters the
   analysis only when it appears in the response table and every imputation
   and is non-missing on every column the formula needs. The included and
   excluded IDs (with reasons) are returned; nothing is dropped per-table.
2. **The feature engineering is learned exactly once.** The formula is
   evaluated on the vertically pooled complete cases via `patsy.dmatrices`,
   producing one frozen `DesignInfo`. All stateful transforms — named-column
   arithmetic and interactions, `C()` levels/contrasts, `center`,
   `standardize`/`scale`, and `bs` knots — are defined by that single object.
3. **One frozen design for every version.** Each imputation's matrix is built
   with `patsy.build_design_matrices([frozen_design_info], ...)`. Columns,
   ordering, contrasts, centering/scaling constants and spline knots are
   identical across versions; nothing is re-learned per version.
4. **Equal-weight OLS per version**, each estimating its own error variance
   with its residual degrees of freedom (`sigma2 = RSS / (n - p)`).
5. **Rubin's rules.** Pooled coefficients are the mean across versions; the
   total covariance is
   `T = U_bar + (1 + 1/m) * B`, keeping the within-imputation component
   `U_bar`, the between-imputation component `B`, and the finite-`m`
   correction explicitly. Rubin df and the Barnard-Rubin small-sample
   adjusted df are both returned.
6. **Hard failure, never silent repair.** If any version is rank deficient or
   has no residual degrees of freedom, `PoolingError` is raised. Columns are
   never deleted and the design is never re-learned to "fix" a version.
7. **Frozen prediction.** `RubinPooledOLS.predict(new_data)` reuses the same
   `DesignInfo`; missing values raise by default and unseen categorical
   levels are reported rather than absorbed.

## Minimal example

```python
from mi_pooling import pool_ols_rubin

result = pool_ols_rubin(
    "y ~ standardize(x2) + center(x3) + bs(x1, df=4) + C(grp) "
    "+ standardize(x2):grp",
    feature_tables=[imp0, imp1, imp2],   # DataFrames sharing case_id
    response_table=response,             # DataFrame with case_id and y
    id_col="case_id",
)

result.cases.included_ids          # cases actually used
result.cases.exclusion_report()    # excluded IDs and why
result.coef_table()                # pooled coefs, SEs, t, df, p, CIs, FMI
result.within_cov                  # U_bar
result.between_cov                 # B
result.total_cov                   # T = U_bar + (1 + 1/m) B
result.predict(new_data)           # frozen-design point predictions
```

A right-hand-side-only formula (`"x1 + x2"`) is accepted when the response
table has exactly one non-ID column; otherwise name the outcome with
`"y ~ ..."`.

## Running the tests and demo

Under isolated Python (the workspace treats input data as untrusted):

```bash
python -I -m pytest mi_pooling/ -q
python -I examples/mi_pooling_demo.py
```

The root `conftest.py` only puts the workspace root on `sys.path` so the
package imports under `python -I`; it does not alter Patsy.
