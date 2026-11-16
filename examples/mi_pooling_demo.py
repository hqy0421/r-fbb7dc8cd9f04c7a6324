"""End-to-end demonstration of Rubin-pooled OLS on imputed feature tables.

Run with (from the workspace root)::

    python -I examples/mi_pooling_demo.py

The script builds three synthetic "completed imputation" feature tables and
one observed-response table, deliberately introduces:
  * two cases absent from the response table,
  * two cases with a missing value in one imputation,
so the complete-case intersection can be seen working. It then fits the
pooled model, prints the audit trail, and predicts on new data using the one
frozen Patsy DesignInfo.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from mi_pooling import pool_ols_rubin

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
rng = np.random.default_rng(20261116)
n = 80
case_id = np.arange(1, n + 1)

x1 = rng.normal(0.0, 1.0, n)
x2 = rng.normal(5.0, 2.0, n)
x3 = rng.normal(-1.0, 1.2, n)
grp = rng.choice(["a", "b", "c"], size=n)

y = (
    0.8
    + 1.5 * x1
    - 0.4 * x2
    + 0.6 * (grp == "b")
    - 0.3 * (grp == "c")
    + rng.normal(0.0, 0.5, n)
)

response = pd.DataFrame({"case_id": case_id, "y": y})
# Cases 7, 8 were never observed for the outcome.
response = response[~response["case_id"].isin([7, 8])].reset_index(drop=True)

m = 3
feature_tables = []
for k in range(m):
    x1_imp = x1 + rng.normal(0.0, 0.6, n)  # imputed, varies per version
    tab = pd.DataFrame(
        {
            "case_id": case_id,
            "x1": x1_imp,
            "x2": x2,
            "x3": x3,
            "grp": grp,
        }
    )
    if k == 1:
        # Impuation 2 could not fill x1 for cases 20, 21 -> listwise exclusion.
        tab.loc[tab["case_id"].isin([20, 21]), "x1"] = np.nan
    feature_tables.append(tab)

# ---------------------------------------------------------------------------
# One formula: named-column arithmetic, center/standardize, bs, C(), a
# named interaction, and a plain additive term. Learned exactly once.
# ---------------------------------------------------------------------------
formula = (
    "y ~ standardize(x2) + center(x3) + bs(x1, df=4) "
    "+ C(grp) + standardize(x2):grp"
)

result = pool_ols_rubin(
    formula,
    feature_tables,
    response,
    id_col="case_id",
    versions=["imp_A", "imp_B", "imp_C"],
)

# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------
print("=" * 72)
print("CASE ALIGNMENT")
print(
    "included: %d | excluded: %d"
    % (result.cases.n_included, result.cases.n_excluded)
)
print(result.cases.exclusion_report().to_string(index=False))

print("\n" + "=" * 72)
print("PER-IMPUTATION EQUAL-WEIGHT OLS")
for fit in result.fits:
    print(
        "version=%-5s n=%d rank=%d p=%d resid_df=%d sigma2=%.4f R2=%.4f"
        % (
            fit.version,
            fit.n,
            fit.rank,
            fit.p,
            int(fit.residual_df),
            fit.sigma2,
            fit.r_squared,
        )
    )

print("\n" + "=" * 72)
print(result.summary())

print("\n" + "=" * 72)
print("RUBIN VARIANCE COMPONENTS (trace)")
print("trace U_bar (within)      : %.5f" % np.trace(result.within_cov))
print("trace B     (between)     : %.5f" % np.trace(result.between_cov))
print(
    "trace T (total, 1+1/m)    : %.5f"
    % np.trace(result.total_cov)
)
print(
    "identity check ||T - (Ubar + (1+1/m)B)|| = %.2e"
    % np.abs(
        result.total_cov
        - (result.within_cov + (1 + 1 / result.m) * result.between_cov)
    ).max()
)

# ---------------------------------------------------------------------------
# Prediction with the frozen DesignInfo (no re-learning)
# ---------------------------------------------------------------------------
new_data = pd.DataFrame(
    {
        "case_id": [1001, 1002, 1003],
        "x1": [0.4, -1.1, 2.0],
        "x2": [4.5, 6.0, 3.2],
        "x3": [-0.5, 0.8, -1.7],
        "grp": ["a", "b", "c"],
    }
)
predictions = result.predict(new_data)
print("\n" + "=" * 72)
print("PREDICTIONS ON NEW DATA (frozen design)")
print(predictions.to_string())

# Missing values in new data raise rather than silently yielding NaNs.
bad = new_data.copy()
bad.loc[0, "x2"] = np.nan
try:
    result.predict(bad)
except Exception as exc:  # noqa: BLE001 - demo only
    print("\nmissing-value guard raised as expected: %s" % type(exc).__name__)
