"""Rubin-rules pooling of OLS fits over multiple imputed feature tables.

The workflow implemented here is deliberately conservative:

1. All tables (the observed-response table and every imputed feature table)
   are aligned on a case ID, and only cases that are *complete in every
   table* are retained. Nothing is silently dropped table by table.
2. The Patsy formula is learned **exactly once** on the vertically pooled
   complete cases, so every stateful transform (``center``,
   ``standardize``/``scale``, ``bs`` spline knots, ``C()`` categorical
   levels and contrasts, and the named-column arithmetic/interaction
   structure) is defined by a single frozen :class:`~patsy.DesignInfo`.
3. That same ``DesignInfo`` is used to build the design matrix of *every*
   imputation version. Each version then gets its own equal-weight OLS fit
   with its own residual-df error variance.
4. Coefficients are averaged across versions and the total covariance is
   assembled by Rubin's rules, keeping the within-imputation component, the
   between-imputation component, and the finite-number-of-imputations
   correction ``(1 + 1/m)``.

If any single version is rank deficient or has no residual degrees of
freedom, pooling aborts with :class:`PoolingError`. Columns are never deleted
and the feature engineering is never re-learned on a per-version basis.
"""

from dataclasses import dataclass, field
import math

import numpy as np
from scipy import stats

from patsy import PatsyError, build_design_matrices, dmatrices
from patsy.desc import ModelDesc
from patsy.eval import EvalEnvironment

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is a hard dependency below
    pd = None


class PoolingError(ValueError):
    """Raised when the imputed fits cannot validly be pooled.

    Typical causes: a version's design matrix is rank deficient, a version
    has no residual degrees of freedom, no complete cases survive the
    intersection, or the formula cannot be evaluated consistently.
    """


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class CaseAlignment:
    """Which cases entered the analysis and why the rest were excluded."""

    id_col: str
    included_ids: list
    excluded_ids: list
    # id -> list of human-readable reasons
    excluded_reasons: dict

    @property
    def n_included(self):
        return len(self.included_ids)

    @property
    def n_excluded(self):
        return len(self.excluded_ids)

    def exclusion_report(self):
        """Return a DataFrame of excluded IDs and their reasons, if pandas."""
        if pd is None:  # pragma: no cover
            raise PoolingError("pandas is required for exclusion_report()")
        rows = [
            {"id": cid, "reasons": "; ".join(self.excluded_reasons.get(cid, []))}
            for cid in self.excluded_ids
        ]
        return pd.DataFrame(rows, columns=["id", "reasons"])


@dataclass
class ImputationFit:
    """The equal-weight OLS result for one imputation version."""

    version: object
    n: int
    rank: int
    p: int
    residual_df: float
    sigma2: float
    coef: np.ndarray                 # shape (p,)
    cov: np.ndarray                  # shape (p, p), sigma2 * (X'X)^-1
    residual_sum_squares: float
    r_squared: float


@dataclass
class RubinPooledOLS:
    """The pooled linear model plus everything needed to audit it."""

    formula: str
    m: int
    column_names: list
    design_info: object
    outcome_name: str
    id_col: str
    cases: CaseAlignment
    fits: list = field(default_factory=list)

    # pooled quantities
    coef: np.ndarray = None                       # shape (p,)
    within_cov: np.ndarray = None                 # U_bar
    between_cov: np.ndarray = None                # B
    total_cov: np.ndarray = None                  # T
    relative_increase_variance: np.ndarray = None  # lambda_hat, per coefficient
    fraction_missing_info: np.ndarray = None      # FMI, per coefficient
    df_rubin: np.ndarray = None                   # per coefficient
    df_barnard_rubin: np.ndarray = None           # small-sample adjusted df

    # ------------------------------------------------------------------
    # Basic summaries
    # ------------------------------------------------------------------
    @property
    def n(self):
        return self.cases.n_included

    @property
    def p(self):
        return len(self.column_names)

    def std_errors(self):
        return np.sqrt(np.diag(self.total_cov))

    def t_values(self):
        return self.coef / self.std_errors()

    def p_values(self, use="barnard_rubin"):
        """Two-sided p-values from a t reference distribution.

        ``use`` selects the degrees of freedom: ``"barnard_rubin"`` (the
        small-sample adjusted default) or ``"rubin"``.
        """
        df = self._select_df(use)
        return 2.0 * stats.t.sf(np.abs(self.t_values()), df)

    def confidence_intervals(self, alpha=0.05, use="barnard_rubin"):
        """Per-coefficient (1-alpha) t intervals as an (p, 2) array."""
        df = self._select_df(use)
        se = self.std_errors()
        crit = stats.t.ppf(1.0 - alpha / 2.0, df)
        lower = self.coef - crit * se
        upper = self.coef + crit * se
        return np.column_stack([lower, upper])

    def _select_df(self, use):
        if use == "rubin":
            return self.df_rubin
        if use == "barnard_rubin":
            return self.df_barnard_rubin
        raise ValueError("use must be 'rubin' or 'barnard_rubin'")

    def coef_table(self, alpha=0.05, use="barnard_rubin"):
        """Return a tidy pandas DataFrame of the pooled coefficients."""
        if pd is None:  # pragma: no cover
            raise PoolingError("pandas is required for coef_table()")
        ci = self.confidence_intervals(alpha=alpha, use=use)
        df = self._select_df(use)
        return pd.DataFrame(
            {
                "term": self.column_names,
                "coef": self.coef,
                "std_error": self.std_errors(),
                "t": self.t_values(),
                "df": df,
                "p_value": self.p_values(use=use),
                f"ci_{alpha/2:g}": ci[:, 0],
                f"ci_{1 - alpha/2:g}": ci[:, 1],
                "lambda": self.relative_increase_variance,
                "fmi": self.fraction_missing_info,
            }
        )

    def summary(self, alpha=0.05):
        """A readable text summary of the pooled model."""
        lines = []
        lines.append("Rubin-pooled OLS (m = %d imputations)" % self.m)
        lines.append("Formula: %s" % self.formula)
        lines.append(
            "Cases: %d included, %d excluded (id column %r)"
            % (self.cases.n_included, self.cases.n_excluded, self.id_col)
        )
        lines.append("Columns (p = %d): %s" % (self.p, self.column_names))
        lines.append("")
        lines.append(self.coef_table(alpha=alpha).to_string(index=False))
        lines.append("")
        lines.append(
            "Within-imputation covariance U_bar, between B, total T "
            "(with 1+1/m correction) are stored on this object."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Prediction with the *frozen* design
    # ------------------------------------------------------------------
    def predict(self, new_data, na_action="raise"):
        """Predict on new data using the single frozen DesignInfo.

        No re-learning happens: categorical levels, contrasts, centering /
        scaling constants and spline knots are all reused from fitting.

        :arg new_data: a pandas DataFrame or dict-like containing every raw
          variable referenced by the right-hand side of the formula.
        :arg na_action: defaults to ``"raise"`` so a missing value in new
          data is reported rather than silently producing NaN predictions;
          ``"drop"`` is also accepted.
        :returns: a pandas Series (if ``new_data`` is a DataFrame) or a
          numpy array of point predictions.
        """
        data, is_frame = _coerce_frame(new_data)
        try:
            (x,) = build_design_matrices(
                [self.design_info], data, NA_action=na_action
            )
        except PatsyError as exc:
            raise PoolingError(
                "could not build the design matrix for new data using the "
                "frozen DesignInfo: %s" % (exc,)
            ) from exc
        pred = np.asarray(x) @ self.coef
        if is_frame:
            index = data.index
            return pd.Series(pred, index=index, name="prediction")
        return pred


# ---------------------------------------------------------------------------
# Input handling and case alignment
# ---------------------------------------------------------------------------


def _coerce_frame(data):
    """Return (DataFrame, whether_input_was_frame-like)."""
    if pd is None:  # pragma: no cover
        raise PoolingError(
            "pandas is required for ID alignment and is a hard dependency"
        )
    if isinstance(data, pd.DataFrame):
        return data, True
    if isinstance(data, dict):
        return pd.DataFrame(data), False
    if isinstance(data, np.ndarray):
        raise PoolingError(
            "raw ndarray inputs have no column or case-ID information; "
            "pass a pandas DataFrame or a dict of named columns instead"
        )
    # Last resort: let pandas try to construct a frame.
    return pd.DataFrame(data), False


def _check_id_unique(frame, id_col, label):
    dup = frame[id_col].duplicated(keep=False)
    if dup.any():
        ids = frame.loc[dup, id_col].unique().tolist()
        raise PoolingError(
            "%s contains duplicate case IDs %r in column %r; one row per "
            "case is required for unambiguous alignment"
            % (label, ids[:10], id_col)
        )


def _align_cases(frames, id_col, labels, outcome_name):
    """Intersect all tables on complete cases.

    ``frames`` includes the response table first, then every feature table.
    A case is included only when it appears exactly once in every table and
    is non-missing on every column the formula needs (the outcome for the
    response table, all columns for feature tables).
    """
    reasons = {}

    def mark(cid, reason):
        reasons.setdefault(cid, set()).add(reason)

    # Universe of every ID we have seen.
    all_ids_sets = [set(f[id_col].unique()) for f in frames]
    universe = set().union(*all_ids_sets)

    common = set.intersection(*all_ids_sets)
    for cid in universe - common:
        present = [labels[i] for i, ids in enumerate(all_ids_sets) if cid in ids]
        missing = [labels[i] for i, ids in enumerate(all_ids_sets) if cid not in ids]
        mark(
            cid,
            "missing from %s (present in %s)"
            % (", ".join(missing), ", ".join(present) if present else "no table"),
        )

    # Within-table completeness for the common IDs.
    good_per_table = []
    for frame, label in zip(frames, labels):
        sub = frame[frame[id_col].isin(common)]
        if label == "response table":
            value_cols = [outcome_name]
            if outcome_name not in frame.columns:
                raise PoolingError(
                    "response table has no outcome column %r" % (outcome_name,)
                )
        else:
            value_cols = [c for c in frame.columns if c != id_col]
        missing_mask = sub[value_cols].isna().any(axis=1).to_numpy()
        bad_ids = sub.loc[missing_mask, id_col].tolist()
        for cid in bad_ids:
            mark(cid, "missing value in %s" % label)
        good_per_table.append(set(sub.loc[~missing_mask, id_col]))

    complete = set.intersection(*good_per_table) if good_per_table else set()
    # Any common ID that failed a completeness check is already marked.
    included = sorted(complete)
    excluded = sorted(universe - complete)
    # Stable, serializable reason lists.
    reason_lists = {cid: sorted(reasons.get(cid, ())) for cid in excluded}
    return CaseAlignment(
        id_col=id_col,
        included_ids=included,
        excluded_ids=excluded,
        excluded_reasons=reason_lists,
    )


# ---------------------------------------------------------------------------
# OLS per imputation
# ---------------------------------------------------------------------------


def _numeric_rank(x):
    """Rank via SVD with the same threshold numpy's lstsq uses by default."""
    sv = np.linalg.svd(x, compute_uv=False)
    if sv.size == 0:
        return 0
    tol = np.asarray(x).shape[0] * sv.max(initial=0.0) * np.finfo(sv.dtype).eps
    return int(np.count_nonzero(sv > tol))


def _fit_ols(version, x, y):
    """Equal-weight OLS with explicit rank and residual-df guards."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n, p = x.shape
    if n != y.shape[0]:
        raise PoolingError(
            "version %r: design matrix and outcome row counts differ" % (version,)
        )
    rank = _numeric_rank(x)
    residual_df = n - p
    if rank < p:
        raise PoolingError(
            "version %r: rank-deficient design matrix (rank %d < %d columns). "
            "Refusing to pool: columns are not deleted and the design is not "
            "re-learned per version." % (version, rank, p)
        )
    if residual_df <= 0:
        raise PoolingError(
            "version %r: no residual degrees of freedom (n=%d, p=%d, "
            "residual df=%d); an error variance cannot be estimated."
            % (version, n, p, residual_df)
        )

    # Full column rank is guaranteed above, so the normal equations are
    # nonsingular and solve() gives the OLS solution directly.
    xtx = x.T @ x
    coef = np.linalg.solve(xtx, x.T @ y)
    fitted = x @ coef
    resid = y - fitted
    rss = float(resid @ resid)
    sigma2 = rss / residual_df
    cov = sigma2 * np.linalg.inv(xtx)
    tss = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - rss / tss if tss > 0 else float("nan")
    return ImputationFit(
        version=version,
        n=n,
        rank=rank,
        p=p,
        residual_df=float(residual_df),
        sigma2=float(sigma2),
        coef=coef,
        cov=cov,
        residual_sum_squares=rss,
        r_squared=r_squared,
    )


# ---------------------------------------------------------------------------
# Rubin's rules
# ---------------------------------------------------------------------------


def _rubin_pool(fits, m, p):
    coefs = np.vstack([f.coef for f in fits])            # (m, p)
    q_bar = coefs.mean(axis=0)                            # (p,)
    within = np.mean(np.stack([f.cov for f in fits], axis=0), axis=0)  # U_bar
    if m < 2:
        # Guarded by the public entry point, but keep the math honest.
        between = np.zeros((p, p))
    else:
        deviations = coefs - q_bar
        between = (deviations.T @ deviations) / (m - 1)  # B
    total = within + (1.0 + 1.0 / m) * between            # T

    diag_total = np.diag(total)
    diag_between = np.diag(between)
    diag_within = np.diag(within)
    # Relative increase in variance due to nonresponse (Rubin's lambda).
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = (
            (1.0 + 1.0 / m) * diag_between / diag_total
        )
        lam = np.where(diag_total > 0, lam, 0.0)
        fmi = (
            (1.0 + 1.0 / m) * diag_between
            / (diag_within + (1.0 + 1.0 / m) * diag_between)
        )
        fmi = np.where(
            diag_within + (1.0 + 1.0 / m) * diag_between > 0, fmi, 0.0
        )

    # Classic Rubin degrees of freedom, per coefficient.
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = (1.0 + 1.0 / m) * diag_between
        df_rubin = np.where(
            denom > 0,
            (m - 1) * (1.0 + diag_within / denom) ** 2,
            np.inf,
        )

    # Barnard-Rubin small-sample adjustment: fold the residual-df of the
    # complete-data fits into Rubin's df. All versions share n and p here.
    nu_complete = np.array([f.residual_df for f in fits], dtype=float)
    nu_star = nu_complete.mean()
    # df_obs = (p_special - p_special*lam + 1)/(p_special - p_special*lam + 3)
    # * nu_star, with the model dimension equal to the number of columns.
    p_special = float(p)
    df_obs_factor = (p_special + 1.0) / (p_special + 3.0)
    df_obs = df_obs_factor * (1.0 - lam) * nu_star
    with np.errstate(divide="ignore", invalid="ignore"):
        df_adj = np.where(
            (df_rubin < np.inf) & (df_obs > 0),
            (df_rubin * df_obs) / (df_rubin + df_obs),
            df_obs,
        )
    # Keep values finite and well behaved.
    df_adj = np.where(np.isfinite(df_adj), df_adj, df_obs)
    df_rubin = np.where(np.isfinite(df_rubin), df_rubin, np.inf)
    return {
        "coef": q_bar,
        "within": within,
        "between": between,
        "total": total,
        "lambda": lam,
        "fmi": fmi,
        "df_rubin": df_rubin,
        "df_barnard_rubin": df_adj,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def pool_ols_rubin(
    formula,
    feature_tables,
    response_table,
    id_col,
    eval_env=0,
    versions=None,
):
    """Pool equal-weight OLS fits over multiply-imputed feature tables.

    :arg formula: a Patsy formula. Either a full formula ``"y ~ ..."`` whose
      left-hand side names the outcome column in ``response_table``, or a
      right-hand-side-only formula ``"x1 + ..."`` (the outcome is then taken
      from the response table's sole non-ID column).
    :arg feature_tables: a non-empty sequence of imputed feature tables, each
      a pandas DataFrame (or dict-like) containing ``id_col`` plus every raw
      predictor referenced by the formula. At least two are expected for
      meaningful between-imputation variance.
    :arg response_table: the single observed-response table containing
      ``id_col`` and the outcome column.
    :arg id_col: name of the case-ID column shared by every table.
    :arg eval_env: Patsy evaluation environment for names not found in the
      tables; defaults to the caller's namespace, matching :func:`patsy.dmatrix`.
    :arg versions: optional labels for the imputation tables (defaults to
      ``0, 1, ...``).
    :returns: :class:`RubinPooledOLS`.
    :raises PoolingError: if no complete cases remain, any version is rank
      deficient, any version lacks residual df, or inputs are inconsistent.
    """
    if pd is None:  # pragma: no cover
        raise PoolingError("pandas is required but is not installed")
    if not isinstance(feature_tables, (list, tuple)) or not feature_tables:
        raise PoolingError("feature_tables must be a non-empty list of tables")
    m = len(feature_tables)
    if m < 2:
        raise PoolingError(
            "at least two imputed feature tables are required for Rubin's "
            "between-imputation variance (got %d)" % m
        )
    if versions is None:
        versions = list(range(m))
    if len(versions) != m:
        raise PoolingError("versions labels must match the number of tables")

    eval_env = EvalEnvironment.capture(eval_env, reference=1)

    # --- resolve the outcome name from the formula ---------------------
    if not isinstance(formula, str):
        raise PoolingError("formula must be a Patsy formula string")
    desc = ModelDesc.from_formula(formula)
    lhs_terms = desc.lhs_termlist
    if len(lhs_terms) > 1 or any(len(t.factors) != 1 for t in lhs_terms):
        raise PoolingError(
            "exactly one scalar outcome on the left-hand side is supported"
        )
    if lhs_terms:
        lhs_factor = lhs_terms[0].factors[0]
        outcome_name = lhs_factor.code
    else:
        non_id = [c for c in response_table.columns if c != id_col]
        if len(non_id) != 1:
            raise PoolingError(
                "right-hand-side-only formula given but the response table "
                "has %d non-ID columns; use 'y ~ ...' to name the outcome"
                % len(non_id)
            )
        outcome_name = non_id[0]

    # --- coerce and validate every table -------------------------------
    response_frame, _ = _coerce_frame(response_table)
    feature_frames = []
    for i, tab in enumerate(feature_tables):
        frame, _ = _coerce_frame(tab)
        if id_col not in frame.columns:
            raise PoolingError(
                "feature table %r is missing the ID column %r"
                % (versions[i], id_col)
            )
        _check_id_unique(frame, id_col, "feature table %r" % (versions[i],))
        feature_frames.append(frame)
    if id_col not in response_frame.columns:
        raise PoolingError("response table is missing the ID column %r" % id_col)
    _check_id_unique(response_frame, id_col, "response table")
    if outcome_name not in response_frame.columns:
        raise PoolingError(
            "response table is missing the outcome column %r" % (outcome_name,)
        )

    labels = ["response table"] + ["imputation %r" % v for v in versions]
    frames = [response_frame] + feature_frames
    cases = _align_cases(frames, id_col, labels, outcome_name)
    if cases.n_included == 0:
        raise PoolingError(
            "no cases are complete in every table; nothing to fit. Inspect "
            "RubinPooledOLS.cases (or the alignment report) for details."
        )

    included_ids = cases.included_ids

    # --- build one frozen DesignInfo on the pooled complete cases ------
    # Each imputation contributes its rows; the stateful transforms see the
    # union of all versions exactly once, and categorical levels are learned
    # across every version. This single DesignInfo is the only one ever built.
    aligned_features = [
        frame[frame[id_col].isin(included_ids)]
        .sort_values(id_col)
        .reset_index(drop=True)
        for frame in feature_frames
    ]
    # Defensive: identical ID ordering across versions.
    order_ids = aligned_features[0][id_col].tolist()
    for frame in aligned_features[1:]:
        if frame[id_col].tolist() != order_ids:
            raise PoolingError(  # pragma: no cover - internal invariant
                "internal alignment error: case ordering diverged between versions"
            )
    pooled_features = pd.concat(aligned_features, ignore_index=True)

    aligned_response = (
        response_frame[response_frame[id_col].isin(included_ids)]
        .sort_values(id_col)
        .reset_index(drop=True)
    )
    if aligned_response[id_col].tolist() != order_ids:
        raise PoolingError(  # pragma: no cover - internal invariant
            "internal alignment error: response ordering diverged"
        )
    # Repeat the observed outcome once per imputation row for the one-time
    # design-learning call.
    pooled_learning = pooled_features.copy()
    pooled_learning[outcome_name] = np.tile(
        aligned_response[outcome_name].to_numpy(dtype=float), m
    )

    full_formula = formula if lhs_terms else "%s ~ %s" % (
        outcome_name,
        formula.strip(),
    )
    try:
        y_learn, x_learn = dmatrices(
            full_formula,
            pooled_learning,
            eval_env=eval_env,
            NA_action="raise",
            return_type="matrix",
        )
    except PatsyError as exc:
        raise PoolingError(
            "failed to learn a single DesignInfo from the pooled complete "
            "cases: %s" % (exc,)
        ) from exc
    design_info = x_learn.design_info
    column_names = list(design_info.column_names)
    p = len(column_names)

    # --- build each version's matrix with the frozen DesignInfo + OLS --
    y_common = aligned_response[outcome_name].to_numpy(dtype=float)
    if np.isnan(y_common).any():  # pragma: no cover - alignment guarantees this
        raise PoolingError("outcome contains NaN among the aligned cases")
    fits = []
    for version, frame in zip(versions, aligned_features):
        try:
            (x_mat,) = build_design_matrices(
                [design_info], frame, NA_action="raise", return_type="matrix"
            )
        except PatsyError as exc:
            raise PoolingError(
                "could not rebuild the design matrix for imputation %r from "
                "the frozen DesignInfo: %s" % (version, exc)
            ) from exc
        if x_mat.shape[1] != p:
            raise PoolingError(  # pragma: no cover - guaranteed by DesignInfo
                "imputation %r produced %d columns but the frozen design has "
                "%d; per-version re-engineering is not allowed"
                % (version, x_mat.shape[1], p)
            )
        fits.append(_fit_ols(version, x_mat, y_common))

    pooled = _rubin_pool(fits, m, p)

    return RubinPooledOLS(
        formula=full_formula,
        m=m,
        column_names=column_names,
        design_info=design_info,
        outcome_name=outcome_name,
        id_col=id_col,
        cases=cases,
        fits=fits,
        coef=pooled["coef"],
        within_cov=pooled["within"],
        between_cov=pooled["between"],
        total_cov=pooled["total"],
        relative_increase_variance=pooled["lambda"],
        fraction_missing_info=pooled["fmi"],
        df_rubin=pooled["df_rubin"],
        df_barnard_rubin=pooled["df_barnard_rubin"],
    )
