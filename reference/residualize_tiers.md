# Hierarchical residualization of tiered feature matrices with PCA

Compatibility wrapper around
[`fit_hierarchy_transform`](https://bbuchsbaum.github.io/imfeatures/reference/fit_hierarchy_transform.md).
Given a named list of feature matrices, it compresses each tier with PCA
and residualizes later tiers against earlier tiers using a stored
cross-tier mapping. New observations are transformed inductively: each
row depends only on that row's own tier values and the training-fitted
maps.

## Usage

``` r
residualize_tiers(
  feature_list,
  numpcs = NULL,
  pca_method = c("stats", "eigencore"),
  svd_tol = 1e-07,
  scale_scores = TRUE
)
```

## Arguments

- feature_list:

  Named list of numeric matrices, each of dimension N_samples × P_i.

- numpcs:

  Integer scalar or numeric vector, or NULL. If NULL (default), uses up
  to 50 PCs per tier (or fewer, if a tier has \<50 features).

- pca_method:

  Character string, one of "stats" (default, uses
  [`stats::prcomp`](https://rdrr.io/r/stats/prcomp.html)) or "eigencore"
  (uses
  [`eigencore::svd_partial`](https://bbuchsbaum.github.io/eigencore/reference/svd_partial.html)).

- svd_tol:

  Numeric tolerance factor used in determining effective rank via SVD
  for sequential residualization. Singular values `s_i` are considered
  non-zero if `s_i > svd_tol * max(dim(H)) * s[1]`, where `H` is the
  matrix being decomposed and `s[1]` is its largest singular value.
  Default is 1e-7.

- scale_scores:

  Logical. If TRUE (default), PC scores for each tier are z-scored
  (scaled to unit standard deviation, no centering as PCA already
  centers) before residualization. SDs for scaling in `predict` are
  taken from the training data.

## Value

An object of class `c("residualized_tiers", "hierarchy_transform")`, a
list with components:

- pca:

  Named list of PCA objects. If `scale_scores = TRUE`, this will also
  contain `sds_for_scaling` for each tier.

- pc_scores_raw:

  Named list of raw PC score matrices (N_samples × numpcs\[i\]) before
  residualization (but after optional scaling if `scale_scores=TRUE`).

- residuals:

  Named list of final residualized PC matrices.

- projection_bases:

  Named list. For each tier (except the first), the training-set
  orthonormal basis of preceding scores, stored as a diagnostic only.
  Prediction uses stored coefficient maps, not these bases.

- numpcs:

  Integer vector of number of PCs per tier.

- tiers:

  Character vector of tier names.

- svd_tol_info:

  List containing the `svd_tol` value used and a description of the
  tolerance formula.

- scale_scores:

  Logical value of the `scale_scores` argument used.

An attribute `total_rank_kept` (sum of `numpcs`) is also attached.

## Details

Initial PCA is performed on each tier. If `scale_scores = TRUE`, the
resulting PC scores are then z-scored (column-wise, per tier). These
(optionally scaled) PC scores are then sequentially residualized.
Residualization stores the least-squares map from earlier-tier scores to
the current tier, so [`predict()`](https://rdrr.io/r/stats/predict.html)
does not re-estimate that map from the new batch. Exact orthogonality is
expected on the training observations for unregularized residualization
and is not expected on new data. All matrices must contain only finite
values; the function stops if any NA or Inf values are detected.
