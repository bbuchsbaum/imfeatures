# Hierarchical residualization of tiered feature matrices with PCA

Given a named list of feature matrices (e.g., low-, mid-, high-tier
features), perform PCA within each tier to reduce to a specified number
of components, then sequentially residualize each tier's PCs against all
previous tiers using an SVD-based rank-aware approach. Optionally, PC
scores can be z-scored.

## Usage

``` r
residualize_tiers(
  feature_list,
  numpcs = NULL,
  pca_method = c("stats", "irlba"),
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
  [`stats::prcomp`](https://rdrr.io/r/stats/prcomp.html)) or "irlba"
  (uses
  [`irlba::prcomp_irlba`](https://rdrr.io/pkg/irlba/man/prcomp_irlba.html)).

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

An object of class `residualized_tiers`, a list with components:

- pca:

  Named list of PCA objects. If `scale_scores = TRUE`, this will also
  contain `sds_for_scaling` for each tier.

- pc_scores_raw:

  Named list of raw PC score matrices (N_samples × numpcs\[i\]) before
  residualization (but after optional scaling if `scale_scores=TRUE`).

- residuals:

  Named list of final residualized PC matrices.

- projection_bases:

  Named list. For each tier (except the first), stores the orthonormal
  basis matrix Q (from SVD of preceding tiers' cumulative
  non-residualized PC scores, truncated by rank), or NULL.

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
(optionally scaled) PC scores are then sequentially residualized. The
orthogonalization uses SVD to form a rank-aware basis of the preceding
tiers' cumulative scores to ensure numerical stability, using the
`svd_tol` parameter in a LAPACK-style manner. Note: For applications
like fMRI analysis where data might be processed in folds (e.g., for
cross-validation), z-scoring should ideally be performed based on
training fold statistics and then applied to test folds. This function,
when applied to a whole dataset, uses statistics from the entire input
for z-scoring. All matrices must contain only finite values; the
function stops if any NA or Inf values are detected.
