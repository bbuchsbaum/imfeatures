# Sequential orthogonal residualization of score matrices

This helper takes a list of score matrices (each with the same number of
rows) and sequentially residualizes each matrix against the cumulative
scores of all previous matrices using an SVD-based rank-aware approach.

## Usage

``` r
.orthogonal_residuals(scores_list, svd_tol, return_projection_bases = TRUE)
```

## Arguments

- scores_list:

  Named list of numeric matrices.

- svd_tol:

  Numeric tolerance factor for determining effective rank in the SVD
  step.

- return_projection_bases:

  Logical; return the orthonormal projection bases used at each step?

## Value

If `return_projection_bases = TRUE`, a list with elements `residuals`
and `projection_bases`. Otherwise, just the list of residual matrices.
