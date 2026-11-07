# Specify per-stage PCA compression

Specify per-stage PCA compression

## Usage

``` r
block_pca(
  rank = NULL,
  method = c("stats", "eigencore"),
  feature_scale = c("sd", "center", "none"),
  score_weighting = c("singular", "whitened")
)
```

## Arguments

- rank:

  Integer scalar, named integer vector, or `NULL`. A scalar is recycled
  across stages. `NULL` keeps up to 50 components per stage (or fewer
  when a stage has fewer columns).

- method:

  Character; `"stats"` uses
  [`stats::prcomp`](https://rdrr.io/r/stats/prcomp.html), `"eigencore"`
  uses
  [`eigencore::svd_partial`](https://bbuchsbaum.github.io/eigencore/reference/svd_partial.html)
  when installed.

- feature_scale:

  How to standardize raw columns before PCA: `"sd"` centers and scales,
  `"center"` centers only, `"none"` leaves columns unchanged.

- score_weighting:

  `"singular"` keeps PCA scores as \\UD\\; `"whitened"` scales each
  retained score to unit training SD.

## Value

An object of class `block_pca`.
