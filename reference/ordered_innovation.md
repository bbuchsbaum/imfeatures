# Specify ordered innovation residualization

Fits a mapping from earlier-stage scores to the current stage on
training observations and stores the coefficient matrix so new rows can
be residualized independently.

## Usage

``` r
ordered_innovation(method = c("svd", "ridge"), tolerance = 1e-07, lambda = 0)
```

## Arguments

- method:

  `"svd"` uses a rank-aware least-squares mapping; `"ridge"` uses
  \\\\Z - HB\\\_F^2 + \lambda\\B\\\_F^2\\.

- tolerance:

  SVD rank tolerance factor, used as
  `s_i > tolerance * max(dim(H)) * s[1]`.

- lambda:

  Ridge penalty. A non-negative number, or the string `"blocked_cv"`
  (not yet implemented). Ignored when `method = "svd"` except that
  `"blocked_cv"` is still rejected.

## Value

An object of class `ordered_innovation`.
