# Predict method for residualized_tiers

Applies a trained residualized_tiers transformation (Sequential SVD-QR
method) to new data.

## Usage

``` r
# S3 method for class 'residualized_tiers'
predict(object, newdata, ...)
```

## Arguments

- object:

  An object of class `residualized_tiers` produced by
  [`residualize_tiers()`](https://bbuchsbaum.github.io/imfeatures/reference/residualize_tiers.md).

- newdata:

  Named list of matrices with the same tier names and feature columns as
  the training data.

- ...:

  Additional arguments (currently ignored).

## Value

Named list of residualized PC matrices for each tier in `newdata`.
