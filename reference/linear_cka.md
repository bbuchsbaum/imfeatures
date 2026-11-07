# Linear-kernel CKA between two observation-by-feature matrices

Linear-kernel CKA between two observation-by-feature matrices

## Usage

``` r
linear_cka(X, Y)
```

## Arguments

- X, Y:

  Numeric matrices with the same number of rows.

## Value

A scalar in \\\[0, 1\]\\, or `NA` if either matrix has no columns or a
kernel is degenerate.
