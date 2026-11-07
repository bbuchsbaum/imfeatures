# Apply a fitted hierarchy transformation

Row-independent apply path for a
[`fit_hierarchy_transform`](https://bbuchsbaum.github.io/imfeatures/reference/fit_hierarchy_transform.md)
object. `apply_hierarchy_transform()` is the readable alias of
[`predict()`](https://rdrr.io/r/stats/predict.html).

## Usage

``` r
apply_hierarchy_transform(object, blocks, ...)

# S3 method for class 'hierarchy_transform'
predict(object, newdata, ...)
```

## Arguments

- object:

  A `hierarchy_transform`.

- blocks, newdata:

  New blocks with the same stages and columns as the training data. A
  named list or `feature_hierarchy`.

- ...:

  Unused.

## Value

A named list of numeric matrices, one per output band.
