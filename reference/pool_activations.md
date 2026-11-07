# Apply a pooler to an activation tensor

Apply a pooler to an activation tensor

## Usage

``` r
pool_activations(pooler, a, layout = "nhwc", ...)
```

## Arguments

- pooler:

  An `imfeatures_pooler`.

- a:

  Numeric matrix (`N × P`) or 4-D array.

- layout:

  Tensor layout for 4-D inputs; see
  [`as_nhwc`](https://bbuchsbaum.github.io/imfeatures/reference/as_nhwc.md).

- ...:

  Unused.

## Value

An `N × P` numeric matrix.
