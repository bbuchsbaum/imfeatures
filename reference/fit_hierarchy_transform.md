# Fit an inductive hierarchy transformation

Learns per-stage compression, cross-stage residualization maps, and
optional post-residual compression from training blocks. The fitted
object stores coefficient matrices, not observation-space projection
bases, so new rows are transformed independently of batch membership.

## Usage

``` r
fit_hierarchy_transform(recipe, blocks, block_id = NULL)
```

## Arguments

- recipe:

  A
  [`hierarchy_recipe`](https://bbuchsbaum.github.io/imfeatures/reference/hierarchy_recipe.md).

- blocks:

  A named list of numeric matrices or a
  [`as_feature_hierarchy`](https://bbuchsbaum.github.io/imfeatures/reference/as_feature_hierarchy.md)
  object. Names must include every stage in `recipe$order`.

- block_id:

  Optional blocking factor reserved for future `lambda = "blocked_cv"`
  support. Currently unused.

## Value

An object of class `hierarchy_transform`.

## Examples

``` r
set.seed(1)
blocks <- list(
  b1 = matrix(rnorm(80), 40, 2, dimnames = list(NULL, c("a", "b"))),
  b2 = matrix(rnorm(120), 40, 3, dimnames = list(NULL, c("x", "y", "z")))
)
recipe <- hierarchy_recipe(
  order = c("b1", "b2"),
  compress = block_pca(rank = 2),
  decomposition = ordered_innovation(method = "svd")
)
fit <- fit_hierarchy_transform(recipe, blocks)
transformed <- predict(fit, blocks)
```
