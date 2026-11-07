# Compile a hierarchy transformation recipe

A recipe records stage order, per-stage compression, an optional
innovation decomposition, and an optional post-residual compression.

## Usage

``` r
hierarchy_recipe(
  order,
  compress,
  decomposition = NULL,
  post_compress = NULL,
  name_prefix = "",
  innovation_suffix = TRUE
)
```

## Arguments

- order:

  Character vector of stage names, in residualization order.

- compress:

  A
  [`block_pca`](https://bbuchsbaum.github.io/imfeatures/reference/block_pca.md)
  specification applied to each raw stage before decomposition.

- decomposition:

  An
  [`ordered_innovation`](https://bbuchsbaum.github.io/imfeatures/reference/ordered_innovation.md)
  specification, or `NULL` to emit compressed scores only.

- post_compress:

  Optional
  [`block_pca`](https://bbuchsbaum.github.io/imfeatures/reference/block_pca.md)
  applied to each innovation (or compressed score if `decomposition` is
  `NULL`).

- name_prefix:

  Optional prefix for output band names, for example `"vgg"` yields
  `vgg_b1`, `vgg_b2_innovation`, ...

- innovation_suffix:

  Logical; if `TRUE` (default), stages after the first are named
  `*_innovation` when a decomposition is present.

## Value

An object of class `hierarchy_recipe`.

## Examples

``` r
hierarchy_recipe(
  order = c("b1", "b2"),
  compress = block_pca(rank = 8),
  decomposition = ordered_innovation(method = "svd")
)
#> Hierarchy recipe
#>   Order: b1 -> b2
#>   Compress: block_pca(rank=8, scale=sd, weighting=singular)
#>   Decomposition: ordered_innovation(method=svd)
#>   Post-compress: none
```
