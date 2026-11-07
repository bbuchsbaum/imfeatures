# Build a named feature-hierarchy object

Wraps a named list of aligned feature matrices (one per stage) into an
object that records sample identifiers, stage order, and optional schema
or provenance metadata.

## Usage

``` r
as_feature_hierarchy(
  blocks,
  sample_id = NULL,
  stage_order = NULL,
  schema = NULL,
  provenance = NULL
)
```

## Arguments

- blocks:

  Named list of numeric matrices, each with the same number of rows
  (observations). A `feature_hierarchy` is returned unchanged.

- sample_id:

  Optional vector of sample identifiers of length `nrow`. If omitted,
  row names of the first block are used, otherwise `seq_len(nrow)`.

- stage_order:

  Optional character vector giving the intended stage order. Must be a
  permutation of `names(blocks)`. Defaults to the list order.

- schema:

  Optional list describing column-name conventions or pooling.

- provenance:

  Optional list of extraction metadata.

## Value

An object of class `feature_hierarchy` with elements `blocks`,
`sample_id`, `stage_order`, `schema`, and `provenance`.

## Examples

``` r
blocks <- list(
  b1 = matrix(rnorm(20), 10, 2),
  b2 = matrix(rnorm(30), 10, 3)
)
as_feature_hierarchy(blocks)
#> Feature hierarchy
#>   Samples: 10
#>   Stages:  b1, b2
#>   Dims:    b1=2, b2=3
```
