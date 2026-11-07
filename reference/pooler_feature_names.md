# Feature names for a pooled stage

Feature names for a pooled stage

## Usage

``` r
pooler_feature_names(pooler, stage, n_channels, spatial = NULL)
```

## Arguments

- pooler:

  An `imfeatures_pooler`.

- stage:

  Stage name used as the first token.

- n_channels:

  Number of channels, or the number of dense features when `spatial` is
  `NULL` and the pooler is identity.

- spatial:

  Optional integer length-2 spatial size `c(H, W)` for identity
  flattening of 4-D maps.
