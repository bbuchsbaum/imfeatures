# Feature names for a pooled stage

Builds the column names that match the output of
[`pool_activations`](https://bbuchsbaum.github.io/imfeatures/reference/pool_activations.md),
in the same order. Names have three tokens separated by `"::"`: the
stage, the zero-padded channel, and the spatial feature, for example
`b3::channel_017::dct_y0_x1`.

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

## Value

A character vector with one name per pooled column.

## Details

|  |  |
|----|----|
| **Pooler** | **Spatial token** |
| [`global_average_pool()`](https://bbuchsbaum.github.io/imfeatures/reference/poolers.md) | `gap` |
| [`global_max_pool()`](https://bbuchsbaum.github.io/imfeatures/reference/poolers.md) | `gmp` |
| [`adaptive_grid_pool()`](https://bbuchsbaum.github.io/imfeatures/reference/adaptive_grid_pool.md) | `grid_y<row>_x<col>` (one-based cell) |
| [`dct_pool()`](https://bbuchsbaum.github.io/imfeatures/reference/dct_pool.md) | `dct_y<v>_x<u>` (zero-based frequency) |
| [`radial_pool()`](https://bbuchsbaum.github.io/imfeatures/reference/radial_pool.md) | `radial_r<ring>` (one-based, centre outwards) |
| [`identity_pool()`](https://bbuchsbaum.github.io/imfeatures/reference/poolers.md) on a 4-D map | `y<row>_x<col>` (one-based position) |
| [`identity_pool()`](https://bbuchsbaum.github.io/imfeatures/reference/poolers.md) on a dense layer | `stage::f_001` (two tokens) |

## See also

[`pool_activations`](https://bbuchsbaum.github.io/imfeatures/reference/pool_activations.md),
[`poolers`](https://bbuchsbaum.github.io/imfeatures/reference/poolers.md).

## Examples

``` r
pooler_feature_names(global_average_pool(), "b5", 2)
#> [1] "b5::channel_001::gap" "b5::channel_002::gap"
pooler_feature_names(dct_pool(2, 2), "b5", 1)
#> [1] "b5::channel_001::dct_y0_x0" "b5::channel_001::dct_y0_x1"
#> [3] "b5::channel_001::dct_y1_x0" "b5::channel_001::dct_y1_x1"
pooler_feature_names(identity_pool(), "b5", 1, spatial = c(2, 2))
#> [1] "b5::channel_001::y1_x1" "b5::channel_001::y2_x1" "b5::channel_001::y1_x2"
#> [4] "b5::channel_001::y2_x2"
pooler_feature_names(identity_pool(), "fc1", 3)
#> [1] "fc1::f_001" "fc1::f_002" "fc1::f_003"
```
