# Spatial poolers for convolutional activations

A pooler turns a 4-D activation tensor (`N` images x `H` x `W` x `C`
channels) into an `N x P` feature matrix. The constructors below only
describe *how* to pool;
[`pool_activations`](https://bbuchsbaum.github.io/imfeatures/reference/pool_activations.md)
does the work, and
[`vgg_hierarchy_spec`](https://bbuchsbaum.github.io/imfeatures/reference/vgg_hierarchy_spec.md)
assigns one pooler to each network stage.

## Usage

``` r
global_average_pool()

global_max_pool()

identity_pool()
```

## Value

An object of class `imfeatures_pooler`: a list holding the pooler `type`
and its parameters.

## Details

|  |  |  |
|----|----|----|
| **Constructor** | **Columns per channel** | **What is kept** |
| `global_average_pool()` | 1 | Mean over all positions |
| `global_max_pool()` | 1 | Maximum over all positions |
| `adaptive_grid_pool(ny, nx)` | `ny * nx` | Mean in each cell of a coarse grid |
| `dct_pool(ny, nx)` | `ny * nx` | Mean plus low spatial frequencies |
| `radial_pool(breaks)` | `length(breaks) - 1` | Mean in each centre-to-periphery ring |
| `identity_pool()` | `H * W` | Every position (flattened) |

Every pooler writes its columns channel by channel: all spatial features
of channel 1, then all of channel 2, and so on.
[`pooler_feature_names`](https://bbuchsbaum.github.io/imfeatures/reference/pooler_feature_names.md)
returns the matching column names.

Dense activations (`N x P` matrices, such as `fc1` or logits) have no
spatial axes. Every pooler returns them unchanged.

## Functions

- `global_average_pool()`: Mean over height and width; one column per
  channel.

- `global_max_pool()`: Maximum over height and width; one column per
  channel.

- `identity_pool()`: No pooling. Dense matrices pass through unchanged;
  4-D maps are flattened to `H * W * C` columns ordered by channel, then
  width, then height (height varies fastest).

## Choosing a pooler

- Use `global_average_pool()` when you want one number per channel and
  do not care where in the image a feature occurred.

- Use `global_max_pool()` for the same size when presence of a feature
  matters more than its extent.

- Use
  [`adaptive_grid_pool()`](https://bbuchsbaum.github.io/imfeatures/reference/adaptive_grid_pool.md)
  to keep coarse location in non-negative, directly interpretable cells.

- Use
  [`dct_pool()`](https://bbuchsbaum.github.io/imfeatures/reference/dct_pool.md)
  to keep coarse layout as a mean plus smooth spatial contrasts. The
  first column per channel equals global average pooling, so the
  remaining columns add layout to the usual summary.

- Use
  [`radial_pool()`](https://bbuchsbaum.github.io/imfeatures/reference/radial_pool.md)
  when eccentricity (centre versus periphery) is the spatial variable of
  interest.

- Use `identity_pool()` for dense layers, or for small maps where you
  want every position.

## See also

[`pool_activations`](https://bbuchsbaum.github.io/imfeatures/reference/pool_activations.md),
[`pooler_feature_names`](https://bbuchsbaum.github.io/imfeatures/reference/pooler_feature_names.md),
[`vgg_hierarchy_spec`](https://bbuchsbaum.github.io/imfeatures/reference/vgg_hierarchy_spec.md),
and
[`vignette("spatial-pooling", package = "imfeatures")`](https://bbuchsbaum.github.io/imfeatures/articles/spatial-pooling.md).

## Examples

``` r
# Two images, an 8 x 8 map, three channels (N x H x W x C)
a <- array(rnorm(2 * 8 * 8 * 3), dim = c(2, 8, 8, 3))

dim(pool_activations(global_average_pool(), a))    # 2 x 3
#> [1] 2 3
dim(pool_activations(adaptive_grid_pool(2, 2), a)) # 2 x 12
#> [1]  2 12
dim(pool_activations(dct_pool(3, 3), a))           # 2 x 27
#> [1]  2 27
dim(pool_activations(radial_pool(), a))            # 2 x 9
#> [1] 2 9
dim(pool_activations(identity_pool(), a))          # 2 x 192
#> [1]   2 192

# The first DCT column of each channel is the global average
all.equal(
  pool_activations(dct_pool(3, 3), a)[, 1],
  pool_activations(global_average_pool(), a)[, 1]
)
#> [1] TRUE
```
