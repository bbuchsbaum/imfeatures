# Adaptive rectangular grid pooling

Averages each channel within the cells of an `ny x nx` grid laid over
the activation map. The grid adapts to the map: the same pooler works on
a 224 x 224 map and on a 14 x 14 map.

## Usage

``` r
adaptive_grid_pool(ny, nx)
```

## Arguments

- ny, nx:

  Number of bins along height and width. Neither may exceed the size of
  the activation map it is applied to.

## Value

An `imfeatures_pooler` that yields `ny * nx` columns per channel.

## Details

Each spatial axis is cut into contiguous, non-overlapping bins that
together cover every position. When the map size is a multiple of the
number of bins, all bins have the same size and the mean of the cells
equals global average pooling. Otherwise the bins differ in size (14
positions in 4 bins gives 4, 3, 3, 4), and the mean of the cells is only
close to the global average. This differs from PyTorch's
`AdaptiveAvgPool2d`, which lets neighbouring windows overlap in that
case.

Columns are ordered by channel, then grid row (top to bottom), then grid
column (left to right). Names take the form
`stage::channel_001::grid_y1_x1`, with one-based cell indices.

## See also

[`poolers`](https://bbuchsbaum.github.io/imfeatures/reference/poolers.md)
for the other pooling options.

## Examples

``` r
# One image, a 4 x 4 map, one channel, with a bright top-right quadrant
a <- array(0, dim = c(1, 4, 4, 1))
a[1, 1:2, 3:4, 1] <- 1
out <- pool_activations(adaptive_grid_pool(2, 2), a)
colnames(out) <- pooler_feature_names(adaptive_grid_pool(2, 2), "b5", 1)
out
#>      b5::channel_001::grid_y1_x1 b5::channel_001::grid_y1_x2
#> [1,]                           0                           1
#>      b5::channel_001::grid_y2_x1 b5::channel_001::grid_y2_x2
#> [1,]                           0                           0
```
