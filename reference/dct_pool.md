# Low-frequency 2-D DCT pooling

Projects each channel's activation map onto the lowest `ny x nx` modes
of a two-dimensional discrete cosine transform (DCT-II). The result
summarizes *where* a channel is active as a mean plus a few smooth
spatial contrasts (left versus right, top versus bottom, centre versus
edges, and so on).

## Usage

``` r
dct_pool(ny, nx, order = c("zigzag", "raster"))
```

## Arguments

- ny, nx:

  Number of retained DCT modes along height and width. Neither may
  exceed the size of the activation map it is applied to.

- order:

  `"zigzag"` (default) or `"raster"`.

## Value

An `imfeatures_pooler` that yields `ny * nx` columns per channel.

## Details

Mode `(v, u)` has `v` half-cycles along height and `u` half-cycles along
width. All `ny * nx` modes with `v < ny` and `u < nx` are kept.

## Scale of the coefficients

The mode `(0, 0)` is rescaled to be the spatial mean, so the first
column of each channel equals
[`global_average_pool`](https://bbuchsbaum.github.io/imfeatures/reference/poolers.md).
All other modes use the orthonormal DCT-II basis. Their coefficients
grow with the square root of the number of positions: a unit-amplitude
cosine pattern gives a coefficient near 9.9 on a 14 x 14 map and near
158 on a 224 x 224 map. Columns from different modes or different stages
are therefore not on a common scale. Standardize the columns before you
combine them.
[`block_pca`](https://bbuchsbaum.github.io/imfeatures/reference/block_pca.md)
does so by default (`feature_scale = "sd"`).

## Column order and names

Columns are ordered by channel, then by mode. `order = "zigzag"` sorts
modes from low to high frequency along anti-diagonals, as in JPEG:
`(0,0), (0,1), (1,0), (2,0), (1,1), (0,2), ...`. `order = "raster"`
sorts them row by row. Both orders keep the same modes; only the column
order differs. Names take the form `stage::channel_001::dct_y0_x1`,
where `y` and `x` are the zero-based frequencies `v` and `u`.

## See also

[`poolers`](https://bbuchsbaum.github.io/imfeatures/reference/poolers.md)
for the other pooling options, and
[`vgg16_block_end_spec`](https://bbuchsbaum.github.io/imfeatures/reference/vgg16_block_end_spec.md)
for a ready-made VGG-16 spec that uses DCT pooling.

## Examples

``` r
# A channel that is active only on the left half of a 6 x 6 map
a <- array(0, dim = c(1, 6, 6, 1))
a[1, , 1:3, 1] <- 1
out <- pool_activations(dct_pool(2, 2), a)
colnames(out) <- pooler_feature_names(dct_pool(2, 2), "b3", 1)
round(out, 3)
#>      b3::channel_001::dct_y0_x0 b3::channel_001::dct_y0_x1
#> [1,]                        0.5                      2.732
#>      b3::channel_001::dct_y1_x0 b3::channel_001::dct_y1_x1
#> [1,]                          0                          0
# dct_y0_x0 is the mean (0.5); dct_y0_x1 carries the left-right contrast.
```
