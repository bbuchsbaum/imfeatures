# Apply a pooler to an activation tensor

Reduces a batch of activations to one feature row per image. This is the
function
[`extract_feature_hierarchy`](https://bbuchsbaum.github.io/imfeatures/reference/extract_feature_hierarchy.md)
calls for each stage; you can also call it directly on activations from
any backend.

## Usage

``` r
pool_activations(pooler, a, layout = "nhwc", ...)
```

## Arguments

- pooler:

  An `imfeatures_pooler`; see
  [`poolers`](https://bbuchsbaum.github.io/imfeatures/reference/poolers.md).

- a:

  Numeric matrix (`N x P`) or 4-D array with one image per slice of the
  first dimension.

- layout:

  Tensor layout for 4-D inputs: `"nhwc"` (Keras, default) or `"nchw"`
  (PyTorch). See
  [`as_nhwc`](https://bbuchsbaum.github.io/imfeatures/reference/as_nhwc.md).

- ...:

  Unused.

## Value

An `N x P` numeric matrix without column names. `P` is the number of
channels times the pooler's columns per channel. Use
[`pooler_feature_names`](https://bbuchsbaum.github.io/imfeatures/reference/pooler_feature_names.md)
to label the columns.

## Details

A 4-D array is first put in `N x H x W x C` order (see
[`as_nhwc`](https://bbuchsbaum.github.io/imfeatures/reference/as_nhwc.md))
and then pooled over height and width. A 2-D `N x P` matrix has no
spatial axes and is returned unchanged by every pooler, which is how
dense layers pass through a spec.

## See also

[`poolers`](https://bbuchsbaum.github.io/imfeatures/reference/poolers.md),
[`pooler_feature_names`](https://bbuchsbaum.github.io/imfeatures/reference/pooler_feature_names.md).

## Examples

``` r
# PyTorch-style activations: 2 images, 3 channels, 8 x 8 map
nchw <- array(rnorm(2 * 3 * 8 * 8), dim = c(2, 3, 8, 8))
feats <- pool_activations(adaptive_grid_pool(2, 2), nchw, layout = "nchw")
colnames(feats) <- pooler_feature_names(adaptive_grid_pool(2, 2), "layer4", 3)
dim(feats)
#> [1]  2 12
head(colnames(feats))
#> [1] "layer4::channel_001::grid_y1_x1" "layer4::channel_001::grid_y1_x2"
#> [3] "layer4::channel_001::grid_y2_x1" "layer4::channel_001::grid_y2_x2"
#> [5] "layer4::channel_002::grid_y1_x1" "layer4::channel_002::grid_y1_x2"
```
