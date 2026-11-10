# Specify a VGG feature hierarchy

A spec names the layers to tap and says how each one is pooled over
space. Pass it to
[`extract_feature_hierarchy`](https://bbuchsbaum.github.io/imfeatures/reference/extract_feature_hierarchy.md)
to get one feature block per stage.

## Usage

``` r
vgg_hierarchy_spec(
  architecture = c("vgg16", "vgg19"),
  taps = NULL,
  pooling = NULL,
  target_size = c(224, 224)
)
```

## Arguments

- architecture:

  `"vgg16"` or `"vgg19"`.

- taps:

  Named character vector mapping stage names to Keras layer names. The
  virtual tap `"classifier_logits"` requests pre-softmax ImageNet
  logits. Defaults to
  [`vgg_block_end_taps`](https://bbuchsbaum.github.io/imfeatures/reference/vgg_block_end_taps.md).

- pooling:

  How to pool each stage over space; see
  [`poolers`](https://bbuchsbaum.github.io/imfeatures/reference/poolers.md)
  for the options. Supply one of:

  - `NULL` (default):
    [`global_average_pool`](https://bbuchsbaum.github.io/imfeatures/reference/poolers.md)
    for convolutional taps and
    [`identity_pool`](https://bbuchsbaum.github.io/imfeatures/reference/poolers.md)
    for dense taps.

  - A single pooler, used for every stage. Dense stages have no spatial
    axes and pass through unchanged whatever the pooler.

  - A named list with one pooler per stage. Names must cover every stage
    in `taps`.

- target_size:

  Image resize dimensions passed to Keras.

## Value

An object of class `c("vgg_hierarchy_spec", "feature_hierarchy_spec")`.

## See also

[`vgg16_block_end_spec`](https://bbuchsbaum.github.io/imfeatures/reference/vgg16_block_end_spec.md)
for a ready-made spec,
[`poolers`](https://bbuchsbaum.github.io/imfeatures/reference/poolers.md),
and
[`vignette("spatial-pooling", package = "imfeatures")`](https://bbuchsbaum.github.io/imfeatures/articles/spatial-pooling.md).

## Examples

``` r
# Default: global average pooling on convolutional stages
vgg_hierarchy_spec("vgg16")
#> VGG hierarchy spec (vgg16)
#>   Stages: b1, b2, b3, b4, b5, fc1, logits
#>   Taps:   block1_conv2, block2_conv2, block3_conv3, block4_conv3, block5_conv3, fc1, classifier_logits
#>   Pooling:
#>     b1      Global average pool
#>     b2      Global average pool
#>     b3      Global average pool
#>     b4      Global average pool
#>     b5      Global average pool
#>     fc1     Identity pool
#>     logits  Identity pool

# One pooler for every stage
vgg_hierarchy_spec("vgg16", pooling = adaptive_grid_pool(2, 2))
#> VGG hierarchy spec (vgg16)
#>   Stages: b1, b2, b3, b4, b5, fc1, logits
#>   Taps:   block1_conv2, block2_conv2, block3_conv3, block4_conv3, block5_conv3, fc1, classifier_logits
#>   Pooling:
#>     b1      Adaptive grid pool 2 x 2
#>     b2      Adaptive grid pool 2 x 2
#>     b3      Adaptive grid pool 2 x 2
#>     b4      Adaptive grid pool 2 x 2
#>     b5      Adaptive grid pool 2 x 2
#>     fc1     Adaptive grid pool 2 x 2
#>     logits  Adaptive grid pool 2 x 2

# A different pooler per stage
vgg_hierarchy_spec(
  "vgg16",
  taps = c(b4 = "block4_conv3", b5 = "block5_conv3", fc1 = "fc1"),
  pooling = list(
    b4 = dct_pool(3, 3),
    b5 = radial_pool(),
    fc1 = identity_pool()
  )
)
#> VGG hierarchy spec (vgg16)
#>   Stages: b4, b5, fc1
#>   Taps:   block4_conv3, block5_conv3, fc1
#>   Pooling:
#>     b4   DCT pool 3 x 3 (zigzag)
#>     b5   Radial pool (0, 0.25, 0.5, 1)
#>     fc1  Identity pool
```
