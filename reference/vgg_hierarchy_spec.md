# Specify a VGG feature hierarchy

Specify a VGG feature hierarchy

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

  Named list of `imfeatures_pooler` objects, one per stage. A single
  pooler is recycled across stages. Dense taps default to
  [`identity_pool`](https://bbuchsbaum.github.io/imfeatures/reference/identity_pool.md);
  convolutional taps default to
  [`global_average_pool`](https://bbuchsbaum.github.io/imfeatures/reference/global_average_pool.md).

- target_size:

  Image resize dimensions passed to Keras.

## Value

An object of class `c("vgg_hierarchy_spec", "feature_hierarchy_spec")`.
