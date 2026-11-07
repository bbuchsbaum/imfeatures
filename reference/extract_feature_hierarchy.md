# Extract a named feature hierarchy from images

Runs one batched multi-output forward pass and applies a per-stage
pooler. Convolutional activations are converted to NHWC before pooling.
The virtual tap `classifier_logits` is computed from `fc2` and the
ImageNet classifier weights without applying softmax.

## Usage

``` r
extract_feature_hierarchy(
  impaths,
  spec,
  model = NULL,
  batch_size = 16L,
  layout = c("nhwc", "nchw")
)
```

## Arguments

- impaths:

  Character vector of image paths, or a directory containing
  `jpg`/`jpeg`/`png` files.

- spec:

  A
  [`vgg_hierarchy_spec`](https://bbuchsbaum.github.io/imfeatures/reference/vgg_hierarchy_spec.md)
  or other `feature_hierarchy_spec`.

- model:

  Optional preloaded Keras model. If `NULL`, the spec architecture is
  loaded with ImageNet weights.

- batch_size:

  Images per forward pass.

- layout:

  Layout of 4-D backend outputs; `"nhwc"` for Keras.

## Value

A
[`as_feature_hierarchy`](https://bbuchsbaum.github.io/imfeatures/reference/as_feature_hierarchy.md)
object.
