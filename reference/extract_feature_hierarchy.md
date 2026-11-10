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
object. Its `schema` records the taps, the poolers, and the layout that
produced the blocks.

## Details

Spatial pooling is set per stage in `spec`; see
[`poolers`](https://bbuchsbaum.github.io/imfeatures/reference/poolers.md)
for the options. Each block has one row per image and one column per
pooled feature, named as described in
[`pooler_feature_names`](https://bbuchsbaum.github.io/imfeatures/reference/pooler_feature_names.md).
Full-resolution maps are pooled batch by batch and never held for all
images at once, so memory use depends on `batch_size` and on the pooled
width, not on the number of images.

## See also

[`vgg_hierarchy_spec`](https://bbuchsbaum.github.io/imfeatures/reference/vgg_hierarchy_spec.md),
[`poolers`](https://bbuchsbaum.github.io/imfeatures/reference/poolers.md),
and
[`vignette("spatial-pooling", package = "imfeatures")`](https://bbuchsbaum.github.io/imfeatures/articles/spatial-pooling.md).

## Examples

``` r
if (FALSE) { # \dontrun{
imgs <- list.files("stimuli", pattern = "\\.jpg$", full.names = TRUE)

# Depth-adaptive DCT pooling at the end of each VGG-16 block
raw <- extract_feature_hierarchy(imgs, spec = vgg16_block_end_spec())
vapply(raw$blocks, ncol, integer(1))

# The same taps, with a 3 x 3 grid on every convolutional stage
taps <- vgg_block_end_taps("vgg16")
pooling <- lapply(taps, function(tap) adaptive_grid_pool(3, 3))
pooling[c("fc1", "logits")] <- list(identity_pool(), identity_pool())
spec <- vgg_hierarchy_spec("vgg16", taps = taps, pooling = pooling)
raw_grid <- extract_feature_hierarchy(imgs, spec = spec)
} # }
```
